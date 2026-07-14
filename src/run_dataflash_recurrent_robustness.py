#!/usr/bin/env python3
"""Run purged, multi-seed DataFlash recurrent feature ablations."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from dataflash_baseline import read_pos
from dataflash_physical_features import physical_feature_cube
from run_dataflash_recurrent_models import fit_predict
from run_dataflash_rolling_validation import block_indices, concatenate_blocks
from run_dataflash_sequence_baseline import (
    SequenceDataset,
    build_prediction_rows,
    make_sequence_dataset,
    write_prediction_rows,
)
from run_dataflash_sweep import load_series, metrics, parse_window


ABLATIONS = (
    "imu_raw",
    "imu_engineered",
    "imu_att",
    "imu_att_crt",
    "imu_att_baro",
    "imu_att_crt_motors",
    "all_direct",
    "all_physical",
)


@dataclass
class FoldSplit:
    fold: int
    val_block: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--window", default="5000:5000")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--purge-ms", type=float, default=10000.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260713, 20260714, 20260715, 20260716, 20260717])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows from --summary-csv.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_recurrent_robustness/summary.csv"),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("derived/predictions/dataflash_recurrent_purged"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/dataflash_recurrent_robustness.md"),
    )
    return parser.parse_args()


def feature_indices(names: list[str], ablation: str) -> np.ndarray:
    imu_raw = {
        "IMU.GyrX", "IMU.GyrY", "IMU.GyrZ", "IMU.AccX", "IMU.AccY", "IMU.AccZ",
    }
    imu_derived = {"gyro_norm", "acc_norm", "jerk_norm", "angular_acc_norm"}
    attitude = {"ATT.Roll", "ATT.Pitch", "ATT.att_sin_yaw", "ATT.att_cos_yaw"}
    linear = {"linear_acc_east", "linear_acc_north", "linear_acc_up", "linear_acc_norm"}
    baro = {"BARO.Alt", "BARO.CRt"}
    climb_rate = {"BARO.CRt"}
    motors = {
        "MOTB.ThrOut", "MOTB.ThLimit", "RCOU_motor_features.motor_mean_norm",
        "RCOU_motor_features.motor_std", "RCOU_motor_features.motor_range",
        "RCOU_motor_features.motor_diff_c1_c3", "RCOU_motor_features.motor_diff_c2_c4",
        "thrust_x_gyro_norm", "thrust_x_acc_norm", "thrust_x_climb_rate",
    }
    direct = {name for name in names if "." in name}
    selected = {
        "imu_raw": imu_raw,
        "imu_engineered": imu_raw | imu_derived,
        "imu_att": imu_raw | imu_derived | attitude | linear,
        "imu_att_crt": imu_raw | imu_derived | attitude | linear | climb_rate,
        "imu_att_baro": imu_raw | imu_derived | attitude | linear | baro,
        "imu_att_crt_motors": imu_raw | imu_derived | attitude | linear | climb_rate | motors,
        "all_direct": direct,
        "all_physical": set(names),
    }[ablation]
    indices = np.asarray([idx for idx, name in enumerate(names) if name in selected], dtype=np.int64)
    if len(indices) != len(selected):
        missing = sorted(selected - {names[idx] for idx in indices})
        raise ValueError(f"Missing channels for {ablation}: {missing}")
    return indices


def purged_splits(
    dataset: SequenceDataset,
    block_count: int,
    min_train_blocks: int,
    purge_us: float,
) -> list[FoldSplit]:
    blocks = block_indices(len(dataset.x), block_count)
    result: list[FoldSplit] = []
    for fold, val_block in enumerate(range(min_train_blocks, block_count - 1), start=1):
        train = concatenate_blocks(blocks, 0, val_block)
        val = blocks[val_block]
        test = blocks[val_block + 1]
        val_start = float(dataset.time_us[val[0]])
        test_start = float(dataset.time_us[test[0]])
        train = train[dataset.time_us[train] < val_start - purge_us]
        val = val[dataset.time_us[val] < test_start - purge_us]
        if min(len(train), len(val), len(test)) == 0:
            raise ValueError(f"Purge removed an entire split in fold {fold}")
        result.append(FoldSplit(fold, val_block, train, val, test))
    return result


def non_overlapping(dataset: SequenceDataset, indices: np.ndarray) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    next_time = -math.inf
    for pred_idx in np.argsort(dataset.time_us[indices]):
        data_idx = int(indices[pred_idx])
        if float(dataset.time_us[data_idx]) + 1e-3 < next_time:
            continue
        selected.append((int(pred_idx), data_idx))
        next_time = float(dataset.future_time_us[data_idx])
    return selected


def rollout_metrics(dataset: SequenceDataset, indices: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    selected = non_overlapping(dataset, indices)
    current = dataset.pos[selected[0][1]].astype(np.float64).copy()
    errors: list[float] = []
    for pred_idx, data_idx in selected:
        current += pred[pred_idx]
        errors.append(float(np.linalg.norm(current - dataset.future_pos[data_idx])))

    dense_order = np.argsort(dataset.time_us[indices])
    dense_pos = dataset.pos[indices[dense_order]]
    dense_pos = np.vstack([dense_pos, dataset.future_pos[indices[dense_order[-1]]]])
    differences = np.diff(dense_pos, axis=0)
    distance_xy = float(np.linalg.norm(differences[:, :2], axis=1).sum())
    distance_3d = float(np.linalg.norm(differences, axis=1).sum())
    final_error = errors[-1]
    return {
        "rollout_steps": float(len(selected)),
        "rollout_mean_m": float(np.mean(errors)),
        "rollout_max_m": float(np.max(errors)),
        "rollout_final_m": final_error,
        "distance_xy_m": distance_xy,
        "distance_3d_m": distance_3d,
        "final_error_per_km": final_error / max(distance_xy / 1000.0, 1e-9),
    }


def result_row(
    model: str,
    feature_set: str,
    seed: int,
    split: FoldSplit,
    channel_count: int,
    val_mae: float,
    test_values: dict[str, float | list[float]],
    rollout: dict[str, float],
    trained_epochs: int,
    best_epoch: int,
) -> dict[str, str]:
    return {
        "model": model,
        "feature_set": feature_set,
        "seed": str(seed),
        "fold": str(split.fold),
        "channels": str(channel_count),
        "train_rows": str(len(split.train)),
        "val_rows": str(len(split.val)),
        "test_rows": str(len(split.test)),
        "validation_mae_3d": f"{val_mae:.9f}",
        "test_mae_3d": f"{float(test_values['mae_3d']):.9f}",
        "test_rmse_3d": f"{float(test_values['rmse_3d']):.9f}",
        "test_p95_3d": f"{float(test_values['p95_3d']):.9f}",
        **{name: f"{value:.9f}" for name, value in rollout.items()},
        "trained_epochs": str(trained_epochs),
        "best_epoch": str(best_epoch),
    }


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate_seed_rows(rows: list[dict[str, str]], model: str, feature_set: str) -> list[dict[str, float]]:
    result = []
    for seed in sorted({int(row["seed"]) for row in rows if row["model"] == model and row["feature_set"] == feature_set}):
        selected = [row for row in rows if row["model"] == model and row["feature_set"] == feature_set and int(row["seed"]) == seed]
        weights = np.asarray([int(row["test_rows"]) for row in selected], dtype=np.float64)
        result.append({
            "seed": float(seed),
            "mae": float(np.average([float(row["test_mae_3d"]) for row in selected], weights=weights)),
            "rmse": float(np.sqrt(np.average([float(row["test_rmse_3d"]) ** 2 for row in selected], weights=weights))),
            "p95": float(np.mean([float(row["test_p95_3d"]) for row in selected])),
            "rollout_mean": float(np.mean([float(row["rollout_mean_m"]) for row in selected])),
            "rollout_final": float(np.mean([float(row["rollout_final_m"]) for row in selected])),
            "distance_xy": float(np.sum([float(row["distance_xy_m"]) for row in selected])),
            "final_per_km": float(np.mean([float(row["final_error_per_km"]) for row in selected])),
        })
    return result


def aggregate_line(label: str, seed_rows: list[dict[str, float]]) -> str:
    mae = mean_std([row["mae"] for row in seed_rows])
    rollout = mean_std([row["rollout_mean"] for row in seed_rows])
    final = mean_std([row["rollout_final"] for row in seed_rows])
    per_km = mean_std([row["final_per_km"] for row in seed_rows])
    distance = float(np.mean([row["distance_xy"] for row in seed_rows]))
    return (
        f"| `{label}` | {mae[0]:.3f} +/- {mae[1]:.3f} | {rollout[0]:.3f} +/- {rollout[1]:.3f} | "
        f"{final[0]:.3f} +/- {final[1]:.3f} | {distance:.1f} | {per_km[0]:.1f} +/- {per_km[1]:.1f} |"
    )


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    splits: list[FoldSplit],
    names: list[str],
    args: argparse.Namespace,
) -> None:
    ablation_lines = [
        aggregate_line(feature_set, aggregate_seed_rows(rows, "lstm_64", feature_set))
        for feature_set in ABLATIONS
    ]
    comparison_lines = [
        aggregate_line("lstm_64/imu_att_crt", aggregate_seed_rows(rows, "lstm_64", "imu_att_crt")),
        aggregate_line("gru_64/imu_att_crt", aggregate_seed_rows(rows, "gru_64", "imu_att_crt")),
        aggregate_line("lstm_64/all_physical", aggregate_seed_rows(rows, "lstm_64", "all_physical")),
        aggregate_line("gru_64/all_physical", aggregate_seed_rows(rows, "gru_64", "all_physical")),
    ]
    split_lines = [
        f"| {split.fold} | {len(split.train)} | {len(split.val)} | {len(split.test)} | "
        f"{split.test[0]}-{split.test[-1]} |"
        for split in splits
    ]
    lines = [
        "# DataFlash Recurrent Robustness", "",
        "Strict within-flight check with a purged rolling split, five random seeds, and feature ablations.", "",
        f"- lookback/horizon: `5000/5000 ms`", f"- purge between split anchors: `{args.purge_ms:g} ms`",
        f"- seeds: `{', '.join(map(str, args.seeds))}`", f"- full physical channels: `{len(names)}`",
        "- no test-time bias correction, clipping, or state gating", "",
        "The purge removes the last 10 seconds of train and validation blocks. Therefore the complete sensor/target interval of a retained window does not overlap the next split.", "",
        "## LSTM feature ablation", "",
        "| feature set | MAE 3D, m | rollout mean, m | rollout final, m | total test POS distance, m | mean fold final error per km, m/km |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *ablation_lines, "",
        "## Recurrent comparison", "",
        "| model/features | MAE 3D, m | rollout mean, m | rollout final, m | total test POS distance, m | mean fold final error per km, m/km |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *comparison_lines, "",
        "Values are mean +/- sample standard deviation across seeds. Rollout values first aggregate the three independent folds within a seed.", "",
        "## Purged split sizes", "",
        "| fold | train rows | validation rows | test rows | test index range |",
        "| ---: | ---: | ---: | ---: | --- |", *split_lines, "",
        "## Feature sets", "",
        "- `imu_raw`: accelerometer and gyroscope axes only.",
        "- `imu_engineered`: IMU plus norms, jerk norm, and angular-acceleration norm.",
        "- `imu_att`: engineered IMU plus attitude and gravity-compensated ENU acceleration.",
        "- `imu_att_crt`: adds barometric climb rate without absolute altitude.",
        "- `imu_att_baro`: adds barometric altitude and climb rate.",
        "- `imu_att_crt_motors`: adds climb rate and selected motor channels/interactions, without battery or absolute altitude.",
        "- `all_direct`: adds direct battery and motor channels, without interaction features.",
        "- `all_physical`: adds thrust interactions and all derived channels.", "",
        "Distance is the dense POS path length over the three test blocks. It can include EKF/GPS jitter and is not directly comparable to a separately defined 3 km benchmark.", "",
        "This is still validation inside one flight. A holdout DataFlash flight remains required before claiming cross-flight generalization.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    horizon_ms, lookback_ms = parse_window(args.window)
    series, _ = load_series(args.data_dir, "all")
    pos_times, positions = read_pos(args.data_dir / "POS.csv")
    dataset = make_sequence_dataset(
        pos_times, positions, series, horizon_ms * 1000.0, lookback_ms * 1000.0, args.sequence_len
    )
    full_x, names = physical_feature_cube(dataset, args.sequence_len, lookback_ms / 1000.0)
    splits = purged_splits(dataset, args.block_count, args.min_train_blocks, args.purge_ms * 1000.0)
    rows = read_summary(args.summary_csv) if args.resume else []
    completed_keys = {
        (row["model"], row["feature_set"], int(row["seed"]), int(row["fold"])) for row in rows
    }
    representative: dict[str, list[dict[str, str]]] = {
        "lstm_64_imu_att_crt": [],
        "gru_64_imu_att_crt": [],
        "lstm_64_all_physical": [],
        "gru_64_all_physical": [],
    }

    cases = [("lstm_64", feature_set) for feature_set in ABLATIONS] + [
        ("gru_64", "imu_att_crt"),
        ("gru_64", "all_physical"),
    ]
    total = len(cases) * len(args.seeds) * len(splits)
    completed = 0
    for model_name, feature_set in cases:
        channel_idx = feature_indices(names, feature_set)
        x = full_x[:, :, channel_idx]
        for seed in args.seeds:
            for split in splits:
                key = (model_name, feature_set, seed, split.fold)
                representative_key = f"{model_name}_{feature_set}"
                needs_representative = (
                    seed == args.seeds[0] and representative_key in representative
                )
                if key in completed_keys and not needs_representative:
                    completed += 1
                    continue
                model_offset = 0 if model_name == "lstm_64" else 1
                val_pred, test_pred, trained_epochs, best_epoch = fit_predict(
                    model_name,
                    x[split.train], dataset.y[split.train], x[split.val], dataset.y[split.val],
                    x[split.test], args, seed + split.fold * 100 + model_offset,
                )
                val_mae = float(metrics(dataset.y[split.val], val_pred)["mae_3d"])
                test_values = metrics(dataset.y[split.test], test_pred)
                rollout = rollout_metrics(dataset, split.test, test_pred)
                if key not in completed_keys:
                    rows.append(result_row(
                        model_name, feature_set, seed, split, len(channel_idx), val_mae,
                        test_values, rollout, trained_epochs, best_epoch,
                    ))
                    completed_keys.add(key)
                    write_summary(args.summary_csv, rows)
                if needs_representative:
                    representative[representative_key].extend(build_prediction_rows(
                        dataset, split.test, test_pred, feature_set, horizon_ms, lookback_ms,
                        args.sequence_len, f"{model_name}_purged", split.fold,
                        f"0-{split.val_block - 1} purged", split.val_block, split.val_block + 1, 0.0,
                    ))
                completed += 1
                print(
                    f"[{completed}/{total}] {model_name}/{feature_set} seed={seed} fold={split.fold} "
                    f"mae={float(test_values['mae_3d']):.3f} rollout={rollout['rollout_mean_m']:.3f}",
                    flush=True,
                )

    for case_name, prediction_rows in representative.items():
        write_prediction_rows(args.pred_dir / f"{case_name}_pred.csv", prediction_rows)
    write_report(args.report, rows, splits, names, args)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
