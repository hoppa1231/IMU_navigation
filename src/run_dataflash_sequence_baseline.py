#!/usr/bin/env python3
"""Run a pure-NumPy sequence baseline for DataFlash rolling validation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataflash_baseline import Series, fit_ridge, predict_ridge, read_pos
from run_dataflash_rolling_validation import block_indices, concatenate_blocks
from run_dataflash_sweep import ALPHAS, FEATURE_SETS, load_series, metrics, parse_window


SHRINKS = [0.25, 0.5, 0.75, 1.0]


@dataclass
class SequenceDataset:
    x: np.ndarray
    y: np.ndarray
    time_us: np.ndarray
    future_time_us: np.ndarray
    pos: np.ndarray
    future_pos: np.ndarray
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="imu_att")
    parser.add_argument("--window", default="5000:5000", help="Pair as horizon_ms:lookback_ms.")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--fixed-alpha", type=float, default=0.0, help="Use this ridge alpha instead of per-fold validation selection.")
    parser.add_argument("--tune-bias-shrink", action="store_true", help="Pick validation-residual correction strength on validation MAE.")
    parser.add_argument(
        "--bias-shrink-metric",
        choices=["mae", "rollout_mean", "rollout_final", "rollout_max"],
        default="mae",
        help="Validation metric used when --tune-bias-shrink is enabled.",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/dataflash_sequence_imu_att_h5000.md"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/dataflash_sequence_rolling"))
    return parser.parse_args()


def sequence_feature_names(series_list: list[Series], sequence_len: int) -> list[str]:
    names: list[str] = []
    for step_idx in range(sequence_len):
        for series in series_list:
            names.extend([f"t{step_idx:02d}.{name}" for name in series.feature_names])
    return names


def sample_sequence(series_list: list[Series], time_us: float, lookback_us: float, sequence_len: int) -> np.ndarray | None:
    offsets = np.linspace(-lookback_us, 0.0, sequence_len)
    parts: list[np.ndarray] = []
    for offset in offsets:
        sample_time = time_us + offset
        for series in series_list:
            idx = int(np.searchsorted(series.times_us, sample_time, side="right")) - 1
            if idx < 0:
                return None
            parts.append(series.values[idx])
    return np.concatenate(parts)


def make_sequence_dataset(
    pos_times: np.ndarray,
    positions: np.ndarray,
    series_list: list[Series],
    horizon_us: float,
    lookback_us: float,
    sequence_len: int,
) -> SequenceDataset:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    times: list[float] = []
    future_times: list[float] = []
    current_pos: list[np.ndarray] = []
    future_pos: list[np.ndarray] = []

    future_indices = np.searchsorted(pos_times, pos_times + horizon_us, side="left")
    for idx, future_idx in enumerate(future_indices):
        if future_idx >= len(pos_times):
            continue
        features = sample_sequence(series_list, pos_times[idx], lookback_us, sequence_len)
        if features is None:
            continue
        rows.append(features)
        targets.append(positions[future_idx] - positions[idx])
        times.append(float(pos_times[idx]))
        future_times.append(float(pos_times[future_idx]))
        current_pos.append(positions[idx])
        future_pos.append(positions[future_idx])

    return SequenceDataset(
        x=np.vstack(rows),
        y=np.vstack(targets),
        time_us=np.asarray(times, dtype=np.float64),
        future_time_us=np.asarray(future_times, dtype=np.float64),
        pos=np.vstack(current_pos),
        future_pos=np.vstack(future_pos),
        feature_names=sequence_feature_names(series_list, sequence_len),
    )


def write_prediction_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_prediction_rows(
    dataset: SequenceDataset,
    indices: np.ndarray,
    pred: np.ndarray,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    sequence_len: int,
    model: str,
    fold_id: int,
    train_blocks: str,
    val_block: int,
    test_block: int,
    selected_alpha: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row_idx, data_idx in enumerate(indices):
        current = dataset.pos[data_idx]
        true_future = dataset.future_pos[data_idx]
        pred_delta = pred[row_idx]
        pred_future = current + pred_delta
        rows.append(
            {
                "source": "dataflash",
                "feature_set": feature_set,
                "horizon_ms": f"{horizon_ms:g}",
                "lookback_ms": f"{lookback_ms:g}",
                "sequence_len": str(sequence_len),
                "model": model,
                "fold_id": str(fold_id),
                "train_blocks": train_blocks,
                "val_block": str(val_block),
                "test_block": str(test_block),
                "selected_alpha": f"{selected_alpha:g}",
                "time_s": f"{dataset.time_us[data_idx] / 1_000_000.0:.6f}",
                "future_time_s": f"{dataset.future_time_us[data_idx] / 1_000_000.0:.6f}",
                "current_east_m": f"{current[0]:.6f}",
                "current_north_m": f"{current[1]:.6f}",
                "current_up_m": f"{current[2]:.6f}",
                "true_future_east_m": f"{true_future[0]:.6f}",
                "true_future_north_m": f"{true_future[1]:.6f}",
                "true_future_up_m": f"{true_future[2]:.6f}",
                "pred_future_east_m": f"{pred_future[0]:.6f}",
                "pred_future_north_m": f"{pred_future[1]:.6f}",
                "pred_future_up_m": f"{pred_future[2]:.6f}",
                "true_dx_east_m": f"{dataset.y[data_idx, 0]:.6f}",
                "true_dy_north_m": f"{dataset.y[data_idx, 1]:.6f}",
                "true_dz_up_m": f"{dataset.y[data_idx, 2]:.6f}",
                "pred_dx_east_m": f"{pred_delta[0]:.6f}",
                "pred_dy_north_m": f"{pred_delta[1]:.6f}",
                "pred_dz_up_m": f"{pred_delta[2]:.6f}",
            }
        )
    return rows


def metric_row(model: str, folds: int, windows: int, values: dict[str, float | list[float]]) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, list)
    assert isinstance(rmse, list)
    return (
        f"| `{model}` | {folds} | {windows} | {mae[0]:.3f} | {mae[1]:.3f} | {mae[2]:.3f} | "
        f"{values['mae_3d']:.3f} | {rmse[0]:.3f} | {rmse[1]:.3f} | {rmse[2]:.3f} | "
        f"{values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def fold_row(
    fold_id: int,
    train_blocks: str,
    val_block: int,
    test_block: int,
    selected_alpha: float,
    model: str,
    values: dict[str, float | list[float]],
) -> str:
    return (
        f"| {fold_id} | `{train_blocks}` | {val_block} | {test_block} | {selected_alpha:g} | `{model}` | "
        f"{values['mae_3d']:.3f} | {values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def rollout_score(dataset: SequenceDataset, indices: np.ndarray, pred: np.ndarray, metric: str) -> float:
    order = np.argsort(dataset.time_us[indices])
    selected: list[tuple[int, int]] = []
    next_time = -math.inf
    for pred_idx in order:
        data_idx = int(indices[pred_idx])
        time_us = float(dataset.time_us[data_idx])
        if time_us + 1e-3 < next_time:
            continue
        selected.append((pred_idx, data_idx))
        next_time = float(dataset.future_time_us[data_idx])
    if not selected:
        return math.inf

    first_data_idx = selected[0][1]
    current = dataset.pos[first_data_idx].astype(np.float64).copy()
    errors: list[float] = []
    for pred_idx, data_idx in selected:
        current = current + pred[pred_idx]
        err = current - dataset.future_pos[data_idx]
        errors.append(float(np.linalg.norm(err)))

    if metric == "rollout_final":
        return errors[-1]
    if metric == "rollout_max":
        return max(errors)
    return float(np.mean(errors))


def run_experiment(
    dataset: SequenceDataset,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    sequence_len: int,
    block_count: int,
    min_train_blocks: int,
    fixed_alpha: float,
    tune_bias_shrink: bool,
    bias_shrink_metric: str,
    pred_dir: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    blocks = block_indices(len(dataset.x), block_count)
    by_model_true: dict[str, list[np.ndarray]] = defaultdict(list)
    by_model_pred: dict[str, list[np.ndarray]] = defaultdict(list)
    pred_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    fold_rows: list[str] = []
    selected_alphas: list[float] = []
    selected_shrinks: list[float] = []
    alpha_val_scores: dict[float, list[float]] = defaultdict(list)
    alpha_test_scores: dict[float, list[float]] = defaultdict(list)

    fold_count = 0
    for val_block in range(min_train_blocks, block_count - 1):
        fold_id = fold_count + 1
        test_block = val_block + 1
        train_idx = concatenate_blocks(blocks, 0, val_block)
        val_idx = blocks[val_block]
        test_idx = blocks[test_block]
        train_blocks = f"0-{val_block - 1}"

        x_train, y_train = dataset.x[train_idx], dataset.y[train_idx]
        x_val, y_val = dataset.x[val_idx], dataset.y[val_idx]
        x_test, y_test = dataset.x[test_idx], dataset.y[test_idx]

        best_alpha = ALPHAS[0]
        best_params = fit_ridge(x_train, y_train, best_alpha)
        best_score = math.inf
        fixed_params = None
        for alpha in ALPHAS:
            params = fit_ridge(x_train, y_train, alpha)
            val_pred = predict_ridge(x_val, *params)
            test_pred = predict_ridge(x_test, *params)
            val_score = float(metrics(y_val, val_pred)["mae_3d"])
            test_score = float(metrics(y_test, test_pred)["mae_3d"])
            alpha_val_scores[alpha].append(val_score)
            alpha_test_scores[alpha].append(test_score)
            if fixed_alpha and abs(alpha - fixed_alpha) < 1e-12:
                fixed_params = params
            if val_score < best_score:
                best_score = val_score
                best_alpha = alpha
                best_params = params
        if fixed_alpha:
            if fixed_params is None:
                fixed_params = fit_ridge(x_train, y_train, fixed_alpha)
            best_alpha = fixed_alpha
            best_params = fixed_params

        selected_alphas.append(best_alpha)
        ridge_val_pred = predict_ridge(x_val, *best_params)
        ridge_test_pred = predict_ridge(x_test, *best_params)
        bias_correction = np.mean(y_val - ridge_val_pred, axis=0, keepdims=True)
        best_shrink = 1.0
        if tune_bias_shrink:
            best_shrink = 0.0
            if bias_shrink_metric == "mae":
                best_shrink_score = float(metrics(y_val, ridge_val_pred)["mae_3d"])
            else:
                best_shrink_score = rollout_score(dataset, val_idx, ridge_val_pred, bias_shrink_metric)
            for shrink in SHRINKS:
                candidate = ridge_val_pred + shrink * bias_correction
                if bias_shrink_metric == "mae":
                    score = float(metrics(y_val, candidate)["mae_3d"])
                else:
                    score = rollout_score(dataset, val_idx, candidate, bias_shrink_metric)
                if score < best_shrink_score:
                    best_shrink_score = score
                    best_shrink = shrink
        selected_shrinks.append(best_shrink)
        predictions = {
            "zero": np.zeros_like(y_test),
            "train_mean": np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0),
            "sequence_ridge": ridge_test_pred,
            "sequence_ridge_bias_corrected": ridge_test_pred + bias_correction,
            "sequence_ridge_bias_tuned": ridge_test_pred + best_shrink * bias_correction,
        }
        for model_name, pred in predictions.items():
            values = metrics(y_test, pred)
            fold_rows.append(fold_row(fold_id, train_blocks, val_block, test_block, best_alpha, model_name, values))
            by_model_true[model_name].append(y_test)
            by_model_pred[model_name].append(pred)
            pred_rows[model_name].extend(
                build_prediction_rows(
                    dataset,
                    test_idx,
                    pred,
                    feature_set,
                    horizon_ms,
                    lookback_ms,
                    sequence_len,
                    model_name,
                    fold_id,
                    train_blocks,
                    val_block,
                    test_block,
                    best_alpha,
                )
            )
        fold_count += 1

    overall_rows: list[str] = []
    case_name = f"{feature_set}_h{int(horizon_ms)}_l{int(lookback_ms)}_s{sequence_len}"
    for model_name in ["zero", "train_mean", "sequence_ridge", "sequence_ridge_bias_corrected", "sequence_ridge_bias_tuned"]:
        y_true = np.vstack(by_model_true[model_name])
        y_pred = np.vstack(by_model_pred[model_name])
        overall_rows.append(metric_row(model_name, fold_count, len(y_true), metrics(y_true, y_pred)))
        write_prediction_rows(pred_dir / case_name / f"{model_name}_pred.csv", pred_rows[model_name])

    alpha_counts = ", ".join(f"{alpha:g}: {count}" for alpha, count in sorted(Counter(selected_alphas).items()))
    shrink_counts = ", ".join(f"{shrink:g}: {count}" for shrink, count in sorted(Counter(selected_shrinks).items()))
    notes = [
        f"dataset windows={len(dataset.x)} features={dataset.x.shape[1]} blocks={block_count} folds={fold_count}",
        f"alpha policy: {'fixed' if fixed_alpha else 'per-fold validation'}",
        f"selected alpha counts: {{ {alpha_counts} }}",
        f"bias shrink policy: {'validation tuned' if tune_bias_shrink else 'full correction'}",
        f"bias shrink metric: {bias_shrink_metric}",
        f"selected shrink counts: {{ {shrink_counts} }}",
    ]
    alpha_rows = ["| alpha | mean validation MAE 3D | mean test MAE 3D |", "| ---: | ---: | ---: |"]
    for alpha in ALPHAS:
        alpha_rows.append(f"| {alpha:g} | {np.mean(alpha_val_scores[alpha]):.3f} | {np.mean(alpha_test_scores[alpha]):.3f} |")
    return overall_rows, fold_rows, notes, alpha_rows


def write_report(
    path: Path,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    sequence_len: int,
    fixed_alpha: float,
    tune_bias_shrink: bool,
    bias_shrink_metric: str,
    pred_dir: Path,
    overall_rows: list[str],
    fold_rows: list[str],
    notes: list[str],
    alpha_rows: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash Sequence Baseline",
        "",
        "This report is generated by `src/run_dataflash_sequence_baseline.py`.",
        "",
        f"- feature set: `{feature_set}`",
        f"- horizon: `{horizon_ms:g}` ms",
        f"- lookback: `{lookback_ms:g}` ms",
        f"- sequence length: `{sequence_len}`",
        f"- alpha policy: `{'fixed ' + format(fixed_alpha, 'g') if fixed_alpha else 'per-fold validation'}`",
        f"- bias shrink policy: `{'validation tuned' if tune_bias_shrink else 'full correction'}`",
        f"- bias shrink metric: `{bias_shrink_metric}`",
        f"- predictions: `{pred_dir}`",
        "",
        "Features are ordered sensor samples from the lookback interval, flattened and fitted with ridge regression. GPS/POS is used only as future displacement target.",
        "",
        "## Overall Test Metrics",
        "",
        "| model | folds | windows | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D | P95 3D |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(overall_rows)
    lines.extend(
        [
            "",
            "## Fold Test Metrics",
            "",
            "| fold | train blocks | val block | test block | selected alpha | model | MAE 3D | RMSE 3D | P95 3D |",
            "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(fold_rows)
    lines.extend(["", "## Details", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Mean Alpha Sensitivity", ""])
    lines.extend(alpha_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    horizon_ms, lookback_ms = parse_window(args.window)
    series_list, _ = load_series(args.data_dir, args.feature_set)
    pos_times, positions = read_pos(args.data_dir / "POS.csv")
    dataset = make_sequence_dataset(
        pos_times,
        positions,
        series_list,
        horizon_ms * 1000.0,
        lookback_ms * 1000.0,
        args.sequence_len,
    )
    overall_rows, fold_rows, notes, alpha_rows = run_experiment(
        dataset,
        args.feature_set,
        horizon_ms,
        lookback_ms,
        args.sequence_len,
        args.block_count,
        args.min_train_blocks,
        args.fixed_alpha,
        args.tune_bias_shrink,
        args.bias_shrink_metric,
        args.pred_dir,
    )
    write_report(
        args.report,
        args.feature_set,
        horizon_ms,
        lookback_ms,
        args.sequence_len,
        args.fixed_alpha,
        args.tune_bias_shrink,
        args.bias_shrink_metric,
        args.pred_dir,
        overall_rows,
        fold_rows,
        notes,
        alpha_rows,
    )
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
