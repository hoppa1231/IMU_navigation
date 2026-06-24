#!/usr/bin/env python3
"""Run chronological DataFlash baseline sweeps with validation tuning."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataflash_baseline import (
    FEATURE_SETS,
    Series,
    aggregate_feature_names,
    aggregate_for_time,
    append_synthetic,
    fit_ridge,
    predict_ridge,
    read_numeric_csv,
    read_pos,
    top_features,
)


ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


@dataclass
class WindowDataset:
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
    parser.add_argument("--feature-sets", nargs="*", choices=sorted(FEATURE_SETS), default=["imu", "imu_att", "all"])
    parser.add_argument(
        "--windows",
        nargs="*",
        default=["1000:1000", "3000:3000", "5000:5000"],
        help="Pairs as horizon_ms:lookback_ms.",
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/dataflash_sweep.md"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/dataflash_sweep"))
    return parser.parse_args()


def parse_window(text: str) -> tuple[float, float]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Window must be horizon_ms:lookback_ms, got {text!r}")
    return float(parts[0]), float(parts[1])


def load_series(data_dir: Path, feature_set: str) -> tuple[list[Series], list[Path]]:
    source_files = [data_dir / f"{name}.csv" for name in FEATURE_SETS[feature_set]]
    series_list = [
        append_synthetic(read_numeric_csv(path, path.stem, drop_columns={"I", "Inst", "Health", "H", "SH", "FailFlags"}))
        for path in source_files
    ]
    return series_list, source_files


def make_dataset(
    pos_times: np.ndarray,
    positions: np.ndarray,
    series_list: list[Series],
    horizon_us: float,
    lookback_us: float,
) -> WindowDataset:
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
        parts = [aggregate_for_time(series, pos_times[idx], lookback_us) for series in series_list]
        if any(part is None for part in parts):
            continue
        rows.append(np.concatenate([part for part in parts if part is not None]))
        targets.append(positions[future_idx] - positions[idx])
        times.append(float(pos_times[idx]))
        future_times.append(float(pos_times[future_idx]))
        current_pos.append(positions[idx])
        future_pos.append(positions[future_idx])

    return WindowDataset(
        x=np.vstack(rows),
        y=np.vstack(targets),
        time_us=np.asarray(times, dtype=np.float64),
        future_time_us=np.asarray(future_times, dtype=np.float64),
        pos=np.vstack(current_pos),
        future_pos=np.vstack(future_pos),
        feature_names=aggregate_feature_names(series_list),
    )


def split_indices(count: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(1, min(count - 2, int(count * train_frac)))
    val_end = max(train_end + 1, min(count - 1, int(count * (train_frac + val_frac))))
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, count)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    err = y_pred - y_true
    mae_axis = np.mean(np.abs(err), axis=0)
    rmse_axis = np.sqrt(np.mean(err * err, axis=0))
    dist_err = np.linalg.norm(err, axis=1)
    return {
        "mae_axis": [float(value) for value in mae_axis],
        "rmse_axis": [float(value) for value in rmse_axis],
        "mae_3d": float(np.mean(dist_err)),
        "rmse_3d": float(np.sqrt(np.mean(dist_err * dist_err))),
        "p95_3d": float(np.percentile(dist_err, 95)),
    }


def metric_row(
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    subset: str,
    model: str,
    alpha: float | None,
    values: dict[str, float | list[float]],
) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, list)
    assert isinstance(rmse, list)
    alpha_text = "" if alpha is None else f"{alpha:g}"
    return (
        f"| `{feature_set}` | {horizon_ms:g} | {lookback_ms:g} | `{subset}` | `{model}` | {alpha_text} | "
        f"{mae[0]:.3f} | {mae[1]:.3f} | {mae[2]:.3f} | {values['mae_3d']:.3f} | "
        f"{rmse[0]:.3f} | {rmse[1]:.3f} | {rmse[2]:.3f} | {values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def write_predictions(
    path: Path,
    dataset: WindowDataset,
    indices: np.ndarray,
    pred: np.ndarray,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "feature_set",
        "horizon_ms",
        "lookback_ms",
        "model",
        "time_s",
        "future_time_s",
        "current_east_m",
        "current_north_m",
        "current_up_m",
        "true_future_east_m",
        "true_future_north_m",
        "true_future_up_m",
        "pred_future_east_m",
        "pred_future_north_m",
        "pred_future_up_m",
        "true_dx_east_m",
        "true_dy_north_m",
        "true_dz_up_m",
        "pred_dx_east_m",
        "pred_dy_north_m",
        "pred_dz_up_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, data_idx in enumerate(indices):
            current = dataset.pos[data_idx]
            true_future = dataset.future_pos[data_idx]
            pred_delta = pred[row_idx]
            pred_future = current + pred_delta
            writer.writerow(
                {
                    "source": "dataflash",
                    "feature_set": feature_set,
                    "horizon_ms": f"{horizon_ms:g}",
                    "lookback_ms": f"{lookback_ms:g}",
                    "model": model,
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


def run_case(
    data_dir: Path,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    train_frac: float,
    val_frac: float,
    pred_dir: Path,
) -> tuple[list[str], str, str, list[str], list[str]]:
    series_list, source_files = load_series(data_dir, feature_set)
    pos_times, positions = read_pos(data_dir / "POS.csv")
    dataset = make_dataset(pos_times, positions, series_list, horizon_ms * 1000.0, lookback_ms * 1000.0)
    train_idx, val_idx, test_idx = split_indices(len(dataset.x), train_frac, val_frac)

    x_train, y_train = dataset.x[train_idx], dataset.y[train_idx]
    x_val, y_val = dataset.x[val_idx], dataset.y[val_idx]
    x_test, y_test = dataset.x[test_idx], dataset.y[test_idx]

    best_alpha = ALPHAS[0]
    best_params = fit_ridge(x_train, y_train, best_alpha)
    best_val_score = math.inf
    alpha_lines = [
        f"### `{feature_set}` h={horizon_ms:g} l={lookback_ms:g}",
        "",
        "| alpha | validation MAE 3D | test MAE 3D |",
        "| ---: | ---: | ---: |",
    ]
    for alpha in ALPHAS:
        params = fit_ridge(x_train, y_train, alpha)
        val_pred = predict_ridge(x_val, *params)
        score = float(metrics(y_val, val_pred)["mae_3d"])
        test_score = float(metrics(y_test, predict_ridge(x_test, *params))["mae_3d"])
        alpha_lines.append(f"| {alpha:g} | {score:.3f} | {test_score:.3f} |")
        if score < best_val_score:
            best_val_score = score
            best_alpha = alpha
            best_params = params
    alpha_lines.append("")

    rows: list[str] = []
    best_test_items: list[tuple[str, dict[str, float | list[float]]]] = []
    for subset_name, indices, x_current, y_current in [
        ("val", val_idx, x_val, y_val),
        ("test", test_idx, x_test, y_test),
    ]:
        predictions = {
            "zero": (np.zeros_like(y_current), None),
            "train_mean": (np.repeat(y_train.mean(axis=0, keepdims=True), len(y_current), axis=0), None),
            "ridge": (predict_ridge(x_current, *best_params), best_alpha),
        }
        for model_name, (pred, alpha) in predictions.items():
            values = metrics(y_current, pred)
            rows.append(metric_row(feature_set, horizon_ms, lookback_ms, subset_name, model_name, alpha, values))
            if subset_name == "test":
                best_test_items.append((model_name, values))
                pred_path = pred_dir / f"{feature_set}_h{int(horizon_ms)}_l{int(lookback_ms)}" / f"{model_name}_pred.csv"
                write_predictions(pred_path, dataset, indices, pred, feature_set, horizon_ms, lookback_ms, model_name)

    best_model, best_values = min(best_test_items, key=lambda item: float(item[1]["mae_3d"]))
    best_row = (
        f"| `{feature_set}` | {horizon_ms:g} | {lookback_ms:g} | `{best_model}` | "
        f"{float(best_values['mae_3d']):.3f} | {float(best_values['rmse_3d']):.3f} | {float(best_values['p95_3d']):.3f} |"
    )
    note = (
        f"`{feature_set}` h={horizon_ms:g}ms l={lookback_ms:g}ms windows={len(dataset.x)} "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} best_alpha={best_alpha:g} "
        f"val_ridge_mae_3d={best_val_score:.3f}"
    )
    feature_lines = [f"### `{feature_set}` h={horizon_ms:g} l={lookback_ms:g}", ""]
    feature_lines.extend(f"- source: `{path}`" for path in source_files)
    feature_lines.extend(["", "| feature | score |", "| --- | ---: |"])
    for name, score in top_features(dataset.feature_names, best_params[0], limit=12):
        feature_lines.append(f"| `{name}` | {score:.5f} |")
    feature_lines.append("")
    return rows, best_row, note, feature_lines, alpha_lines


def write_report(
    path: Path,
    rows: list[str],
    best_rows: list[str],
    notes: list[str],
    feature_lines: list[str],
    alpha_lines: list[str],
    pred_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash Sweep",
        "",
        "This report is generated by `src/run_dataflash_sweep.py`.",
        "",
        "Scope: one DataFlash log only. Sources are `derived/dataflash/*.csv`; no module-data rows are mixed in.",
        "",
        "Split: chronological train/validation/test inside the DataFlash log. Ridge alpha is selected on validation only.",
        "",
        f"Predictions: `{pred_dir}`",
        "",
        "## Metrics",
        "",
        "| feature set | horizon ms | lookback ms | subset | model | alpha | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D | P95 3D |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(rows)
    lines.extend(
        [
            "",
            "## Best Test Baseline",
            "",
            "| feature set | horizon ms | lookback ms | model | MAE 3D | RMSE 3D | P95 3D |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(best_rows)
    lines.extend(["", "## Split Details", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Alpha Sensitivity", ""])
    lines.extend(alpha_lines)
    lines.extend(["", "## Top Ridge Features", ""])
    lines.extend(feature_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[str] = []
    best_rows: list[str] = []
    notes: list[str] = []
    feature_lines: list[str] = []
    alpha_lines: list[str] = []
    for feature_set in args.feature_sets:
        for horizon_ms, lookback_ms in [parse_window(text) for text in args.windows]:
            case_rows, best_row, note, case_features, case_alpha_lines = run_case(
                args.data_dir,
                feature_set,
                horizon_ms,
                lookback_ms,
                args.train_frac,
                args.val_frac,
                args.pred_dir,
            )
            rows.extend(case_rows)
            best_rows.append(best_row)
            notes.append(note)
            feature_lines.extend(case_features)
            alpha_lines.extend(case_alpha_lines)
            print(f"Ran {feature_set} h={horizon_ms:g} l={lookback_ms:g}")
    write_report(args.report, rows, best_rows, notes, feature_lines, alpha_lines, args.pred_dir)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
