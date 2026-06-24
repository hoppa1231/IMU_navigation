#!/usr/bin/env python3
"""Run rolling time-block validation for DataFlash window baselines."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataflash_baseline import fit_ridge, predict_ridge, read_pos
from run_dataflash_sweep import (
    ALPHAS,
    FEATURE_SETS,
    WindowDataset,
    load_series,
    make_dataset,
    metrics,
    parse_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--feature-sets", nargs="*", choices=sorted(FEATURE_SETS), default=["imu", "imu_att", "all"])
    parser.add_argument("--windows", nargs="*", default=["1000:1000", "3000:3000", "5000:5000"])
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/dataflash_rolling_validation.md"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/dataflash_rolling_validation"))
    return parser.parse_args()


def block_indices(count: int, block_count: int) -> list[np.ndarray]:
    if block_count < 4:
        raise ValueError("--block-count must be >= 4")
    return [block.astype(np.int64) for block in np.array_split(np.arange(count), block_count)]


def concatenate_blocks(blocks: list[np.ndarray], start: int, stop: int) -> np.ndarray:
    selected = blocks[start:stop]
    if not selected:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(selected)


def metric_row(
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    model: str,
    fold_count: int,
    window_count: int,
    values: dict[str, float | list[float]],
) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, list)
    assert isinstance(rmse, list)
    return (
        f"| `{feature_set}` | {horizon_ms:g} | {lookback_ms:g} | `{model}` | "
        f"{fold_count} | {window_count} | {mae[0]:.3f} | {mae[1]:.3f} | {mae[2]:.3f} | "
        f"{values['mae_3d']:.3f} | {rmse[0]:.3f} | {rmse[1]:.3f} | {rmse[2]:.3f} | "
        f"{values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def fold_metric_row(
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    fold_id: int,
    train_blocks: str,
    val_block: int,
    test_block: int,
    selected_alpha: float,
    model: str,
    values: dict[str, float | list[float]],
) -> str:
    return (
        f"| `{feature_set}` | {horizon_ms:g} | {lookback_ms:g} | {fold_id} | `{train_blocks}` | "
        f"{val_block} | {test_block} | {selected_alpha:g} | `{model}` | "
        f"{values['mae_3d']:.3f} | {values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def write_prediction_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prediction_rows(
    dataset: WindowDataset,
    indices: np.ndarray,
    pred: np.ndarray,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
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


def run_case(
    data_dir: Path,
    feature_set: str,
    horizon_ms: float,
    lookback_ms: float,
    block_count: int,
    min_train_blocks: int,
    pred_dir: Path,
) -> tuple[list[str], list[str], str, list[str], list[str]]:
    series_list, _ = load_series(data_dir, feature_set)
    pos_times, positions = read_pos(data_dir / "POS.csv")
    dataset = make_dataset(pos_times, positions, series_list, horizon_ms * 1000.0, lookback_ms * 1000.0)
    blocks = block_indices(len(dataset.x), block_count)

    by_model_true: dict[str, list[np.ndarray]] = defaultdict(list)
    by_model_pred: dict[str, list[np.ndarray]] = defaultdict(list)
    pred_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    fold_rows: list[str] = []
    selected_alphas: list[float] = []
    alpha_val_scores: dict[float, list[float]] = defaultdict(list)
    alpha_test_scores: dict[float, list[float]] = defaultdict(list)

    fold_count = 0
    max_val_block = block_count - 2
    for val_block in range(min_train_blocks, max_val_block + 1):
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
        for alpha in ALPHAS:
            params = fit_ridge(x_train, y_train, alpha)
            val_pred = predict_ridge(x_val, *params)
            test_pred = predict_ridge(x_test, *params)
            val_score = float(metrics(y_val, val_pred)["mae_3d"])
            test_score = float(metrics(y_test, test_pred)["mae_3d"])
            alpha_val_scores[alpha].append(val_score)
            alpha_test_scores[alpha].append(test_score)
            if val_score < best_score:
                best_score = val_score
                best_alpha = alpha
                best_params = params

        selected_alphas.append(best_alpha)
        predictions = {
            "zero": np.zeros_like(y_test),
            "train_mean": np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0),
            "ridge": predict_ridge(x_test, *best_params),
        }
        for model_name, pred in predictions.items():
            values = metrics(y_test, pred)
            fold_rows.append(
                fold_metric_row(
                    feature_set,
                    horizon_ms,
                    lookback_ms,
                    fold_id,
                    train_blocks,
                    val_block,
                    test_block,
                    best_alpha,
                    model_name,
                    values,
                )
            )
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
    best_items: list[tuple[str, dict[str, float | list[float]]]] = []
    for model_name in ["zero", "train_mean", "ridge"]:
        y_true = np.vstack(by_model_true[model_name])
        y_pred = np.vstack(by_model_pred[model_name])
        values = metrics(y_true, y_pred)
        overall_rows.append(metric_row(feature_set, horizon_ms, lookback_ms, model_name, fold_count, len(y_true), values))
        best_items.append((model_name, values))
        pred_path = pred_dir / f"{feature_set}_h{int(horizon_ms)}_l{int(lookback_ms)}" / f"{model_name}_pred.csv"
        write_prediction_rows(pred_path, pred_rows[model_name])

    best_model, best_values = min(best_items, key=lambda item: float(item[1]["mae_3d"]))
    best_row = (
        f"| `{feature_set}` | {horizon_ms:g} | {lookback_ms:g} | `{best_model}` | "
        f"{float(best_values['mae_3d']):.3f} | {float(best_values['rmse_3d']):.3f} | {float(best_values['p95_3d']):.3f} |"
    )
    alpha_counts = ", ".join(f"{alpha:g}: {count}" for alpha, count in sorted(Counter(selected_alphas).items()))
    note = (
        f"`{feature_set}` h={horizon_ms:g}ms l={lookback_ms:g}ms windows={len(dataset.x)} "
        f"blocks={block_count} folds={fold_count} selected_alpha_counts={{ {alpha_counts} }}"
    )
    alpha_rows = [f"### `{feature_set}` h={horizon_ms:g} l={lookback_ms:g}", ""]
    alpha_rows.extend(["| alpha | mean validation MAE 3D | mean test MAE 3D |", "| ---: | ---: | ---: |"])
    for alpha in ALPHAS:
        alpha_rows.append(
            f"| {alpha:g} | {np.mean(alpha_val_scores[alpha]):.3f} | {np.mean(alpha_test_scores[alpha]):.3f} |"
        )
    alpha_rows.append("")
    return overall_rows, fold_rows, best_row, [note], alpha_rows


def write_report(
    path: Path,
    overall_rows: list[str],
    fold_rows: list[str],
    best_rows: list[str],
    notes: list[str],
    alpha_rows: list[str],
    pred_dir: Path,
    block_count: int,
    min_train_blocks: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash Rolling Validation",
        "",
        "This report is generated by `src/run_dataflash_rolling_validation.py`.",
        "",
        "Scope: one DataFlash log only. Sources are `derived/dataflash/*.csv`; no module-data rows are mixed in.",
        "",
        f"Blocks: `{block_count}` contiguous chronological blocks.",
        f"Fold rule: train uses all blocks before validation, starting with `{min_train_blocks}` train blocks; test is the block immediately after validation.",
        "Ridge alpha is selected on each fold validation block only.",
        "",
        f"Predictions: `{pred_dir}`",
        "",
        "## Overall Test Metrics",
        "",
        "| feature set | horizon ms | lookback ms | model | folds | windows | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D | P95 3D |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(overall_rows)
    lines.extend(
        [
            "",
            "## Best Overall Test Baseline",
            "",
            "| feature set | horizon ms | lookback ms | model | MAE 3D | RMSE 3D | P95 3D |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(best_rows)
    lines.extend(["", "## Fold Test Metrics", ""])
    lines.extend(
        [
            "| feature set | horizon ms | lookback ms | fold | train blocks | val block | test block | selected alpha | model | MAE 3D | RMSE 3D | P95 3D |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(fold_rows)
    lines.extend(["", "## Split Details", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Mean Alpha Sensitivity", ""])
    lines.extend(alpha_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    overall_rows: list[str] = []
    fold_rows: list[str] = []
    best_rows: list[str] = []
    notes: list[str] = []
    alpha_rows: list[str] = []
    for feature_set in args.feature_sets:
        for horizon_ms, lookback_ms in [parse_window(text) for text in args.windows]:
            case_overall, case_folds, case_best, case_notes, case_alpha = run_case(
                args.data_dir,
                feature_set,
                horizon_ms,
                lookback_ms,
                args.block_count,
                args.min_train_blocks,
                args.pred_dir,
            )
            overall_rows.extend(case_overall)
            fold_rows.extend(case_folds)
            best_rows.append(case_best)
            notes.extend(case_notes)
            alpha_rows.extend(case_alpha)
            print(f"Ran {feature_set} h={horizon_ms:g} l={lookback_ms:g}")
    write_report(
        args.report,
        overall_rows,
        fold_rows,
        best_rows,
        notes,
        alpha_rows,
        args.pred_dir,
        args.block_count,
        args.min_train_blocks,
    )
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
