#!/usr/bin/env python3
"""Run hover/moving two-stage baselines on window datasets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from run_window_baselines import (
    ALPHAS,
    Dataset,
    Split,
    default_splits,
    fit_ridge,
    mask_for,
    metrics,
    predict_ridge,
    read_dataset,
)


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=Path,
        default=sorted(
            path
            for path in Path("derived/datasets").glob("windows_module_h*_l*.npz")
            if "_trim" not in path.stem and "_pathrel" not in path.stem and "_move" not in path.stem
        ),
    )
    parser.add_argument("--moving-threshold-m", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/module_window_two_stage.md"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/module_window_two_stage"))
    return parser.parse_args()


def horizontal_norm(y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y[:, :2], axis=1)


def moving_labels(y: np.ndarray, threshold_m: float) -> np.ndarray:
    return horizontal_norm(y) >= threshold_m


def fit_ridge_binary(x: np.ndarray, labels: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    params = fit_ridge(x, labels.astype(np.float64)[:, None], alpha)
    weights, mean, std, y_mean = params
    return weights[:, 0], mean, std, float(y_mean[0])


def predict_ridge_binary(x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> np.ndarray:
    weights, mean, std, y_mean = params
    return ((x - mean) / std) @ weights + y_mean


def classifier_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(true & pred))
    tn = int(np.sum(~true & ~pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))
    total = max(len(true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "moving_rate_true": float(np.mean(true)),
        "moving_rate_pred": float(np.mean(pred)),
    }


def choose_classifier(
    dataset: Dataset,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    moving_threshold_m: float,
) -> tuple[float, float, tuple[np.ndarray, np.ndarray, np.ndarray, float], dict[str, float]]:
    y_train_label = moving_labels(dataset.y[train_mask], moving_threshold_m)
    y_val_label = moving_labels(dataset.y[val_mask], moving_threshold_m)
    best_alpha = ALPHAS[0]
    best_threshold = THRESHOLDS[0]
    best_params = fit_ridge_binary(dataset.x[train_mask], y_train_label, best_alpha)
    best_metrics = classifier_metrics(y_val_label, predict_ridge_binary(dataset.x[val_mask], best_params) >= best_threshold)
    best_f1 = -1.0
    for alpha in ALPHAS:
        params = fit_ridge_binary(dataset.x[train_mask], y_train_label, alpha)
        scores = predict_ridge_binary(dataset.x[val_mask], params)
        for threshold in THRESHOLDS:
            pred = scores >= threshold
            current = classifier_metrics(y_val_label, pred)
            if current["f1"] > best_f1:
                best_f1 = current["f1"]
                best_alpha = alpha
                best_threshold = threshold
                best_params = params
                best_metrics = current
    return best_alpha, best_threshold, best_params, best_metrics


def choose_regressor(
    dataset: Dataset,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    moving_threshold_m: float,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    train_moving = train_mask & moving_labels(dataset.y, moving_threshold_m)
    val_moving = val_mask & moving_labels(dataset.y, moving_threshold_m)
    if not train_moving.any():
        raise ValueError(f"{dataset.name}: no moving train windows")
    if not val_moving.any():
        val_moving = val_mask
    best_alpha = ALPHAS[0]
    best_params = fit_ridge(dataset.x[train_moving], dataset.y[train_moving], best_alpha)
    best_score = math.inf
    for alpha in ALPHAS:
        params = fit_ridge(dataset.x[train_moving], dataset.y[train_moving], alpha)
        pred = predict_ridge(dataset.x[val_moving], params)
        score = float(metrics(dataset.y[val_moving], pred)["mae_3d"])
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_params = params
    return best_alpha, best_params


def two_stage_predict(
    x: np.ndarray,
    classifier_params: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    classifier_threshold: float,
    regressor_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    moving_pred = predict_ridge_binary(x, classifier_params) >= classifier_threshold
    pred = np.zeros((len(x), regressor_params[0].shape[1]), dtype=np.float64)
    if np.any(moving_pred):
        pred[moving_pred] = predict_ridge(x[moving_pred], regressor_params)
    return pred, moving_pred


def oracle_gate_predict(
    dataset: Dataset,
    mask: np.ndarray,
    moving_threshold_m: float,
    regressor_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    y_true = dataset.y[mask]
    moving_true = moving_labels(y_true, moving_threshold_m)
    x = dataset.x[mask]
    pred = np.zeros_like(y_true)
    if np.any(moving_true):
        pred[moving_true] = predict_ridge(x[moving_true], regressor_params)
    return pred


def clip_prediction_norm(pred: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(pred, axis=1)
    scale = np.ones_like(norm)
    over = norm > max_norm
    scale[over] = max_norm / np.maximum(norm[over], 1e-9)
    return pred * scale[:, None]


def metric_row(dataset: str, split: str, subset: str, model: str, values: dict[str, float | list[float]]) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, list)
    assert isinstance(rmse, list)
    return (
        f"| `{dataset}` | `{split}` | `{subset}` | `{model}` | "
        f"{mae[0]:.3f} | {mae[1]:.3f} | {mae[2]:.3f} | {values['mae_3d']:.3f} | "
        f"{rmse[0]:.3f} | {rmse[1]:.3f} | {rmse[2]:.3f} | {values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def write_predictions(
    path: Path,
    dataset: Dataset,
    mask: np.ndarray,
    pred: np.ndarray,
    moving_pred: np.ndarray,
    split_name: str,
    model_name: str,
    moving_threshold_m: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.nonzero(mask)[0]
    fieldnames = [
        "dataset",
        "split",
        "model",
        "flight_id",
        "time_s",
        "future_time_s",
        "true_moving",
        "pred_moving",
        "true_dx_east_m",
        "true_dy_north_m",
        "true_dz_up_m",
        "pred_dx_east_m",
        "pred_dy_north_m",
        "pred_dz_up_m",
    ]
    y_true_moving = moving_labels(dataset.y[mask], moving_threshold_m)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, data_idx in enumerate(indices):
            writer.writerow(
                {
                    "dataset": dataset.name,
                    "split": split_name,
                    "model": model_name,
                    "flight_id": dataset.flight_id[data_idx],
                    "time_s": f"{dataset.time_s[data_idx]:.6f}",
                    "future_time_s": f"{dataset.future_time_s[data_idx]:.6f}",
                    "true_moving": "1" if y_true_moving[row_idx] else "0",
                    "pred_moving": "1" if moving_pred[row_idx] else "0",
                    "true_dx_east_m": f"{dataset.y[data_idx, 0]:.6f}",
                    "true_dy_north_m": f"{dataset.y[data_idx, 1]:.6f}",
                    "true_dz_up_m": f"{dataset.y[data_idx, 2]:.6f}",
                    "pred_dx_east_m": f"{pred[row_idx, 0]:.6f}",
                    "pred_dy_north_m": f"{pred[row_idx, 1]:.6f}",
                    "pred_dz_up_m": f"{pred[row_idx, 2]:.6f}",
                }
            )


def run_split(
    dataset: Dataset,
    split: Split,
    moving_threshold_m: float,
    pred_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    train_mask = mask_for(dataset.flight_id, split.train)
    val_mask = mask_for(dataset.flight_id, split.val)
    test_mask = mask_for(dataset.flight_id, split.test)
    cls_alpha, cls_threshold, cls_params, cls_val_metrics = choose_classifier(
        dataset,
        train_mask,
        val_mask,
        moving_threshold_m,
    )
    reg_alpha, reg_params = choose_regressor(dataset, train_mask, val_mask, moving_threshold_m)
    train_moving = train_mask & moving_labels(dataset.y, moving_threshold_m)
    clip_norm = float(np.percentile(np.linalg.norm(dataset.y[train_moving], axis=1), 95)) if train_moving.any() else math.inf

    rows: list[str] = []
    best_rows: list[str] = []
    notes: list[str] = []
    for subset_name, mask in [("val", val_mask), ("test", test_mask)]:
        y_true = dataset.y[mask]
        zero_pred = np.zeros_like(y_true)
        train_mean = np.repeat(dataset.y[train_mask].mean(axis=0)[None, :], len(y_true), axis=0)
        two_pred, moving_pred = two_stage_predict(dataset.x[mask], cls_params, cls_threshold, reg_params)
        oracle_pred = oracle_gate_predict(dataset, mask, moving_threshold_m, reg_params)
        two_pred_clipped = clip_prediction_norm(two_pred, clip_norm)
        oracle_pred_clipped = clip_prediction_norm(oracle_pred, clip_norm)
        oracle_moving = moving_labels(y_true, moving_threshold_m)
        models = {
            "zero": zero_pred,
            "train_mean": train_mean,
            "two_stage": two_pred,
            "two_stage_clipped": two_pred_clipped,
            "oracle_gate": oracle_pred,
            "oracle_gate_clipped": oracle_pred_clipped,
        }
        scored: list[tuple[str, dict[str, float | list[float]]]] = []
        for model_name, pred in models.items():
            values = metrics(y_true, pred)
            rows.append(metric_row(dataset.name, split.name, subset_name, model_name, values))
            scored.append((model_name, values))
            if subset_name == "test" and model_name in {"two_stage", "two_stage_clipped", "oracle_gate", "oracle_gate_clipped"}:
                pred_path = pred_dir / dataset.name / split.name / f"{model_name}_pred.csv"
                pred_moving = moving_pred if model_name.startswith("two_stage") else oracle_moving
                write_predictions(
                    pred_path,
                    dataset,
                    mask,
                    pred,
                    pred_moving,
                    split.name,
                    model_name,
                    moving_threshold_m,
                )
        if subset_name == "test":
            best_model, best_values = min(scored, key=lambda item: float(item[1]["mae_3d"]))
            best_rows.append(
                f"| `{dataset.name}` | `{split.name}` | `{best_model}` | "
                f"{float(best_values['mae_3d']):.3f} | {float(best_values['rmse_3d']):.3f} |"
            )
            true_test_moving = moving_labels(y_true, moving_threshold_m)
            cls_test = classifier_metrics(true_test_moving, moving_pred)
            notes.append(
                f"`{dataset.name}/{split.name}` train={int(train_mask.sum())} val={int(val_mask.sum())} "
                f"test={int(test_mask.sum())} moving_train={int(np.sum(moving_labels(dataset.y[train_mask], moving_threshold_m)))} "
                f"cls_alpha={cls_alpha:g} cls_threshold={cls_threshold:g} reg_alpha={reg_alpha:g} "
                f"clip_norm_p95={clip_norm:.3f} "
                f"test_cls_f1={cls_test['f1']:.3f} test_moving_true={cls_test['moving_rate_true']:.3f} "
                f"test_moving_pred={cls_test['moving_rate_pred']:.3f}"
            )
    return rows, best_rows, notes


def write_report(path: Path, metric_rows: list[str], best_rows: list[str], notes: list[str], pred_dir: Path, moving_threshold_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Module Window Two-Stage Baseline",
        "",
        "This report is generated by `src/run_two_stage_baseline.py`.",
        "",
        f"Moving label: horizontal target displacement >= `{moving_threshold_m}` m.",
        "",
        "Models:",
        "",
        "- `zero`: always predicts no displacement.",
        "- `train_mean`: predicts mean train displacement.",
        "- `two_stage`: predicts hover/moving from sensor features; if moving, uses ridge regression.",
        "- `two_stage_clipped`: same as two_stage, with prediction norm clipped to train moving target p95.",
        "- `oracle_gate`: uses the true hover/moving label and is only a diagnostic upper bound for the gate.",
        "- `oracle_gate_clipped`: same as oracle_gate, with prediction norm clipped to train moving target p95.",
        "",
        f"Predictions: `{pred_dir}`",
        "",
        "## Metrics",
        "",
        "| dataset | split | subset | model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D | P95 3D |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(metric_rows)
    lines.extend(
        [
            "",
            "## Best Test Baseline",
            "",
            "| dataset | split | model | MAE 3D | RMSE 3D |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    lines.extend(best_rows)
    lines.extend(["", "## Split Details", ""])
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metric_rows: list[str] = []
    best_rows: list[str] = []
    notes: list[str] = []
    for path in args.datasets:
        dataset = read_dataset(path)
        splits = default_splits(set(dataset.flight_id.tolist()))
        for split in splits:
            rows, best, split_notes = run_split(dataset, split, args.moving_threshold_m, args.pred_dir)
            metric_rows.extend(rows)
            best_rows.extend(best)
            notes.extend(split_notes)
            print(f"Ran {dataset.name}/{split.name}")
    write_report(args.report, metric_rows, best_rows, notes, args.pred_dir, args.moving_threshold_m)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
