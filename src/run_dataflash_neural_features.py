#!/usr/bin/env python3
"""Compare physical sequence features and small MLPs on DataFlash folds."""

from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from dataflash_baseline import fit_ridge, predict_ridge, read_pos
from dataflash_physical_features import physical_features
from run_dataflash_rolling_validation import block_indices, concatenate_blocks
from run_dataflash_sequence_baseline import (
    SequenceDataset,
    build_prediction_rows,
    make_sequence_dataset,
    rollout_score,
    write_prediction_rows,
)
from run_dataflash_sweep import load_series, metrics, parse_window


RIDGE_ALPHAS = (10.0, 100.0, 1000.0)
MLP_CONFIGS = {
    "mlp_64": (64,),
    "mlp_128_64": (128, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--window", default="5000:5000")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/dataflash_neural_physical_features.md"),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("derived/predictions/dataflash_neural_physical_features"),
    )
    return parser.parse_args()


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden: tuple[int, ...],
    max_iter: int,
    seed: int,
) -> tuple[MLPRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit((x_train - x_mean) / x_std, (y_train - y_mean) / y_std)
    return model, x_mean, x_std, y_mean, y_std


def predict_mlp(x: np.ndarray, params: tuple[MLPRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    model, x_mean, x_std, y_mean, y_std = params
    return model.predict((x - x_mean) / x_std) * y_std + y_mean


def format_metric(model: str, fold: str, values: dict[str, float | list[float]], rollout: float) -> str:
    return (
        f"| `{model}` | {fold} | {values['mae_3d']:.3f} | {values['rmse_3d']:.3f} | "
        f"{values['p95_3d']:.3f} | {rollout:.3f} |"
    )


def main() -> None:
    args = parse_args()
    horizon_ms, lookback_ms = parse_window(args.window)
    series, _ = load_series(args.data_dir, "all")
    pos_times, positions = read_pos(args.data_dir / "POS.csv")
    dataset = make_sequence_dataset(
        pos_times, positions, series, horizon_ms * 1000.0, lookback_ms * 1000.0, args.sequence_len
    )
    physical_x, physical_names = physical_features(dataset, args.sequence_len, lookback_ms / 1000.0)
    blocks = block_indices(len(dataset.x), args.block_count)
    rows: list[str] = []
    predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    truths: dict[str, list[np.ndarray]] = defaultdict(list)
    prediction_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    details: list[str] = []

    for fold_offset, val_block in enumerate(range(args.min_train_blocks, args.block_count - 1), 1):
        train_idx = concatenate_blocks(blocks, 0, val_block)
        val_idx, test_idx = blocks[val_block], blocks[val_block + 1]
        y_train, y_val, y_test = dataset.y[train_idx], dataset.y[val_idx], dataset.y[test_idx]
        model_predictions: dict[str, np.ndarray] = {}

        for label, x in (("raw_ridge", dataset.x), ("physical_ridge", physical_x)):
            best = None
            for alpha in RIDGE_ALPHAS:
                params = fit_ridge(x[train_idx], y_train, alpha)
                score = float(metrics(y_val, predict_ridge(x[val_idx], *params))["mae_3d"])
                if best is None or score < best[0]:
                    best = (score, alpha, params)
            assert best is not None
            model_predictions[label] = predict_ridge(x[test_idx], *best[2])
            details.append(f"fold {fold_offset} {label}: validation alpha={best[1]:g}, MAE={best[0]:.3f}")

        for config_idx, (label, hidden) in enumerate(MLP_CONFIGS.items()):
            params = fit_mlp(
                physical_x[train_idx], y_train, hidden, args.max_iter, args.seed + fold_offset * 10 + config_idx
            )
            val_pred = predict_mlp(physical_x[val_idx], params)
            model_predictions[label] = predict_mlp(physical_x[test_idx], params)
            details.append(
                f"fold {fold_offset} {label}: validation MAE={metrics(y_val, val_pred)['mae_3d']:.3f}, "
                f"epochs={params[0].n_iter_}"
            )

        for label, pred in model_predictions.items():
            values = metrics(y_test, pred)
            rollout = rollout_score(dataset, test_idx, pred, "rollout_mean")
            rows.append(format_metric(label, str(fold_offset), values, rollout))
            predictions[label].append(pred)
            truths[label].append(y_test)
            prediction_rows[label].extend(
                build_prediction_rows(
                    dataset, test_idx, pred, "all_physical", horizon_ms, lookback_ms,
                    args.sequence_len, label, fold_offset, f"0-{val_block - 1}", val_block,
                    val_block + 1, 0.0,
                )
            )

    overall: list[str] = []
    for label in ("raw_ridge", "physical_ridge", *MLP_CONFIGS):
        y_true, y_pred = np.vstack(truths[label]), np.vstack(predictions[label])
        fold_rollouts = []
        for fold_offset, val_block in enumerate(range(args.min_train_blocks, args.block_count - 1)):
            fold_rollouts.append(
                rollout_score(dataset, blocks[val_block + 1], predictions[label][fold_offset], "rollout_mean")
            )
        overall.append(format_metric(label, "all", metrics(y_true, y_pred), float(np.mean(fold_rollouts))))
        write_prediction_rows(args.pred_dir / f"{label}_pred.csv", prediction_rows[label])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# DataFlash Neural and Physical Feature Experiment", "",
        "No test-time bias correction or state gating is applied. GPS/POS is used only as the displacement target.", "",
        f"- windows: `{len(dataset.x)}`", f"- sequence: `{args.sequence_len}` samples over `{lookback_ms:g}` ms",
        f"- raw features: `{dataset.x.shape[1]}`", f"- curated physical features: `{physical_x.shape[1]}` ({len(physical_names) // args.sequence_len} channels)",
        "- evaluation: three rolling folds inside one DataFlash flight", "",
        f"- random seed: `{args.seed}`", f"- maximum MLP epochs: `{args.max_iter}`", "",
        "## Overall test metrics", "",
        "| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |",
        "| --- | --- | ---: | ---: | ---: | ---: |", *overall, "",
        "## Per-fold test metrics", "",
        "| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |",
        "| --- | --- | ---: | ---: | ---: | ---: |", *rows, "", "## Validation details", "",
        *[f"- {line}" for line in details], "", "## Interpretation", "",
        "`raw_ridge` uses the original flattened all-sensor sequence. `physical_ridge` uses the same samples after channel selection and explicit physical transforms. The MLP variants use the physical features and predict displacement directly; no residual bias is added afterwards.", "",
        "The rotation-based linear acceleration assumes ArduPilot body FRD and NED attitude conventions. This is a testable feature hypothesis, not a calibrated INS mechanization.", "",
    ]
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
