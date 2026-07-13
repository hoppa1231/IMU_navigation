#!/usr/bin/env python3
"""Evaluate compact GRU and LSTM models on DataFlash rolling folds."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from dataflash_baseline import read_pos
from dataflash_physical_features import physical_feature_cube
from run_dataflash_rolling_validation import block_indices, concatenate_blocks
from run_dataflash_sequence_baseline import (
    build_prediction_rows,
    make_sequence_dataset,
    rollout_score,
    write_prediction_rows,
)
from run_dataflash_sweep import load_series, metrics, parse_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--window", default="5000:5000")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=["gru_64", "lstm_64"], default=["gru_64", "lstm_64"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/experiments/dataflash_recurrent_models.md")
    )
    parser.add_argument(
        "--pred-dir", type=Path, default=Path("derived/predictions/dataflash_recurrent_models")
    )
    return parser.parse_args()


def standardize(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_std = x_train.std(axis=(0, 1), keepdims=True)
    x_std[x_std < 1e-7] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-7] = 1.0
    return x_mean, x_std, y_mean, y_std


def make_model(kind: str, sequence_len: int, channel_count: int, seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(sequence_len, channel_count), name="physical_sequence")
    recurrent = tf.keras.layers.GRU if kind == "gru_64" else tf.keras.layers.LSTM
    x = recurrent(64, dropout=0.1, name=kind)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", name="shared_dense")(x)
    outputs = tf.keras.layers.Dense(3, name="displacement_enu")(x)
    model = tf.keras.Model(inputs, outputs, name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))
    return model


def fit_predict(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    x_mean, x_std, y_mean, y_std = standardize(x_train, y_train)
    train_x = ((x_train - x_mean) / x_std).astype(np.float32)
    val_x = ((x_val - x_mean) / x_std).astype(np.float32)
    test_x = ((x_test - x_mean) / x_std).astype(np.float32)
    train_y = ((y_train - y_mean) / y_std).astype(np.float32)
    val_y = ((y_val - y_mean) / y_std).astype(np.float32)
    model = make_model(kind, train_x.shape[1], train_x.shape[2], seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, min_delta=1e-4
        )
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    val_pred = model.predict(val_x, batch_size=args.batch_size, verbose=0) * y_std + y_mean
    test_pred = model.predict(test_x, batch_size=args.batch_size, verbose=0) * y_std + y_mean
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return val_pred, test_pred, len(history.history["loss"]), best_epoch


def metric_row(model: str, fold: str, values: dict[str, float | list[float]], rollout: float) -> str:
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
    x, feature_names = physical_feature_cube(dataset, args.sequence_len, lookback_ms / 1000.0)
    blocks = block_indices(len(dataset.x), args.block_count)
    predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    truths: dict[str, list[np.ndarray]] = defaultdict(list)
    prediction_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    fold_rows: list[str] = []
    details: list[str] = []

    for fold_idx, val_block in enumerate(range(args.min_train_blocks, args.block_count - 1), start=1):
        train_idx = concatenate_blocks(blocks, 0, val_block)
        val_idx, test_idx = blocks[val_block], blocks[val_block + 1]
        for model_idx, model_name in enumerate(args.models):
            val_pred, test_pred, trained_epochs, best_epoch = fit_predict(
                model_name,
                x[train_idx], dataset.y[train_idx], x[val_idx], dataset.y[val_idx], x[test_idx],
                args, args.seed + fold_idx * 10 + model_idx,
            )
            val_mae = float(metrics(dataset.y[val_idx], val_pred)["mae_3d"])
            test_metrics = metrics(dataset.y[test_idx], test_pred)
            rollout = rollout_score(dataset, test_idx, test_pred, "rollout_mean")
            fold_rows.append(metric_row(model_name, str(fold_idx), test_metrics, rollout))
            details.append(
                f"fold {fold_idx} {model_name}: validation MAE={val_mae:.3f}, "
                f"best epoch={best_epoch}, trained epochs={trained_epochs}"
            )
            predictions[model_name].append(test_pred)
            truths[model_name].append(dataset.y[test_idx])
            prediction_rows[model_name].extend(
                build_prediction_rows(
                    dataset, test_idx, test_pred, "all_physical", horizon_ms, lookback_ms,
                    args.sequence_len, model_name, fold_idx, f"0-{val_block - 1}", val_block,
                    val_block + 1, 0.0,
                )
            )
            print(f"fold={fold_idx} model={model_name} val_mae={val_mae:.3f} test_mae={test_metrics['mae_3d']:.3f}")

    overall_rows: list[str] = []
    for model_name in args.models:
        y_true, y_pred = np.vstack(truths[model_name]), np.vstack(predictions[model_name])
        rollout_values = [
            rollout_score(dataset, blocks[val_block + 1], predictions[model_name][fold_idx], "rollout_mean")
            for fold_idx, val_block in enumerate(range(args.min_train_blocks, args.block_count - 1))
        ]
        overall_rows.append(metric_row(model_name, "all", metrics(y_true, y_pred), float(np.mean(rollout_values))))
        write_prediction_rows(args.pred_dir / f"{model_name}_pred.csv", prediction_rows[model_name])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash Recurrent Models", "",
        "GRU and LSTM predict 5-second ENU displacement directly. No bias correction or state gating is applied.", "",
        f"- input: `{args.sequence_len} x {len(feature_names)}` physical feature sequence",
        f"- windows: `{len(dataset.x)}`", "- evaluation: three rolling folds inside one DataFlash flight",
        f"- seed: `{args.seed}`", f"- maximum epochs: `{args.epochs}`", f"- early-stopping patience: `{args.patience}`", "",
        "## Overall test metrics", "",
        "| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |",
        "| --- | --- | ---: | ---: | ---: | ---: |", *overall_rows, "",
        "## Per-fold test metrics", "",
        "| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |",
        "| --- | --- | ---: | ---: | ---: | ---: |", *fold_rows, "", "## Validation details", "",
        *[f"- {detail}" for detail in details], "",
        "GPS/POS is used only as the supervised displacement target and for evaluation. Validation blocks control early stopping; test blocks are not used during training.", "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
