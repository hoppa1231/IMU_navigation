#!/usr/bin/env python3
"""Train GRU/LSTM on module sensor sequences with route-level holdouts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("derived/datasets/windows_module_h1000_l1000.npz"))
    p.add_argument("--sequence-len", type=int, default=20)
    p.add_argument("--test-flights", nargs="+", default=["circle_07_02_2025", "square_07_02_2025"])
    p.add_argument("--validation-flight", default="module_data_s07")
    p.add_argument("--models", nargs="+", choices=["gru_64", "lstm_64"], default=["gru_64", "lstm_64"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-train-sequences", type=int, default=12000)
    p.add_argument("--feature-mode", choices=["last", "last_mean", "all"], default="last_mean")
    p.add_argument("--seed", type=int, default=20260719)
    p.add_argument("--report", type=Path, default=Path("reports/experiments/module_recurrent_route_holdout.md"))
    p.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/module_recurrent_route_holdout"))
    return p.parse_args()


def sequence_indices(flights: np.ndarray, times: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    histories, targets = [], []
    for flight in sorted(set(flights.tolist())):
        idx = np.flatnonzero(flights == flight)
        idx = idx[np.argsort(times[idx])]
        for end in range(length - 1, len(idx)):
            histories.append(idx[end - length + 1 : end + 1])
            targets.append(idx[end])
    return np.asarray(histories), np.asarray(targets)


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    error = np.linalg.norm(pred - y, axis=1)
    return float(error.mean()), float(np.sqrt(np.mean(error * error))), float(np.percentile(error, 95))


def select_features(x: np.ndarray, names: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    if mode == "all":
        return x, len(names)
    suffixes = {"last"} if mode == "last" else {"last", "mean"}
    indices = [idx for idx, name in enumerate(names.astype(str)) if name.rsplit("_", 1)[-1] in suffixes]
    return x[:, indices], len(indices)


def model(kind: str, length: int, features: int, seed: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    x = tf.keras.Input((length, features), name="sensor_sequence")
    recurrent = tf.keras.layers.GRU if kind == "gru_64" else tf.keras.layers.LSTM
    h = recurrent(64, dropout=0.1, name=kind)(x)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    y = tf.keras.layers.Dense(3, name="enu_displacement")(h)
    net = tf.keras.Model(x, y)
    net.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))
    return net


def write_predictions(path: Path, source: np.lib.npyio.NpzFile, target_idx: np.ndarray, pred: np.ndarray, name: str, test: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "split", "model", "flight_id", "time_s", "future_time_s", "true_dx_east_m", "true_dy_north_m", "true_dz_up_m", "pred_dx_east_m", "pred_dy_north_m", "pred_dz_up_m"]
    truth = source["y"][target_idx]
    lines = [",".join(fields)]
    for row, idx in enumerate(target_idx):
        lines.append(
            f"{Path(source.fid.name).stem},route_holdout_{test},{name},{source['flight_id'][idx]},"
            f"{source['time_s'][idx]:.6f},{source['future_time_s'][idx]:.6f},"
            f"{truth[row, 0]:.6f},{truth[row, 1]:.6f},{truth[row, 2]:.6f},"
            f"{pred[row, 0]:.6f},{pred[row, 1]:.6f},{pred[row, 2]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    a = args()
    import tensorflow as tf
    data = np.load(a.dataset, allow_pickle=False)
    x, y, flight, time = data["x"].astype(np.float32), data["y"].astype(np.float32), data["flight_id"].astype(str), data["time_s"]
    x, channel_count = select_features(x, data["feature_names"], a.feature_mode)
    hist, target = sequence_indices(flight, time, a.sequence_len)
    rows = []
    for test in a.test_flights:
        if test not in set(flight): raise ValueError(f"Unknown test flight: {test}")
        test_mask = flight[target] == test
        val_name = a.validation_flight if test != a.validation_flight else "module_data_s06"
        val_mask = flight[target] == val_name
        train_mask = ~(test_mask | val_mask)
        if a.max_train_sequences and int(train_mask.sum()) > a.max_train_sequences:
            train_indices = np.flatnonzero(train_mask)
            sampled = train_indices[np.linspace(0, len(train_indices) - 1, a.max_train_sequences, dtype=int)]
            train_mask = np.zeros_like(train_mask)
            train_mask[sampled] = True
        x_mean, x_std = x[hist[train_mask]].mean((0, 1), keepdims=True), x[hist[train_mask]].std((0, 1), keepdims=True)
        x_std[x_std < 1e-6] = 1.0
        y_mean, y_std = y[target[train_mask]].mean(0, keepdims=True), y[target[train_mask]].std(0, keepdims=True)
        y_std[y_std < 1e-6] = 1.0
        seq = ((x[hist] - x_mean) / x_std).astype(np.float32)
        target_y = ((y[target] - y_mean) / y_std).astype(np.float32)
        for model_idx, name in enumerate(a.models):
            tf.keras.backend.clear_session()
            net = model(name, a.sequence_len, channel_count, a.seed + model_idx)
            cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True, min_delta=1e-4)
            history = net.fit(seq[train_mask], target_y[train_mask], validation_data=(seq[val_mask], target_y[val_mask]), epochs=a.epochs, batch_size=a.batch_size, shuffle=True, callbacks=[cb], verbose=0)
            pred = net.predict(seq[test_mask], batch_size=a.batch_size, verbose=0) * y_std + y_mean
            mae, rmse, p95 = metrics(y[target[test_mask]], pred)
            write_predictions(a.pred_dir / Path(a.dataset).stem / f"route_holdout_{test}" / f"{name}_pred.csv", data, target[test_mask], pred, name, test)
            rows.append((test, name, int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()), len(history.history["loss"]), mae, rmse, p95))
            print(f"{test}/{name}: MAE={mae:.3f} RMSE={rmse:.3f} P95={p95:.3f}")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Module recurrent route holdout", "", "GRU/LSTM use chronological sensor-window vectors. GPS is absent from X and used only as ENU target.", "", f"- dataset: `{a.dataset}`", f"- sequence length: {a.sequence_len}", f"- feature mode: `{a.feature_mode}` ({channel_count} channels)", f"- validation route: `{a.validation_flight}` (excluded from train)", "", "| test route | model | train sequences | validation | test | epochs | MAE 3D m | RMSE 3D m | P95 3D m |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| `{t}` | `{m}` | {tr} | {va} | {te} | {ep} | {mae:.3f} | {rmse:.3f} | {p95:.3f} |" for t, m, tr, va, te, ep, mae, rmse, p95 in rows]
    a.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {a.report}")


if __name__ == "__main__": main()
