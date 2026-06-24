#!/usr/bin/env python3
"""Windowed baseline on exported ArduPilot DataFlash CSV files.

This experiment is intentionally lightweight: it uses only numpy, synchronizes
already exported DataFlash messages by TimeUS, aggregates feature windows before
each POS point, and predicts future local displacement from POS.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_SETS = {
    "imu": ["IMU", "BARO"],
    "imu_att": ["IMU", "ATT", "BARO"],
    "no_motors": ["IMU", "ATT", "BARO"],
    "all": ["IMU", "ATT", "BARO", "BAT", "MOTB", "RCOU_motor_features"],
}


@dataclass
class Series:
    name: str
    times_us: np.ndarray
    values: np.ndarray
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="all")
    parser.add_argument("--horizon-ms", type=float, default=1000.0)
    parser.add_argument("--lookback-ms", type=float, default=1000.0)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--report", type=Path, default=Path("reports/dataflash_baseline.md"))
    return parser.parse_args()


def as_float(value: str) -> float:
    value = value.strip()
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_numeric_csv(path: Path, prefix: str, drop_columns: set[str] | None = None) -> Series:
    drop_columns = drop_columns or set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if "TimeUS" not in fieldnames:
            raise ValueError(f"{path} has no TimeUS column")
        value_columns = [name for name in fieldnames if name != "TimeUS" and name not in drop_columns]
        times: list[float] = []
        rows: list[list[float]] = []
        for row in reader:
            ts = as_float(row.get("TimeUS", ""))
            if math.isnan(ts):
                continue
            values = [as_float(row.get(name, "")) for name in value_columns]
            if any(math.isnan(value) for value in values):
                continue
            times.append(ts)
            rows.append(values)

    if not rows:
        raise ValueError(f"No numeric rows in {path}")

    return Series(
        name=prefix,
        times_us=np.asarray(times, dtype=np.float64),
        values=np.asarray(rows, dtype=np.float64),
        feature_names=[f"{prefix}.{name}" for name in value_columns],
    )


def read_pos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        times: list[float] = []
        gps: list[tuple[float, float, float]] = []
        for row in reader:
            ts = as_float(row.get("TimeUS", ""))
            lat = as_float(row.get("Lat", ""))
            lng = as_float(row.get("Lng", ""))
            alt = as_float(row.get("Alt", ""))
            if any(math.isnan(value) for value in (ts, lat, lng, alt)):
                continue
            times.append(ts)
            gps.append((lat, lng, alt))

    if not gps:
        raise ValueError(f"No POS rows in {path}")

    origin = gps[0]
    positions = np.asarray([gps_to_local_m(lat, lng, alt, origin) for lat, lng, alt in gps])
    return np.asarray(times, dtype=np.float64), positions


def gps_to_local_m(
    lat: float,
    lon: float,
    alt: float,
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    lat0, lon0, alt0 = origin
    earth_radius_m = 6_371_000.0
    north = math.radians(lat - lat0) * earth_radius_m
    east = math.radians(lon - lon0) * earth_radius_m * math.cos(math.radians(lat0))
    up = alt - alt0
    return east, north, up


def append_synthetic(series: Series) -> Series:
    names = list(series.feature_names)
    values = series.values
    extra_values: list[np.ndarray] = []
    extra_names: list[str] = []

    name_to_idx = {name.split(".", 1)[1]: idx for idx, name in enumerate(names)}
    for label, columns in {
        "gyro_norm": ["GyrX", "GyrY", "GyrZ"],
        "acc_norm": ["AccX", "AccY", "AccZ"],
        "att_sin_yaw": ["Yaw"],
        "att_cos_yaw": ["Yaw"],
        "motor_power_proxy": ["motor_mean_norm", "motor_std"],
    }.items():
        if all(column in name_to_idx for column in columns):
            data = values[:, [name_to_idx[column] for column in columns]]
            if label.endswith("sin_yaw"):
                extra = np.sin(np.deg2rad(data[:, 0]))
            elif label.endswith("cos_yaw"):
                extra = np.cos(np.deg2rad(data[:, 0]))
            elif label == "motor_power_proxy":
                extra = data[:, 0] * data[:, 0] + data[:, 1] / 1000.0
            else:
                extra = np.sqrt(np.sum(data * data, axis=1))
            extra_values.append(extra[:, None])
            extra_names.append(f"{series.name}.{label}")

    if not extra_values:
        return series

    return Series(
        name=series.name,
        times_us=series.times_us,
        values=np.hstack([values, *extra_values]),
        feature_names=names + extra_names,
    )


def aggregate_for_time(series: Series, time_us: float, lookback_us: float) -> np.ndarray | None:
    end = np.searchsorted(series.times_us, time_us, side="right")
    start = np.searchsorted(series.times_us, time_us - lookback_us, side="left")
    if end <= start:
        return None
    window = series.values[start:end]
    return np.concatenate([window[-1], window.mean(axis=0), window.std(axis=0), window[-1] - window[0]])


def aggregate_feature_names(series_list: list[Series]) -> list[str]:
    names: list[str] = []
    for series in series_list:
        for suffix in ("last", "mean", "std", "delta"):
            names.extend([f"{name}_{suffix}" for name in series.feature_names])
    return names


def make_dataset(
    pos_times: np.ndarray,
    positions: np.ndarray,
    series_list: list[Series],
    horizon_us: float,
    lookback_us: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    future_indices = np.searchsorted(pos_times, pos_times + horizon_us, side="left")
    for idx, future_idx in enumerate(future_indices):
        if future_idx >= len(pos_times):
            continue
        parts = [aggregate_for_time(series, pos_times[idx], lookback_us) for series in series_list]
        if any(part is None for part in parts):
            continue
        rows.append(np.concatenate([part for part in parts if part is not None]))
        targets.append(positions[future_idx] - positions[idx])

    return np.vstack(rows), np.vstack(targets), aggregate_feature_names(series_list)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-9] = 1.0
    y_mean = y.mean(axis=0)
    xz = (x - x_mean) / x_std
    yc = y - y_mean
    weights = np.linalg.solve(xz.T @ xz + alpha * np.eye(xz.shape[1]), xz.T @ yc)
    return weights, x_mean, x_std, y_mean


def predict_ridge(x: np.ndarray, weights: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    return ((x - x_mean) / x_std) @ weights + y_mean


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray | float]:
    err = y_pred - y_true
    mae_axis = np.mean(np.abs(err), axis=0)
    rmse_axis = np.sqrt(np.mean(err * err, axis=0))
    dist_err = np.linalg.norm(err, axis=1)
    return {
        "mae_axis": mae_axis,
        "rmse_axis": rmse_axis,
        "mae_3d": float(np.mean(dist_err)),
        "rmse_3d": float(np.sqrt(np.mean(dist_err * dist_err))),
    }


def format_metric_row(name: str, values: dict[str, np.ndarray | float]) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, np.ndarray)
    assert isinstance(rmse, np.ndarray)
    return (
        f"| {name} | {mae[0]:.3f} | {mae[1]:.3f} | {mae[2]:.3f} | "
        f"{values['mae_3d']:.3f} | {rmse[0]:.3f} | {rmse[1]:.3f} | "
        f"{rmse[2]:.3f} | {values['rmse_3d']:.3f} |"
    )


def top_features(feature_names: list[str], weights: np.ndarray, limit: int = 20) -> list[tuple[str, float]]:
    scores = np.linalg.norm(weights, axis=1)
    order = np.argsort(scores)[::-1][:limit]
    return [(feature_names[idx], float(scores[idx])) for idx in order]


def write_report(
    path: Path,
    feature_set: str,
    source_files: list[Path],
    horizon_ms: float,
    lookback_ms: float,
    train_count: int,
    test_count: int,
    zero_metrics: dict[str, np.ndarray | float],
    mean_metrics: dict[str, np.ndarray | float],
    ridge_metrics: dict[str, np.ndarray | float],
    important: list[tuple[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash baseline",
        "",
        f"- feature set: `{feature_set}`",
        f"- horizon: `{horizon_ms:g}` ms",
        f"- lookback: `{lookback_ms:g}` ms",
        f"- train windows: `{train_count}`",
        f"- test windows: `{test_count}`",
        "",
        "## Sources",
        "",
    ]
    lines.extend([f"- `{source}`" for source in source_files])
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "Chronological split inside one DataFlash log. Error is future POS displacement in meters.",
            "",
            "| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            format_metric_row("zero displacement", zero_metrics),
            format_metric_row("train mean displacement", mean_metrics),
            format_metric_row("ridge baseline", ridge_metrics),
            "",
            "## Feature importance",
            "",
            "| feature | score |",
            "| --- | ---: |",
        ]
    )
    for name, score in important:
        lines.append(f"| `{name}` | {score:.5f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    message_names = FEATURE_SETS[args.feature_set]
    source_files = [args.data_dir / f"{name}.csv" for name in message_names]
    series_list = [
        append_synthetic(read_numeric_csv(path, path.stem, drop_columns={"I", "Inst", "Health", "H", "SH", "FailFlags"}))
        for path in source_files
    ]

    pos_times, positions = read_pos(args.data_dir / "POS.csv")
    x, y, feature_names = make_dataset(
        pos_times,
        positions,
        series_list,
        horizon_us=args.horizon_ms * 1000.0,
        lookback_us=args.lookback_ms * 1000.0,
    )

    split = max(1, min(len(x) - 1, int(len(x) * args.train_frac)))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    weights, x_mean, x_std, y_mean = fit_ridge(x_train, y_train, args.ridge_alpha)
    ridge_pred = predict_ridge(x_test, weights, x_mean, x_std, y_mean)
    zero_pred = np.zeros_like(y_test)
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)

    zero_metrics = metrics(y_test, zero_pred)
    mean_metrics = metrics(y_test, mean_pred)
    ridge_metrics = metrics(y_test, ridge_pred)

    write_report(
        args.report,
        args.feature_set,
        source_files,
        args.horizon_ms,
        args.lookback_ms,
        len(x_train),
        len(x_test),
        zero_metrics,
        mean_metrics,
        ridge_metrics,
        top_features(feature_names, weights),
    )

    print(f"Wrote {args.report}")
    print(format_metric_row("zero displacement", zero_metrics))
    print(format_metric_row("train mean displacement", mean_metrics))
    print(format_metric_row("ridge baseline", ridge_metrics))


if __name__ == "__main__":
    main()
