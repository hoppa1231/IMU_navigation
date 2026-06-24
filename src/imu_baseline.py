#!/usr/bin/env python3
"""Streaming baseline for IMU/GPS displacement prediction.

The script intentionally depends only on numpy from the scientific stack.
It reads large semicolon-separated telemetry CSV files in a streaming way,
converts GPS coordinates to local meters, builds fixed-horizon displacement
targets, trains a ridge regression baseline, and writes a compact Markdown
report with metrics and feature importance.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GPS_COLUMNS = {
    "LatMinutes",
    "LatDegrees",
    "LonMinutes",
    "LonDegrees",
    "Altitude, m",
}

DROP_FEATURES = GPS_COLUMNS | {"TimeStamp", "Time"}


@dataclass
class FlightSamples:
    path: Path
    times_ms: np.ndarray
    features: np.ndarray
    positions_m: np.ndarray
    feature_names: list[str]
    rows_seen: int
    rows_with_gps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        nargs="+",
        type=Path,
        default=[
            Path("artifacts/linear_15_01_2025.csv"),
            Path("artifacts/triangle_15_01_2025.csv"),
        ],
        help="Telemetry CSV files.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("artifacts/triangle_15_01_2025.csv"),
        help="File used as holdout. Other files are used for training.",
    )
    parser.add_argument("--sample-ms", type=float, default=100.0)
    parser.add_argument("--horizon-ms", type=float, default=1000.0)
    parser.add_argument(
        "--lookback-ms",
        type=float,
        default=0.0,
        help="Past window used for feature aggregates. 0 means current sample only.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-samples-per-file", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_report.md"))
    return parser.parse_args()


def as_float(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def has_gps(row: dict[str, str]) -> bool:
    return any(abs(as_float(row.get(col, "0"))) > 1e-12 for col in GPS_COLUMNS)


def gps_decimal(row: dict[str, str]) -> tuple[float, float, float]:
    lat = as_float(row["LatDegrees"]) + as_float(row["LatMinutes"]) / 60.0
    lon = as_float(row["LonDegrees"]) + as_float(row["LonMinutes"]) / 60.0
    alt = as_float(row["Altitude, m"])
    return lat, lon, alt


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


def build_feature_names(header: Iterable[str]) -> list[str]:
    base = [name.strip() for name in header if name.strip() and name.strip() not in DROP_FEATURES]
    synthetic = [
        "acc_norm",
        "gyro_norm",
        "mag1_norm",
        "mag2_norm",
        "flow_norm",
        "lidar_m",
        "altbar_minus_lidar_m",
    ]
    return base + synthetic


def build_features(row: dict[str, str], base_names: list[str]) -> list[float]:
    values = [as_float(row.get(name, "0")) for name in base_names]

    xacc = as_float(row.get("Xacc, g", "0"))
    yacc = as_float(row.get("Yacc, g", "0"))
    zacc = as_float(row.get("Zacc, g", "0"))
    xgyro = as_float(row.get("Xgyro, DPS", "0"))
    ygyro = as_float(row.get("Ygyro, DPS", "0"))
    zgyro = as_float(row.get("Zgyro, DPS", "0"))
    xmag1 = as_float(row.get("Xmag1, mG", "0"))
    ymag1 = as_float(row.get("Ymag1, mG", "0"))
    zmag1 = as_float(row.get("Zmag1, mG", "0"))
    xmag2 = as_float(row.get("Xmag2, uT", "0"))
    ymag2 = as_float(row.get("Ymag2, uT", "0"))
    zmag2 = as_float(row.get("Zmag2, uT", "0"))
    xflow = as_float(row.get("Xflow", "0"))
    yflow = as_float(row.get("Yflow", "0"))
    lidar_m = as_float(row.get("Lidar, sm", "0")) / 100.0
    altbar = as_float(row.get("AltBar, m", "0"))

    values.extend(
        [
            math.sqrt(xacc * xacc + yacc * yacc + zacc * zacc),
            math.sqrt(xgyro * xgyro + ygyro * ygyro + zgyro * zgyro),
            math.sqrt(xmag1 * xmag1 + ymag1 * ymag1 + zmag1 * zmag1),
            math.sqrt(xmag2 * xmag2 + ymag2 * ymag2 + zmag2 * zmag2),
            math.sqrt(xflow * xflow + yflow * yflow),
            lidar_m,
            altbar - lidar_m,
        ]
    )
    return values


def read_flight(path: Path, sample_ms: float, max_samples: int = 0) -> FlightSamples:
    times: list[float] = []
    features: list[list[float]] = []
    positions: list[tuple[float, float, float]] = []

    rows_seen = 0
    rows_with_gps = 0
    origin: tuple[float, float, float] | None = None
    next_sample_ts: float | None = None

    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        feature_names = build_feature_names(reader.fieldnames)
        base_names = [name for name in feature_names if name not in {
            "acc_norm",
            "gyro_norm",
            "mag1_norm",
            "mag2_norm",
            "flow_norm",
            "lidar_m",
            "altbar_minus_lidar_m",
        }]

        for raw_row in reader:
            rows_seen += 1
            row = {str(k).strip(): str(v).strip() for k, v in raw_row.items() if k is not None}
            if not has_gps(row):
                continue
            rows_with_gps += 1

            ts = as_float(row.get("TimeStamp", "0"))
            if next_sample_ts is None:
                next_sample_ts = ts
            if ts + 1e-9 < next_sample_ts:
                continue

            lat, lon, alt = gps_decimal(row)
            if origin is None:
                origin = (lat, lon, alt)

            times.append(ts)
            positions.append(gps_to_local_m(lat, lon, alt, origin))
            features.append(build_features(row, base_names))
            next_sample_ts = ts + sample_ms

            if max_samples and len(times) >= max_samples:
                break

    if not times:
        raise ValueError(f"No GPS samples found in {path}")

    return FlightSamples(
        path=path,
        times_ms=np.asarray(times, dtype=np.float64),
        features=np.asarray(features, dtype=np.float64),
        positions_m=np.asarray(positions, dtype=np.float64),
        feature_names=feature_names,
        rows_seen=rows_seen,
        rows_with_gps=rows_with_gps,
    )


def expanded_feature_names(feature_names: list[str], lookback_steps: int) -> list[str]:
    if lookback_steps <= 0:
        return feature_names
    names: list[str] = []
    for suffix in ("last", "mean", "std", "delta"):
        names.extend([f"{name}_{suffix}" for name in feature_names])
    return names


def aggregate_window(window: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            window[-1],
            window.mean(axis=0),
            window.std(axis=0),
            window[-1] - window[0],
        ]
    )


def make_supervised(
    samples: FlightSamples,
    horizon_ms: float,
    lookback_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    target_indices = np.searchsorted(samples.times_ms, samples.times_ms + horizon_ms, side="left")
    valid = target_indices < len(samples.times_ms)
    if lookback_steps > 0:
        valid[:lookback_steps] = False
    idx = np.nonzero(valid)[0]
    future_idx = target_indices[valid]
    if lookback_steps > 0:
        x = np.vstack([
            aggregate_window(samples.features[i - lookback_steps:i + 1])
            for i in idx
        ])
    else:
        x = samples.features[idx]
    y = samples.positions_m[future_idx] - samples.positions_m[idx]
    return x, y


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-9] = 1.0
    y_mean = y.mean(axis=0)

    xz = (x - x_mean) / x_std
    yc = y - y_mean
    xtx = xz.T @ xz
    reg = alpha * np.eye(xtx.shape[0])
    weights = np.linalg.solve(xtx + reg, xz.T @ yc)
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


def top_weight_features(feature_names: list[str], weights: np.ndarray, limit: int = 15) -> list[tuple[str, float]]:
    scores = np.linalg.norm(weights, axis=1)
    order = np.argsort(scores)[::-1][:limit]
    return [(feature_names[i], float(scores[i])) for i in order]


def top_correlations(feature_names: list[str], x: np.ndarray, y: np.ndarray, limit: int = 15) -> list[tuple[str, float]]:
    y_norm = np.linalg.norm(y, axis=1)
    result: list[tuple[str, float]] = []
    for i, name in enumerate(feature_names):
        col = x[:, i]
        if np.std(col) < 1e-9 or np.std(y_norm) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y_norm)[0, 1])
            if math.isnan(corr):
                corr = 0.0
        result.append((name, abs(corr)))
    result.sort(key=lambda item: item[1], reverse=True)
    return result[:limit]


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


def write_report(
    report_path: Path,
    sample_ms: float,
    horizon_ms: float,
    lookback_ms: float,
    train_files: list[FlightSamples],
    test_files: list[FlightSamples],
    train_count: int,
    test_count: int,
    model_metrics: dict[str, np.ndarray | float],
    zero_metrics: dict[str, np.ndarray | float],
    important: list[tuple[str, float]],
    correlations: list[tuple[str, float]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline report",
        "",
        f"- sample step: `{sample_ms:g}` ms",
        f"- target horizon: `{horizon_ms:g}` ms",
        f"- lookback window: `{lookback_ms:g}` ms",
        f"- train windows: `{train_count}`",
        f"- test windows: `{test_count}`",
        "",
        "## Files",
        "",
        "| split | file | rows seen | rows with GPS | sampled GPS points |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in train_files:
        lines.append(
            f"| train | `{item.path}` | {item.rows_seen} | {item.rows_with_gps} | {len(item.times_ms)} |"
        )
    for item in test_files:
        lines.append(
            f"| test | `{item.path}` | {item.rows_seen} | {item.rows_with_gps} | {len(item.times_ms)} |"
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "Error is measured for GPS displacement over the selected horizon, in meters.",
            "",
            "| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            format_metric_row("zero displacement", zero_metrics),
            format_metric_row("ridge baseline", model_metrics),
            "",
            "## Feature importance",
            "",
            "Importance is the norm of standardized ridge weights over `dx/dy/dz`.",
            "",
            "| feature | score |",
            "| --- | ---: |",
        ]
    )
    for name, score in important:
        lines.append(f"| `{name}` | {score:.5f} |")

    lines.extend(
        [
            "",
            "## Simple correlations",
            "",
            "Absolute Pearson correlation with the 3D target displacement length on train data.",
            "",
            "| feature | abs corr |",
            "| --- | ---: |",
        ]
    )
    for name, score in correlations:
        lines.append(f"| `{name}` | {score:.5f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a first sanity-check, not the final navigation model.",
            "- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.",
            "- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples = [
        read_flight(path, args.sample_ms, args.max_samples_per_file)
        for path in args.csv
    ]

    lookback_steps = int(round(args.lookback_ms / args.sample_ms)) if args.lookback_ms > 0 else 0
    feature_names = expanded_feature_names(samples[0].feature_names, lookback_steps)
    for item in samples[1:]:
        if expanded_feature_names(item.feature_names, lookback_steps) != feature_names:
            raise ValueError("CSV files have different feature columns")

    test_path = args.test_file.resolve()
    test_samples = [item for item in samples if item.path.resolve() == test_path]
    train_samples = [item for item in samples if item.path.resolve() != test_path]
    if not test_samples:
        test_samples = [samples[-1]]
        train_samples = samples[:-1]
    if not train_samples:
        item = samples[0]
        x, y = make_supervised(item, args.horizon_ms, lookback_steps)
        split = max(1, int(len(x) * 0.8))
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]
        train_samples = [item]
        test_samples = [item]
    else:
        train_xy = [make_supervised(item, args.horizon_ms, lookback_steps) for item in train_samples]
        test_xy = [make_supervised(item, args.horizon_ms, lookback_steps) for item in test_samples]
        x_train = np.vstack([item[0] for item in train_xy])
        y_train = np.vstack([item[1] for item in train_xy])
        x_test = np.vstack([item[0] for item in test_xy])
        y_test = np.vstack([item[1] for item in test_xy])

    weights, x_mean, x_std, y_mean = fit_ridge(x_train, y_train, args.ridge_alpha)
    pred = predict_ridge(x_test, weights, x_mean, x_std, y_mean)
    zero = np.zeros_like(y_test)

    model_metrics = metrics(y_test, pred)
    zero_metrics = metrics(y_test, zero)
    important = top_weight_features(feature_names, weights)
    correlations = top_correlations(feature_names, x_train, y_train)

    write_report(
        args.report,
        args.sample_ms,
        args.horizon_ms,
        args.lookback_ms,
        train_samples,
        test_samples,
        len(x_train),
        len(x_test),
        model_metrics,
        zero_metrics,
        important,
        correlations,
    )

    print(f"Wrote {args.report}")
    print(format_metric_row("zero displacement", zero_metrics))
    print(format_metric_row("ridge baseline", model_metrics))


if __name__ == "__main__":
    main()
