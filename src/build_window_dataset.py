#!/usr/bin/env python3
"""Build supervised window datasets for GNSS-free displacement prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_flight_index import read_module_points_and_indices, split_points_with_indices


GPS_COLUMNS = {"LatMinutes", "LatDegrees", "LonMinutes", "LonDegrees", "Altitude, m"}
DROP_COLUMNS = GPS_COLUMNS | {"TimeStamp", "Time"}
SYNTHETIC_FEATURES = [
    "acc_norm",
    "gyro_norm",
    "mag1_norm",
    "mag2_norm",
    "flow_norm",
    "lidar_m",
    "altbar_minus_lidar_m",
]
AGGREGATIONS = ["last", "mean", "std", "min", "max", "delta", "integral_s"]


@dataclass
class SegmentBounds:
    row_start: int
    row_end: int


@dataclass
class SensorSeries:
    times_s: np.ndarray
    values: np.ndarray
    feature_names: list[str]
    rows_read: int
    rows_kept: int


@dataclass
class Track:
    time_s: np.ndarray
    source_time_s: np.ndarray
    positions_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-index", type=Path, default=Path("derived/datasets/flight_index.csv"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--out-dir", type=Path, default=Path("derived/datasets"))
    parser.add_argument("--report", type=Path, default=Path("reports/window_datasets.md"))
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["1000:1000", "3000:3000", "5000:5000"],
        help="Window configs as horizon_ms:lookback_ms.",
    )
    parser.add_argument("--source-format", choices=["module"], default="module")
    parser.add_argument("--sensor-sample-ms", type=float, default=20.0)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    parser.add_argument("--max-windows-per-flight", type=int, default=0)
    return parser.parse_args()


def parse_configs(values: list[str]) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected horizon_ms:lookback_ms, got {value!r}")
        horizon, lookback = value.split(":", 1)
        configs.append((int(horizon), int(lookback)))
    return configs


def as_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    try:
        number = float(value)
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def read_index(path: Path, source_format: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["source_format"] == source_format]


def build_feature_names(fieldnames: list[str]) -> tuple[list[str], list[str]]:
    base_names = [name for name in fieldnames if name and name not in DROP_COLUMNS]
    return base_names, base_names + SYNTHETIC_FEATURES


def row_features(row: dict[str, str], base_names: list[str]) -> list[float]:
    values = [as_float(row.get(name), default=0.0) for name in base_names]

    xacc = as_float(row.get("Xacc, g"))
    yacc = as_float(row.get("Yacc, g"))
    zacc = as_float(row.get("Zacc, g"))
    xgyro = as_float(row.get("Xgyro, DPS"))
    ygyro = as_float(row.get("Ygyro, DPS"))
    zgyro = as_float(row.get("Zgyro, DPS"))
    xmag1 = as_float(row.get("Xmag1, mG"))
    ymag1 = as_float(row.get("Ymag1, mG"))
    zmag1 = as_float(row.get("Zmag1, mG"))
    xmag2 = as_float(row.get("Xmag2, uT"))
    ymag2 = as_float(row.get("Ymag2, uT"))
    zmag2 = as_float(row.get("Zmag2, uT"))
    xflow = as_float(row.get("Xflow"))
    yflow = as_float(row.get("Yflow"))
    lidar_m = as_float(row.get("Lidar, sm")) / 100.0
    altbar = as_float(row.get("AltBar, m"))

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


def source_segment_bounds(
    source_path: Path,
    max_gap_s: float,
    max_jump_m: float,
) -> tuple[list[str], list[SegmentBounds]]:
    fieldnames, gps_points, row_indices = read_module_points_and_indices(source_path)
    point_segments, index_segments = split_points_with_indices(gps_points, row_indices, max_gap_s, max_jump_m)
    bounds = [
        SegmentBounds(row_start=indices[0], row_end=indices[-1])
        for points, indices in zip(point_segments, index_segments)
        if len(points) >= 2
    ]
    return fieldnames, bounds


def read_sensor_segment(
    source_path: Path,
    bounds: SegmentBounds,
    fieldnames: list[str],
    sensor_sample_ms: float,
) -> SensorSeries:
    base_names, feature_names = build_feature_names(fieldnames)
    times: list[float] = []
    values: list[list[float]] = []
    rows_read = 0
    next_sample_s: float | None = None

    with source_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for row_index, raw_row in enumerate(reader):
            if row_index < bounds.row_start:
                continue
            if row_index > bounds.row_end:
                break
            rows_read += 1
            row = {key.strip(): str(value).strip() for key, value in raw_row.items() if key is not None}
            timestamp_ms = as_float(row.get("TimeStamp"), default=math.nan)
            if not math.isfinite(timestamp_ms):
                continue
            time_s = timestamp_ms / 1000.0
            if next_sample_s is None:
                next_sample_s = time_s
            if time_s + 1e-12 < next_sample_s:
                continue
            times.append(time_s)
            values.append(row_features(row, base_names))
            next_sample_s = time_s + sensor_sample_ms / 1000.0

    if not values:
        raise ValueError(f"No sensor rows selected from {source_path} rows {bounds.row_start}..{bounds.row_end}")

    return SensorSeries(
        times_s=np.asarray(times, dtype=np.float64),
        values=np.asarray(values, dtype=np.float32),
        feature_names=feature_names,
        rows_read=rows_read,
        rows_kept=len(values),
    )


def read_track(path: Path) -> Track:
    time_s: list[float] = []
    source_time_s: list[float] = []
    positions: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_s.append(as_float(row.get("time_s")))
            source_time_s.append(as_float(row.get("source_time_s")))
            positions.append(
                (
                    as_float(row.get("east_m")),
                    as_float(row.get("north_m")),
                    as_float(row.get("up_m")),
                )
            )
    return Track(
        time_s=np.asarray(time_s, dtype=np.float64),
        source_time_s=np.asarray(source_time_s, dtype=np.float64),
        positions_m=np.asarray(positions, dtype=np.float32),
    )


def expanded_feature_names(feature_names: list[str]) -> list[str]:
    return [f"{name}_{aggregation}" for aggregation in AGGREGATIONS for name in feature_names]


def aggregate_window(times_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(times_s) >= 2:
        integral = np.trapezoid(values.astype(np.float64), times_s, axis=0)
    else:
        integral = np.zeros(values.shape[1], dtype=np.float64)
    parts = [
        values[-1],
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        values[-1] - values[0],
        integral.astype(np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


def build_windows_for_flight(
    flight_id: str,
    track: Track,
    sensor: SensorSeries,
    horizon_ms: int,
    lookback_ms: int,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    horizon_s = horizon_ms / 1000.0
    lookback_s = lookback_ms / 1000.0
    future_indices = np.searchsorted(track.time_s, track.time_s + horizon_s, side="left")
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta: list[dict[str, str]] = []

    for idx, future_idx in enumerate(future_indices):
        if future_idx >= len(track.time_s):
            continue
        end_s = track.source_time_s[idx]
        start_s = end_s - lookback_s
        sensor_start = np.searchsorted(sensor.times_s, start_s, side="left")
        sensor_end = np.searchsorted(sensor.times_s, end_s, side="right")
        if sensor_end <= sensor_start:
            continue
        window_times = sensor.times_s[sensor_start:sensor_end]
        if len(window_times) < 2 or window_times[0] - start_s > 0.25:
            continue

        x_row = aggregate_window(window_times, sensor.values[sensor_start:sensor_end])
        y_row = track.positions_m[future_idx] - track.positions_m[idx]
        rows.append(x_row)
        targets.append(y_row.astype(np.float32))
        meta.append(
            {
                "flight_id": flight_id,
                "time_s": f"{track.time_s[idx]:.6f}",
                "future_time_s": f"{track.time_s[future_idx]:.6f}",
                "source_time_s": f"{track.source_time_s[idx]:.6f}",
                "future_source_time_s": f"{track.source_time_s[future_idx]:.6f}",
                "sensor_window_start_s": f"{window_times[0]:.6f}",
                "sensor_window_end_s": f"{window_times[-1]:.6f}",
                "sensor_samples": str(len(window_times)),
            }
        )
        if max_windows and len(rows) >= max_windows:
            break

    if not rows:
        return (
            np.empty((0, len(expanded_feature_names(sensor.feature_names))), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )
    return np.vstack(rows), np.vstack(targets), meta


def write_meta_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "flight_id",
        "time_s",
        "future_time_s",
        "source_time_s",
        "future_source_time_s",
        "sensor_window_start_s",
        "sensor_window_end_s",
        "sensor_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_features_json(
    path: Path,
    feature_names: list[str],
    base_feature_names: list[str],
    horizon_ms: int,
    lookback_ms: int,
    sensor_sample_ms: float,
    source_format: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_format": source_format,
        "horizon_ms": horizon_ms,
        "lookback_ms": lookback_ms,
        "sensor_sample_ms": sensor_sample_ms,
        "target": ["dx_east_m", "dy_north_m", "dz_up_m"],
        "gps_usage": "GPS is used only for target displacement and metadata; GPS columns are excluded from X.",
        "aggregations": AGGREGATIONS,
        "base_feature_names": base_feature_names,
        "feature_names": feature_names,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_stem(horizon_ms: int, lookback_ms: int, source_format: str) -> str:
    return f"windows_{source_format}_h{horizon_ms}_l{lookback_ms}"


def write_report(path: Path, summary_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Window datasets",
        "",
        "This report is generated by `src/build_window_dataset.py`.",
        "",
        "Current scope: module CSV flights only. DataFlash is intentionally not mixed into this dataset because its sensor set and sampling rates differ.",
        "",
        "GPS/GNSS columns are excluded from input features. GPS tracks are used only to compute `dx, dy, dz` targets.",
        "",
        "## Outputs",
        "",
        "| dataset | flights | windows | features | horizon | lookback | sensor sample | X shape | y shape |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['flight_count']} | {row['windows']} | {row['features']} | "
            f"{row['horizon_ms']} ms | {row['lookback_ms']} ms | {row['sensor_sample_ms']} ms | "
            f"{row['x_shape']} | {row['y_shape']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
        ]
    )
    for row in summary_rows:
        lines.extend(
            [
                f"### `{row['dataset']}`",
                "",
                f"- NPZ: `{row['npz']}`",
                f"- features: `{row['features_json']}`",
                f"- meta: `{row['meta_csv']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configs = parse_configs(args.configs)
    records = read_index(args.flight_index, args.source_format)
    if not records:
        raise ValueError(f"No {args.source_format} records found in {args.flight_index}")

    source_cache: dict[Path, tuple[list[str], list[SegmentBounds]]] = {}
    tracks: dict[str, Track] = {}
    for record in records:
        source_path = Path(record["source_file"])
        if source_path not in source_cache:
            source_cache[source_path] = source_segment_bounds(source_path, args.max_gap_s, args.max_jump_m)
        tracks[record["flight_id"]] = read_track(args.tracks_dir / f"{record['flight_id']}_track.csv")

    # A full mission contains millions of module samples.  Read one flight at
    # a time and immediately distribute its windows to all requested configs;
    # retaining every SensorSeries at once needlessly exhausts memory.
    base_feature_names: list[str] | None = None
    feature_names: list[str] | None = None
    config_parts: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray], list[dict[str, str]]]] = {
        config: ([], [], []) for config in configs
    }
    for record in records:
        flight_id = record["flight_id"]
        source_path = Path(record["source_file"])
        fieldnames, bounds = source_cache[source_path]
        segment_index = int(record["segment_index"]) - 1
        if segment_index < 0 or segment_index >= len(bounds):
            raise ValueError(f"{flight_id}: segment index is out of range")
        sensor = read_sensor_segment(
            source_path,
            bounds[segment_index],
            fieldnames,
            args.sensor_sample_ms,
        )
        if base_feature_names is None:
            base_feature_names = sensor.feature_names
            feature_names = expanded_feature_names(base_feature_names)
        elif sensor.feature_names != base_feature_names:
            raise ValueError(f"{flight_id}: feature schema differs from first module flight")
        for horizon_ms, lookback_ms in configs:
            x_flight, y_flight, meta_flight = build_windows_for_flight(
                flight_id,
                tracks[flight_id],
                sensor,
                horizon_ms,
                lookback_ms,
                args.max_windows_per_flight,
            )
            if meta_flight:
                x_parts, y_parts, meta_rows = config_parts[(horizon_ms, lookback_ms)]
                x_parts.append(x_flight)
                y_parts.append(y_flight)
                meta_rows.extend(meta_flight)

    if base_feature_names is None or feature_names is None:
        raise ValueError("No sensor features were read")
    summary_rows: list[dict[str, object]] = []

    for horizon_ms, lookback_ms in configs:
        x_parts, y_parts, meta_rows = config_parts[(horizon_ms, lookback_ms)]

        if not x_parts:
            raise ValueError(f"No windows created for horizon={horizon_ms}, lookback={lookback_ms}")

        x = np.vstack(x_parts).astype(np.float32)
        y = np.vstack(y_parts).astype(np.float32)
        flight_ids = np.asarray([row["flight_id"] for row in meta_rows])
        time_s = np.asarray([float(row["time_s"]) for row in meta_rows], dtype=np.float64)
        future_time_s = np.asarray([float(row["future_time_s"]) for row in meta_rows], dtype=np.float64)

        stem = dataset_stem(horizon_ms, lookback_ms, args.source_format)
        npz_path = args.out_dir / f"{stem}.npz"
        features_path = args.out_dir / f"{stem}_features.json"
        meta_path = args.out_dir / f"{stem}_meta.csv"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            x=x,
            y=y,
            flight_id=flight_ids,
            time_s=time_s,
            future_time_s=future_time_s,
            feature_names=np.asarray(feature_names),
            target_names=np.asarray(["dx_east_m", "dy_north_m", "dz_up_m"]),
        )
        write_features_json(
            features_path,
            feature_names,
            base_feature_names,
            horizon_ms,
            lookback_ms,
            args.sensor_sample_ms,
            args.source_format,
        )
        write_meta_csv(meta_path, meta_rows)
        summary_rows.append(
            {
                "dataset": stem,
                "flight_count": len(set(row["flight_id"] for row in meta_rows)),
                "windows": len(meta_rows),
                "features": x.shape[1],
                "horizon_ms": horizon_ms,
                "lookback_ms": lookback_ms,
                "sensor_sample_ms": args.sensor_sample_ms,
                "x_shape": tuple(x.shape),
                "y_shape": tuple(y.shape),
                "npz": npz_path,
                "features_json": features_path,
                "meta_csv": meta_path,
            }
        )
        print(f"Wrote {npz_path} {x.shape} -> {y.shape}")

    write_report(args.report, summary_rows)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
