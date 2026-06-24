#!/usr/bin/env python3
"""Build a stable flight catalog from original GPS-capable telemetry files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from gps_flight_map import (
    GpsPoint,
    build_local_track,
    decimal_from_degrees_minutes,
    format_seconds,
    horizontal_distance_m,
    read_gps,
    split_points,
    summarize,
)


MODULE_SOURCES = [
    Path("artifacts/data.csv"),
    Path("artifacts/linear_15_01_2025.csv"),
    Path("artifacts/triangle_15_01_2025.csv"),
]

DATAFLASH_DIR = Path("derived/dataflash")
DATAFLASH_GPS = DATAFLASH_DIR / "GPS.csv"
DATAFLASH_POS = DATAFLASH_DIR / "POS.csv"
DATAFLASH_SENSOR_FILES = [
    "IMU.csv",
    "ATT.csv",
    "BARO.csv",
    "BAT.csv",
    "MOTB.csv",
    "RCOU.csv",
    "RCOU_motor_features.csv",
]

GPS_COLUMNS = {"LatMinutes", "LatDegrees", "LonMinutes", "LonDegrees", "Altitude, m"}
NON_FEATURE_COLUMNS = GPS_COLUMNS | {"TimeStamp", "Time"}

MODULE_FEATURE_GROUPS = {
    "imu_acc": {"Xacc, g", "Yacc, g", "Zacc, g"},
    "imu_gyro": {"Xgyro, DPS", "Ygyro, DPS", "Zgyro, DPS"},
    "mag1": {"Xmag1, mG", "Ymag1, mG", "Zmag1, mG"},
    "mag2": {"Xmag2, uT", "Ymag2, uT", "Zmag2, uT"},
    "optical_flow": {"Xflow", "Yflow"},
    "barometer": {"Baro, bar", "AltBar, m"},
    "lidar": {"Lidar, sm"},
    "gps": GPS_COLUMNS,
}

DATAFLASH_FEATURE_GROUPS = {
    "IMU": "imu_acc,imu_gyro,temperature",
    "ATT": "attitude",
    "BARO": "barometer",
    "BAT": "battery",
    "MOTB": "motor_telemetry",
    "RCOU": "motor_outputs",
    "RCOU_motor_features": "motor_output_features",
    "GPS": "gps_target",
    "POS": "position_target",
}


@dataclass
class FlightRecord:
    flight_id: str
    source_file: str
    source_role: str
    source_format: str
    segment_index: int
    segment_count: int
    gps_source: str
    gps_points: int
    sensor_rows: int
    duration_s: float
    gps_distance_2d_m: float
    min_alt_m: float | None
    max_alt_m: float | None
    max_internal_gap_s: float
    start_time_s: float
    end_time_s: float
    start_lat: float | None
    start_lon: float | None
    end_lat: float | None
    end_lon: float | None
    sensor_rates_hz: dict[str, float | None]
    available_feature_groups: list[str]
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-sources", type=Path, nargs="*", default=MODULE_SOURCES)
    parser.add_argument("--dataflash-dir", type=Path, default=DATAFLASH_DIR)
    parser.add_argument("--out-csv", type=Path, default=Path("derived/datasets/flight_index.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("derived/datasets/flight_index.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/flight_index.md"))
    parser.add_argument("--min-status", type=float, default=3.0)
    parser.add_argument("--max-hdop", type=float, default=None)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def slug_from_path(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def read_module_points_and_indices(path: Path) -> tuple[list[str], list[GpsPoint], list[int]]:
    points: list[GpsPoint] = []
    point_row_indices: list[int] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        reader.fieldnames = fieldnames

        for row_index, raw_row in enumerate(reader):
            row = {key.strip(): str(value).strip() for key, value in raw_row.items() if key is not None}
            ts = as_float(row.get("TimeStamp"))

            lat_minutes = as_float(row.get("LatMinutes"))
            lat_degrees = as_float(row.get("LatDegrees"))
            lon_minutes = as_float(row.get("LonMinutes"))
            lon_degrees = as_float(row.get("LonDegrees"))
            alt = as_float(row.get("Altitude, m"))
            if (
                ts is not None
                and lat_minutes is not None
                and lat_degrees is not None
                and lon_minutes is not None
                and lon_degrees is not None
                and alt is not None
            ):
                lat = decimal_from_degrees_minutes(lat_degrees, lat_minutes)
                lon = decimal_from_degrees_minutes(lon_degrees, lon_minutes)
                if abs(lat) >= 1e-12 or abs(lon) >= 1e-12:
                    points.append(
                        GpsPoint(
                            time_us=ts * 1000.0,
                            lat=lat,
                            lon=lon,
                            alt=alt,
                            speed=None,
                            status=None,
                            nsats=None,
                            hdop=None,
                        )
                    )
                    point_row_indices.append(row_index)

    return fieldnames, points, point_row_indices


def split_points_with_indices(
    points: list[GpsPoint],
    row_indices: list[int],
    max_gap_s: float,
    max_jump_m: float,
) -> tuple[list[list[GpsPoint]], list[list[int]]]:
    if not points:
        return [], []

    point_segments: list[list[GpsPoint]] = []
    index_segments: list[list[int]] = []
    current_points = [points[0]]
    current_indices = [row_indices[0]]

    for previous, point, row_index in zip(points, points[1:], row_indices[1:]):
        dt_s = (point.time_us - previous.time_us) / 1_000_000.0
        jump_m = horizontal_distance_m(previous, point)
        should_split = dt_s < 0.0 or dt_s > max_gap_s or jump_m > max_jump_m
        if should_split:
            point_segments.append(current_points)
            index_segments.append(current_indices)
            current_points = [point]
            current_indices = [row_index]
        else:
            current_points.append(point)
            current_indices.append(row_index)

    point_segments.append(current_points)
    index_segments.append(current_indices)
    return point_segments, index_segments


def rate_from_count_duration(count: int, duration_s: float) -> float | None:
    if count < 2 or duration_s <= 0.0:
        return None
    return (count - 1) / duration_s


def present_groups(fieldnames: list[str], groups: dict[str, set[str]]) -> list[str]:
    names = set(fieldnames)
    return [group for group, columns in groups.items() if columns.issubset(names)]


def feature_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name and name not in NON_FEATURE_COLUMNS]


def build_module_records(
    path: Path,
    min_status: float,
    max_hdop: float | None,
    max_gap_s: float,
    max_jump_m: float,
) -> list[FlightRecord]:
    if not path.exists():
        return []

    _ = min_status, max_hdop
    fieldnames, gps_points, gps_row_indices = read_module_points_and_indices(path)
    source_format = "module"
    point_segments, index_segments = split_points_with_indices(gps_points, gps_row_indices, max_gap_s, max_jump_m)
    paired_segments = [
        (point_segment, index_segment)
        for point_segment, index_segment in zip(point_segments, index_segments)
        if len(point_segment) >= 2
    ]
    segment_count = len(paired_segments)
    records: list[FlightRecord] = []
    groups = present_groups(fieldnames, MODULE_FEATURE_GROUPS)
    columns = feature_columns(fieldnames)
    source_slug = slug_from_path(path)

    for idx in range(segment_count):
        gps_segment, index_segment = paired_segments[idx]
        local = build_local_track(gps_segment)
        summary = summarize(gps_segment, local, source_segments=segment_count)
        row_count = index_segment[-1] - index_segment[0] + 1
        duration_s = float(summary["duration_s"] or 0.0)
        if source_slug == "data":
            flight_id = f"module_data_s{idx + 1:02d}"
        else:
            flight_id = source_slug

        records.append(
            FlightRecord(
                flight_id=flight_id,
                source_file=str(path),
                source_role="original_module_csv",
                source_format=source_format,
                segment_index=idx + 1,
                segment_count=segment_count,
                gps_source=str(path),
                gps_points=int(summary["points"] or 0),
                sensor_rows=row_count,
                duration_s=duration_s,
                gps_distance_2d_m=float(summary["distance_m"] or 0.0),
                min_alt_m=optional_float(summary["min_alt_m"]),
                max_alt_m=optional_float(summary["max_alt_m"]),
                max_internal_gap_s=float(summary["max_gap_s"] or 0.0),
                start_time_s=gps_segment[0].time_us / 1_000_000.0,
                end_time_s=gps_segment[-1].time_us / 1_000_000.0,
                start_lat=optional_float(summary["start_lat"]),
                start_lon=optional_float(summary["start_lon"]),
                end_lat=optional_float(summary["end_lat"]),
                end_lon=optional_float(summary["end_lon"]),
                sensor_rates_hz={"module_rows": rate_from_count_duration(row_count, duration_s)},
                available_feature_groups=groups,
                feature_columns=columns,
            )
        )

    return records


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_csv_time_stats(path: Path, time_column: str = "TimeUS") -> tuple[int, float, float, float | None, list[str]]:
    if not path.exists():
        return 0, 0.0, 0.0, None, []
    times: list[float] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        value_columns = [name for name in fieldnames if name != time_column]
        for row in reader:
            ts = as_float(row.get(time_column))
            if ts is not None:
                times.append(ts)
    if len(times) < 2:
        return len(times), 0.0, 0.0, None, value_columns
    start_s = min(times) / 1_000_000.0
    end_s = max(times) / 1_000_000.0
    duration_s = max(0.0, end_s - start_s)
    return len(times), start_s, end_s, rate_from_count_duration(len(times), duration_s), value_columns


def build_dataflash_record(
    dataflash_dir: Path,
    min_status: float,
    max_hdop: float | None,
    max_gap_s: float,
    max_jump_m: float,
) -> list[FlightRecord]:
    gps_path = dataflash_dir / "GPS.csv"
    pos_path = dataflash_dir / "POS.csv"
    if not gps_path.exists() and not pos_path.exists():
        return []

    target_path = gps_path if gps_path.exists() else pos_path
    points, source_format = read_gps(target_path, "dataflash", min_status=min_status, max_hdop=max_hdop)
    segments = [segment for segment in split_points(points, max_gap_s, max_jump_m) if len(segment) >= 2]
    if not segments:
        return []
    gps_segment = segments[0]
    local = build_local_track(gps_segment)
    summary = summarize(gps_segment, local, source_segments=len(segments))

    sensor_rates: dict[str, float | None] = {}
    all_feature_columns: list[str] = []
    total_sensor_rows = 0
    start_times: list[float] = []
    end_times: list[float] = []
    groups: list[str] = []

    for filename in [*DATAFLASH_SENSOR_FILES, "GPS.csv", "POS.csv"]:
        path = dataflash_dir / filename
        count, start_s, end_s, rate_hz, columns = read_csv_time_stats(path)
        if count == 0:
            continue
        table = path.stem
        sensor_rates[table] = rate_hz
        total_sensor_rows += count
        start_times.append(start_s)
        end_times.append(end_s)
        if table in DATAFLASH_FEATURE_GROUPS:
            groups.append(f"{table}:{DATAFLASH_FEATURE_GROUPS[table]}")
        all_feature_columns.extend(f"{table}.{column}" for column in columns)

    start_s = min(start_times) if start_times else 0.0
    end_s = max(end_times) if end_times else 0.0
    duration_s = float(summary["duration_s"] or 0.0)

    return [
        FlightRecord(
            flight_id="dataflash_2025_01_15",
            source_file=str(dataflash_dir),
            source_role="derived_dataflash_export",
            source_format=source_format,
            segment_index=1,
            segment_count=len(segments),
            gps_source=str(target_path),
            gps_points=int(summary["points"] or 0),
            sensor_rows=total_sensor_rows,
            duration_s=duration_s,
            gps_distance_2d_m=float(summary["distance_m"] or 0.0),
            min_alt_m=optional_float(summary["min_alt_m"]),
            max_alt_m=optional_float(summary["max_alt_m"]),
            max_internal_gap_s=float(summary["max_gap_s"] or 0.0),
            start_time_s=start_s,
            end_time_s=end_s,
            start_lat=optional_float(summary["start_lat"]),
            start_lon=optional_float(summary["start_lon"]),
            end_lat=optional_float(summary["end_lat"]),
            end_lon=optional_float(summary["end_lon"]),
            sensor_rates_hz=sensor_rates,
            available_feature_groups=groups,
            feature_columns=sorted(set(all_feature_columns)),
        )
    ]


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, records: list[FlightRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "flight_id",
        "source_file",
        "source_role",
        "source_format",
        "segment_index",
        "segment_count",
        "gps_source",
        "gps_points",
        "sensor_rows",
        "duration_s",
        "gps_distance_2d_m",
        "min_alt_m",
        "max_alt_m",
        "max_internal_gap_s",
        "start_time_s",
        "end_time_s",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "sensor_rates_hz",
        "available_feature_groups",
        "feature_columns",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["sensor_rates_hz"] = compact_json(record.sensor_rates_hz)
            row["available_feature_groups"] = compact_json(record.available_feature_groups)
            row["feature_columns"] = compact_json(record.feature_columns)
            writer.writerow(row)


def write_json(path: Path, records: list[FlightRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_note": "Original telemetry is not modified. Generated datasets live under derived/datasets/.",
        "flights": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_float(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def format_rates(rates: dict[str, float | None]) -> str:
    parts = []
    for name, rate in rates.items():
        parts.append(f"{name}={format_float(rate, 2, ' Hz') if rate is not None else 'n/a'}")
    return "<br>".join(parts) if parts else "n/a"


def write_report(path: Path, records: list[FlightRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Flight index",
        "",
        "This report is generated from original GPS-capable telemetry and DataFlash exports.",
        "",
        "Original / source-like inputs:",
        "",
        "- `artifacts/data.csv` - original module CSV, split into 7 flights by `TimeStamp` resets.",
        "- `artifacts/linear_15_01_2025.csv` - original module CSV, one flight.",
        "- `artifacts/triangle_15_01_2025.csv` - original module CSV, one flight.",
        "- `derived/dataflash/*.csv` - exported tables from the original DataFlash log `artifacts/2025-01-15 16-46-48.log`.",
        "",
        "Generated outputs from this step:",
        "",
        "- `derived/datasets/flight_index.csv`",
        "- `derived/datasets/flight_index.json`",
        "- `reports/flight_index.md`",
        "",
        "No new files are written to the root of `artifacts/`.",
        "",
        "## Flights",
        "",
        "| flight_id | source | segment | format | GPS points | sensor rows | duration | 2D distance | altitude | rates | feature groups |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        altitude = f"{format_float(record.min_alt_m, 1)}..{format_float(record.max_alt_m, 1)} m"
        lines.append(
            f"| `{record.flight_id}` | `{record.source_file}` | {record.segment_index}/{record.segment_count} | "
            f"`{record.source_format}` | {record.gps_points} | {record.sensor_rows} | "
            f"{format_seconds(record.duration_s)} | {record.gps_distance_2d_m:.1f} m | {altitude} | "
            f"{format_rates(record.sensor_rates_hz)} | {', '.join(record.available_feature_groups)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `flight_id` is stable and should be used by later dataset, split, model, and prediction scripts.",
            "- GPS/GNSS columns are cataloged as target/reference data, not as input features for GNSS-free navigation.",
            "- `module_data_s01` ... `module_data_s07` are separate flights from `artifacts/data.csv`; they must not be joined into one trajectory.",
            "- `dataflash_2025_01_15` uses `derived/dataflash/GPS.csv` as the GPS reference and catalogs sensor rates from the other exported DataFlash tables.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records: list[FlightRecord] = []
    for source in args.module_sources:
        records.extend(
            build_module_records(
                source,
                min_status=args.min_status,
                max_hdop=args.max_hdop,
                max_gap_s=args.max_gap_s,
                max_jump_m=args.max_jump_m,
            )
        )
    records.extend(
        build_dataflash_record(
            args.dataflash_dir,
            min_status=args.min_status,
            max_hdop=args.max_hdop,
            max_gap_s=args.max_gap_s,
            max_jump_m=args.max_jump_m,
        )
    )
    records.sort(key=lambda record: record.flight_id)

    write_csv(args.out_csv, records)
    write_json(args.out_json, records)
    write_report(args.report, records)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.report}")
    print(f"Indexed {len(records)} flights")


if __name__ == "__main__":
    main()
