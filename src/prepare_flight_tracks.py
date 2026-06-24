#!/usr/bin/env python3
"""Prepare per-flight GPS tracks in local ENU meters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from build_track_viewer import build_payload, write_viewer
from gps_flight_map import (
    GpsPoint,
    LocalPoint,
    build_local_track,
    format_seconds,
    read_gps,
    split_points,
    summarize,
    write_geojson,
    write_html,
    write_manifest,
    write_simulation_html,
    write_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-index", type=Path, default=Path("derived/datasets/flight_index.csv"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--visual-dir", type=Path, default=Path("artifacts/generated/gps_flights"))
    parser.add_argument("--viewer-html", type=Path, default=Path("artifacts/generated/gps_flights/index.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/flight_tracks.md"))
    parser.add_argument("--min-status", type=float, default=3.0)
    parser.add_argument("--max-hdop", type=float, default=None)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    return parser.parse_args()


def as_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def as_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        return math.nan
    return result if math.isfinite(result) else math.nan


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_segment(
    record: dict[str, str],
    min_status: float,
    max_hdop: float | None,
    max_gap_s: float,
    max_jump_m: float,
    segment_cache: dict[tuple[str, str], tuple[list[list[GpsPoint]], str]],
) -> tuple[list[GpsPoint], str]:
    gps_source = Path(record["gps_source"])
    source_format = record["source_format"]
    cache_key = (str(gps_source), source_format)
    if cache_key not in segment_cache:
        points, detected_format = read_gps(gps_source, source_format, min_status=min_status, max_hdop=max_hdop)
        segments = [segment for segment in split_points(points, max_gap_s=max_gap_s, max_jump_m=max_jump_m) if len(segment) >= 2]
        segment_cache[cache_key] = (segments, detected_format)
    else:
        segments, detected_format = segment_cache[cache_key]
    segment_index = as_int(record["segment_index"])
    if segment_index < 1 or segment_index > len(segments):
        raise ValueError(
            f"{record['flight_id']}: segment {segment_index} is unavailable in {gps_source}; "
            f"found {len(segments)} GPS segments"
        )
    return segments[segment_index - 1], detected_format


def write_track_csv(path: Path, flight_id: str, points: list[GpsPoint], local: list[LocalPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "flight_id",
        "time_s",
        "source_time_s",
        "lat",
        "lon",
        "alt_m",
        "east_m",
        "north_m",
        "up_m",
        "speed_mps",
        "distance_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for point, local_point in zip(points, local):
            writer.writerow(
                {
                    "flight_id": flight_id,
                    "time_s": f"{local_point.time_s:.6f}",
                    "source_time_s": f"{point.time_us / 1_000_000.0:.6f}",
                    "lat": f"{point.lat:.12f}",
                    "lon": f"{point.lon:.12f}",
                    "alt_m": f"{point.alt:.6f}",
                    "east_m": f"{local_point.east_m:.6f}",
                    "north_m": f"{local_point.north_m:.6f}",
                    "up_m": f"{local_point.up_m:.6f}",
                    "speed_mps": "" if local_point.speed is None else f"{local_point.speed:.6f}",
                    "distance_m": f"{local_point.distance_m:.6f}",
                }
            )


def max_track_gap_s(points: list[GpsPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return max((points[idx].time_us - points[idx - 1].time_us) / 1_000_000.0 for idx in range(1, len(points)))


def has_monotonic_track(local: list[LocalPoint]) -> bool:
    return all(local[idx].time_s >= local[idx - 1].time_s for idx in range(1, len(local))) and all(
        local[idx].distance_m + 1e-9 >= local[idx - 1].distance_m for idx in range(1, len(local))
    )


def relative_error(reference: float, value: float) -> float:
    if abs(reference) < 1e-9:
        return abs(value)
    return abs(value - reference) / abs(reference)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Flight tracks",
        "",
        "This report is generated by `src/prepare_flight_tracks.py` from `derived/datasets/flight_index.csv`.",
        "",
        "Generated outputs:",
        "",
        "- `derived/datasets/tracks/{flight_id}_track.csv` - local ENU GPS trajectory in meters.",
        "- `artifacts/generated/gps_flights/{flight_id}/map.html` - map for the same `flight_id`.",
        "- `artifacts/generated/gps_flights/{flight_id}/simulation.html` - replay for the same `flight_id`.",
        "- `artifacts/generated/gps_flights/index.html` - one-page track selector and viewer.",
        "",
        "Original telemetry files are not modified.",
        "",
        "## Tracks",
        "",
        "| flight_id | points | duration | distance | max gap | distance check | track csv | replay |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['flight_id']}` | {row['points']} | {format_seconds(float(row['duration_s']))} | "
            f"{float(row['distance_m']):.1f} m | {float(row['max_gap_s']):.3f} s | "
            f"{float(row['distance_error_m']):.6f} m | `{row['track_csv']}` | `{row['simulation_html']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `time_s` starts at zero for every flight.",
            "- `east_m`, `north_m`, `up_m` use the first valid GPS point of the same flight as origin.",
            "- GPS coordinates are kept in the track files as target/reference data; later feature builders must not use them as GNSS-free navigation inputs.",
            "- `source_time_s` preserves the original telemetry timestamp for later sensor synchronization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_record(
    record: dict[str, str],
    tracks_dir: Path,
    visual_dir: Path,
    min_status: float,
    max_hdop: float | None,
    max_gap_s: float,
    max_jump_m: float,
    segment_cache: dict[tuple[str, str], tuple[list[list[GpsPoint]], str]],
) -> dict[str, object]:
    flight_id = record["flight_id"]
    points, detected_format = select_segment(record, min_status, max_hdop, max_gap_s, max_jump_m, segment_cache)
    local = build_local_track(points)
    summary = summarize(points, local, source_segments=as_int(record["segment_count"]))

    track_csv = tracks_dir / f"{flight_id}_track.csv"
    flight_visual_dir = visual_dir / flight_id
    map_html = flight_visual_dir / "map.html"
    simulation_html = flight_visual_dir / "simulation.html"
    path_svg = flight_visual_dir / "path.svg"
    track_geojson = flight_visual_dir / "track.geojson"
    manifest_json = flight_visual_dir / "manifest.json"

    write_track_csv(track_csv, flight_id, points, local)
    write_html(map_html, points, local, summary, flight_id, detected_format)
    write_simulation_html(simulation_html, local, summary, flight_id)
    write_svg(path_svg, local, summary)
    write_geojson(track_geojson, points, summary)
    write_manifest(
        manifest_json,
        Path(record["gps_source"]),
        flight_id,
        detected_format,
        summary,
        {
            "track_csv": track_csv,
            "map_html": map_html,
            "simulation_html": simulation_html,
            "path_svg": path_svg,
            "track_geojson": track_geojson,
        },
        as_int(record["segment_index"]),
        as_int(record["segment_count"]),
    )

    index_distance = as_float(record.get("gps_distance_2d_m", ""))
    distance = float(summary["distance_m"] or 0.0)
    distance_error_m = abs(distance - index_distance) if math.isfinite(index_distance) else math.nan
    if not has_monotonic_track(local):
        raise ValueError(f"{flight_id}: non-monotonic time or distance in local track")
    if math.isfinite(index_distance) and relative_error(index_distance, distance) > 0.001:
        raise ValueError(
            f"{flight_id}: distance mismatch against flight_index.csv: "
            f"{distance:.3f} m vs {index_distance:.3f} m"
        )

    return {
        "flight_id": flight_id,
        "points": len(points),
        "duration_s": float(summary["duration_s"] or 0.0),
        "distance_m": distance,
        "max_gap_s": max_track_gap_s(points),
        "distance_error_m": distance_error_m,
        "track_csv": str(track_csv),
        "simulation_html": str(simulation_html),
    }


def main() -> None:
    args = parse_args()
    records = read_index(args.flight_index)
    if not records:
        raise ValueError(f"No records found in {args.flight_index}")

    segment_cache: dict[tuple[str, str], tuple[list[list[GpsPoint]], str]] = {}
    report_rows = [
        prepare_record(
            record,
            tracks_dir=args.tracks_dir,
            visual_dir=args.visual_dir,
            min_status=args.min_status,
            max_hdop=args.max_hdop,
            max_gap_s=args.max_gap_s,
            max_jump_m=args.max_jump_m,
            segment_cache=segment_cache,
        )
        for record in records
    ]
    flights = build_payload(records, args.tracks_dir, args.visual_dir, args.viewer_html)
    write_viewer(args.viewer_html, flights)
    write_report(args.report, report_rows)
    print(f"Wrote {args.viewer_html}")
    print(f"Wrote {args.report}")
    print(f"Wrote {len(report_rows)} tracks to {args.tracks_dir}")


if __name__ == "__main__":
    main()
