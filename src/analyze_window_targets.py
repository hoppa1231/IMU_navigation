#!/usr/bin/env python3
"""Diagnose target distributions and obvious GPS/altitude jumps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--datasets", nargs="*", type=Path, default=sorted(Path("derived/datasets").glob("windows_module_h*_l*.npz")))
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/module_window_target_diagnostics.md"))
    return parser.parse_args()


def read_track(path: Path) -> dict[str, np.ndarray | str]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    flight_id = rows[0]["flight_id"] if rows else path.stem.replace("_track", "")
    return {
        "flight_id": flight_id,
        "time_s": np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64),
        "east_m": np.asarray([float(row["east_m"]) for row in rows], dtype=np.float64),
        "north_m": np.asarray([float(row["north_m"]) for row in rows], dtype=np.float64),
        "up_m": np.asarray([float(row["up_m"]) for row in rows], dtype=np.float64),
        "distance_m": np.asarray([float(row["distance_m"]) for row in rows], dtype=np.float64),
    }


def quantiles(values: np.ndarray, qs: list[float]) -> list[float]:
    if values.size == 0:
        return [math.nan for _ in qs]
    return [float(np.percentile(values, q)) for q in qs]


def track_rows(tracks_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(tracks_dir.glob("*_track.csv")):
        track = read_track(path)
        time_s = track["time_s"]
        east = track["east_m"]
        north = track["north_m"]
        up = track["up_m"]
        distance = track["distance_m"]
        assert isinstance(time_s, np.ndarray)
        assert isinstance(east, np.ndarray)
        assert isinstance(north, np.ndarray)
        assert isinstance(up, np.ndarray)
        assert isinstance(distance, np.ndarray)
        if len(time_s) < 2:
            continue
        dt = np.diff(time_s)
        dt[dt <= 1e-9] = np.nan
        step_horizontal = np.sqrt(np.diff(east) ** 2 + np.diff(north) ** 2)
        step_up = np.diff(up)
        speed = step_horizontal / dt
        vertical_speed = np.abs(step_up) / dt
        first_5 = time_s <= 5.0
        rows.append(
            {
                "flight_id": track["flight_id"],
                "points": len(time_s),
                "duration_s": float(time_s[-1]),
                "distance_m": float(distance[-1]),
                "up_start_m": float(up[0]),
                "up_end_m": float(up[-1]),
                "up_min_m": float(np.min(up)),
                "up_max_m": float(np.max(up)),
                "up_delta_first_5s_m": float(up[first_5][-1] - up[0]) if np.any(first_5) else math.nan,
                "max_step_horizontal_m": float(np.nanmax(step_horizontal)),
                "max_step_up_m": float(np.nanmax(np.abs(step_up))),
                "max_horizontal_speed_mps": float(np.nanmax(speed)),
                "max_vertical_speed_mps": float(np.nanmax(vertical_speed)),
            }
        )
    return rows


def dataset_rows(path: Path) -> list[dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    y = data["y"].astype(np.float64)
    flight_id = data["flight_id"].astype(str)
    time_s = data["time_s"].astype(np.float64)
    result: list[dict[str, object]] = []
    target_3d = np.linalg.norm(y, axis=1)
    abs_dz = np.abs(y[:, 2])
    for flight in sorted(set(flight_id.tolist())):
        mask = flight_id == flight
        y_f = y[mask]
        target_f = target_3d[mask]
        abs_dz_f = abs_dz[mask]
        time_f = time_s[mask]
        result.append(
            {
                "dataset": path.stem,
                "flight_id": flight,
                "windows": int(mask.sum()),
                "target_3d_median_m": float(np.median(target_f)),
                "target_3d_p95_m": float(np.percentile(target_f, 95)),
                "target_3d_max_m": float(np.max(target_f)),
                "abs_dz_p95_m": float(np.percentile(abs_dz_f, 95)),
                "abs_dz_max_m": float(np.max(abs_dz_f)),
                "count_abs_dz_gt_5m": int(np.sum(abs_dz_f > 5.0)),
                "count_abs_dz_gt_10m": int(np.sum(abs_dz_f > 10.0)),
                "count_abs_dz_gt_20m": int(np.sum(abs_dz_f > 20.0)),
                "count_first_5s": int(np.sum(time_f < 5.0)),
            }
        )
    return result


def write_report(path: Path, track_stats: list[dict[str, object]], target_stats: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Module Window Target Diagnostics",
        "",
        "This report checks whether the GPS-derived targets contain obvious jumps or startup transients.",
        "",
        "## Track-Level Checks",
        "",
        "| flight_id | duration | distance | up range | first 5s up delta | max horizontal step | max vertical step | max h speed | max v speed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in track_stats:
        lines.append(
            f"| `{row['flight_id']}` | {float(row['duration_s']):.1f} s | {float(row['distance_m']):.1f} m | "
            f"{float(row['up_min_m']):.1f}..{float(row['up_max_m']):.1f} m | "
            f"{float(row['up_delta_first_5s_m']):.1f} m | {float(row['max_step_horizontal_m']):.2f} m | "
            f"{float(row['max_step_up_m']):.2f} m | {float(row['max_horizontal_speed_mps']):.1f} m/s | "
            f"{float(row['max_vertical_speed_mps']):.1f} m/s |"
        )

    lines.extend(
        [
            "",
            "## Window Target Checks",
            "",
            "| dataset | flight_id | windows | target 3D median | target 3D p95 | target 3D max | abs dz p95 | abs dz max | dz>5m | dz>10m | dz>20m | first 5s windows |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in target_stats:
        lines.append(
            f"| `{row['dataset']}` | `{row['flight_id']}` | {row['windows']} | "
            f"{float(row['target_3d_median_m']):.2f} m | {float(row['target_3d_p95_m']):.2f} m | "
            f"{float(row['target_3d_max_m']):.2f} m | {float(row['abs_dz_p95_m']):.2f} m | "
            f"{float(row['abs_dz_max_m']):.2f} m | {row['count_abs_dz_gt_5m']} | "
            f"{row['count_abs_dz_gt_10m']} | {row['count_abs_dz_gt_20m']} | {row['count_first_5s']} |"
        )

    suspicious_tracks = [
        row for row in track_stats
        if abs(float(row["up_delta_first_5s_m"])) > 10.0 or float(row["max_vertical_speed_mps"]) > 20.0
    ]
    lines.extend(["", "## Interpretation", ""])
    if suspicious_tracks:
        lines.append("Suspicious startup/altitude behavior was found:")
        lines.append("")
        for row in suspicious_tracks:
            lines.append(
                f"- `{row['flight_id']}`: first 5s up delta {float(row['up_delta_first_5s_m']):.1f} m, "
                f"max vertical speed {float(row['max_vertical_speed_mps']):.1f} m/s."
            )
        lines.extend(
            [
                "",
                "Next practical check: rebuild module window datasets with the first 5 seconds of each flight excluded, then rerun baselines.",
            ]
        )
    else:
        lines.append("No large startup altitude jumps were detected by the current thresholds.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    track_stats = track_rows(args.tracks_dir)
    target_stats: list[dict[str, object]] = []
    for dataset in args.datasets:
        target_stats.extend(dataset_rows(dataset))
    write_report(args.report, track_stats, target_stats)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
