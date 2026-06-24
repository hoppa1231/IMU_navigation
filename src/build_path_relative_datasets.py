#!/usr/bin/env python3
"""Create path-relative horizontal target datasets from module window datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=Path,
        default=sorted(
            path
            for path in Path("derived/datasets").glob("windows_module_h*_l*.npz")
            if "_trim" not in path.stem and "_pathrel" not in path.stem
        ),
    )
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--out-dir", type=Path, default=Path("derived/datasets"))
    parser.add_argument("--heading-lookback-s", type=float, default=1.0)
    parser.add_argument("--min-heading-distance-m", type=float, default=0.2)
    parser.add_argument("--report", type=Path, default=Path("reports/path_relative_datasets.md"))
    return parser.parse_args()


def read_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    xy: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            times.append(float(row["time_s"]))
            xy.append((float(row["east_m"]), float(row["north_m"])))
    return np.asarray(times, dtype=np.float64), np.asarray(xy, dtype=np.float64)


def interpolate_xy(times: np.ndarray, xy: np.ndarray, query: np.ndarray) -> np.ndarray:
    east = np.interp(query, times, xy[:, 0])
    north = np.interp(query, times, xy[:, 1])
    return np.column_stack([east, north])


def build_heading_units(
    flight_ids: np.ndarray,
    time_s: np.ndarray,
    tracks_dir: Path,
    lookback_s: float,
    min_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    units = np.zeros((len(time_s), 2), dtype=np.float64)
    valid = np.zeros(len(time_s), dtype=bool)
    for flight in sorted(set(flight_ids.tolist())):
        mask = flight_ids == flight
        track_times, track_xy = read_track(tracks_dir / f"{flight}_track.csv")
        current = interpolate_xy(track_times, track_xy, time_s[mask])
        previous = interpolate_xy(track_times, track_xy, np.maximum(0.0, time_s[mask] - lookback_s))
        heading = current - previous
        distance = np.linalg.norm(heading, axis=1)
        local_valid = distance >= min_distance_m
        unit = np.zeros_like(heading)
        unit[local_valid] = heading[local_valid] / distance[local_valid, None]
        unit[~local_valid] = np.asarray([1.0, 0.0])
        units[mask] = unit
        valid[mask] = local_valid
    return units, valid


def transform_dataset(
    path: Path,
    tracks_dir: Path,
    out_dir: Path,
    heading_lookback_s: float,
    min_heading_distance_m: float,
) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    flight_id = data["flight_id"].astype(str)
    time_s = data["time_s"].astype(np.float64)
    y_xy = data["y"][:, :2].astype(np.float64)
    units, heading_valid = build_heading_units(
        flight_id,
        time_s,
        tracks_dir,
        heading_lookback_s,
        min_heading_distance_m,
    )
    left = np.column_stack([-units[:, 1], units[:, 0]])
    along = np.sum(y_xy * units, axis=1)
    cross = np.sum(y_xy * left, axis=1)
    y_path = np.column_stack([along, cross]).astype(np.float32)

    out_npz = out_dir / f"{path.stem}_pathrel.npz"
    payload = {
        name: data[name]
        for name in data.files
        if name not in {"y", "target_names"}
    }
    payload["y"] = y_path
    payload["target_names"] = np.asarray(["along_m", "cross_m"])
    payload["heading_valid"] = heading_valid
    payload["heading_unit_east"] = units[:, 0].astype(np.float32)
    payload["heading_unit_north"] = units[:, 1].astype(np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    features_src = path.with_name(f"{path.stem}_features.json")
    features_dst = out_dir / f"{path.stem}_pathrel_features.json"
    if features_src.exists():
        features = json.loads(features_src.read_text(encoding="utf-8"))
        features["target"] = ["along_m", "cross_m"]
        features["target_transform"] = {
            "type": "path_relative_horizontal",
            "heading_lookback_s": heading_lookback_s,
            "min_heading_distance_m": min_heading_distance_m,
            "note": "Heading is computed from past GPS track for diagnostic target analysis.",
        }
        features_dst.write_text(json.dumps(features, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta_src = path.with_name(f"{path.stem}_meta.csv")
    meta_dst = out_dir / f"{path.stem}_pathrel_meta.csv"
    if meta_src.exists():
        with meta_src.open(newline="", encoding="utf-8") as src, meta_dst.open("w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            fieldnames = list(reader.fieldnames or []) + ["heading_valid", "heading_unit_east", "heading_unit_north"]
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row, is_valid, unit in zip(reader, heading_valid, units):
                row["heading_valid"] = "1" if is_valid else "0"
                row["heading_unit_east"] = f"{unit[0]:.8f}"
                row["heading_unit_north"] = f"{unit[1]:.8f}"
                writer.writerow(row)

    return {
        "source": path,
        "output": out_npz,
        "windows": len(time_s),
        "valid_heading": int(np.sum(heading_valid)),
        "invalid_heading": int(len(time_s) - np.sum(heading_valid)),
        "along_abs_mean": float(np.mean(np.abs(along))),
        "cross_abs_mean": float(np.mean(np.abs(cross))),
    }


def write_report(path: Path, rows: list[dict[str, object]], heading_lookback_s: float, min_heading_distance_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Path-Relative Datasets",
        "",
        "This report is generated by `src/build_path_relative_datasets.py`.",
        "",
        "These datasets are diagnostic: the path heading is computed from past GPS track positions, so this target transform is meant to test whether global east/north axes are hurting the baseline.",
        "",
        f"- heading lookback: `{heading_lookback_s}` s",
        f"- minimum heading distance: `{min_heading_distance_m}` m",
        "",
        "| source | output | windows | valid heading | invalid heading | mean abs along | mean abs cross |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source']}` | `{row['output']}` | {row['windows']} | {row['valid_heading']} | "
            f"{row['invalid_heading']} | {float(row['along_abs_mean']):.3f} m | {float(row['cross_abs_mean']):.3f} m |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [
        transform_dataset(
            path,
            args.tracks_dir,
            args.out_dir,
            args.heading_lookback_s,
            args.min_heading_distance_m,
        )
        for path in args.datasets
    ]
    write_report(args.report, rows, args.heading_lookback_s, args.min_heading_distance_m)
    print(f"Wrote {args.report}")
    for row in rows:
        print(f"Wrote {row['output']}")


if __name__ == "__main__":
    main()
