#!/usr/bin/env python3
"""Extract selected ArduPilot DataFlash text-log messages to CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


DEFAULT_MESSAGES = ["RCOU", "MOTB", "BAT", "POS", "GPS", "IMU", "ATT", "BARO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("artifacts/2025-01-15 16-46-48.log"))
    parser.add_argument("--out-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--messages", nargs="+", default=DEFAULT_MESSAGES)
    parser.add_argument("--summary", type=Path, default=Path("reports/dataflash_summary.md"))
    return parser.parse_args()


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_log(path: Path, selected: set[str]) -> tuple[dict[str, list[str]], dict[str, list[list[str]]], Counter[str]]:
    formats: dict[str, list[str]] = {}
    rows: dict[str, list[list[str]]] = {name: [] for name in selected}
    counts: Counter[str] = Counter()

    with path.open(encoding="ascii", errors="replace") as file:
        for line in file:
            parts = [part.strip() for part in line.strip().split(",")]
            if not parts or not parts[0]:
                continue
            kind = parts[0]
            counts[kind] += 1
            if kind == "FMT" and len(parts) >= 6:
                name = parts[3]
                formats[name] = parts[5:]
                continue
            if kind in selected:
                rows[kind].append(parts[1:])

    return formats, rows, counts


def write_message_csv(out_dir: Path, name: str, columns: list[str], rows: list[list[str]]) -> Path:
    path = out_dir / f"{name}.csv"
    width = len(columns)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            if len(row) < width:
                row = row + [""] * (width - len(row))
            writer.writerow(row[:width])
    return path


def write_motor_features(out_dir: Path, columns: list[str], rows: list[list[str]]) -> Path | None:
    if not rows:
        return None
    path = out_dir / "RCOU_motor_features.csv"
    channel_indices = [columns.index(name) for name in ("C1", "C2", "C3", "C4") if name in columns]
    if len(channel_indices) < 4 or "TimeUS" not in columns:
        return None
    time_index = columns.index("TimeUS")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "TimeUS",
                "motor_mean",
                "motor_std",
                "motor_min",
                "motor_max",
                "motor_range",
                "motor_diff_c1_c3",
                "motor_diff_c2_c4",
                "motor_mean_norm",
            ]
        )
        for row in rows:
            if len(row) <= max(channel_indices + [time_index]):
                continue
            motors = [parse_float(row[i]) for i in channel_indices]
            if any(math.isnan(value) for value in motors):
                continue
            mean = sum(motors) / len(motors)
            variance = sum((value - mean) ** 2 for value in motors) / len(motors)
            writer.writerow(
                [
                    row[time_index],
                    f"{mean:.6f}",
                    f"{math.sqrt(variance):.6f}",
                    f"{min(motors):.6f}",
                    f"{max(motors):.6f}",
                    f"{(max(motors) - min(motors)):.6f}",
                    f"{(motors[0] - motors[2]):.6f}",
                    f"{(motors[1] - motors[3]):.6f}",
                    f"{((mean - 1000.0) / 1000.0):.6f}",
                ]
            )
    return path


def write_summary(
    summary_path: Path,
    log_path: Path,
    counts: Counter[str],
    exported: dict[str, Path],
    motor_features: Path | None,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash summary",
        "",
        f"- source log: `{log_path}`",
        "",
        "## Exported CSV",
        "",
        "| message | rows | file |",
        "| --- | ---: | --- |",
    ]
    for name, path in exported.items():
        lines.append(f"| `{name}` | {counts[name]} | `{path}` |")
    if motor_features is not None:
        lines.extend(["", f"Motor feature file: `{motor_features}`"])

    lines.extend(
        [
            "",
            "## Most frequent messages",
            "",
            "| message | rows |",
            "| --- | ---: |",
        ]
    )
    for name, count in counts.most_common(30):
        lines.append(f"| `{name}` | {count} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected = set(args.messages)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    formats, rows, counts = read_log(args.log, selected)
    exported: dict[str, Path] = {}
    for name in args.messages:
        if name not in formats:
            continue
        exported[name] = write_message_csv(args.out_dir, name, formats[name], rows.get(name, []))

    motor_features = None
    if "RCOU" in formats:
        motor_features = write_motor_features(args.out_dir, formats["RCOU"], rows.get("RCOU", []))

    write_summary(args.summary, args.log, counts, exported, motor_features)
    print(f"Wrote {len(exported)} message CSV files to {args.out_dir}")
    if motor_features:
        print(f"Wrote {motor_features}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
