#!/usr/bin/env python3
"""Inventory original GPS sources and report whether they contain split flights."""

from __future__ import annotations

import argparse
from pathlib import Path

from gps_flight_map import build_local_track, format_seconds, read_gps, split_points, summarize


DEFAULT_SOURCES = [
    Path("derived/dataflash/GPS.csv"),
    Path("derived/dataflash/POS.csv"),
    Path("artifacts/linear_15_01_2025.csv"),
    Path("artifacts/triangle_15_01_2025.csv"),
    Path("artifacts/data.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs="*", default=DEFAULT_SOURCES)
    parser.add_argument("--report", type=Path, default=Path("reports/gps_flight_inventory.md"))
    parser.add_argument("--min-status", type=float, default=3.0)
    parser.add_argument("--max-hdop", type=float, default=None)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    return parser.parse_args()


def aggregate_segment_summaries(summaries: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    return {
        "points": sum(int(summary["points"] or 0) for summary in summaries),
        "duration_s": sum(float(summary["duration_s"] or 0.0) for summary in summaries),
        "distance_m": sum(float(summary["distance_m"] or 0.0) for summary in summaries),
        "min_alt_m": min(float(summary["min_alt_m"] or 0.0) for summary in summaries),
        "max_alt_m": max(float(summary["max_alt_m"] or 0.0) for summary in summaries),
        "max_gap_s": max(float(summary["max_gap_s"] or 0.0) for summary in summaries),
        "source_segments": len(summaries),
    }


def main() -> None:
    args = parse_args()
    rows: list[tuple[Path, str, dict[str, float | int | None]]] = []
    for source in args.sources:
        if not source.exists():
            continue
        points, source_format = read_gps(source, "auto", args.min_status, args.max_hdop)
        if len(points) < 2:
            continue
        segments = [segment for segment in split_points(points, args.max_gap_s, args.max_jump_m) if len(segment) >= 2]
        summaries = [
            summarize(segment, build_local_track(segment), source_segments=len(segments))
            for segment in segments
        ]
        rows.append((source, source_format, aggregate_segment_summaries(summaries)))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GPS flight inventory",
        "",
        "This report separates original input files from generated visualization artifacts.",
        "",
        "Original GPS-capable files inspected:",
        "",
    ]
    lines.extend(f"- `{source}`" for source, _, _ in rows)
    lines.extend(
        [
            "",
            "Generated files are written under `artifacts/generated/`.",
            "",
            "## Continuous tracks",
            "",
            "| source | format | auto segments | points | duration sum | distance 2D sum | altitude | max internal gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source, source_format, summary in rows:
        lines.append(
            f"| `{source}` | `{source_format}` | {summary['source_segments']} | {summary['points']} | "
            f"{format_seconds(float(summary['duration_s'] or 0.0))} | "
            f"{float(summary['distance_m'] or 0.0):.1f} m | "
            f"{float(summary['min_alt_m'] or 0.0):.1f}..{float(summary['max_alt_m'] or 0.0):.1f} m | "
            f"{float(summary['max_gap_s'] or 0.0):.3f} s |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `derived/dataflash/GPS.csv` and `derived/dataflash/POS.csv` come from one DataFlash log: `artifacts/2025-01-15 16-46-48.log`.",
            "- `artifacts/linear_15_01_2025.csv`, `artifacts/triangle_15_01_2025.csv`, and `artifacts/data.csv` are separate original module CSV recordings.",
            "- `artifacts/data.csv` contains 7 auto-detected segments caused by `TimeStamp` resets. This matches the chat note about 7 flights with coordinates.",
            "- `artifacts/linear_15_01_2025.csv`, `artifacts/triangle_15_01_2025.csv`, and the DataFlash GPS/POS exports each look like one continuous track.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
