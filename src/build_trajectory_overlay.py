#!/usr/bin/env python3
"""Build GPS/POS trajectory overlays from displacement prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PREDICTIONS = [
    Path("derived/predictions/module_window_baselines_move1/windows_module_h5000_l5000_move1/route_holdout_triangle/ridge_pred.csv"),
    Path("derived/predictions/module_window_baselines_move1/windows_module_h5000_l5000_move1/route_holdout_linear/ridge_pred.csv"),
    Path("derived/predictions/dataflash_sequence_fixed100_shrink_rollout/imu_att_h5000_l5000_s20/sequence_ridge_bias_tuned_pred.csv"),
]


@dataclass
class Track:
    flight_id: str
    time_s: list[float]
    east_m: list[float]
    north_m: list[float]
    up_m: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-csv",
        nargs="*",
        type=Path,
        default=[path for path in DEFAULT_PREDICTIONS if path.exists()],
        help="Prediction CSV files with true/predicted dx,dy,dz columns.",
    )
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--rollout-csv", type=Path, default=Path("derived/predictions/trajectory_overlay/rollout.csv"))
    parser.add_argument("--html", type=Path, default=Path("artifacts/generated/trajectory_overlay/index.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/trajectory_overlay.md"))
    parser.add_argument("--max-track-points", type=int, default=2500)
    parser.add_argument(
        "--max-rollout-gap-s",
        type=float,
        default=1.0,
        help="Start a new rollout segment when the gap between non-overlapping prediction steps is larger.",
    )
    return parser.parse_args()


def as_float(value: str | None, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    return result if math.isfinite(result) else default


def sanitize_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return result or "case"


def case_id_from_path(path: Path) -> str:
    parts = path.with_suffix("").parts
    tail = parts[-4:] if len(parts) >= 4 else parts
    return sanitize_id("__".join(tail))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_track(path: Path) -> Track:
    time_s: list[float] = []
    east_m: list[float] = []
    north_m: list[float] = []
    up_m: list[float] = []
    flight_id = path.stem.removesuffix("_track")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            flight_id = row.get("flight_id") or flight_id
            time_s.append(as_float(row.get("time_s")))
            east_m.append(as_float(row.get("east_m")))
            north_m.append(as_float(row.get("north_m")))
            up_m.append(as_float(row.get("up_m")))
    if not time_s:
        raise ValueError(f"No track points in {path}")
    return Track(flight_id=flight_id, time_s=time_s, east_m=east_m, north_m=north_m, up_m=up_m)


def load_tracks(tracks_dir: Path) -> dict[str, Track]:
    tracks: dict[str, Track] = {}
    for path in sorted(tracks_dir.glob("*_track.csv")):
        track = read_track(path)
        tracks[track.flight_id] = track
    return tracks


def interpolate(track: Track, time_s: float) -> tuple[float, float, float]:
    if time_s <= track.time_s[0]:
        return track.east_m[0], track.north_m[0], track.up_m[0]
    if time_s >= track.time_s[-1]:
        return track.east_m[-1], track.north_m[-1], track.up_m[-1]

    lo = 0
    hi = len(track.time_s) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if track.time_s[mid] <= time_s:
            lo = mid
        else:
            hi = mid

    t0 = track.time_s[lo]
    t1 = track.time_s[hi]
    ratio = 0.0 if t1 <= t0 else (time_s - t0) / (t1 - t0)
    east = track.east_m[lo] + (track.east_m[hi] - track.east_m[lo]) * ratio
    north = track.north_m[lo] + (track.north_m[hi] - track.north_m[lo]) * ratio
    up = track.up_m[lo] + (track.up_m[hi] - track.up_m[lo]) * ratio
    return east, north, up


def position_from_row(
    row: dict[str, str],
    prefix: str,
    track: Track | None,
    time_key: str,
) -> tuple[float, float, float] | None:
    east = as_float(row.get(f"{prefix}_east_m"))
    north = as_float(row.get(f"{prefix}_north_m"))
    up = as_float(row.get(f"{prefix}_up_m"))
    if all(math.isfinite(value) for value in (east, north, up)):
        return east, north, up
    if track is None:
        return None
    time_s = as_float(row.get(time_key))
    if not math.isfinite(time_s):
        return None
    return interpolate(track, time_s)


def true_future_position(row: dict[str, str], track: Track | None) -> tuple[float, float, float] | None:
    direct = position_from_row(row, "true_future", track, "future_time_s")
    if direct is not None:
        return direct
    current = position_from_row(row, "current", track, "time_s")
    if current is None:
        return None
    dx = as_float(row.get("true_dx_east_m"), 0.0)
    dy = as_float(row.get("true_dy_north_m"), 0.0)
    dz = as_float(row.get("true_dz_up_m"), 0.0)
    return current[0] + dx, current[1] + dy, current[2] + dz


def pred_delta(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        as_float(row.get("pred_dx_east_m"), 0.0),
        as_float(row.get("pred_dy_north_m"), 0.0),
        as_float(row.get("pred_dz_up_m"), 0.0),
    )


def selected_non_overlapping(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    next_time_s = -math.inf
    for row in sorted(rows, key=lambda item: as_float(item.get("time_s"))):
        time_s = as_float(row.get("time_s"))
        future_time_s = as_float(row.get("future_time_s"))
        if not math.isfinite(time_s) or not math.isfinite(future_time_s):
            continue
        if time_s + 1e-9 < next_time_s:
            continue
        selected.append(row)
        next_time_s = future_time_s
    return selected


def source_label(row: dict[str, str], pred_path: Path) -> str:
    parts = []
    for key in ("source", "dataset", "split", "feature_set", "horizon_ms", "lookback_ms", "sequence_len", "model"):
        value = row.get(key, "")
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or pred_path.as_posix()


def group_key(row: dict[str, str]) -> str:
    flight_id = row.get("flight_id", "")
    if flight_id:
        return flight_id
    fold_id = row.get("fold_id", "")
    if fold_id:
        return f"fold_{fold_id}"
    return "all"


def build_rollout(
    pred_path: Path,
    rows: list[dict[str, str]],
    tracks: dict[str, Track],
    max_rollout_gap_s: float,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    case_id = case_id_from_path(pred_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    rollout_rows: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    for group, group_rows in sorted(grouped.items()):
        selected = selected_non_overlapping(group_rows)
        if not selected:
            continue

        segments: list[list[dict[str, str]]] = []
        current_segment: list[dict[str, str]] = []
        previous_future_time = -math.inf
        for row in selected:
            time_s = as_float(row.get("time_s"))
            gap_s = time_s - previous_future_time if math.isfinite(previous_future_time) else 0.0
            if current_segment and gap_s > max_rollout_gap_s:
                segments.append(current_segment)
                current_segment = []
            current_segment.append(row)
            previous_future_time = as_float(row.get("future_time_s"))
        if current_segment:
            segments.append(current_segment)

        multiple_segments = len(segments) > 1
        for segment_idx, segment in enumerate(segments, start=1):
            first = segment[0]
            segment_group = f"{group}_seg{segment_idx:02d}" if multiple_segments else group
            flight_id = first.get("flight_id", "")
            track = tracks.get(flight_id) if flight_id else None
            current = position_from_row(first, "current", track, "time_s")
            if current is None:
                raise ValueError(f"{pred_path}: cannot recover current position for group {segment_group}")

            pred_e, pred_n, pred_u = current
            previous_future_time = as_float(first.get("time_s"))
            errors_3d: list[float] = []
            errors_h: list[float] = []
            case_rows: list[dict[str, str]] = []
            for step, row in enumerate(segment, start=1):
                true_future = true_future_position(row, track)
                if true_future is None:
                    continue
                dx, dy, dz = pred_delta(row)
                pred_e += dx
                pred_n += dy
                pred_u += dz
                err_e = pred_e - true_future[0]
                err_n = pred_n - true_future[1]
                err_u = pred_u - true_future[2]
                err_h = math.hypot(err_e, err_n)
                err_3d = math.sqrt(err_e * err_e + err_n * err_n + err_u * err_u)
                time_s = as_float(row.get("time_s"))
                future_time_s = as_float(row.get("future_time_s"))
                gap_s = max(0.0, time_s - previous_future_time) if math.isfinite(previous_future_time) else 0.0
                previous_future_time = future_time_s
                errors_h.append(err_h)
                errors_3d.append(err_3d)
                out = {
                    "case_id": case_id,
                    "group_id": segment_group,
                    "source_file": pred_path.as_posix(),
                    "source_label": source_label(row, pred_path),
                    "flight_id": flight_id,
                    "step": str(step),
                    "time_s": f"{time_s:.6f}",
                    "future_time_s": f"{future_time_s:.6f}",
                    "gap_s": f"{gap_s:.6f}",
                    "true_east_m": f"{true_future[0]:.6f}",
                    "true_north_m": f"{true_future[1]:.6f}",
                    "true_up_m": f"{true_future[2]:.6f}",
                    "pred_east_m": f"{pred_e:.6f}",
                    "pred_north_m": f"{pred_n:.6f}",
                    "pred_up_m": f"{pred_u:.6f}",
                    "err_east_m": f"{err_e:.6f}",
                    "err_north_m": f"{err_n:.6f}",
                    "err_up_m": f"{err_u:.6f}",
                    "err_horizontal_m": f"{err_h:.6f}",
                    "err_3d_m": f"{err_3d:.6f}",
                }
                case_rows.append(out)
                rollout_rows.append(out)

            if errors_3d:
                summaries.append(
                    {
                        "case_id": case_id,
                        "group_id": segment_group,
                        "source_file": pred_path.as_posix(),
                        "source_label": source_label(first, pred_path),
                        "flight_id": flight_id,
                        "steps": len(errors_3d),
                        "start_time_s": as_float(case_rows[0]["time_s"]),
                        "end_time_s": as_float(case_rows[-1]["future_time_s"]),
                        "final_error_3d_m": errors_3d[-1],
                        "mean_error_3d_m": sum(errors_3d) / len(errors_3d),
                        "max_error_3d_m": max(errors_3d),
                        "p95_error_3d_m": percentile(errors_3d, 95.0),
                        "final_error_horizontal_m": errors_h[-1],
                        "mean_error_horizontal_m": sum(errors_h) / len(errors_h),
                        "max_gap_s": max(as_float(row["gap_s"], 0.0) for row in case_rows),
                    }
                )
    return rollout_rows, summaries


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def write_rollout_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rollout rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_track(track: Track, max_points: int) -> list[dict[str, float]]:
    stride = max(1, math.ceil(len(track.time_s) / max_points))
    return [
        {
            "time_s": track.time_s[idx],
            "east_m": track.east_m[idx],
            "north_m": track.north_m[idx],
            "up_m": track.up_m[idx],
        }
        for idx in range(0, len(track.time_s), stride)
    ]


def html_payload(
    rollout_rows: list[dict[str, str]],
    summaries: list[dict[str, object]],
    tracks: dict[str, Track],
    max_track_points: int,
) -> list[dict[str, object]]:
    rows_by_case: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rollout_rows:
        rows_by_case[(row["case_id"], row["group_id"])].append(row)
    summaries_by_case = {(str(row["case_id"]), str(row["group_id"])): row for row in summaries}

    cases: list[dict[str, object]] = []
    for key, rows in sorted(rows_by_case.items()):
        summary = summaries_by_case[key]
        flight_id = str(summary.get("flight_id", ""))
        track_points = sample_track(tracks[flight_id], max_track_points) if flight_id in tracks else []
        cases.append(
            {
                "id": f"{key[0]}::{key[1]}",
                "label": f"{key[0]} / {key[1]}",
                "sourceLabel": summary["source_label"],
                "summary": summary,
                "track": track_points,
                "rows": [
                    {
                        "time_s": as_float(row["future_time_s"]),
                        "true_east_m": as_float(row["true_east_m"]),
                        "true_north_m": as_float(row["true_north_m"]),
                        "true_up_m": as_float(row["true_up_m"]),
                        "pred_east_m": as_float(row["pred_east_m"]),
                        "pred_north_m": as_float(row["pred_north_m"]),
                        "pred_up_m": as_float(row["pred_up_m"]),
                        "err_3d_m": as_float(row["err_3d_m"]),
                        "err_horizontal_m": as_float(row["err_horizontal_m"]),
                    }
                    for row in rows
                ],
            }
        )
    return cases


def html_template(cases: list[dict[str, object]]) -> str:
    payload = json.dumps(cases, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trajectory Overlay</title>
  <style>
    :root {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #17202a;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 14px 18px;
      background: #ffffff;
      border-bottom: 1px solid #d8dde5;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    select {{
      min-width: 360px;
      max-width: 70vw;
      font: inherit;
      padding: 7px 9px;
      border: 1px solid #bac3cf;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      min-height: 0;
    }}
    svg {{
      width: 100%;
      height: calc(100vh - 62px);
      background: #eef1f5;
    }}
    aside {{
      padding: 16px;
      border-left: 1px solid #d8dde5;
      background: #ffffff;
      overflow: auto;
    }}
    .metric {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid #edf0f4;
      font-size: 14px;
    }}
    .source {{
      margin: 4px 0 14px;
      color: #526071;
      font-size: 13px;
      line-height: 1.35;
    }}
    .legend {{
      display: grid;
      gap: 8px;
      margin-top: 16px;
      font-size: 14px;
    }}
    .key {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .swatch {{
      width: 22px;
      height: 4px;
      border-radius: 2px;
    }}
    @media (max-width: 840px) {{
      header {{ align-items: stretch; flex-direction: column; }}
      main {{ grid-template-columns: 1fr; }}
      svg {{ height: 66vh; }}
      aside {{ border-left: 0; border-top: 1px solid #d8dde5; }}
      select {{ min-width: 0; max-width: none; width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Trajectory Overlay</h1>
    <select id="caseSelect" aria-label="Trajectory case"></select>
  </header>
  <main>
    <svg id="plot" role="img" aria-label="GPS and predicted trajectory overlay"></svg>
    <aside>
      <div class="source" id="source"></div>
      <div class="metric"><span>Steps</span><strong id="steps">-</strong></div>
      <div class="metric"><span>Duration</span><strong id="duration">-</strong></div>
      <div class="metric"><span>Final 3D error</span><strong id="final3d">-</strong></div>
      <div class="metric"><span>Mean 3D error</span><strong id="mean3d">-</strong></div>
      <div class="metric"><span>P95 3D error</span><strong id="p953d">-</strong></div>
      <div class="metric"><span>Final horizontal error</span><strong id="finalh">-</strong></div>
      <div class="metric"><span>Max gap</span><strong id="gap">-</strong></div>
      <div class="legend">
        <div class="key"><span class="swatch" style="background:#334155"></span><span>full GPS track</span></div>
        <div class="key"><span class="swatch" style="background:#2563eb"></span><span>reference rollout points</span></div>
        <div class="key"><span class="swatch" style="background:#dc2626"></span><span>IMU/flow accumulated path</span></div>
      </div>
    </aside>
  </main>
  <script>
    const cases = {payload};
    const select = document.getElementById('caseSelect');
    const svg = document.getElementById('plot');
    for (const item of cases) {{
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.label;
      select.appendChild(option);
    }}
    function pathFor(points, xKey, yKey, scale) {{
      if (!points.length) return '';
      return points.map((p, i) => `${{i ? 'L' : 'M'}} ${{scale.x(p[xKey]).toFixed(2)}} ${{scale.y(p[yKey]).toFixed(2)}}`).join(' ');
    }}
    function formatMeters(value) {{
      return `${{Number(value || 0).toFixed(3)}} m`;
    }}
    function render() {{
      const item = cases.find((candidate) => candidate.id === select.value) || cases[0];
      if (!item) return;
      const rows = item.rows;
      const allPoints = [
        ...item.track.map((p) => [p.east_m, p.north_m]),
        ...rows.flatMap((p) => [[p.true_east_m, p.true_north_m], [p.pred_east_m, p.pred_north_m]]),
      ];
      const xs = allPoints.map((p) => p[0]);
      const ys = allPoints.map((p) => p[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const rect = svg.getBoundingClientRect();
      const w = Math.max(rect.width, 320);
      const h = Math.max(rect.height, 320);
      const pad = 34;
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const scale = {{
        x: (x) => pad + (x - minX) / spanX * (w - pad * 2),
        y: (y) => h - pad - (y - minY) / spanY * (h - pad * 2),
      }};
      svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
      svg.innerHTML = `
        <path d="${{pathFor(item.track, 'east_m', 'north_m', scale)}}" fill="none" stroke="#334155" stroke-width="1.8" opacity="0.45"/>
        <path d="${{pathFor(rows, 'true_east_m', 'true_north_m', scale)}}" fill="none" stroke="#2563eb" stroke-width="2.6"/>
        <path d="${{pathFor(rows, 'pred_east_m', 'pred_north_m', scale)}}" fill="none" stroke="#dc2626" stroke-width="2.6"/>
      `;
      const s = item.summary;
      document.getElementById('source').textContent = item.sourceLabel;
      document.getElementById('steps').textContent = Number(s.steps || 0).toLocaleString();
      document.getElementById('duration').textContent = `${{Math.max(0, s.end_time_s - s.start_time_s).toFixed(1)}} s`;
      document.getElementById('final3d').textContent = formatMeters(s.final_error_3d_m);
      document.getElementById('mean3d').textContent = formatMeters(s.mean_error_3d_m);
      document.getElementById('p953d').textContent = formatMeters(s.p95_error_3d_m);
      document.getElementById('finalh').textContent = formatMeters(s.final_error_horizontal_m);
      document.getElementById('gap').textContent = `${{Number(s.max_gap_s || 0).toFixed(3)}} s`;
    }}
    select.addEventListener('change', render);
    window.addEventListener('resize', render);
    render();
  </script>
</body>
</html>
"""


def write_html(path: Path, cases: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_template(cases), encoding="utf-8")


def write_report(
    path: Path,
    pred_paths: list[Path],
    rollout_csv: Path,
    html: Path,
    summaries: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trajectory overlay",
        "",
        "This report is generated by `src/build_trajectory_overlay.py`.",
        "",
        "Method: prediction rows are sorted by time and converted to a sparse non-overlapping rollout. The predicted displacement vectors are accumulated from the first reference position, then compared with GPS/POS future positions.",
        "",
        "This is a trajectory-level drift diagnostic, not raw IMU strapdown integration.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in pred_paths)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- rollout CSV: `{rollout_csv}`",
            f"- HTML overlay: `{html}`",
            "",
            "## Metrics",
            "",
            "| case | group | steps | duration s | final 3D m | mean 3D m | P95 3D m | final horizontal m | max gap s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        duration = float(row["end_time_s"]) - float(row["start_time_s"])
        lines.append(
            f"| `{row['case_id']}` | `{row['group_id']}` | {int(row['steps'])} | {duration:.1f} | "
            f"{float(row['final_error_3d_m']):.3f} | {float(row['mean_error_3d_m']):.3f} | "
            f"{float(row['p95_error_3d_m']):.3f} | {float(row['final_error_horizontal_m']):.3f} | "
            f"{float(row['max_gap_s']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Module CSV cases use the prepared GPS track from `derived/datasets/tracks` as the reference trajectory.",
            "- DataFlash cases use the POS-derived positions already stored in the prediction CSV, so their coordinates can have a small origin offset against the separate GPS export.",
            "- Large final error means displacement predictions contain drift when accumulated, even if individual short-window errors look moderate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.pred_csv:
        raise ValueError("No prediction CSV files were provided or found in defaults")

    tracks = load_tracks(args.tracks_dir)
    all_rollout_rows: list[dict[str, str]] = []
    all_summaries: list[dict[str, object]] = []
    for pred_path in args.pred_csv:
        rows = read_rows(pred_path)
        rollout_rows, summaries = build_rollout(pred_path, rows, tracks, args.max_rollout_gap_s)
        all_rollout_rows.extend(rollout_rows)
        all_summaries.extend(summaries)

    write_rollout_csv(args.rollout_csv, all_rollout_rows)
    cases = html_payload(all_rollout_rows, all_summaries, tracks, args.max_track_points)
    write_html(args.html, cases)
    write_report(args.report, args.pred_csv, args.rollout_csv, args.html, all_summaries)
    print(f"Wrote {args.rollout_csv}")
    print(f"Wrote {args.html}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
