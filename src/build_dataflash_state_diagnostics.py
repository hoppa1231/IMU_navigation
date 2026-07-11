#!/usr/bin/env python3
"""Build state-based diagnostics for the best DataFlash rollout."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


STATE_COLORS = {
    "hover": "#64748b",
    "climb": "#2563eb",
    "descent": "#dc2626",
    "translate": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=Path(
            "derived/predictions/dataflash_sequence_fixed100_shrink/imu_att_h5000_l5000_s20/sequence_ridge_bias_tuned_pred.csv"
        ),
    )
    parser.add_argument(
        "--rollout-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv"),
    )
    parser.add_argument("--baro-csv", type=Path, default=Path("derived/dataflash/BARO.csv"))
    parser.add_argument("--motor-csv", type=Path, default=Path("derived/dataflash/RCOU_motor_features.csv"))
    parser.add_argument("--motb-csv", type=Path, default=Path("derived/dataflash/MOTB.csv"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_diagnostics/sequence_fixed100_shrink_state_rows.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/final/dataflash_state_diagnostics.md"),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("artifacts/generated/dataflash/diagnostics/index.html"),
    )
    parser.add_argument("--hover-horizontal-speed", type=float, default=0.6)
    parser.add_argument("--vertical-speed-threshold", type=float, default=0.35)
    parser.add_argument("--vertical-emphasis-speed", type=float, default=0.8)
    return parser.parse_args()


def as_float(value: str | None, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    return result if math.isfinite(result) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_time_series(path: Path, mapping: dict[str, str]) -> tuple[list[float], list[dict[str, float]]]:
    times: list[float] = []
    rows: list[dict[str, float]] = []
    for row in read_rows(path):
        time_s = as_float(row.get("TimeUS")) / 1_000_000.0
        if not math.isfinite(time_s):
            continue
        parsed = {"time_s": time_s}
        for out_name, in_name in mapping.items():
            parsed[out_name] = as_float(row.get(in_name))
        times.append(time_s)
        rows.append(parsed)
    return times, rows


def nearest_row(time_s: float, times: list[float], rows: list[dict[str, float]]) -> dict[str, float]:
    if not times:
        return {}
    index = bisect.bisect_left(times, time_s)
    candidates: list[int] = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best_index = min(candidates, key=lambda item: abs(times[item] - time_s))
    return rows[best_index]


def build_rollout_index(path: Path) -> dict[str, tuple[list[float], list[dict[str, str]]]]:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(path):
        grouped_rows[row.get("fold_id", "1")].append(row)
    result: dict[str, tuple[list[float], list[dict[str, str]]]] = {}
    for fold_id, rows in grouped_rows.items():
        rows.sort(key=lambda item: as_float(item.get("time_s")))
        times = [as_float(item.get("time_s")) for item in rows]
        result[fold_id] = (times, rows)
    return result


def p95(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def classify_state(
    horizontal_speed_mps: float,
    vertical_speed_mps: float,
    hover_horizontal_speed: float,
    vertical_speed_threshold: float,
    vertical_emphasis_speed: float,
) -> str:
    if abs(vertical_speed_mps) < vertical_speed_threshold and horizontal_speed_mps < hover_horizontal_speed:
        return "hover"
    if vertical_speed_mps >= vertical_speed_threshold and horizontal_speed_mps < vertical_emphasis_speed:
        return "climb"
    if vertical_speed_mps <= -vertical_speed_threshold and horizontal_speed_mps < vertical_emphasis_speed:
        return "descent"
    return "translate"


def build_diagnostics(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rollout_index = build_rollout_index(args.rollout_csv)

    baro_times, baro_rows = read_time_series(args.baro_csv, {"baro_climb_rate_mps": "CRt", "baro_alt_m": "Alt"})
    motor_times, motor_rows = read_time_series(
        args.motor_csv,
        {
            "motor_mean_norm": "motor_mean_norm",
            "motor_std": "motor_std",
            "motor_range": "motor_range",
        },
    )
    motb_times, motb_rows = read_time_series(
        args.motb_csv,
        {
            "thr_out": "ThrOut",
            "thr_avmx": "ThrAvMx",
            "th_limit": "ThLimit",
        },
    )

    diagnostics: list[dict[str, float | str]] = []
    for row in read_rows(args.pred_csv):
        fold_id = row.get("fold_id", "1")
        future_time_s = as_float(row.get("future_time_s"))
        time_s = as_float(row.get("time_s"))
        horizon_s = future_time_s - time_s
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            continue

        true_dx = as_float(row.get("true_dx_east_m"))
        true_dy = as_float(row.get("true_dy_north_m"))
        true_dz = as_float(row.get("true_dz_up_m"))
        pred_dx = as_float(row.get("pred_dx_east_m"))
        pred_dy = as_float(row.get("pred_dy_north_m"))
        pred_dz = as_float(row.get("pred_dz_up_m"))
        local_err_east = pred_dx - true_dx
        local_err_north = pred_dy - true_dy
        local_err_up = pred_dz - true_dz
        local_err_3d = math.sqrt(local_err_east * local_err_east + local_err_north * local_err_north + local_err_up * local_err_up)
        horizontal_distance = math.hypot(true_dx, true_dy)
        horizontal_speed = horizontal_distance / horizon_s
        vertical_speed = true_dz / horizon_s

        state = classify_state(
            horizontal_speed,
            vertical_speed,
            hover_horizontal_speed=args.hover_horizontal_speed,
            vertical_speed_threshold=args.vertical_speed_threshold,
            vertical_emphasis_speed=args.vertical_emphasis_speed,
        )

        rollout_times, rollout_rows = rollout_index.get(fold_id, ([], []))
        rollout_row = nearest_row(future_time_s, rollout_times, rollout_rows) if rollout_times else {}
        rollout_err_3d = as_float(rollout_row.get("err_3d_m"))
        rollout_err_east = as_float(rollout_row.get("err_east_m"))
        rollout_err_north = as_float(rollout_row.get("err_north_m"))
        rollout_err_up = as_float(rollout_row.get("err_up_m"))
        local_err_horizontal = math.hypot(local_err_east, local_err_north)
        rollout_err_horizontal = math.hypot(rollout_err_east, rollout_err_north)

        baro = nearest_row(future_time_s, baro_times, baro_rows)
        motor = nearest_row(future_time_s, motor_times, motor_rows)
        motb = nearest_row(future_time_s, motb_times, motb_rows)

        diagnostics.append(
            {
                "fold_id": fold_id,
                "step": as_float(rollout_row.get("step")),
                "time_s": future_time_s,
                "origin_time_s": time_s,
                "horizon_s": horizon_s,
                "state": state,
                "state_color": STATE_COLORS[state],
                "true_east_m": as_float(row.get("true_future_east_m")),
                "true_north_m": as_float(row.get("true_future_north_m")),
                "true_up_m": as_float(row.get("true_future_up_m")),
                "pred_east_m": as_float(row.get("pred_future_east_m")),
                "pred_north_m": as_float(row.get("pred_future_north_m")),
                "pred_up_m": as_float(row.get("pred_future_up_m")),
                "true_dx_east_m": true_dx,
                "true_dy_north_m": true_dy,
                "true_dz_up_m": true_dz,
                "pred_dx_east_m": pred_dx,
                "pred_dy_north_m": pred_dy,
                "pred_dz_up_m": pred_dz,
                "horizontal_distance_m": horizontal_distance,
                "horizontal_speed_mps": horizontal_speed,
                "vertical_speed_mps": vertical_speed,
                "local_err_east_m": local_err_east,
                "local_err_north_m": local_err_north,
                "local_err_up_m": local_err_up,
                "local_err_3d_m": local_err_3d,
                "local_err_horizontal_m": local_err_horizontal,
                "rollout_err_east_m": rollout_err_east,
                "rollout_err_north_m": rollout_err_north,
                "rollout_err_up_m": rollout_err_up,
                "rollout_err_3d_m": rollout_err_3d,
                "rollout_err_horizontal_m": rollout_err_horizontal,
                "baro_climb_rate_mps": float(baro.get("baro_climb_rate_mps", math.nan)),
                "baro_alt_m": float(baro.get("baro_alt_m", math.nan)),
                "motor_mean_norm": float(motor.get("motor_mean_norm", math.nan)),
                "motor_std": float(motor.get("motor_std", math.nan)),
                "motor_range": float(motor.get("motor_range", math.nan)),
                "thr_out": float(motb.get("thr_out", math.nan)),
                "thr_avmx": float(motb.get("thr_avmx", math.nan)),
                "th_limit": float(motb.get("th_limit", math.nan)),
            }
        )
    diagnostics.sort(key=lambda item: (int(str(item["fold_id"])), float(item["time_s"])))
    return diagnostics


def summarize(rows: list[dict[str, float | str]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result: dict[str, dict[str, float]] = {}
    for name, group in grouped.items():
        rollout_errors = [float(row["rollout_err_3d_m"]) for row in group if math.isfinite(float(row["rollout_err_3d_m"]))]
        rollout_horizontal_errors = [
            float(row["rollout_err_horizontal_m"]) for row in group if math.isfinite(float(row["rollout_err_horizontal_m"]))
        ]
        local_errors = [float(row["local_err_3d_m"]) for row in group if math.isfinite(float(row["local_err_3d_m"]))]
        local_horizontal_errors = [
            float(row["local_err_horizontal_m"]) for row in group if math.isfinite(float(row["local_err_horizontal_m"]))
        ]
        vertical_speeds = [abs(float(row["vertical_speed_mps"])) for row in group if math.isfinite(float(row["vertical_speed_mps"]))]
        result[name] = {
            "count": float(len(group)),
            "rollout_mean": statistics.fmean(rollout_errors) if rollout_errors else math.nan,
            "rollout_p95": p95(rollout_errors),
            "rollout_max": max(rollout_errors) if rollout_errors else math.nan,
            "rollout_horizontal_mean": statistics.fmean(rollout_horizontal_errors) if rollout_horizontal_errors else math.nan,
            "rollout_horizontal_p95": p95(rollout_horizontal_errors),
            "local_mean": statistics.fmean(local_errors) if local_errors else math.nan,
            "local_p95": p95(local_errors),
            "local_horizontal_mean": statistics.fmean(local_horizontal_errors) if local_horizontal_errors else math.nan,
            "local_horizontal_p95": p95(local_horizontal_errors),
            "vertical_abs_mean": statistics.fmean(vertical_speeds) if vertical_speeds else math.nan,
        }
    return result


def build_payload(rows: list[dict[str, float | str]]) -> dict[str, object]:
    state_summary = summarize(rows, "state")
    fold_summary = summarize(rows, "fold_id")
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: value
                for key, value in row.items()
                if isinstance(value, str) or isinstance(value, float)
            }
        )
    return {
        "rows": scalar_rows,
        "state_summary": state_summary,
        "fold_summary": fold_summary,
        "state_colors": STATE_COLORS,
        "counts": dict(Counter(str(row["state"]) for row in rows)),
        "notes": [
            "Optical flow magnitude is unavailable in the current DataFlash export. No OF/optical-flow message is present in derived/dataflash.",
            "State labels are derived from true 5-second target motion, not from model output.",
            "Scatter plots use local delta error; trajectory and state table use rollout cumulative error.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_summary = payload["state_summary"]
    counts = payload["counts"]
    rows = payload["rows"]
    rollout_errors = [float(row["rollout_err_3d_m"]) for row in rows if math.isfinite(float(row["rollout_err_3d_m"]))]
    local_errors = [float(row["local_err_3d_m"]) for row in rows if math.isfinite(float(row["local_err_3d_m"]))]
    lines = [
        "# DataFlash State Diagnostics",
        "",
        "This report is generated by `src/build_dataflash_state_diagnostics.py`.",
        "",
        f"Predictions: `{args.pred_csv}`",
        f"Rollout: `{args.rollout_csv}`",
        f"Row diagnostics CSV: `{args.out_csv}`",
        f"HTML: `{args.html}`",
        "",
        "## Overall",
        "",
        f"- rows: {len(rows)}",
        f"- rollout mean error: {statistics.fmean(rollout_errors):.3f} m" if rollout_errors else "- rollout mean error: n/a",
        f"- rollout p95 error: {p95(rollout_errors):.3f} m" if rollout_errors else "- rollout p95 error: n/a",
        f"- local delta mean error: {statistics.fmean(local_errors):.3f} m" if local_errors else "- local delta mean error: n/a",
        f"- local delta p95 error: {p95(local_errors):.3f} m" if local_errors else "- local delta p95 error: n/a",
        "",
        "## State Counts",
        "",
    ]
    for state in ("hover", "climb", "descent", "translate"):
        lines.append(f"- `{state}`: {counts.get(state, 0)}")
    lines.extend(
        [
            "",
            "## By State",
            "",
            "| state | rows | rollout mean 3D | rollout p95 3D | rollout mean horiz | rollout p95 horiz | local mean 3D | local mean horiz | abs vertical speed mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for state in ("hover", "climb", "descent", "translate"):
        metrics = state_summary.get(state, {})
        if not metrics:
            continue
        lines.append(
            f"| `{state}` | {metrics['count']:.0f} | {metrics['rollout_mean']:.3f} | {metrics['rollout_p95']:.3f} | "
            f"{metrics['rollout_horizontal_mean']:.3f} | {metrics['rollout_horizontal_p95']:.3f} | "
            f"{metrics['local_mean']:.3f} | {metrics['local_horizontal_mean']:.3f} | "
            f"{metrics['vertical_abs_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Optical flow magnitude is unavailable in the current DataFlash export, so this diagnostic uses BARO climb rate and motor telemetry instead.",
            "- `hover`, `climb`, `descent`, `translate` are labeled from true 5-second motion with thresholds:",
            f"  - hover horizontal speed < {args.hover_horizontal_speed:.2f} m/s",
            f"  - climb/descent vertical speed magnitude >= {args.vertical_speed_threshold:.2f} m/s and horizontal speed < {args.vertical_emphasis_speed:.2f} m/s",
            "- `rollout error` is the cumulative navigation error at the rollout endpoint for that step.",
            "- `local delta error` is the single-step 5-second displacement error before accumulation.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_template(payload: dict[str, object]) -> str:
    data = json.dumps(payload, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFlash State Diagnostics</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #16202a;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f4f6f8; }}
    header {{
      padding: 16px 20px;
      background: #ffffff;
      border-bottom: 1px solid #d7dee8;
    }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .muted {{ color: #52606d; font-size: 14px; }}
    main {{
      display: grid;
      gap: 16px;
      padding: 16px;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.9fr);
      align-items: start;
    }}
    .left {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}
    .right {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d7dee8;
      border-radius: 8px;
      padding: 14px;
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .panel p {{
      margin: 0;
      font-size: 14px;
      color: #52606d;
    }}
    .grid2 {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: #fbfcfd;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid #edf1f5;
    }}
    th {{ font-size: 12px; color: #52606d; }}
    .metric {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 7px 0;
      border-bottom: 1px solid #edf1f5;
      font-size: 14px;
    }}
    .metric:last-child {{ border-bottom: 0; }}
    .legend {{
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }}
    .key {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      flex: 0 0 auto;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      color: #52606d;
      font-size: 14px;
    }}
    @media (max-width: 1080px) {{
      main {{ grid-template-columns: 1fr; }}
      .grid2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DataFlash State Diagnostics</h1>
    <div class="muted">Best rollout: `sequence_ridge_bias_tuned` on 5-second horizon. Scatter plots use local step error; trajectory uses accumulated rollout error.</div>
  </header>
  <main>
    <section class="left">
      <div class="panel">
        <h2>Trajectory Colored by State</h2>
        <svg id="trajectory" viewBox="0 0 860 420" role="img" aria-label="Trajectory colored by state"></svg>
      </div>
      <div class="grid2">
        <div class="panel">
          <h2>Error vs Vertical Speed</h2>
          <svg id="verticalScatter" viewBox="0 0 420 320" role="img" aria-label="Rollout error vs vertical speed"></svg>
        </div>
        <div class="panel">
          <h2>Error vs BARO Climb Rate</h2>
          <svg id="baroScatter" viewBox="0 0 420 320" role="img" aria-label="Rollout error vs barometer climb rate"></svg>
        </div>
      </div>
      <div class="grid2">
        <div class="panel">
          <h2>Error vs Motor Mean Norm</h2>
          <svg id="motorScatter" viewBox="0 0 420 320" role="img" aria-label="Rollout error vs motor mean norm"></svg>
        </div>
        <div class="panel">
          <h2>Error vs ThrOut</h2>
          <svg id="thrScatter" viewBox="0 0 420 320" role="img" aria-label="Rollout error vs throttle output"></svg>
        </div>
      </div>
      <div class="panel">
        <h2>Timeline</h2>
        <svg id="timeline" viewBox="0 0 860 180" role="img" aria-label="Timeline of states and rollout error"></svg>
      </div>
    </section>
    <aside class="right">
      <div class="panel">
        <h2>By State</h2>
        <table id="stateTable"></table>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel">
        <h2>Overall</h2>
        <div id="overall"></div>
      </div>
      <div class="panel">
        <h2>Notes</h2>
        <ul id="notes"></ul>
      </div>
    </aside>
  </main>
  <script>
    const payload = {data};
    const rows = payload.rows;
    const stateColors = payload.state_colors;

    function fmt(value, digits = 2) {{
      return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
    }}

    function extents(values) {{
      const filtered = values.filter(Number.isFinite);
      if (!filtered.length) return [0, 1];
      let min = Math.min(...filtered);
      let max = Math.max(...filtered);
      if (min === max) {{
        min -= 1;
        max += 1;
      }}
      return [min, max];
    }}

    function scale(value, inMin, inMax, outMin, outMax) {{
      if (!Number.isFinite(value)) return NaN;
      const t = (value - inMin) / (inMax - inMin);
      return outMin + t * (outMax - outMin);
    }}

    function create(tag, attrs = {{}}, text = "") {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text) node.textContent = text;
      return node;
    }}

    function drawScatter(svgId, xKey, yKey, xLabel, yLabel) {{
      const svg = document.getElementById(svgId);
      const width = 420;
      const height = 320;
      const margin = {{ left: 48, right: 16, top: 16, bottom: 40 }};
      const [xMin, xMax] = extents(rows.map((row) => Number(row[xKey])));
      const [yMin, yMax] = extents(rows.map((row) => Number(row[yKey])));
      svg.innerHTML = "";
      svg.appendChild(create("rect", {{ x: 0, y: 0, width, height, fill: "#fbfcfd" }}));
      svg.appendChild(create("line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8" }}));
      svg.appendChild(create("line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8" }}));
      for (const row of rows) {{
        const x = scale(Number(row[xKey]), xMin, xMax, margin.left, width - margin.right);
        const y = scale(Number(row[yKey]), yMin, yMax, height - margin.bottom, margin.top);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        svg.appendChild(create("circle", {{
          cx: x,
          cy: y,
          r: 3.2,
          fill: stateColors[row.state] || "#334155",
          opacity: 0.76
        }}));
      }}
      svg.appendChild(create("text", {{ x: width / 2, y: height - 10, "text-anchor": "middle", fill: "#52606d", "font-size": "12" }}, xLabel));
      svg.appendChild(create("text", {{
        x: 14,
        y: height / 2,
        fill: "#52606d",
        "font-size": "12",
        transform: `rotate(-90 14 ${{height / 2}})`,
        "text-anchor": "middle"
      }}, yLabel));
      svg.appendChild(create("text", {{ x: margin.left, y: height - 20, fill: "#52606d", "font-size": "11" }}, fmt(xMin)));
      svg.appendChild(create("text", {{ x: width - margin.right, y: height - 20, fill: "#52606d", "font-size": "11", "text-anchor": "end" }}, fmt(xMax)));
      svg.appendChild(create("text", {{ x: margin.left - 8, y: height - margin.bottom + 4, fill: "#52606d", "font-size": "11", "text-anchor": "end" }}, fmt(yMin)));
      svg.appendChild(create("text", {{ x: margin.left - 8, y: margin.top + 4, fill: "#52606d", "font-size": "11", "text-anchor": "end" }}, fmt(yMax)));
    }}

    function drawTrajectory() {{
      const svg = document.getElementById("trajectory");
      const width = 860;
      const height = 420;
      const margin = {{ left: 48, right: 20, top: 20, bottom: 40 }};
      const xs = rows.flatMap((row) => [Number(row.true_east_m), Number(row.pred_east_m)]);
      const ys = rows.flatMap((row) => [Number(row.true_north_m), Number(row.pred_north_m)]);
      const [xMin, xMax] = extents(xs);
      const [yMin, yMax] = extents(ys);
      svg.innerHTML = "";
      svg.appendChild(create("rect", {{ x: 0, y: 0, width, height, fill: "#fbfcfd" }}));
      svg.appendChild(create("line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8" }}));
      svg.appendChild(create("line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8" }}));

      let truePath = "";
      let predPath = "";
      rows.forEach((row, index) => {{
        const tx = scale(Number(row.true_east_m), xMin, xMax, margin.left, width - margin.right);
        const ty = scale(Number(row.true_north_m), yMin, yMax, height - margin.bottom, margin.top);
        const px = scale(Number(row.pred_east_m), xMin, xMax, margin.left, width - margin.right);
        const py = scale(Number(row.pred_north_m), yMin, yMax, height - margin.bottom, margin.top);
        truePath += `${{index ? " L" : "M"}}${{tx}} ${{ty}}`;
        predPath += `${{index ? " L" : "M"}}${{px}} ${{py}}`;
      }});
      svg.appendChild(create("path", {{ d: truePath, fill: "none", stroke: "#0f172a", "stroke-width": 2.2 }}));
      svg.appendChild(create("path", {{ d: predPath, fill: "none", stroke: "#cbd5e1", "stroke-width": 2.0, "stroke-dasharray": "5 4" }}));
      rows.forEach((row) => {{
        const px = scale(Number(row.pred_east_m), xMin, xMax, margin.left, width - margin.right);
        const py = scale(Number(row.pred_north_m), yMin, yMax, height - margin.bottom, margin.top);
        svg.appendChild(create("circle", {{
          cx: px,
          cy: py,
          r: 4.2,
          fill: stateColors[row.state] || "#334155",
          opacity: 0.85
        }}));
      }});
      svg.appendChild(create("text", {{ x: width / 2, y: height - 10, "text-anchor": "middle", fill: "#52606d", "font-size": "12" }}, "east, m"));
      svg.appendChild(create("text", {{
        x: 16, y: height / 2, "text-anchor": "middle", fill: "#52606d", "font-size": "12",
        transform: `rotate(-90 16 ${{height / 2}})`
      }}, "north, m"));
    }}

    function drawTimeline() {{
      const svg = document.getElementById("timeline");
      const width = 860;
      const height = 180;
      const margin = {{ left: 44, right: 20, top: 16, bottom: 28 }};
      const [tMin, tMax] = extents(rows.map((row) => Number(row.time_s)));
      const [eMin, eMax] = extents(rows.map((row) => Number(row.rollout_err_3d_m)));
      svg.innerHTML = "";
      svg.appendChild(create("rect", {{ x: 0, y: 0, width, height, fill: "#fbfcfd" }}));
      svg.appendChild(create("line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8" }}));
      svg.appendChild(create("line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8" }}));
      let path = "";
      rows.forEach((row, index) => {{
        const x = scale(Number(row.time_s), tMin, tMax, margin.left, width - margin.right);
        const y = scale(Number(row.rollout_err_3d_m), eMin, eMax, height - margin.bottom, margin.top);
        path += `${{index ? " L" : "M"}}${{x}} ${{y}}`;
      }});
      svg.appendChild(create("path", {{ d: path, fill: "none", stroke: "#0f172a", "stroke-width": 1.8 }}));
      rows.forEach((row) => {{
        const x = scale(Number(row.time_s), tMin, tMax, margin.left, width - margin.right);
        const y = scale(Number(row.rollout_err_3d_m), eMin, eMax, height - margin.bottom, margin.top);
        const baseY = height - margin.bottom + 2;
        svg.appendChild(create("line", {{
          x1: x, y1: baseY, x2: x, y2: baseY + 14, stroke: stateColors[row.state] || "#334155", "stroke-width": 2
        }}));
        svg.appendChild(create("circle", {{ cx: x, cy: y, r: 3, fill: stateColors[row.state] || "#334155" }}));
      }});
      svg.appendChild(create("text", {{ x: width / 2, y: height - 8, "text-anchor": "middle", fill: "#52606d", "font-size": "12" }}, "time, s"));
      svg.appendChild(create("text", {{
        x: 14, y: height / 2, "text-anchor": "middle", fill: "#52606d", "font-size": "12",
        transform: `rotate(-90 14 ${{height / 2}})`
      }}, "rollout error, m"));
    }}

    function renderTable() {{
      const table = document.getElementById("stateTable");
      const order = ["hover", "climb", "descent", "translate"];
      const rowsHtml = order
        .filter((name) => payload.state_summary[name])
        .map((name) => {{
          const item = payload.state_summary[name];
          return `
            <tr>
              <td><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${{stateColors[name]}};margin-right:6px;"></span>${{name}}</td>
              <td>${{item.count.toFixed(0)}}</td>
              <td>${{fmt(item.rollout_mean)}}</td>
              <td>${{fmt(item.rollout_p95)}}</td>
              <td>${{fmt(item.local_mean)}}</td>
            </tr>
          `;
        }})
        .join("");
      table.innerHTML = `
        <thead>
          <tr><th>state</th><th>rows</th><th>rollout mean</th><th>rollout p95</th><th>local mean</th></tr>
        </thead>
        <tbody>${{rowsHtml}}</tbody>
      `;
    }}

    function renderLegend() {{
      const legend = document.getElementById("legend");
      legend.innerHTML = Object.entries(stateColors)
        .map(([name, color]) => `<div class="key"><span class="swatch" style="background:${{color}}"></span><span>${{name}}</span></div>`)
        .join("");
    }}

    function renderOverall() {{
      const node = document.getElementById("overall");
      const rollout = rows.map((row) => Number(row.rollout_err_3d_m)).filter(Number.isFinite);
      const local = rows.map((row) => Number(row.local_err_3d_m)).filter(Number.isFinite);
      const maxRollout = rollout.length ? Math.max(...rollout) : NaN;
      const finalRollout = rollout.length ? rollout[rollout.length - 1] : NaN;
      node.innerHTML = `
        <div class="metric"><span>rows</span><strong>${{rows.length}}</strong></div>
        <div class="metric"><span>final rollout error</span><strong>${{fmt(finalRollout)}} m</strong></div>
        <div class="metric"><span>max rollout error</span><strong>${{fmt(maxRollout)}} m</strong></div>
        <div class="metric"><span>mean local delta error</span><strong>${{fmt(local.reduce((a, b) => a + b, 0) / Math.max(local.length, 1))}} m</strong></div>
      `;
    }}

    function renderNotes() {{
      const notes = document.getElementById("notes");
      notes.innerHTML = payload.notes.map((line) => `<li>${{line}}</li>`).join("");
    }}

    drawTrajectory();
    drawScatter("verticalScatter", "vertical_speed_mps", "local_err_3d_m", "true vertical speed, m/s", "local delta error, m");
    drawScatter("baroScatter", "baro_climb_rate_mps", "local_err_3d_m", "BARO CRt, m/s", "local delta error, m");
    drawScatter("motorScatter", "motor_mean_norm", "local_err_3d_m", "motor mean norm", "local delta error, m");
    drawScatter("thrScatter", "thr_out", "local_err_3d_m", "ThrOut", "local delta error, m");
    drawTimeline();
    renderTable();
    renderLegend();
    renderOverall();
    renderNotes();
  </script>
</body>
</html>
"""


def write_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_template(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    diagnostics = build_diagnostics(args)
    if not diagnostics:
        raise SystemExit("No diagnostic rows produced.")
    write_csv(args.out_csv, diagnostics)
    payload = build_payload(diagnostics)
    write_report(args.report, args, payload)
    write_html(args.html, payload)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
