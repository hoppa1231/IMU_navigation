#!/usr/bin/env python3
"""Evaluate a simple state-gated sparse rollout for the best DataFlash predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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
        "--baseline-rollout-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_state_gated_rollout.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink_state_gated.md"),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("artifacts/generated/dataflash/rollouts/sequence_fixed100_shrink_state_gated/index.html"),
    )
    parser.add_argument("--hover-horizontal-speed", type=float, default=0.6)
    parser.add_argument("--vertical-speed-threshold", type=float, default=0.35)
    parser.add_argument("--vertical-emphasis-speed", type=float, default=0.8)
    parser.add_argument("--hover-horizontal-scale", type=float, default=0.5)
    parser.add_argument("--vertical-horizontal-scale", type=float, default=1.0)
    parser.add_argument("--translate-horizontal-scale", type=float, default=1.0)
    parser.add_argument("--translate-fast-threshold", type=float, default=math.inf)
    parser.add_argument("--translate-fast-horizontal-scale", type=float, default=1.0)
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


def select_non_overlapping(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    next_time = -math.inf
    for row in sorted(rows, key=lambda item: as_float(item["time_s"])):
        time_s = as_float(row["time_s"])
        if time_s + 1e-9 < next_time:
            continue
        selected.append(row)
        next_time = as_float(row["future_time_s"])
    return selected


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


def horizontal_scale_for_state(args: argparse.Namespace, state: str, horizontal_speed_mps: float) -> float:
    if state == "hover":
        return args.hover_horizontal_scale
    if state in {"climb", "descent"}:
        return args.vertical_horizontal_scale
    if state == "translate" and horizontal_speed_mps >= args.translate_fast_threshold:
        return args.translate_fast_horizontal_scale
    return args.translate_horizontal_scale


def build_baseline_index(path: Path) -> dict[tuple[str, float], dict[str, str]]:
    index: dict[tuple[str, float], dict[str, str]] = {}
    for row in read_rows(path):
        index[(row.get("fold_id", "1"), round(as_float(row["time_s"]), 6))] = row
    return index


def build_fold_rollout(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    baseline_index: dict[tuple[str, float], dict[str, str]],
) -> list[dict[str, float | str]]:
    selected = select_non_overlapping(rows)
    if not selected:
        return []
    pred_e = as_float(selected[0]["current_east_m"])
    pred_n = as_float(selected[0]["current_north_m"])
    pred_u = as_float(selected[0]["current_up_m"])
    result: list[dict[str, float | str]] = []
    for step_idx, row in enumerate(selected, start=1):
        horizon_s = as_float(row["future_time_s"]) - as_float(row["time_s"])
        true_dx_e = as_float(row["true_dx_east_m"])
        true_dx_n = as_float(row["true_dy_north_m"])
        true_dz_u = as_float(row["true_dz_up_m"])
        pred_dx_e = as_float(row["pred_dx_east_m"])
        pred_dx_n = as_float(row["pred_dy_north_m"])
        pred_dz_u = as_float(row["pred_dz_up_m"])

        horizontal_speed = math.hypot(true_dx_e, true_dx_n) / horizon_s if horizon_s > 0 else math.nan
        vertical_speed = true_dz_u / horizon_s if horizon_s > 0 else math.nan
        state = classify_state(
            horizontal_speed,
            vertical_speed,
            hover_horizontal_speed=args.hover_horizontal_speed,
            vertical_speed_threshold=args.vertical_speed_threshold,
            vertical_emphasis_speed=args.vertical_emphasis_speed,
        )
        horizontal_scale = horizontal_scale_for_state(args, state, horizontal_speed)
        gated_dx_e = pred_dx_e * horizontal_scale
        gated_dx_n = pred_dx_n * horizontal_scale
        gated_dz_u = pred_dz_u

        pred_e += gated_dx_e
        pred_n += gated_dx_n
        pred_u += gated_dz_u

        true_e = as_float(row["true_future_east_m"])
        true_n = as_float(row["true_future_north_m"])
        true_u = as_float(row["true_future_up_m"])
        err_e = pred_e - true_e
        err_n = pred_n - true_n
        err_u = pred_u - true_u
        err_h = math.hypot(err_e, err_n)
        err_3d = math.sqrt(err_h * err_h + err_u * err_u)

        baseline = baseline_index.get((row.get("fold_id", "1"), round(as_float(row["future_time_s"]), 6)), {})
        baseline_err_3d = as_float(baseline.get("err_3d_m"))
        baseline_err_h = math.hypot(as_float(baseline.get("err_east_m")), as_float(baseline.get("err_north_m")))

        result.append(
            {
                "fold_id": row.get("fold_id", "1"),
                "step": float(step_idx),
                "time_s": as_float(row["future_time_s"]),
                "state": state,
                "state_color": STATE_COLORS[state],
                "horizontal_scale": horizontal_scale,
                "true_east_m": true_e,
                "true_north_m": true_n,
                "true_up_m": true_u,
                "pred_east_m": pred_e,
                "pred_north_m": pred_n,
                "pred_up_m": pred_u,
                "err_east_m": err_e,
                "err_north_m": err_n,
                "err_up_m": err_u,
                "err_horizontal_m": err_h,
                "err_3d_m": err_3d,
                "baseline_err_horizontal_m": baseline_err_h,
                "baseline_err_3d_m": baseline_err_3d,
                "delta_err_horizontal_m": err_h - baseline_err_h,
                "delta_err_3d_m": err_3d - baseline_err_3d,
                "horizontal_speed_mps": horizontal_speed,
                "vertical_speed_mps": vertical_speed,
                "pred_dx_east_m": pred_dx_e,
                "pred_dy_north_m": pred_dx_n,
                "pred_dz_up_m": pred_dz_u,
                "gated_dx_east_m": gated_dx_e,
                "gated_dy_north_m": gated_dx_n,
                "gated_dz_up_m": gated_dz_u,
            }
        )
    return result


def build_rollout(args: argparse.Namespace) -> list[dict[str, float | str]]:
    source_rows = read_rows(args.pred_csv)
    if not source_rows:
        return []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[row.get("fold_id", "1")].append(row)
    baseline_index = build_baseline_index(args.baseline_rollout_csv)
    result: list[dict[str, float | str]] = []
    for fold_id, fold_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        result.extend(build_fold_rollout(fold_rows, args, baseline_index))
    return result


def p95(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def metric_summary(rows: list[dict[str, float | str]]) -> dict[str, float]:
    err3d = [float(row["err_3d_m"]) for row in rows]
    errh = [float(row["err_horizontal_m"]) for row in rows]
    return {
        "steps": float(len(rows)),
        "final_3d": err3d[-1] if err3d else math.nan,
        "mean_3d": sum(err3d) / len(err3d) if err3d else math.nan,
        "max_3d": max(err3d) if err3d else math.nan,
        "final_h": errh[-1] if errh else math.nan,
        "mean_h": sum(errh) / len(errh) if errh else math.nan,
        "max_h": max(errh) if errh else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(value) for key, value in row.items()})


def format_value(value: float | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state"])].append(row)

    summary = metric_summary(rows)
    base_err3d = [float(row["baseline_err_3d_m"]) for row in rows]
    base_errh = [float(row["baseline_err_horizontal_m"]) for row in rows]
    lines = [
        "# DataFlash State-Gated Sparse Rollout",
        "",
        "This report is generated by `src/build_dataflash_state_gated_rollout.py`.",
        "",
        f"Predictions: `{args.pred_csv}`",
        f"Baseline rollout: `{args.baseline_rollout_csv}`",
        f"Gated rollout CSV: `{args.out_csv}`",
        f"HTML: `{args.html}`",
        "",
        "Horizontal gating policy:",
        f"- `hover` -> scale XY by `{args.hover_horizontal_scale:.2f}`",
        f"- `climb/descent` -> scale XY by `{args.vertical_horizontal_scale:.2f}`",
        f"- `translate` -> scale XY by `{args.translate_horizontal_scale:.2f}`",
        (
            f"- `fast translate` (`horizontal speed >= {args.translate_fast_threshold:.2f} m/s`) "
            f"-> scale XY by `{args.translate_fast_horizontal_scale:.2f}`"
            if math.isfinite(args.translate_fast_threshold)
            else "- `fast translate` override disabled"
        ),
        "",
        "## Overall",
        "",
        f"- steps: {len(rows)}",
        f"- baseline final 3D: {base_err3d[-1]:.3f} m",
        f"- gated final 3D: {summary['final_3d']:.3f} m",
        f"- baseline mean 3D: {sum(base_err3d) / len(base_err3d):.3f} m",
        f"- gated mean 3D: {summary['mean_3d']:.3f} m",
        f"- baseline final horizontal: {base_errh[-1]:.3f} m",
        f"- gated final horizontal: {summary['final_h']:.3f} m",
        f"- baseline mean horizontal: {sum(base_errh) / len(base_errh):.3f} m",
        f"- gated mean horizontal: {summary['mean_h']:.3f} m",
        "",
        "## By State",
        "",
        "| state | rows | gated mean 3D | gated p95 3D | gated mean horiz | gated p95 horiz | mean delta 3D vs baseline | mean delta horiz vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for state in ("hover", "climb", "descent", "translate"):
        state_rows = grouped.get(state, [])
        if not state_rows:
            continue
        err3d = [float(row["err_3d_m"]) for row in state_rows]
        errh = [float(row["err_horizontal_m"]) for row in state_rows]
        delta3d = [float(row["delta_err_3d_m"]) for row in state_rows]
        deltah = [float(row["delta_err_horizontal_m"]) for row in state_rows]
        lines.append(
            f"| `{state}` | {len(state_rows)} | {sum(err3d) / len(err3d):.3f} | {p95(err3d):.3f} | "
            f"{sum(errh) / len(errh):.3f} | {p95(errh):.3f} | "
            f"{sum(delta3d) / len(delta3d):.3f} | {sum(deltah) / len(deltah):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: negative delta means the gated rollout improved error relative to the current best baseline.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_template(rows: list[dict[str, float | str]], args: argparse.Namespace) -> str:
    payload = json.dumps(rows, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFlash State-Gated Rollout</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #16202a;
    }}
    body {{ margin: 0; background: #f4f6f8; }}
    header {{ padding: 16px 20px; background: #fff; border-bottom: 1px solid #d7dee8; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .muted {{ color: #52606d; font-size: 14px; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1.3fr) 360px; gap: 16px; padding: 16px; }}
    .panel {{ background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 14px; overflow: hidden; }}
    .left {{ display: grid; gap: 16px; min-width: 0; }}
    .right {{ display: grid; gap: 16px; min-width: 0; }}
    svg {{ width: 100%; display: block; border: 1px solid #e2e8f0; border-radius: 8px; background: #fbfcfd; }}
    .metric {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; padding: 7px 0; border-bottom: 1px solid #edf1f5; font-size: 14px; }}
    .metric:last-child {{ border-bottom: 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid #edf1f5; }}
    th {{ font-size: 12px; color: #52606d; }}
    @media (max-width: 1080px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>DataFlash State-Gated Rollout</h1>
    <div class="muted">Policy: hover={args.hover_horizontal_scale:.2f}, climb/descent={args.vertical_horizontal_scale:.2f}, translate={args.translate_horizontal_scale:.2f}, fast-translate={args.translate_fast_horizontal_scale:.2f} above {args.translate_fast_threshold if math.isfinite(args.translate_fast_threshold) else float("inf"):.2f} m/s.</div>
  </header>
  <main>
    <section class="left">
      <div class="panel">
        <svg id="traj" viewBox="0 0 880 420" role="img" aria-label="Trajectory comparison"></svg>
      </div>
      <div class="panel">
        <svg id="delta" viewBox="0 0 880 240" role="img" aria-label="Error delta timeline"></svg>
      </div>
    </section>
    <aside class="right">
      <div class="panel"><div id="overall"></div></div>
      <div class="panel"><table id="stateTable"></table></div>
    </aside>
  </main>
  <script>
    const rows = {payload};
    const stateColors = {json.dumps(STATE_COLORS, ensure_ascii=True)};
    function fmt(v) {{ return Number.isFinite(v) ? v.toFixed(2) : "n/a"; }}
    function extents(values) {{
      const filtered = values.filter(Number.isFinite);
      let min = Math.min(...filtered), max = Math.max(...filtered);
      if (min === max) {{ min -= 1; max += 1; }}
      return [min, max];
    }}
    function scale(value, inMin, inMax, outMin, outMax) {{
      return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
    }}
    function create(tag, attrs = {{}}, text = "") {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      if (text) node.textContent = text;
      return node;
    }}
    function drawTraj() {{
      const svg = document.getElementById("traj");
      const width = 880, height = 420, m = {{left: 48, right: 18, top: 18, bottom: 36}};
      const xs = rows.flatMap((r) => [Number(r.true_east_m), Number(r.pred_east_m)]);
      const ys = rows.flatMap((r) => [Number(r.true_north_m), Number(r.pred_north_m)]);
      const [xMin, xMax] = extents(xs), [yMin, yMax] = extents(ys);
      let truePath = "", predPath = "";
      svg.innerHTML = "";
      rows.forEach((r, i) => {{
        const tx = scale(Number(r.true_east_m), xMin, xMax, m.left, width - m.right);
        const ty = scale(Number(r.true_north_m), yMin, yMax, height - m.bottom, m.top);
        const px = scale(Number(r.pred_east_m), xMin, xMax, m.left, width - m.right);
        const py = scale(Number(r.pred_north_m), yMin, yMax, height - m.bottom, m.top);
        truePath += `${{i ? " L" : "M"}}${{tx}} ${{ty}}`;
        predPath += `${{i ? " L" : "M"}}${{px}} ${{py}}`;
      }});
      svg.appendChild(create("path", {{d: truePath, fill: "none", stroke: "#0f172a", "stroke-width": 2.1}}));
      svg.appendChild(create("path", {{d: predPath, fill: "none", stroke: "#94a3b8", "stroke-width": 2, "stroke-dasharray": "5 4"}}));
      rows.forEach((r) => {{
        const px = scale(Number(r.pred_east_m), xMin, xMax, m.left, width - m.right);
        const py = scale(Number(r.pred_north_m), yMin, yMax, height - m.bottom, m.top);
        svg.appendChild(create("circle", {{cx: px, cy: py, r: 3.8, fill: stateColors[r.state] || "#334155", opacity: 0.85}}));
      }});
    }}
    function drawDelta() {{
      const svg = document.getElementById("delta");
      const width = 880, height = 240, m = {{left: 48, right: 18, top: 18, bottom: 30}};
      const [xMin, xMax] = extents(rows.map((r) => Number(r.time_s)));
      const [yMin, yMax] = extents(rows.map((r) => Number(r.delta_err_horizontal_m)));
      svg.innerHTML = "";
      svg.appendChild(create("line", {{x1: m.left, y1: scale(0, yMin, yMax, height - m.bottom, m.top), x2: width - m.right, y2: scale(0, yMin, yMax, height - m.bottom, m.top), stroke: "#cbd5e1"}}));
      rows.forEach((r) => {{
        const x = scale(Number(r.time_s), xMin, xMax, m.left, width - m.right);
        const y = scale(Number(r.delta_err_horizontal_m), yMin, yMax, height - m.bottom, m.top);
        svg.appendChild(create("circle", {{cx: x, cy: y, r: 3.2, fill: stateColors[r.state] || "#334155"}}));
      }});
    }}
    function renderOverall() {{
      const base3d = rows.map((r) => Number(r.baseline_err_3d_m));
      const gated3d = rows.map((r) => Number(r.err_3d_m));
      const baseH = rows.map((r) => Number(r.baseline_err_horizontal_m));
      const gatedH = rows.map((r) => Number(r.err_horizontal_m));
      document.getElementById("overall").innerHTML = `
        <div class="metric"><span>baseline final 3D</span><strong>${{fmt(base3d.at(-1))}} m</strong></div>
        <div class="metric"><span>gated final 3D</span><strong>${{fmt(gated3d.at(-1))}} m</strong></div>
        <div class="metric"><span>baseline final horizontal</span><strong>${{fmt(baseH.at(-1))}} m</strong></div>
        <div class="metric"><span>gated final horizontal</span><strong>${{fmt(gatedH.at(-1))}} m</strong></div>
        <div class="metric"><span>mean delta horizontal</span><strong>${{fmt(gatedH.reduce((a,b)=>a+b,0)/gatedH.length - baseH.reduce((a,b)=>a+b,0)/baseH.length)}} m</strong></div>
      `;
    }}
    function renderTable() {{
      const byState = new Map();
      rows.forEach((r) => {{
        const state = r.state;
        if (!byState.has(state)) byState.set(state, []);
        byState.get(state).push(r);
      }});
      const order = ["hover", "climb", "descent", "translate"];
      document.getElementById("stateTable").innerHTML = `
        <thead><tr><th>state</th><th>rows</th><th>gated mean h</th><th>delta mean h</th></tr></thead>
        <tbody>
          ${{order.filter((s) => byState.has(s)).map((s) => {{
            const items = byState.get(s);
            const meanH = items.reduce((a, r) => a + Number(r.err_horizontal_m), 0) / items.length;
            const deltaH = items.reduce((a, r) => a + Number(r.delta_err_horizontal_m), 0) / items.length;
            return `<tr><td>${{s}}</td><td>${{items.length}}</td><td>${{fmt(meanH)}}</td><td>${{fmt(deltaH)}}</td></tr>`;
          }}).join("")}}
        </tbody>
      `;
    }}
    drawTraj();
    drawDelta();
    renderOverall();
    renderTable();
  </script>
</body>
</html>
"""


def write_html(path: Path, rows: list[dict[str, float | str]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_template(rows, args), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rollout(args)
    if not rows:
        raise SystemExit("No rollout rows produced.")
    write_csv(args.out_csv, rows)
    write_report(args.report, args, rows)
    write_html(args.html, rows, args)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
