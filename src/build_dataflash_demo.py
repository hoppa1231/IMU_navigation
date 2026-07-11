#!/usr/bin/env python3
"""Build an animated DataFlash rollout demo page."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_bias_rollout.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/dataflash/demo/index.html"),
    )
    parser.add_argument("--model-label", default="sequence_ridge_bias_corrected")
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


def build_payload(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_fold: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        fold_id = row.get("fold_id", "1")
        by_fold.setdefault(fold_id, []).append(
            {
                "step": as_float(row["step"]),
                "time_s": as_float(row["time_s"]),
                "true_east_m": as_float(row["true_east_m"]),
                "true_north_m": as_float(row["true_north_m"]),
                "true_up_m": as_float(row["true_up_m"]),
                "pred_east_m": as_float(row["pred_east_m"]),
                "pred_north_m": as_float(row["pred_north_m"]),
                "pred_up_m": as_float(row["pred_up_m"]),
                "err_3d_m": as_float(row["err_3d_m"]),
            }
        )
    payload = []
    for fold_id, fold_rows in sorted(by_fold.items(), key=lambda item: int(item[0])):
        fold_rows.sort(key=lambda row: row["step"])
        times = [row["time_s"] for row in fold_rows]
        errors = [row["err_3d_m"] for row in fold_rows]
        payload.append(
            {
                "id": fold_id,
                "label": f"Fold {fold_id}",
                "rows": fold_rows,
                "duration_s": max(times) - min(times) if len(times) >= 2 else 0.0,
                "final_error_m": errors[-1] if errors else math.nan,
                "mean_error_m": sum(errors) / len(errors) if errors else math.nan,
                "max_error_m": max(errors) if errors else math.nan,
            }
        )
    return payload


def html_template(payload: list[dict[str, object]], model_label: str) -> str:
    data = json.dumps(payload, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFlash Rollout Demo</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #16202a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
      background: #f4f6f8;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 20px;
      background: #ffffff;
      border-bottom: 1px solid #d7dee8;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, auto));
      gap: 10px 14px;
      padding: 14px 20px;
      align-items: center;
      background: #ffffff;
      border-bottom: 1px solid #d7dee8;
    }}
    select, button, input[type="range"] {{
      font: inherit;
    }}
    select, button {{
      border: 1px solid #b8c3cf;
      border-radius: 6px;
      background: #ffffff;
      color: #16202a;
      padding: 8px 10px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      min-height: 0;
    }}
    .plotWrap {{
      padding: 16px;
      min-height: 0;
    }}
    svg {{
      width: 100%;
      height: calc(100vh - 162px);
      background: #eef2f6;
      border: 1px solid #d7dee8;
      border-radius: 8px;
    }}
    aside {{
      padding: 16px;
      border-left: 1px solid #d7dee8;
      background: #ffffff;
      display: grid;
      gap: 12px;
      align-content: start;
      overflow: auto;
    }}
    .panel {{
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 14px;
      background: #fbfcfd;
    }}
    .metric {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 6px 0;
      border-bottom: 1px solid #edf1f5;
      font-size: 14px;
    }}
    .metric:last-child {{
      border-bottom: 0;
    }}
    .legend {{
      display: grid;
      gap: 8px;
      font-size: 14px;
    }}
    .key {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 20px;
      height: 4px;
      border-radius: 4px;
    }}
    .muted {{
      color: #5b6674;
      font-size: 13px;
    }}
    @media (max-width: 960px) {{
      .controls {{
        grid-template-columns: 1fr 1fr;
      }}
      main {{
        grid-template-columns: 1fr;
      }}
      svg {{
        height: 62vh;
      }}
      aside {{
        border-left: 0;
        border-top: 1px solid #d7dee8;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DataFlash 3D Rollout</h1>
    <div class="muted">{model_label}: real POS/GPS and predicted route with relative altitude.</div>
  </header>
  <div class="controls">
    <select id="caseSelect" aria-label="Scenario"></select>
    <button id="playButton" type="button">Play</button>
    <select id="speedSelect" aria-label="Speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="5">5x</option>
    </select>
    <input id="progress" type="range" min="0" max="1000" value="0" aria-label="Progress">
  </div>
  <main>
    <div class="plotWrap">
      <svg id="plot" role="img" aria-label="Animated DataFlash rollout"></svg>
    </div>
    <aside>
      <div class="panel">
        <div class="metric"><span>Step</span><strong id="stepValue">-</strong></div>
        <div class="metric"><span>Time</span><strong id="timeValue">-</strong></div>
        <div class="metric"><span>Current Error</span><strong id="currentError">-</strong></div>
        <div class="metric"><span>Real Altitude</span><strong id="trueAltitude">-</strong></div>
        <div class="metric"><span>Predicted Altitude</span><strong id="predAltitude">-</strong></div>
      </div>
      <div class="panel">
        <div class="metric"><span>Final Error</span><strong id="finalError">-</strong></div>
        <div class="metric"><span>Mean Error</span><strong id="meanError">-</strong></div>
        <div class="metric"><span>Max Error</span><strong id="maxError">-</strong></div>
      </div>
      <div class="panel">
        <div class="legend">
          <div class="key"><span class="swatch" style="background:#94a3b8"></span><span>ground projection</span></div>
          <div class="key"><span class="swatch" style="background:#2563eb"></span><span>real trajectory so far</span></div>
          <div class="key"><span class="swatch" style="background:#dc2626"></span><span>predicted trajectory so far</span></div>
        </div>
      </div>
      <div class="panel">
        <div class="muted" id="caseNote"></div>
      </div>
    </aside>
  </main>
  <script>
    const cases = {data};
    const caseSelect = document.getElementById('caseSelect');
    const playButton = document.getElementById('playButton');
    const speedSelect = document.getElementById('speedSelect');
    const progress = document.getElementById('progress');
    const svg = document.getElementById('plot');
    const stepValue = document.getElementById('stepValue');
    const timeValue = document.getElementById('timeValue');
    const currentError = document.getElementById('currentError');
    const trueAltitude = document.getElementById('trueAltitude');
    const predAltitude = document.getElementById('predAltitude');
    const finalError = document.getElementById('finalError');
    const meanError = document.getElementById('meanError');
    const maxError = document.getElementById('maxError');
    const caseNote = document.getElementById('caseNote');

    for (const item of cases) {{
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.label;
      caseSelect.appendChild(option);
    }}

    let playing = false;
    let playhead = 0;
    let lastFrame = 0;

    function currentCase() {{
      return cases.find((item) => item.id === caseSelect.value) || cases[0];
    }}

    function extent(points, key) {{
      let min = Infinity;
      let max = -Infinity;
      for (const point of points) {{
        min = Math.min(min, point[key]);
        max = Math.max(max, point[key]);
      }}
      return [min, max];
    }}

    function projectFactory(rows, width, height) {{
      const xs = rows.flatMap((row) => [row.true_east_m, row.pred_east_m]);
      const ys = rows.flatMap((row) => [row.true_north_m, row.pred_north_m]);
      const actualMinX = Math.min(...xs);
      const actualMaxX = Math.max(...xs);
      const actualMinY = Math.min(...ys);
      const actualMaxY = Math.max(...ys);
      const zs = rows.flatMap((row) => [row.true_up_m, row.pred_up_m]);
      const minZ = Math.min(...zs);
      const maxZ = Math.max(...zs);
      const pad = 34;
      const spanX = Math.max(1, actualMaxX - actualMinX);
      const spanY = Math.max(1, actualMaxY - actualMinY);
      const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY) * 0.78;
      const offsetX = (width - spanX * scale) / 2;
      const offsetY = (height - spanY * scale) / 2;
      const spanZ = Math.max(1, maxZ - minZ);
      return (east, north, up = minZ) => {{
        const groundX = offsetX + (east - actualMinX) * scale;
        const groundY = height - offsetY - (north - actualMinY) * scale;
        const lift = (up - minZ) / spanZ * Math.min(150, height * 0.24);
        return {{ x: groundX, y: groundY - lift, groundX, groundY }};
      }};
    }}

    function pathString(rows, eastKey, northKey, upKey, project, count) {{
      const active = rows.slice(0, count);
      return active.map((row, index) => {{
        const point = project(row[eastKey], row[northKey], row[upKey]);
        return `${{index ? 'L' : 'M'}} ${{point.x.toFixed(2)}} ${{point.y.toFixed(2)}}`;
      }}).join(' ');
    }}

    function render() {{
      const data = currentCase();
      const rows = data.rows;
      if (!rows.length) return;
      const rect = svg.getBoundingClientRect();
      const width = Math.max(320, rect.width);
      const height = Math.max(320, rect.height);
      const project = projectFactory(rows, width, height);
      const count = Math.max(1, Math.min(rows.length, Math.round(playhead)));
      const current = rows[count - 1];
      const trueEnd = project(current.true_east_m, current.true_north_m, current.true_up_m);
      const predEnd = project(current.pred_east_m, current.pred_north_m, current.pred_up_m);

      const groundReference = pathString(rows, 'true_east_m', 'true_north_m', 'ground_up_m',
        (e, n) => project(e, n), rows.length);
      const trueSoFar = pathString(rows, 'true_east_m', 'true_north_m', 'true_up_m', project, count);
      const predSoFar = pathString(rows, 'pred_east_m', 'pred_north_m', 'pred_up_m', project, count);

      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#eef2f6" rx="8" />
        <path d="${{groundReference}}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-opacity="0.55" stroke-dasharray="5 5"/>
        <path d="${{trueSoFar}}" fill="none" stroke="#2563eb" stroke-width="2.8"/>
        <path d="${{predSoFar}}" fill="none" stroke="#dc2626" stroke-width="2.8"/>
        <line x1="${{trueEnd.x.toFixed(2)}}" y1="${{trueEnd.y.toFixed(2)}}" x2="${{trueEnd.groundX.toFixed(2)}}" y2="${{trueEnd.groundY.toFixed(2)}}" stroke="#2563eb" stroke-opacity="0.45" stroke-width="1.5"/>
        <line x1="${{predEnd.x.toFixed(2)}}" y1="${{predEnd.y.toFixed(2)}}" x2="${{predEnd.groundX.toFixed(2)}}" y2="${{predEnd.groundY.toFixed(2)}}" stroke="#dc2626" stroke-opacity="0.45" stroke-width="1.5"/>
        <ellipse cx="${{trueEnd.groundX.toFixed(2)}}" cy="${{trueEnd.groundY.toFixed(2)}}" rx="8" ry="3" fill="#2563eb" fill-opacity="0.2"/>
        <ellipse cx="${{predEnd.groundX.toFixed(2)}}" cy="${{predEnd.groundY.toFixed(2)}}" rx="8" ry="3" fill="#dc2626" fill-opacity="0.2"/>
        <circle cx="${{trueEnd.x.toFixed(2)}}" cy="${{trueEnd.y.toFixed(2)}}" r="5.5" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>
        <circle cx="${{predEnd.x.toFixed(2)}}" cy="${{predEnd.y.toFixed(2)}}" r="5.5" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
      `;

      progress.value = Math.round((count - 1) / Math.max(rows.length - 1, 1) * 1000);
      stepValue.textContent = `${{count}} / ${{rows.length}}`;
      timeValue.textContent = `${{current.time_s.toFixed(3)}} s`;
      currentError.textContent = `${{current.err_3d_m.toFixed(3)}} m`;
      trueAltitude.textContent = `${{current.true_up_m.toFixed(2)}} m`;
      predAltitude.textContent = `${{current.pred_up_m.toFixed(2)}} m`;
      finalError.textContent = `${{data.final_error_m.toFixed(3)}} m`;
      meanError.textContent = `${{data.mean_error_m.toFixed(3)}} m`;
      maxError.textContent = `${{data.max_error_m.toFixed(3)}} m`;
      caseNote.textContent = 'Each fold is a separate test interval. Altitude is relative to the first POS point of the complete flight.';
    }}

    function tick(timestamp) {{
      if (!lastFrame) lastFrame = timestamp;
      const dt = (timestamp - lastFrame) / 1000;
      lastFrame = timestamp;
      if (playing) {{
        const rows = currentCase().rows;
        const speed = Number(speedSelect.value) || 1;
        playhead += dt * speed * 2.5;
        if (playhead >= rows.length) {{
          playhead = rows.length;
          playing = false;
          playButton.textContent = 'Play';
        }}
        render();
      }}
      requestAnimationFrame(tick);
    }}

    playButton.addEventListener('click', () => {{
      const rows = currentCase().rows;
      if (playhead >= rows.length) playhead = 1;
      playing = !playing;
      playButton.textContent = playing ? 'Pause' : 'Play';
      lastFrame = 0;
      render();
    }});

    progress.addEventListener('input', () => {{
      const rows = currentCase().rows;
      playing = false;
      playButton.textContent = 'Play';
      playhead = 1 + Number(progress.value) / 1000 * Math.max(rows.length - 1, 0);
      render();
    }});

    caseSelect.addEventListener('change', () => {{
      playing = false;
      playButton.textContent = 'Play';
      playhead = 1;
      render();
    }});

    window.addEventListener('resize', render);
    caseSelect.value = cases[0].id;
    playhead = 1;
    render();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = read_rows(args.rollout_csv)
    payload = build_payload(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_template(payload, args.model_label), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
