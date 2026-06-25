#!/usr/bin/env python3
"""Build a static viewer for DataFlash future-position predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/dataflash_sweep"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/dataflash/predictions/sweep/index.html"))
    parser.add_argument("--max-points", type=int, default=1500)
    return parser.parse_args()


def as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_prediction(path: Path, max_points: int) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "time": as_float(row["time_s"]),
                    "current_e": as_float(row["current_east_m"]),
                    "current_n": as_float(row["current_north_m"]),
                    "true_e": as_float(row["true_future_east_m"]),
                    "true_n": as_float(row["true_future_north_m"]),
                    "pred_e": as_float(row["pred_future_east_m"]),
                    "pred_n": as_float(row["pred_future_north_m"]),
                    "err": math.sqrt(
                        (as_float(row["pred_dx_east_m"]) - as_float(row["true_dx_east_m"])) ** 2
                        + (as_float(row["pred_dy_north_m"]) - as_float(row["true_dy_north_m"])) ** 2
                        + (as_float(row["pred_dz_up_m"]) - as_float(row["true_dz_up_m"])) ** 2
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No rows in {path}")
    stride = max(1, math.ceil(len(rows) / max_points))
    sampled = rows[::stride]
    first = sampled[0]
    label = f"{first_label(path)}"
    mae = sum(row["err"] for row in rows) / len(rows)
    p95 = sorted(row["err"] for row in rows)[int(0.95 * (len(rows) - 1))]
    return {
        "id": path.parent.name + "/" + path.stem,
        "label": label,
        "count": len(rows),
        "shown": len(sampled),
        "mae": mae,
        "p95": p95,
        "rows": sampled,
    }


def first_label(path: Path) -> str:
    case = path.parent.name
    model = path.stem.replace("_pred", "")
    return f"{case} / {model}"


def collect_cases(pred_dir: Path, max_points: int) -> list[dict[str, object]]:
    paths = sorted(pred_dir.glob("*/*_pred.csv"))
    return [read_prediction(path, max_points) for path in paths]


def html_template(cases: list[dict[str, object]]) -> str:
    payload = json.dumps(cases, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFlash Prediction Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
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
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      background: #ffffff;
      border-bottom: 1px solid #d8dde5;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    select {{
      min-width: 320px;
      max-width: 72vw;
      font: inherit;
      padding: 7px 9px;
      border: 1px solid #bac3cf;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
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
    @media (max-width: 820px) {{
      header {{
        align-items: stretch;
        flex-direction: column;
      }}
      main {{
        grid-template-columns: 1fr;
      }}
      svg {{
        height: 68vh;
      }}
      aside {{
        border-left: 0;
        border-top: 1px solid #d8dde5;
      }}
      select {{
        min-width: 0;
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DataFlash Prediction Viewer</h1>
    <div class="controls">
      <select id="caseSelect" aria-label="Prediction case"></select>
    </div>
  </header>
  <main>
    <svg id="plot" role="img" aria-label="Prediction plot"></svg>
    <aside>
      <div class="metric"><span>Rows</span><strong id="count">-</strong></div>
      <div class="metric"><span>Shown</span><strong id="shown">-</strong></div>
      <div class="metric"><span>MAE 3D</span><strong id="mae">-</strong></div>
      <div class="metric"><span>P95 3D</span><strong id="p95">-</strong></div>
      <div class="legend">
        <div class="key"><span class="swatch" style="background:#6b7280"></span><span>current POS at prediction time</span></div>
        <div class="key"><span class="swatch" style="background:#2563eb"></span><span>true future POS</span></div>
        <div class="key"><span class="swatch" style="background:#dc2626"></span><span>predicted future POS</span></div>
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
      return points.map((p, i) => `${{i ? 'L' : 'M'}} ${{scale.x(p[xKey]).toFixed(2)}} ${{scale.y(p[yKey]).toFixed(2)}}`).join(' ');
    }}
    function render() {{
      const item = cases.find((candidate) => candidate.id === select.value) || cases[0];
      const rows = item.rows;
      const xs = rows.flatMap((p) => [p.current_e, p.true_e, p.pred_e]);
      const ys = rows.flatMap((p) => [p.current_n, p.true_n, p.pred_n]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const pad = 34;
      const rect = svg.getBoundingClientRect();
      const w = Math.max(rect.width, 320);
      const h = Math.max(rect.height, 320);
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const scale = {{
        x: (x) => pad + (x - minX) / spanX * (w - pad * 2),
        y: (y) => h - pad - (y - minY) / spanY * (h - pad * 2),
      }};
      svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
      svg.innerHTML = `
        <path d="${{pathFor(rows, 'current_e', 'current_n', scale)}}" fill="none" stroke="#6b7280" stroke-width="2" opacity="0.7"/>
        <path d="${{pathFor(rows, 'true_e', 'true_n', scale)}}" fill="none" stroke="#2563eb" stroke-width="2.4"/>
        <path d="${{pathFor(rows, 'pred_e', 'pred_n', scale)}}" fill="none" stroke="#dc2626" stroke-width="2.4"/>
      `;
      document.getElementById('count').textContent = item.count.toLocaleString();
      document.getElementById('shown').textContent = item.shown.toLocaleString();
      document.getElementById('mae').textContent = `${{item.mae.toFixed(3)}} m`;
      document.getElementById('p95').textContent = `${{item.p95.toFixed(3)}} m`;
    }}
    select.addEventListener('change', render);
    window.addEventListener('resize', render);
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    cases = collect_cases(args.pred_dir, args.max_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_template(cases), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
