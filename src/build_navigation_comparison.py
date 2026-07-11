#!/usr/bin/env python3
"""Build a single comparison page for GPS/POS and navigation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


MAX_POINTS = 2500


@dataclass
class SeriesPayload:
    id: str
    label: str
    kind: str
    color: str
    note: str
    detail_html: str
    final_error_m: float
    mean_error_m: float
    max_error_m: float
    points: list[dict[str, float]]


@dataclass
class CasePayload:
    id: str
    title: str
    reference_label: str
    reference_note: str
    reference_points: list[dict[str, float]]
    series: list[SeriesPayload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--flow-csv", type=Path, default=Path("derived/predictions/flow_dead_reckoning/flow_dr.csv"))
    parser.add_argument("--poli-csv", type=Path, default=Path("derived/predictions/poli_na_rollout/poli_na_rollout.csv"))
    parser.add_argument("--imu-csv", type=Path, default=Path("derived/predictions/imu_dead_reckoning/dataflash_imu_dr.csv"))
    parser.add_argument(
        "--dataflash-rollout-csv",
        type=Path,
        default=Path("derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/navigation/comparison/index.html"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/navigation/navigation_comparison.md"),
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sample_rows(rows: list[dict[str, float]], max_points: int = MAX_POINTS) -> list[dict[str, float]]:
    if len(rows) <= max_points:
        return rows
    stride = max(1, math.ceil(len(rows) / max_points))
    sampled = rows[::stride]
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def load_track_reference(path: Path) -> list[dict[str, float]]:
    rows = []
    for row in read_rows(path):
        rows.append(
            {
                "time_s": as_float(row["time_s"]),
                "east_m": as_float(row["east_m"]),
                "north_m": as_float(row["north_m"]),
                "up_m": as_float(row["up_m"]),
            }
        )
    return sample_rows(rows)


def build_series_from_rows(
    *,
    rows: list[dict[str, str]],
    point_fields: tuple[str, str, str],
    err_field: str,
    label: str,
    kind: str,
    color: str,
    note: str,
    detail_html: str,
    series_id: str,
) -> SeriesPayload:
    points = [
        {
            "time_s": as_float(row["time_s"]),
            "east_m": as_float(row[point_fields[0]]),
            "north_m": as_float(row[point_fields[1]]),
            "up_m": as_float(row[point_fields[2]]),
        }
        for row in rows
    ]
    errors = [as_float(row[err_field]) for row in rows if math.isfinite(as_float(row[err_field]))]
    final_error = errors[-1] if errors else math.nan
    mean_error = sum(errors) / len(errors) if errors else math.nan
    max_error = max(errors) if errors else math.nan
    return SeriesPayload(
        id=series_id,
        label=label,
        kind=kind,
        color=color,
        note=note,
        detail_html=detail_html,
        final_error_m=final_error,
        mean_error_m=mean_error,
        max_error_m=max_error,
        points=sample_rows(points),
    )


def select_rows(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key) == value for key, value in filters.items())]


def build_cases(args: argparse.Namespace) -> list[CasePayload]:
    base_dir = args.output.parent
    flow_rows = read_rows(args.flow_csv)
    poli_rows = read_rows(args.poli_csv)
    imu_rows = read_rows(args.imu_csv)
    dataflash_rollout_rows = read_rows(args.dataflash_rollout_csv)

    linear_reference = load_track_reference(args.tracks_dir / "linear_15_01_2025_track.csv")
    triangle_reference = load_track_reference(args.tracks_dir / "triangle_15_01_2025_track.csv")
    dataflash_reference = sample_rows(
        [
            {
                "time_s": as_float(row["time_s"]),
                "east_m": as_float(row["true_east_m"]),
                "north_m": as_float(row["true_north_m"]),
                "up_m": as_float(row["true_up_m"]),
            }
            for row in imu_rows
        ]
    )

    cases = [
        CasePayload(
            id="linear",
            title="Linear module flight",
            reference_label="Real GPS",
            reference_note="Prepared ENU GPS track from the single-flight module CSV.",
            reference_points=linear_reference,
            series=[
                build_series_from_rows(
                    rows=select_rows(flow_rows, flight_id="linear_15_01_2025", model="flow_only"),
                    point_fields=("pred_east_m", "pred_north_m", "pred_up_m"),
                    err_field="err_3d_m",
                    label="Flow only",
                    kind="open-loop velocity integration",
                    color="#ef6c00",
                    note="Ridge velocity model from optical-flow aggregates only.",
                    detail_html=relpath(Path("artifacts/generated/navigation/flow_dead_reckoning/index.html"), base_dir),
                    series_id="linear_flow_only",
                ),
                build_series_from_rows(
                    rows=select_rows(poli_rows, flight_id="linear_15_01_2025", feature_preset="imu_flow_mag10_raw"),
                    point_fields=("pred_east_m", "pred_north_m", "pred_up_m"),
                    err_field="err_3d_m",
                    label="POLI_NA best preset",
                    kind="open-loop displacement rollout",
                    color="#7b1fa2",
                    note="Best available preset without the original preprocessing spec.",
                    detail_html=relpath(Path("artifacts/generated/navigation/poli_na_rollout/index.html"), base_dir),
                    series_id="linear_poli_na",
                ),
            ],
        ),
        CasePayload(
            id="triangle",
            title="Triangle module flight",
            reference_label="Real GPS",
            reference_note="Prepared ENU GPS track from the single-flight module CSV.",
            reference_points=triangle_reference,
            series=[
                build_series_from_rows(
                    rows=select_rows(flow_rows, flight_id="triangle_15_01_2025", model="flow_only"),
                    point_fields=("pred_east_m", "pred_north_m", "pred_up_m"),
                    err_field="err_3d_m",
                    label="Flow only",
                    kind="open-loop velocity integration",
                    color="#ef6c00",
                    note="Ridge velocity model from optical-flow aggregates only.",
                    detail_html=relpath(Path("artifacts/generated/navigation/flow_dead_reckoning/index.html"), base_dir),
                    series_id="triangle_flow_only",
                ),
                build_series_from_rows(
                    rows=select_rows(poli_rows, flight_id="triangle_15_01_2025", feature_preset="imu_flow_mag10_raw"),
                    point_fields=("pred_east_m", "pred_north_m", "pred_up_m"),
                    err_field="err_3d_m",
                    label="POLI_NA best preset",
                    kind="open-loop displacement rollout",
                    color="#7b1fa2",
                    note="Best available preset without the original preprocessing spec.",
                    detail_html=relpath(Path("artifacts/generated/navigation/poli_na_rollout/index.html"), base_dir),
                    series_id="triangle_poli_na",
                ),
            ],
        ),
        CasePayload(
            id="dataflash",
            title="DataFlash flight",
            reference_label="Real POS/GPS",
            reference_note="Reference trajectory from the IMU dead-reckoning evaluation target columns.",
            reference_points=dataflash_reference,
            series=[
                build_series_from_rows(
                    rows=imu_rows,
                    point_fields=("pred_east_m", "pred_north_m", "pred_up_m"),
                    err_field="err_3d_m",
                    label="Pure IMU dead reckoning",
                    kind="strapdown double integration",
                    color="#c62828",
                    note="ATT-based gravity compensation plus double integration, no external correction.",
                    detail_html=relpath(Path("artifacts/generated/navigation/imu_dead_reckoning/index.html"), base_dir),
                    series_id="dataflash_imu",
                ),
                build_series_from_rows(
                    rows=dataflash_rollout_rows,
                    point_fields=("pred_east_m", "pred_north_m", "pred_up_m"),
                    err_field="err_3d_m",
                    label="DataFlash best rollout",
                    kind="sparse 5 s displacement rollout",
                    color="#00897b",
                    note="Current best DataFlash model: sequence_ridge_bias_tuned.",
                    detail_html=relpath(
                        Path("artifacts/generated/dataflash/rollouts/sequence_fixed100_shrink/index.html"),
                        base_dir,
                    ),
                    series_id="dataflash_best",
                ),
            ],
        ),
    ]
    return cases


def html_template(cases: list[CasePayload]) -> str:
    payload = [
        {
            "id": case.id,
            "title": case.title,
            "referenceLabel": case.reference_label,
            "referenceNote": case.reference_note,
            "referencePoints": case.reference_points,
            "series": [
                {
                    "id": series.id,
                    "label": series.label,
                    "kind": series.kind,
                    "color": series.color,
                    "note": series.note,
                    "detailHtml": series.detail_html,
                    "finalErrorM": series.final_error_m,
                    "meanErrorM": series.mean_error_m,
                    "maxErrorM": series.max_error_m,
                    "points": series.points,
                }
                for series in case.series
            ],
        }
        for case in cases
    ]
    data = json.dumps(payload, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Navigation Comparison</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f5f7;
      color: #15202b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      background: #f3f5f7;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      background: #ffffff;
      border-bottom: 1px solid #d8dde5;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .subtle {{
      color: #54606e;
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    select {{
      font: inherit;
      min-width: 220px;
      padding: 8px 10px;
      border: 1px solid #b9c3cf;
      border-radius: 6px;
      background: #ffffff;
      color: #15202b;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 0;
    }}
    .plotWrap {{
      min-height: 0;
      padding: 16px;
    }}
    svg {{
      width: 100%;
      height: calc(100vh - 96px);
      background: #eef2f6;
      border: 1px solid #d8dde5;
      border-radius: 8px;
    }}
    aside {{
      min-height: 0;
      overflow: auto;
      padding: 16px;
      background: #ffffff;
      border-left: 1px solid #d8dde5;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .panel {{
      border: 1px solid #e3e8ef;
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
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
    .seriesList {{
      display: grid;
      gap: 10px;
    }}
    .seriesItem {{
      border: 1px solid #e3e8ef;
      border-radius: 8px;
      padding: 10px 12px;
      background: #ffffff;
      display: grid;
      gap: 8px;
    }}
    .seriesHead {{
      display: flex;
      align-items: center;
      gap: 10px;
      justify-content: space-between;
    }}
    .seriesHead label {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 600;
    }}
    .swatch {{
      width: 18px;
      height: 4px;
      border-radius: 4px;
      flex: 0 0 auto;
    }}
    .mini {{
      font-size: 12px;
      color: #5b6674;
    }}
    a {{
      color: #0f62fe;
      text-decoration: none;
      font-size: 13px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
      color: #495463;
    }}
    .legend .key {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    @media (max-width: 980px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      svg {{
        height: 62vh;
      }}
      aside {{
        border-left: 0;
        border-top: 1px solid #d8dde5;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Navigation Comparison</h1>
      <div class="subtle">Real trajectory is always shown together with the selected open-loop methods.</div>
    </div>
    <div class="controls">
      <label>
        <span class="subtle">Scenario</span><br>
        <select id="caseSelect" aria-label="Scenario"></select>
      </label>
    </div>
  </header>
  <main>
    <div class="plotWrap">
      <svg id="plot" role="img" aria-label="Trajectory comparison"></svg>
    </div>
    <aside>
      <div class="panel">
        <h2 id="caseTitle">-</h2>
        <div class="metric"><span>Reference</span><strong id="referenceLabel">-</strong></div>
        <div class="mini" id="referenceNote">-</div>
      </div>
      <div class="panel">
        <h2>Legend</h2>
        <div class="legend">
          <div class="key"><span class="swatch" style="background:#1f2937"></span><span>real trajectory</span></div>
          <div class="key"><span class="swatch" style="background:#94a3b8"></span><span>start point</span></div>
        </div>
      </div>
      <div class="panel">
        <h2>Methods</h2>
        <div class="seriesList" id="seriesList"></div>
      </div>
    </aside>
  </main>
  <script>
    const cases = {data};
    const caseSelect = document.getElementById('caseSelect');
    const plot = document.getElementById('plot');
    const caseTitle = document.getElementById('caseTitle');
    const referenceLabel = document.getElementById('referenceLabel');
    const referenceNote = document.getElementById('referenceNote');
    const seriesList = document.getElementById('seriesList');
    const selectedSeries = new Set();

    for (const item of cases) {{
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.title;
      caseSelect.appendChild(option);
    }}

    function extent(points, key) {{
      let min = Infinity;
      let max = -Infinity;
      for (const point of points) {{
        const value = point[key];
        if (!Number.isFinite(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
      }}
      return [min, max];
    }}

    function currentCase() {{
      return cases.find((item) => item.id === caseSelect.value) || cases[0];
    }}

    function metricLabel(value) {{
      return Number.isFinite(value) ? `${{value.toFixed(3)}} m` : 'n/a';
    }}

    function populateSeries(caseData) {{
      seriesList.innerHTML = '';
      const defaultIds = caseData.series.map((item) => item.id);
      if (![...selectedSeries].some((id) => defaultIds.includes(id))) {{
        selectedSeries.clear();
        for (const id of defaultIds) selectedSeries.add(id);
      }}
      for (const item of caseData.series) {{
        const wrapper = document.createElement('div');
        wrapper.className = 'seriesItem';

        const head = document.createElement('div');
        head.className = 'seriesHead';

        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = selectedSeries.has(item.id);
        checkbox.addEventListener('change', () => {{
          if (checkbox.checked) selectedSeries.add(item.id);
          else selectedSeries.delete(item.id);
          draw();
        }});
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = item.color;
        const text = document.createElement('span');
        text.textContent = item.label;
        label.appendChild(checkbox);
        label.appendChild(swatch);
        label.appendChild(text);
        head.appendChild(label);

        const link = document.createElement('a');
        link.href = item.detailHtml;
        link.textContent = 'details';
        head.appendChild(link);
        wrapper.appendChild(head);

        const kind = document.createElement('div');
        kind.className = 'mini';
        kind.textContent = `${{item.kind}}. ${{item.note}}`;
        wrapper.appendChild(kind);

        const metrics = document.createElement('div');
        metrics.className = 'mini';
        metrics.textContent = `final ${{metricLabel(item.finalErrorM)}}, mean ${{metricLabel(item.meanErrorM)}}, max ${{metricLabel(item.maxErrorM)}}`;
        wrapper.appendChild(metrics);

        seriesList.appendChild(wrapper);
      }}
    }}

    function draw() {{
      const caseData = currentCase();
      caseTitle.textContent = caseData.title;
      referenceLabel.textContent = caseData.referenceLabel;
      referenceNote.textContent = caseData.referenceNote;
      populateSeries(caseData);

      const activeSeries = caseData.series.filter((item) => selectedSeries.has(item.id));
      const allPoints = [...caseData.referencePoints];
      for (const item of activeSeries) allPoints.push(...item.points);
      const [minEast, maxEast] = extent(allPoints, 'east_m');
      const [minNorth, maxNorth] = extent(allPoints, 'north_m');
      const width = plot.clientWidth || 900;
      const height = plot.clientHeight || 700;
      const pad = 36;
      const eastSpan = Math.max(1e-6, maxEast - minEast);
      const northSpan = Math.max(1e-6, maxNorth - minNorth);
      const scale = Math.min((width - pad * 2) / eastSpan, (height - pad * 2) / northSpan);
      const offsetX = (width - eastSpan * scale) / 2;
      const offsetY = (height - northSpan * scale) / 2;

      function project(point) {{
        return {{
          x: offsetX + (point.east_m - minEast) * scale,
          y: height - offsetY - (point.north_m - minNorth) * scale,
        }};
      }}

      function polyline(points, color, widthPx, dash = '') {{
        const coords = points
          .filter((point) => Number.isFinite(point.east_m) && Number.isFinite(point.north_m))
          .map((point) => {{
            const projected = project(point);
            return `${{projected.x.toFixed(2)}},${{projected.y.toFixed(2)}}`;
          }})
          .join(' ');
        const dashAttr = dash ? ` stroke-dasharray="${{dash}}"` : '';
        return `<polyline fill="none" stroke="${{color}}" stroke-width="${{widthPx}}"${{dashAttr}} points="${{coords}}" />`;
      }}

      const start = caseData.referencePoints[0];
      const startPoint = project(start);
      const grid = `
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="#cdd5df" stroke-width="1" />
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="#cdd5df" stroke-width="1" />
        <text x="${{width - pad}}" y="${{height - pad - 8}}" text-anchor="end" fill="#5a6573" font-size="12">east</text>
        <text x="${{pad + 8}}" y="${{pad + 14}}" fill="#5a6573" font-size="12">north</text>
      `;
      const referencePath = polyline(caseData.referencePoints, '#1f2937', 2.6);
      const overlays = activeSeries.map((item) => polyline(item.points, item.color, item.kind.includes('sparse') ? 3.0 : 2.2, item.kind.includes('sparse') ? '8 6' : '')).join('');
      plot.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#eef2f6" rx="8" />
        ${{grid}}
        ${{referencePath}}
        ${{overlays}}
        <circle cx="${{startPoint.x.toFixed(2)}}" cy="${{startPoint.y.toFixed(2)}}" r="5.5" fill="#94a3b8" stroke="#ffffff" stroke-width="2" />
      `;
    }}

    caseSelect.addEventListener('change', () => {{
      selectedSeries.clear();
      for (const item of currentCase().series) selectedSeries.add(item.id);
      draw();
    }});
    window.addEventListener('resize', draw);
    caseSelect.value = cases[0].id;
    for (const item of cases[0].series) selectedSeries.add(item.id);
    draw();
  </script>
</body>
</html>
"""


def write_report(path: Path, cases: list[CasePayload], output: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Navigation Comparison",
        "",
        "This report is generated by `src/build_navigation_comparison.py`.",
        "",
        f"HTML page: `{output}`",
        "",
        "## Scenarios",
        "",
        "| case | real trajectory | compared methods |",
        "| --- | --- | --- |",
    ]
    for case in cases:
        labels = ", ".join(f"`{series.label}`" for series in case.series)
        lines.append(f"| `{case.id}` | `{case.reference_label}` | {labels} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `linear` and `triangle` use prepared GPS ENU tracks from module CSV flights.",
            "- `dataflash` uses the POS/GPS target trajectory from the IMU dead-reckoning dataset for visual alignment.",
            "- The DataFlash best model is still a sparse 5-second rollout, so its path has fewer points than the full IMU/GPS trajectories.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_template(cases), encoding="utf-8")
    write_report(args.report, cases, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
