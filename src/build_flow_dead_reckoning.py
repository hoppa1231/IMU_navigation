#!/usr/bin/env python3
"""Build calibrated optical-flow dead-reckoning trajectories against real GPS tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_window_dataset import SegmentBounds, source_segment_bounds


DEFAULT_TEST_FLIGHTS = ["triangle_15_01_2025", "linear_15_01_2025"]
FLOW_FEATURES = ["Xflow", "Yflow", "flow_norm"]
FULL_FEATURES = [
    "Xflow",
    "Yflow",
    "flow_norm",
    "Xacc, g",
    "Yacc, g",
    "Zacc, g",
    "acc_norm",
    "Xgyro, DPS",
    "Ygyro, DPS",
    "Zgyro, DPS",
    "gyro_norm",
    "lidar_m",
    "altbar_m",
    "baro_bar",
]
AGGREGATIONS = ["last", "mean", "std", "delta"]


@dataclass
class FlightRecord:
    flight_id: str
    source_file: Path
    segment_index: int


@dataclass
class Track:
    flight_id: str
    time_s: np.ndarray
    source_time_s: np.ndarray
    position_m: np.ndarray


@dataclass
class SensorSeries:
    time_s: np.ndarray
    values: np.ndarray
    feature_names: list[str]


@dataclass
class ModelParams:
    weights: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-index", type=Path, default=Path("derived/datasets/flight_index.csv"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--test-flight", nargs="*", default=DEFAULT_TEST_FLIGHTS)
    parser.add_argument("--lookback-ms", type=float, default=300.0)
    parser.add_argument("--sensor-sample-ms", type=float, default=20.0)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    parser.add_argument(
        "--include-combined-data",
        action="store_true",
        help="Also use segmented flights from combined artifacts/data.csv. Off by default because it contains multiple flights.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("derived/predictions/flow_dead_reckoning/flow_dr.csv"))
    parser.add_argument("--html", type=Path, default=Path("artifacts/generated/navigation/flow_dead_reckoning/index.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/navigation/flow_dead_reckoning.md"))
    parser.add_argument("--max-html-points", type=int, default=4000)
    return parser.parse_args()


def as_float(value: str | None, default: float = math.nan) -> float:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    return result if math.isfinite(result) else default


def read_flight_records(path: Path, include_combined_data: bool) -> dict[str, FlightRecord]:
    records: dict[str, FlightRecord] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["source_format"] != "module":
                continue
            source_file = Path(row["source_file"])
            if not include_combined_data and row["segment_count"] != "1":
                continue
            if not include_combined_data and source_file.name == "data.csv":
                continue
            records[row["flight_id"]] = FlightRecord(
                flight_id=row["flight_id"],
                source_file=source_file,
                segment_index=int(row["segment_index"]) - 1,
            )
    return records


def read_track(path: Path) -> Track:
    times: list[float] = []
    source_times: list[float] = []
    positions: list[tuple[float, float, float]] = []
    flight_id = path.stem.removesuffix("_track")
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            flight_id = row.get("flight_id") or flight_id
            times.append(as_float(row.get("time_s")))
            source_times.append(as_float(row.get("source_time_s")))
            positions.append(
                (
                    as_float(row.get("east_m")),
                    as_float(row.get("north_m")),
                    as_float(row.get("up_m")),
                )
            )
    if len(times) < 2:
        raise ValueError(f"Track has fewer than 2 points: {path}")
    return Track(
        flight_id=flight_id,
        time_s=np.asarray(times, dtype=np.float64),
        source_time_s=np.asarray(source_times, dtype=np.float64),
        position_m=np.asarray(positions, dtype=np.float64),
    )


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def sensor_features(row: dict[str, str]) -> list[float]:
    xflow = as_float(row.get("Xflow"), 0.0)
    yflow = as_float(row.get("Yflow"), 0.0)
    xacc = as_float(row.get("Xacc, g"), 0.0)
    yacc = as_float(row.get("Yacc, g"), 0.0)
    zacc = as_float(row.get("Zacc, g"), 0.0)
    xgyro = as_float(row.get("Xgyro, DPS"), 0.0)
    ygyro = as_float(row.get("Ygyro, DPS"), 0.0)
    zgyro = as_float(row.get("Zgyro, DPS"), 0.0)
    lidar_cm = as_float(row.get("Lidar, sm"), 0.0)
    lidar_m = lidar_cm / 100.0 if 0.0 < lidar_cm < 6000.0 else 0.0
    altbar_m = as_float(row.get("AltBar, m"), 0.0)
    baro_bar = as_float(row.get("Baro, bar"), 0.0)
    return [
        finite_or_zero(xflow),
        finite_or_zero(yflow),
        math.hypot(xflow, yflow),
        finite_or_zero(xacc),
        finite_or_zero(yacc),
        finite_or_zero(zacc),
        math.sqrt(xacc * xacc + yacc * yacc + zacc * zacc),
        finite_or_zero(xgyro),
        finite_or_zero(ygyro),
        finite_or_zero(zgyro),
        math.sqrt(xgyro * xgyro + ygyro * ygyro + zgyro * zgyro),
        finite_or_zero(lidar_m),
        finite_or_zero(altbar_m),
        finite_or_zero(baro_bar),
    ]


def read_sensor_segment(
    source_path: Path,
    bounds: SegmentBounds,
    sensor_sample_ms: float,
) -> SensorSeries:
    times: list[float] = []
    values: list[list[float]] = []
    next_sample_s: float | None = None
    with source_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for row_index, raw_row in enumerate(reader):
            if row_index < bounds.row_start:
                continue
            if row_index > bounds.row_end:
                break
            row = {key.strip(): str(value).strip() for key, value in raw_row.items() if key is not None}
            timestamp_ms = as_float(row.get("TimeStamp"))
            if not math.isfinite(timestamp_ms):
                continue
            time_s = timestamp_ms / 1000.0
            if next_sample_s is None:
                next_sample_s = time_s
            if time_s + 1e-12 < next_sample_s:
                continue
            times.append(time_s)
            values.append(sensor_features(row))
            next_sample_s = time_s + sensor_sample_ms / 1000.0
    if not values:
        raise ValueError(f"No sensor rows for {source_path} rows {bounds.row_start}..{bounds.row_end}")
    return SensorSeries(
        time_s=np.asarray(times, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        feature_names=FULL_FEATURES,
    )


def expanded_names(base_names: list[str]) -> list[str]:
    return [f"{name}_{agg}" for agg in AGGREGATIONS for name in base_names]


def aggregate_window(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values[-1], values.mean(axis=0), values.std(axis=0), values[-1] - values[0]])


def feature_indices_for(feature_names: list[str], selected_base: list[str]) -> list[int]:
    selected: list[int] = []
    for aggregation in AGGREGATIONS:
        for base in selected_base:
            selected.append(feature_names.index(f"{base}_{aggregation}"))
    return selected


def gps_velocity(track: Track) -> np.ndarray:
    velocity = np.zeros_like(track.position_m)
    dt = np.diff(track.time_s)
    dp = np.diff(track.position_m, axis=0)
    step_velocity = dp / np.maximum(dt[:, None], 1e-6)
    velocity[:-1] = step_velocity
    velocity[-1] = step_velocity[-1]
    return velocity


def build_samples(track: Track, sensor: SensorSeries, lookback_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[int] = []
    velocity = gps_velocity(track)
    names = expanded_names(sensor.feature_names)
    _ = names
    for idx, source_time_s in enumerate(track.source_time_s):
        start_s = source_time_s - lookback_s
        start = np.searchsorted(sensor.time_s, start_s, side="left")
        end = np.searchsorted(sensor.time_s, source_time_s, side="right")
        if end <= start or end - start < 2:
            continue
        window = sensor.values[start:end]
        rows.append(aggregate_window(window))
        targets.append(velocity[idx])
        indices.append(idx)
    if not rows:
        return np.empty((0, len(expanded_names(sensor.feature_names)))), np.empty((0, 3)), np.empty((0,), dtype=np.int64)
    return np.vstack(rows), np.vstack(targets), np.asarray(indices, dtype=np.int64)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> ModelParams:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-9] = 1.0
    y_mean = y.mean(axis=0)
    xz = (x - x_mean) / x_std
    yc = y - y_mean
    weights = np.linalg.solve(xz.T @ xz + alpha * np.eye(xz.shape[1]), xz.T @ yc)
    return ModelParams(weights=weights, x_mean=x_mean, x_std=x_std, y_mean=y_mean)


def predict_ridge(x: np.ndarray, params: ModelParams) -> np.ndarray:
    return ((x - params.x_mean) / params.x_std) @ params.weights + params.y_mean


def integrate_velocity(track: Track, sample_indices: np.ndarray, pred_velocity: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if len(sample_indices) == 0:
        return rows
    start_idx = 0
    pred_pos = track.position_m[start_idx].copy()
    prev_time = track.time_s[start_idx]
    prev_velocity = pred_velocity[0]
    start_true = track.position_m[start_idx]
    rows.append(
        {
            "time_s": float(track.time_s[start_idx]),
            "source_time_s": float(track.source_time_s[start_idx]),
            "true_east_m": float(start_true[0]),
            "true_north_m": float(start_true[1]),
            "true_up_m": float(start_true[2]),
            "pred_east_m": float(pred_pos[0]),
            "pred_north_m": float(pred_pos[1]),
            "pred_up_m": float(pred_pos[2]),
            "pred_vel_east_mps": float(prev_velocity[0]),
            "pred_vel_north_mps": float(prev_velocity[1]),
            "pred_vel_up_mps": float(prev_velocity[2]),
            "err_east_m": 0.0,
            "err_north_m": 0.0,
            "err_up_m": 0.0,
            "err_horizontal_m": 0.0,
            "err_3d_m": 0.0,
        }
    )
    for row_idx, track_idx in enumerate(sample_indices):
        if track_idx == start_idx:
            continue
        current_time = track.time_s[track_idx]
        dt = max(0.0, current_time - prev_time)
        velocity = 0.5 * (prev_velocity + pred_velocity[row_idx])
        pred_pos = pred_pos + velocity * dt
        true_pos = track.position_m[track_idx]
        err = pred_pos - true_pos
        rows.append(
            {
                "time_s": float(current_time),
                "source_time_s": float(track.source_time_s[track_idx]),
                "true_east_m": float(true_pos[0]),
                "true_north_m": float(true_pos[1]),
                "true_up_m": float(true_pos[2]),
                "pred_east_m": float(pred_pos[0]),
                "pred_north_m": float(pred_pos[1]),
                "pred_up_m": float(pred_pos[2]),
                "pred_vel_east_mps": float(pred_velocity[row_idx, 0]),
                "pred_vel_north_mps": float(pred_velocity[row_idx, 1]),
                "pred_vel_up_mps": float(pred_velocity[row_idx, 2]),
                "err_east_m": float(err[0]),
                "err_north_m": float(err[1]),
                "err_up_m": float(err[2]),
                "err_horizontal_m": float(math.hypot(err[0], err[1])),
                "err_3d_m": float(np.linalg.norm(err)),
            }
        )
        prev_time = current_time
        prev_velocity = pred_velocity[row_idx]
    return rows


def metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    errors = [row["err_3d_m"] for row in rows]
    horizontal = [row["err_horizontal_m"] for row in rows]
    return {
        "points": float(len(rows)),
        "duration_s": rows[-1]["time_s"] - rows[0]["time_s"],
        "final_error_3d_m": errors[-1],
        "mean_error_3d_m": sum(errors) / len(errors),
        "max_error_3d_m": max(errors),
        "final_error_horizontal_m": horizontal[-1],
        "mean_error_horizontal_m": sum(horizontal) / len(horizontal),
        "max_error_horizontal_m": max(horizontal),
        "final_pred_east_m": rows[-1]["pred_east_m"],
        "final_pred_north_m": rows[-1]["pred_north_m"],
        "final_pred_up_m": rows[-1]["pred_up_m"],
        "final_true_east_m": rows[-1]["true_east_m"],
        "final_true_north_m": rows[-1]["true_north_m"],
        "final_true_up_m": rows[-1]["true_up_m"],
    }


def load_all(records: dict[str, FlightRecord], tracks_dir: Path, max_gap_s: float, max_jump_m: float, sensor_sample_ms: float) -> tuple[dict[str, Track], dict[str, SensorSeries]]:
    tracks: dict[str, Track] = {}
    sensors: dict[str, SensorSeries] = {}
    source_cache: dict[Path, tuple[list[str], list[SegmentBounds]]] = {}
    for flight_id, record in records.items():
        track_path = tracks_dir / f"{flight_id}_track.csv"
        if not track_path.exists():
            continue
        if record.source_file not in source_cache:
            source_cache[record.source_file] = source_segment_bounds(record.source_file, max_gap_s, max_jump_m)
        _, bounds = source_cache[record.source_file]
        if record.segment_index < 0 or record.segment_index >= len(bounds):
            continue
        tracks[flight_id] = read_track(track_path)
        sensors[flight_id] = read_sensor_segment(record.source_file, bounds[record.segment_index], sensor_sample_ms)
    return tracks, sensors


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.9f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def sample_case_rows(rows: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    stride = max(1, math.ceil(len(rows) / max_points))
    return rows[::stride]


def html_template(cases: list[dict[str, object]]) -> str:
    payload = json.dumps(cases, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flow Dead Reckoning</title>
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
      min-width: 320px;
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
    <h1>Flow Dead Reckoning</h1>
    <select id="caseSelect" aria-label="Trajectory case"></select>
  </header>
  <main>
    <svg id="plot" role="img" aria-label="Real GPS and flow dead-reckoning trajectory"></svg>
    <aside>
      <div class="metric"><span>Train flights</span><strong id="train">-</strong></div>
      <div class="metric"><span>Duration</span><strong id="duration">-</strong></div>
      <div class="metric"><span>Final 3D error</span><strong id="final3d">-</strong></div>
      <div class="metric"><span>Final horizontal error</span><strong id="finalh">-</strong></div>
      <div class="metric"><span>Mean 3D error</span><strong id="mean3d">-</strong></div>
      <div class="metric"><span>Max 3D error</span><strong id="max3d">-</strong></div>
      <div class="metric"><span>Final predicted ENU</span><strong id="pred">-</strong></div>
      <div class="metric"><span>Final real GPS ENU</span><strong id="true">-</strong></div>
      <div class="legend">
        <div class="key"><span class="swatch" style="background:#2563eb"></span><span>real GPS trajectory</span></div>
        <div class="key"><span class="swatch" style="background:#dc2626"></span><span>flow/IMU integrated path</span></div>
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
    function fmt(value, digits = 1) {{
      return Number(value || 0).toFixed(digits);
    }}
    function render() {{
      const item = cases.find((candidate) => candidate.id === select.value) || cases[0];
      if (!item) return;
      const rows = item.rows;
      const xs = rows.flatMap((p) => [p.true_east_m, p.pred_east_m]);
      const ys = rows.flatMap((p) => [p.true_north_m, p.pred_north_m]);
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
        <path d="${{pathFor(rows, 'true_east_m', 'true_north_m', scale)}}" fill="none" stroke="#2563eb" stroke-width="2.4"/>
        <path d="${{pathFor(rows, 'pred_east_m', 'pred_north_m', scale)}}" fill="none" stroke="#dc2626" stroke-width="2.4"/>
      `;
      const m = item.metrics;
      document.getElementById('train').textContent = String(item.trainCount);
      document.getElementById('duration').textContent = `${{fmt(m.duration_s)}} s`;
      document.getElementById('final3d').textContent = `${{fmt(m.final_error_3d_m)}} m`;
      document.getElementById('finalh').textContent = `${{fmt(m.final_error_horizontal_m)}} m`;
      document.getElementById('mean3d').textContent = `${{fmt(m.mean_error_3d_m)}} m`;
      document.getElementById('max3d').textContent = `${{fmt(m.max_error_3d_m)}} m`;
      document.getElementById('pred').textContent = `${{fmt(m.final_pred_east_m)}}, ${{fmt(m.final_pred_north_m)}}, ${{fmt(m.final_pred_up_m)}}`;
      document.getElementById('true').textContent = `${{fmt(m.final_true_east_m)}}, ${{fmt(m.final_true_north_m)}}, ${{fmt(m.final_true_up_m)}}`;
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
    out_csv: Path,
    html: Path,
    summary_rows: list[dict[str, object]],
    lookback_ms: float,
    sensor_sample_ms: float,
    alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Flow dead reckoning",
        "",
        "This report is generated by `src/build_flow_dead_reckoning.py`.",
        "",
        "Method: train a ridge mapping from module sensor windows to local GPS velocity on separate source CSV flights except the test flight, then integrate predicted velocity from the first GPS point of the test flight. The real GPS trajectory is used only as the plotted reference and error target for the test flight.",
        "",
        f"- lookback window: `{lookback_ms:g}` ms",
        f"- sensor sample: `{sensor_sample_ms:g}` ms",
        f"- ridge alpha: `{alpha:g}`",
        "",
        "## Outputs",
        "",
        f"- trajectory CSV: `{out_csv}`",
        f"- HTML overlay: `{html}`",
        "",
        "## Metrics",
        "",
        "| test flight | model | train flights | train samples | duration s | final 3D m | final horizontal m | mean 3D m | max 3D m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        values = row["metrics"]
        assert isinstance(values, dict)
        lines.append(
            f"| `{row['flight_id']}` | `{row['model']}` | {row['train_flights']} | {row['train_samples']} | "
            f"{float(values['duration_s']):.1f} | {float(values['final_error_3d_m']):.3f} | "
            f"{float(values['final_error_horizontal_m']):.3f} | {float(values['mean_error_3d_m']):.3f} | "
            f"{float(values['max_error_3d_m']):.3f} |"
        )
    lines.extend(
        [
            "",
        "## Notes",
        "",
            "- By default, combined `artifacts/data.csv` segments are excluded. Use `--include-combined-data` only when you intentionally want those segmented flights.",
        "- `flow_only` uses only `Xflow/Yflow/flow_norm` window aggregates.",
        "- `flow_imu_alt` adds accelerometer, gyro, lidar and barometer aggregates.",
        "- This is still not a full EKF: it is a calibrated velocity constraint and open-loop integration on the test flight.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = read_flight_records(args.flight_index, args.include_combined_data)
    tracks, sensors = load_all(records, args.tracks_dir, args.max_gap_s, args.max_jump_m, args.sensor_sample_ms)
    available = sorted(set(records) & set(tracks) & set(sensors))
    test_flights = [flight for flight in args.test_flight if flight in available]
    if not test_flights:
        raise ValueError(f"No requested test flights are available. Available: {available}")

    lookback_s = args.lookback_ms / 1000.0
    feature_names = expanded_names(FULL_FEATURES)
    model_defs = {
        "flow_only": feature_indices_for(feature_names, FLOW_FEATURES),
        "flow_imu_alt": feature_indices_for(feature_names, FULL_FEATURES),
    }

    output_rows: list[dict[str, str | float]] = []
    html_cases: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    samples_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for flight_id in available:
        samples_cache[flight_id] = build_samples(tracks[flight_id], sensors[flight_id], lookback_s)

    for test_flight in test_flights:
        train_flights = [flight for flight in available if flight != test_flight]
        if not train_flights:
            raise ValueError(f"No train flights remain for {test_flight}")
        x_train_all = np.vstack([samples_cache[flight][0] for flight in train_flights if len(samples_cache[flight][0])])
        y_train = np.vstack([samples_cache[flight][1] for flight in train_flights if len(samples_cache[flight][1])])
        x_test_all, _, test_indices = samples_cache[test_flight]
        if len(x_test_all) == 0:
            raise ValueError(f"No test samples for {test_flight}")

        for model_name, selected_indices in model_defs.items():
            params = fit_ridge(x_train_all[:, selected_indices], y_train, args.ridge_alpha)
            pred_velocity = predict_ridge(x_test_all[:, selected_indices], params)
            rows = integrate_velocity(tracks[test_flight], test_indices, pred_velocity)
            values = metrics(rows)
            case_id = f"{test_flight}__{model_name}"
            for row in rows:
                output_rows.append({"case_id": case_id, "flight_id": test_flight, "model": model_name, **row})
            html_cases.append(
                {
                    "id": case_id,
                    "label": f"{test_flight} / {model_name}",
                    "trainCount": len(train_flights),
                    "metrics": values,
                    "rows": sample_case_rows(rows, args.max_html_points),
                }
            )
            summary_rows.append(
                {
                    "flight_id": test_flight,
                    "model": model_name,
                    "train_flights": len(train_flights),
                    "train_samples": len(y_train),
                    "metrics": values,
                }
            )

    write_csv(args.out_csv, output_rows)
    write_html(args.html, html_cases)
    write_report(args.report, args.out_csv, args.html, summary_rows, args.lookback_ms, args.sensor_sample_ms, args.ridge_alpha)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.html}")
    print(f"Wrote {args.report}")
    for row in summary_rows:
        values = row["metrics"]
        assert isinstance(values, dict)
        print(f"{row['flight_id']} {row['model']}: final 3D error {float(values['final_error_3d_m']):.3f} m")


if __name__ == "__main__":
    main()
