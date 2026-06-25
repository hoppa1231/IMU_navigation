#!/usr/bin/env python3
"""Evaluate POLI_NA LSTM as an open-loop displacement rollout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if Path("/tmp/poli_deps").exists():
    sys.path.insert(0, "/tmp/poli_deps")

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
    raise SystemExit(
        "onnxruntime is required for POLI_NA. Install it with: "
        "python3 -m pip install --target /tmp/poli_deps onnxruntime"
    ) from exc

from build_window_dataset import SegmentBounds, source_segment_bounds


DEFAULT_TEST_FLIGHTS = ["triangle_15_01_2025", "linear_15_01_2025"]

FEATURE_PRESETS = {
    "csv_first10_raw": [
        "Lidar, sm",
        "Baro, bar",
        "AltBar, m",
        "Xflow",
        "Yflow",
        "Xmag1, mG",
        "Ymag1, mG",
        "Zmag1, mG",
        "Xacc, g",
        "Yacc, g",
    ],
    "module_motion10_raw": [
        "Lidar, sm",
        "AltBar, m",
        "Xflow",
        "Yflow",
        "Xacc, g",
        "Yacc, g",
        "Zacc, g",
        "Xgyro, DPS",
        "Ygyro, DPS",
        "Zgyro, DPS",
    ],
    "imu_flow_mag10_raw": [
        "Xflow",
        "Yflow",
        "Xacc, g",
        "Yacc, g",
        "Zacc, g",
        "Xgyro, DPS",
        "Ygyro, DPS",
        "Zgyro, DPS",
        "Xmag1, mG",
        "Ymag1, mG",
    ],
}


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
    rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("artifacts/POLI_NA.zip"))
    parser.add_argument("--flight-index", type=Path, default=Path("derived/datasets/flight_index.csv"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--test-flight", nargs="*", default=DEFAULT_TEST_FLIGHTS)
    parser.add_argument("--feature-preset", nargs="*", default=list(FEATURE_PRESETS))
    parser.add_argument("--sequence-len", type=int, default=100)
    parser.add_argument("--sensor-sample-ms", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    parser.add_argument(
        "--include-combined-data",
        action="store_true",
        help="Also use segmented flights from combined artifacts/data.csv. Off by default.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("derived/predictions/poli_na_rollout/poli_na_rollout.csv"))
    parser.add_argument("--html", type=Path, default=Path("artifacts/generated/navigation/poli_na_rollout/index.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/poli_na_rollout.md"))
    parser.add_argument("--max-html-points", type=int, default=3000)
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
            if not include_combined_data and (row["segment_count"] != "1" or source_file.name == "data.csv"):
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
    return Track(
        flight_id=flight_id,
        time_s=np.asarray(times, dtype=np.float64),
        source_time_s=np.asarray(source_times, dtype=np.float64),
        position_m=np.asarray(positions, dtype=np.float64),
    )


def read_sensor_segment(source_path: Path, bounds: SegmentBounds, sensor_sample_ms: float) -> SensorSeries:
    times: list[float] = []
    rows: list[dict[str, str]] = []
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
            rows.append(row)
            next_sample_s = time_s + sensor_sample_ms / 1000.0
    if len(rows) < 2:
        raise ValueError(f"No sensor rows for {source_path} rows {bounds.row_start}..{bounds.row_end}")
    return SensorSeries(time_s=np.asarray(times, dtype=np.float64), rows=rows)


def load_sensors(
    records: dict[str, FlightRecord],
    tracks_dir: Path,
    max_gap_s: float,
    max_jump_m: float,
    sensor_sample_ms: float,
) -> tuple[dict[str, Track], dict[str, SensorSeries]]:
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


def interp_position(track: Track, source_time_s: float) -> tuple[float, np.ndarray]:
    idx = np.searchsorted(track.source_time_s, source_time_s, side="left")
    if idx <= 0:
        return float(track.time_s[0]), track.position_m[0].copy()
    if idx >= len(track.source_time_s):
        return float(track.time_s[-1]), track.position_m[-1].copy()
    t0 = track.source_time_s[idx - 1]
    t1 = track.source_time_s[idx]
    ratio = 0.0 if t1 <= t0 else (source_time_s - t0) / (t1 - t0)
    time_s = track.time_s[idx - 1] + (track.time_s[idx] - track.time_s[idx - 1]) * ratio
    pos = track.position_m[idx - 1] + (track.position_m[idx] - track.position_m[idx - 1]) * ratio
    return float(time_s), pos


def build_sequences(sensor: SensorSeries, columns: list[str], sequence_len: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(sensor.rows) // sequence_len
    if count <= 0:
        return np.empty((0, sequence_len, len(columns)), dtype=np.float32), np.empty((0,), dtype=np.float64)
    x = np.empty((count, sequence_len, len(columns)), dtype=np.float32)
    end_times = np.empty((count,), dtype=np.float64)
    for seq_idx in range(count):
        start = seq_idx * sequence_len
        end = start + sequence_len
        for t_idx, row in enumerate(sensor.rows[start:end]):
            for col_idx, column in enumerate(columns):
                x[seq_idx, t_idx, col_idx] = as_float(row.get(column), 0.0)
        end_times[seq_idx] = sensor.time_s[end - 1]
    return x, end_times


def load_onnx_session(zip_path: Path) -> tuple[tempfile.TemporaryDirectory[str], ort.InferenceSession]:
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("POLI_NA/ONNX/POLI_NA.onnx", temp_dir.name)
    model_path = Path(temp_dir.name) / "POLI_NA/ONNX/POLI_NA.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return temp_dir, session


def predict(session: ort.InferenceSession, x: np.ndarray, batch_size: int) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        # MATLAB-exported ONNX uses sequence-major layout: T x B x C.
        model_input = np.transpose(batch, (1, 0, 2)).astype(np.float32, copy=False)
        outputs.append(session.run([output_name], {input_name: model_input})[0])
    return np.vstack(outputs).astype(np.float64)


def rollout(
    flight_id: str,
    preset: str,
    track: Track,
    sensor: SensorSeries,
    columns: list[str],
    session: ort.InferenceSession,
    sequence_len: int,
    batch_size: int,
) -> list[dict[str, float | str]]:
    x, end_source_times = build_sequences(sensor, columns, sequence_len)
    if len(x) == 0:
        return []
    pred_delta = predict(session, x, batch_size)
    pred_pos = track.position_m[0].astype(np.float64).copy()
    rows: list[dict[str, float | str]] = [
        {
            "case_id": f"{flight_id}__{preset}",
            "flight_id": flight_id,
            "feature_preset": preset,
            "step": 0.0,
            "time_s": float(track.time_s[0]),
            "source_time_s": float(track.source_time_s[0]),
            "true_east_m": float(track.position_m[0, 0]),
            "true_north_m": float(track.position_m[0, 1]),
            "true_up_m": float(track.position_m[0, 2]),
            "pred_east_m": float(pred_pos[0]),
            "pred_north_m": float(pred_pos[1]),
            "pred_up_m": float(pred_pos[2]),
            "pred_dx_east_m": 0.0,
            "pred_dy_north_m": 0.0,
            "pred_dz_up_m": 0.0,
            "err_east_m": 0.0,
            "err_north_m": 0.0,
            "err_up_m": 0.0,
            "err_horizontal_m": 0.0,
            "err_3d_m": 0.0,
        }
    ]
    for idx, (source_time_s, delta) in enumerate(zip(end_source_times, pred_delta), start=1):
        pred_pos += delta
        time_s, true_pos = interp_position(track, float(source_time_s))
        err = pred_pos - true_pos
        rows.append(
            {
                "case_id": f"{flight_id}__{preset}",
                "flight_id": flight_id,
                "feature_preset": preset,
                "step": float(idx),
                "time_s": time_s,
                "source_time_s": float(source_time_s),
                "true_east_m": float(true_pos[0]),
                "true_north_m": float(true_pos[1]),
                "true_up_m": float(true_pos[2]),
                "pred_east_m": float(pred_pos[0]),
                "pred_north_m": float(pred_pos[1]),
                "pred_up_m": float(pred_pos[2]),
                "pred_dx_east_m": float(delta[0]),
                "pred_dy_north_m": float(delta[1]),
                "pred_dz_up_m": float(delta[2]),
                "err_east_m": float(err[0]),
                "err_north_m": float(err[1]),
                "err_up_m": float(err[2]),
                "err_horizontal_m": float(math.hypot(err[0], err[1])),
                "err_3d_m": float(np.linalg.norm(err)),
            }
        )
    return rows


def metrics(rows: list[dict[str, float | str]]) -> dict[str, float]:
    errors = [float(row["err_3d_m"]) for row in rows]
    horizontal = [float(row["err_horizontal_m"]) for row in rows]
    last = rows[-1]
    first = rows[0]
    return {
        "steps": float(len(rows) - 1),
        "duration_s": float(last["time_s"]) - float(first["time_s"]),
        "final_error_3d_m": errors[-1],
        "mean_error_3d_m": sum(errors) / len(errors),
        "max_error_3d_m": max(errors),
        "final_error_horizontal_m": horizontal[-1],
        "final_pred_east_m": float(last["pred_east_m"]),
        "final_pred_north_m": float(last["pred_north_m"]),
        "final_pred_up_m": float(last["pred_up_m"]),
        "final_true_east_m": float(last["true_east_m"]),
        "final_true_north_m": float(last["true_north_m"]),
        "final_true_up_m": float(last["true_up_m"]),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def sample_rows(rows: list[dict[str, float | str]], max_points: int) -> list[dict[str, float | str]]:
    stride = max(1, math.ceil(len(rows) / max_points))
    return rows[::stride]


def html_template(cases: list[dict[str, object]]) -> str:
    payload = json.dumps(cases, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>POLI_NA Rollout</title>
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
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
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
    svg {{ width: 100%; height: calc(100vh - 62px); background: #eef1f5; }}
    aside {{ padding: 16px; border-left: 1px solid #d8dde5; background: #ffffff; overflow: auto; }}
    .metric {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid #edf0f4;
      font-size: 14px;
    }}
    .source {{ margin: 4px 0 14px; color: #526071; font-size: 13px; line-height: 1.35; }}
    .legend {{ display: grid; gap: 8px; margin-top: 16px; font-size: 14px; }}
    .key {{ display: flex; gap: 8px; align-items: center; }}
    .swatch {{ width: 22px; height: 4px; border-radius: 2px; }}
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
    <h1>POLI_NA Rollout</h1>
    <select id="caseSelect" aria-label="Trajectory case"></select>
  </header>
  <main>
    <svg id="plot" role="img" aria-label="Real GPS and POLI_NA predicted trajectory"></svg>
    <aside>
      <div class="source" id="features"></div>
      <div class="metric"><span>Steps</span><strong id="steps">-</strong></div>
      <div class="metric"><span>Duration</span><strong id="duration">-</strong></div>
      <div class="metric"><span>Final 3D error</span><strong id="final3d">-</strong></div>
      <div class="metric"><span>Final horizontal error</span><strong id="finalh">-</strong></div>
      <div class="metric"><span>Mean 3D error</span><strong id="mean3d">-</strong></div>
      <div class="metric"><span>Max 3D error</span><strong id="max3d">-</strong></div>
      <div class="metric"><span>Final predicted ENU</span><strong id="pred">-</strong></div>
      <div class="metric"><span>Final real GPS ENU</span><strong id="true">-</strong></div>
      <div class="legend">
        <div class="key"><span class="swatch" style="background:#2563eb"></span><span>real GPS trajectory</span></div>
        <div class="key"><span class="swatch" style="background:#dc2626"></span><span>POLI_NA accumulated path</span></div>
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
      return points.map((p, i) => `${{i ? 'L' : 'M'}} ${{scale.x(Number(p[xKey])).toFixed(2)}} ${{scale.y(Number(p[yKey])).toFixed(2)}}`).join(' ');
    }}
    function fmt(value, digits = 1) {{ return Number(value || 0).toFixed(digits); }}
    function render() {{
      const item = cases.find((candidate) => candidate.id === select.value) || cases[0];
      if (!item) return;
      const rows = item.rows;
      const xs = rows.flatMap((p) => [Number(p.true_east_m), Number(p.pred_east_m)]);
      const ys = rows.flatMap((p) => [Number(p.true_north_m), Number(p.pred_north_m)]);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const rect = svg.getBoundingClientRect();
      const w = Math.max(rect.width, 320), h = Math.max(rect.height, 320);
      const pad = 34;
      const spanX = Math.max(maxX - minX, 1), spanY = Math.max(maxY - minY, 1);
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
      document.getElementById('features').textContent = item.features.join(', ');
      document.getElementById('steps').textContent = fmt(m.steps, 0);
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
    zip_path: Path,
    out_csv: Path,
    html: Path,
    summaries: list[dict[str, object]],
    sequence_len: int,
    sensor_sample_ms: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# POLI_NA rollout",
        "",
        "This report is generated by `src/run_poli_na_rollout.py`.",
        "",
        f"Model archive: `{zip_path}`",
        "",
        "POLI_NA exposes only the input shape `(sequence, batch, 10)` and output `dx, dy, dz`; the archive does not document the semantic order or normalization of the 10 input channels. Therefore this report tests several plausible raw feature presets from the module CSV files.",
        "",
        f"- sequence length: `{sequence_len}` sampled rows",
        f"- sensor sample step: `{sensor_sample_ms:g}` ms",
        "",
        "GPS is used only as the real trajectory/reference. The predicted trajectory starts at the first GPS point and accumulates POLI_NA outputs.",
        "",
        "## Outputs",
        "",
        f"- rollout CSV: `{out_csv}`",
        f"- HTML overlay: `{html}`",
        "",
        "## Metrics",
        "",
        "| flight | feature preset | steps | duration s | final 3D m | final horizontal m | mean 3D m | max 3D m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = row["metrics"]
        assert isinstance(values, dict)
        lines.append(
            f"| `{row['flight_id']}` | `{row['preset']}` | {float(values['steps']):.0f} | "
            f"{float(values['duration_s']):.1f} | {float(values['final_error_3d_m']):.3f} | "
            f"{float(values['final_error_horizontal_m']):.3f} | {float(values['mean_error_3d_m']):.3f} | "
            f"{float(values['max_error_3d_m']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feature presets",
            "",
        ]
    )
    for name, columns in FEATURE_PRESETS.items():
        lines.append(f"- `{name}`: `{', '.join(columns)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If errors are extremely large, the most likely cause is feature-order or normalization mismatch with the original MATLAB training data.",
            "- A conclusive POLI_NA evaluation needs the original preprocessing pipeline or the exact list and scaling of the 10 channels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.feature_preset if name not in FEATURE_PRESETS]
    if unknown:
        raise ValueError(f"Unknown feature presets: {unknown}. Available: {sorted(FEATURE_PRESETS)}")

    records = read_flight_records(args.flight_index, args.include_combined_data)
    tracks, sensors = load_sensors(records, args.tracks_dir, args.max_gap_s, args.max_jump_m, args.sensor_sample_ms)
    test_flights = [flight for flight in args.test_flight if flight in tracks and flight in sensors]
    if not test_flights:
        raise ValueError(f"No requested test flights are available. Available: {sorted(tracks)}")

    temp_dir, session = load_onnx_session(args.zip)
    try:
        all_rows: list[dict[str, float | str]] = []
        html_cases: list[dict[str, object]] = []
        summaries: list[dict[str, object]] = []
        for flight_id in test_flights:
            for preset in args.feature_preset:
                columns = FEATURE_PRESETS[preset]
                rows = rollout(
                    flight_id,
                    preset,
                    tracks[flight_id],
                    sensors[flight_id],
                    columns,
                    session,
                    args.sequence_len,
                    args.batch_size,
                )
                if not rows:
                    continue
                all_rows.extend(rows)
                values = metrics(rows)
                case_id = f"{flight_id}__{preset}"
                html_cases.append(
                    {
                        "id": case_id,
                        "label": f"{flight_id} / {preset}",
                        "features": columns,
                        "metrics": values,
                        "rows": sample_rows(rows, args.max_html_points),
                    }
                )
                summaries.append(
                    {
                        "flight_id": flight_id,
                        "preset": preset,
                        "metrics": values,
                    }
                )
                print(f"{flight_id} {preset}: final 3D error {values['final_error_3d_m']:.3f} m")
        write_csv(args.out_csv, all_rows)
        write_html(args.html, html_cases)
        write_report(args.report, args.zip, args.out_csv, args.html, summaries, args.sequence_len, args.sensor_sample_ms)
    finally:
        temp_dir.cleanup()

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.html}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
