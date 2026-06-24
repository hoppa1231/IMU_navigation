#!/usr/bin/env python3
"""Build a pure IMU dead-reckoning trajectory and compare it with POS/GPS."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path


GRAVITY_MPS2 = 9.80665


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--out-csv", type=Path, default=Path("derived/predictions/imu_dead_reckoning/dataflash_imu_dr.csv"))
    parser.add_argument("--html", type=Path, default=Path("artifacts/generated/imu_dead_reckoning/index.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/imu_dead_reckoning.md"))
    parser.add_argument(
        "--calibration-s",
        type=float,
        default=5.0,
        help="Initial interval used to estimate residual acceleration bias after gravity compensation.",
    )
    parser.add_argument(
        "--max-dt-s",
        type=float,
        default=0.2,
        help="Skip integration across larger IMU timestamp gaps.",
    )
    parser.add_argument("--max-html-points", type=int, default=4000)
    return parser.parse_args()


def as_float(value: str | None, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    return result if math.isfinite(result) else default


def read_numeric_rows(path: Path, columns: list[str]) -> dict[str, list[float]]:
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed = [as_float(row.get(column)) for column in columns]
            if any(not math.isfinite(value) for value in parsed):
                continue
            for column, value in zip(columns, parsed):
                values[column].append(value)
    if not values[columns[0]]:
        raise ValueError(f"No valid rows in {path}")
    return values


def gps_to_local_m(
    lat: float,
    lon: float,
    alt: float,
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    lat0, lon0, alt0 = origin
    earth_radius_m = 6_371_000.0
    north = math.radians(lat - lat0) * earth_radius_m
    east = math.radians(lon - lon0) * earth_radius_m * math.cos(math.radians(lat0))
    up = alt - alt0
    return east, north, up


def read_pos(path: Path) -> dict[str, list[float]]:
    raw = read_numeric_rows(path, ["TimeUS", "Lat", "Lng", "Alt"])
    origin = (raw["Lat"][0], raw["Lng"][0], raw["Alt"][0])
    east: list[float] = []
    north: list[float] = []
    up: list[float] = []
    for lat, lon, alt in zip(raw["Lat"], raw["Lng"], raw["Alt"]):
        e, n, u = gps_to_local_m(lat, lon, alt, origin)
        east.append(e)
        north.append(n)
        up.append(u)
    return {
        "time_s": [value / 1_000_000.0 for value in raw["TimeUS"]],
        "east_m": east,
        "north_m": north,
        "up_m": up,
    }


def interp_scalar(times: list[float], values: list[float], time_s: float) -> float:
    if time_s <= times[0]:
        return values[0]
    if time_s >= times[-1]:
        return values[-1]
    idx = bisect_left(times, time_s)
    t0 = times[idx - 1]
    t1 = times[idx]
    if t1 <= t0:
        return values[idx - 1]
    ratio = (time_s - t0) / (t1 - t0)
    return values[idx - 1] + (values[idx] - values[idx - 1]) * ratio


def interp_angle_deg(times: list[float], values: list[float], time_s: float) -> float:
    if time_s <= times[0]:
        return values[0]
    if time_s >= times[-1]:
        return values[-1]
    idx = bisect_left(times, time_s)
    t0 = times[idx - 1]
    t1 = times[idx]
    if t1 <= t0:
        return values[idx - 1]
    a0 = values[idx - 1]
    a1 = values[idx]
    delta = (a1 - a0 + 180.0) % 360.0 - 180.0
    ratio = (time_s - t0) / (t1 - t0)
    return (a0 + delta * ratio) % 360.0


def body_to_ned(
    acc_body: tuple[float, float, float],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[float, float, float]:
    """Rotate body FRD acceleration to NED using ArduPilot roll/pitch/yaw angles."""

    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    ax, ay, az = acc_body

    # R_ned_body = Rz(yaw) * Ry(pitch) * Rx(roll)
    r00 = cp * cy
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = cp * sy
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    return (
        r00 * ax + r01 * ay + r02 * az,
        r10 * ax + r11 * ay + r12 * az,
        r20 * ax + r21 * ay + r22 * az,
    )


def imu_acc_to_enu(
    acc_body: tuple[float, float, float],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[float, float, float]:
    ax_n, ay_e, az_d = body_to_ned(acc_body, roll_deg, pitch_deg, yaw_deg)
    # IMU measures specific force. At rest it is about -g on body/down axis;
    # adding gravity in NED gives translational acceleration.
    az_d += GRAVITY_MPS2
    return ay_e, ax_n, -az_d


def read_imu_att(data_dir: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    imu = read_numeric_rows(data_dir / "IMU.csv", ["TimeUS", "AccX", "AccY", "AccZ"])
    att = read_numeric_rows(data_dir / "ATT.csv", ["TimeUS", "Roll", "Pitch", "Yaw"])
    imu["time_s"] = [value / 1_000_000.0 for value in imu["TimeUS"]]
    att["time_s"] = [value / 1_000_000.0 for value in att["TimeUS"]]
    return imu, att


def build_accel_series(
    imu: dict[str, list[float]],
    att: dict[str, list[float]],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for time_s, ax, ay, az in zip(imu["time_s"], imu["AccX"], imu["AccY"], imu["AccZ"]):
        roll = interp_scalar(att["time_s"], att["Roll"], time_s)
        pitch = interp_scalar(att["time_s"], att["Pitch"], time_s)
        yaw = interp_angle_deg(att["time_s"], att["Yaw"], time_s)
        ae, an, au = imu_acc_to_enu((ax, ay, az), roll, pitch, yaw)
        rows.append(
            {
                "time_s": time_s,
                "acc_east_mps2": ae,
                "acc_north_mps2": an,
                "acc_up_mps2": au,
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
            }
        )
    return rows


def estimate_bias(rows: list[dict[str, float]], calibration_s: float) -> tuple[float, float, float]:
    if calibration_s <= 0.0:
        return 0.0, 0.0, 0.0
    start = rows[0]["time_s"]
    selected = [row for row in rows if row["time_s"] - start <= calibration_s]
    if not selected:
        return 0.0, 0.0, 0.0
    return (
        sum(row["acc_east_mps2"] for row in selected) / len(selected),
        sum(row["acc_north_mps2"] for row in selected) / len(selected),
        sum(row["acc_up_mps2"] for row in selected) / len(selected),
    )


def integrate(
    accel_rows: list[dict[str, float]],
    pos: dict[str, list[float]],
    calibration_s: float,
    max_dt_s: float,
) -> tuple[list[dict[str, float]], tuple[float, float, float]]:
    pos_start_t = pos["time_s"][0]
    pos_end_t = pos["time_s"][-1]
    rows = [row for row in accel_rows if pos_start_t <= row["time_s"] <= pos_end_t]
    if len(rows) < 2:
        raise ValueError("Not enough IMU rows inside POS time span")
    bias_e, bias_n, bias_u = estimate_bias(rows, calibration_s)

    pred_e = interp_scalar(pos["time_s"], pos["east_m"], rows[0]["time_s"])
    pred_n = interp_scalar(pos["time_s"], pos["north_m"], rows[0]["time_s"])
    pred_u = interp_scalar(pos["time_s"], pos["up_m"], rows[0]["time_s"])
    vel_e = 0.0
    vel_n = 0.0
    vel_u = 0.0

    out: list[dict[str, float]] = []
    prev = rows[0]
    for idx, row in enumerate(rows):
        dt = 0.0 if idx == 0 else row["time_s"] - prev["time_s"]
        if idx > 0 and 0.0 < dt <= max_dt_s:
            ae0 = prev["acc_east_mps2"] - bias_e
            an0 = prev["acc_north_mps2"] - bias_n
            au0 = prev["acc_up_mps2"] - bias_u
            ae1 = row["acc_east_mps2"] - bias_e
            an1 = row["acc_north_mps2"] - bias_n
            au1 = row["acc_up_mps2"] - bias_u
            ae = 0.5 * (ae0 + ae1)
            an = 0.5 * (an0 + an1)
            au = 0.5 * (au0 + au1)
            pred_e += vel_e * dt + 0.5 * ae * dt * dt
            pred_n += vel_n * dt + 0.5 * an * dt * dt
            pred_u += vel_u * dt + 0.5 * au * dt * dt
            vel_e += ae * dt
            vel_n += an * dt
            vel_u += au * dt
        elif idx > 0:
            vel_e = vel_n = vel_u = 0.0

        true_e = interp_scalar(pos["time_s"], pos["east_m"], row["time_s"])
        true_n = interp_scalar(pos["time_s"], pos["north_m"], row["time_s"])
        true_u = interp_scalar(pos["time_s"], pos["up_m"], row["time_s"])
        err_e = pred_e - true_e
        err_n = pred_n - true_n
        err_u = pred_u - true_u
        out.append(
            {
                "time_s": row["time_s"] - rows[0]["time_s"],
                "source_time_s": row["time_s"],
                "true_east_m": true_e,
                "true_north_m": true_n,
                "true_up_m": true_u,
                "pred_east_m": pred_e,
                "pred_north_m": pred_n,
                "pred_up_m": pred_u,
                "vel_east_mps": vel_e,
                "vel_north_mps": vel_n,
                "vel_up_mps": vel_u,
                "acc_east_mps2": row["acc_east_mps2"] - bias_e,
                "acc_north_mps2": row["acc_north_mps2"] - bias_n,
                "acc_up_mps2": row["acc_up_mps2"] - bias_u,
                "err_east_m": err_e,
                "err_north_m": err_n,
                "err_up_m": err_u,
                "err_horizontal_m": math.hypot(err_e, err_n),
                "err_3d_m": math.sqrt(err_e * err_e + err_n * err_n + err_u * err_u),
            }
        )
        prev = row
    return out, (bias_e, bias_n, bias_u)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.9f}" for key, value in row.items()})


def sample_rows(rows: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    stride = max(1, math.ceil(len(rows) / max_points))
    return rows[::stride]


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


def html_template(rows: list[dict[str, float]], values: dict[str, float], bias: tuple[float, float, float]) -> str:
    payload = json.dumps({"rows": rows, "metrics": values, "bias": bias}, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IMU Dead Reckoning</title>
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
      main {{ grid-template-columns: 1fr; }}
      svg {{ height: 66vh; }}
      aside {{ border-left: 0; border-top: 1px solid #d8dde5; }}
    }}
  </style>
</head>
<body>
  <header><h1>IMU Dead Reckoning</h1></header>
  <main>
    <svg id="plot" role="img" aria-label="GPS and IMU dead-reckoning trajectory"></svg>
    <aside>
      <div class="metric"><span>Duration</span><strong id="duration">-</strong></div>
      <div class="metric"><span>Final 3D error</span><strong id="final3d">-</strong></div>
      <div class="metric"><span>Final horizontal error</span><strong id="finalh">-</strong></div>
      <div class="metric"><span>Mean 3D error</span><strong id="mean3d">-</strong></div>
      <div class="metric"><span>Max 3D error</span><strong id="max3d">-</strong></div>
      <div class="metric"><span>Final predicted ENU</span><strong id="pred">-</strong></div>
      <div class="metric"><span>Final GPS ENU</span><strong id="true">-</strong></div>
      <div class="legend">
        <div class="key"><span class="swatch" style="background:#2563eb"></span><span>GPS/POS reference</span></div>
        <div class="key"><span class="swatch" style="background:#dc2626"></span><span>IMU integrated path</span></div>
      </div>
    </aside>
  </main>
  <script>
    const payload = {payload};
    const rows = payload.rows;
    const svg = document.getElementById('plot');
    function pathFor(points, xKey, yKey, scale) {{
      return points.map((p, i) => `${{i ? 'L' : 'M'}} ${{scale.x(p[xKey]).toFixed(2)}} ${{scale.y(p[yKey]).toFixed(2)}}`).join(' ');
    }}
    function fmt(value, digits = 1) {{
      return Number(value || 0).toFixed(digits);
    }}
    function render() {{
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
      const m = payload.metrics;
      document.getElementById('duration').textContent = `${{fmt(m.duration_s, 1)}} s`;
      document.getElementById('final3d').textContent = `${{fmt(m.final_error_3d_m, 1)}} m`;
      document.getElementById('finalh').textContent = `${{fmt(m.final_error_horizontal_m, 1)}} m`;
      document.getElementById('mean3d').textContent = `${{fmt(m.mean_error_3d_m, 1)}} m`;
      document.getElementById('max3d').textContent = `${{fmt(m.max_error_3d_m, 1)}} m`;
      document.getElementById('pred').textContent = `${{fmt(m.final_pred_east_m)}}, ${{fmt(m.final_pred_north_m)}}, ${{fmt(m.final_pred_up_m)}}`;
      document.getElementById('true').textContent = `${{fmt(m.final_true_east_m)}}, ${{fmt(m.final_true_north_m)}}, ${{fmt(m.final_true_up_m)}}`;
    }}
    window.addEventListener('resize', render);
    render();
  </script>
</body>
</html>
"""


def write_html(path: Path, rows: list[dict[str, float]], values: dict[str, float], bias: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_template(rows, values, bias), encoding="utf-8")


def write_report(
    path: Path,
    data_dir: Path,
    out_csv: Path,
    html: Path,
    values: dict[str, float],
    bias: tuple[float, float, float],
    calibration_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IMU dead reckoning",
        "",
        "This report is generated by `src/build_imu_dead_reckoning.py`.",
        "",
        "Method: start at the first POS/GPS reference point, rotate IMU accelerations from body frame to ENU using ATT roll/pitch/yaw, subtract gravity, estimate a constant residual acceleration bias from the initial calibration interval, then double-integrate acceleration. GPS/POS is used only for the initial point and for error measurement.",
        "",
        "This is pure dead reckoning. No GPS updates, Kalman correction, optical-flow velocity, or zero-velocity updates are applied during the trajectory.",
        "",
        "## Inputs",
        "",
        f"- data dir: `{data_dir}`",
        f"- calibration interval: `{calibration_s:g}` s",
        f"- estimated acceleration bias ENU: `{bias[0]:.6f}`, `{bias[1]:.6f}`, `{bias[2]:.6f}` m/s^2",
        "",
        "## Outputs",
        "",
        f"- trajectory CSV: `{out_csv}`",
        f"- HTML overlay: `{html}`",
        "",
        "## Result",
        "",
        "| duration s | points | final 3D error m | final horizontal error m | mean 3D error m | max 3D error m |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {values['duration_s']:.1f} | {values['points']:.0f} | {values['final_error_3d_m']:.3f} | "
        f"{values['final_error_horizontal_m']:.3f} | {values['mean_error_3d_m']:.3f} | {values['max_error_3d_m']:.3f} |",
        "",
        "## Final position",
        "",
        "| source | east m | north m | up m |",
        "| --- | ---: | ---: | ---: |",
        f"| IMU integrated | {values['final_pred_east_m']:.3f} | {values['final_pred_north_m']:.3f} | {values['final_pred_up_m']:.3f} |",
        f"| POS/GPS reference | {values['final_true_east_m']:.3f} | {values['final_true_north_m']:.3f} | {values['final_true_up_m']:.3f} |",
        "",
        "## Interpretation",
        "",
        "- Double integration of low-cost IMU acceleration drifts quickly; even tiny residual acceleration bias grows quadratically in position.",
        "- The next practical step is to add velocity constraints: optical flow, barometer/lidar altitude constraints, zero-velocity periods, or an EKF-style correction model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    imu, att = read_imu_att(args.data_dir)
    pos = read_pos(args.data_dir / "POS.csv")
    accel_rows = build_accel_series(imu, att)
    rows, bias = integrate(accel_rows, pos, args.calibration_s, args.max_dt_s)
    values = metrics(rows)
    write_csv(args.out_csv, rows)
    write_html(args.html, sample_rows(rows, args.max_html_points), values, bias)
    write_report(args.report, args.data_dir, args.out_csv, args.html, values, bias, args.calibration_s)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.html}")
    print(f"Wrote {args.report}")
    print(f"Final 3D error: {values['final_error_3d_m']:.3f} m")


if __name__ == "__main__":
    main()
