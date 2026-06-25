#!/usr/bin/env python3
"""Build an interactive flight map and a static path plot from GPS CSV data."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass
class GpsPoint:
    time_us: float
    lat: float
    lon: float
    alt: float
    speed: float | None
    status: float | None
    nsats: float | None
    hdop: float | None


@dataclass
class LocalPoint:
    time_s: float
    east_m: float
    north_m: float
    up_m: float
    distance_m: float
    speed: float | None
    lat: float
    lon: float
    alt: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps", type=Path, default=Path("derived/dataflash/GPS.csv"))
    parser.add_argument("--format", choices=["auto", "dataflash", "module"], default="auto")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--out-sim-html", type=Path, default=None)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-svg", type=Path, default=None)
    parser.add_argument("--out-geojson", type=Path, default=None)
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--split-all", action="store_true")
    parser.add_argument("--segment-index", type=int, default=None, help="1-based segment index after automatic splitting.")
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--max-jump-m", type=float, default=50.0)
    parser.add_argument("--min-status", type=float, default=3.0, help="Minimum GPS fix status to keep.")
    parser.add_argument("--max-hdop", type=float, default=None, help="Optional maximum HDOP filter.")
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or "flight"


def decimal_from_degrees_minutes(degrees: float, minutes: float) -> float:
    sign = -1.0 if degrees < 0 else 1.0
    return degrees + sign * minutes / 60.0


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value for key, value in row.items() if key is not None}


def detect_format(fieldnames: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    names = {name.strip() for name in fieldnames}
    if {"TimeUS", "Lat", "Lng", "Alt"}.issubset(names):
        return "dataflash"
    if {"TimeStamp", "LatMinutes", "LatDegrees", "LonMinutes", "LonDegrees", "Altitude, m"}.issubset(names):
        return "module"
    raise ValueError(f"Cannot detect GPS CSV format from columns: {', '.join(fieldnames)}")


def read_gps(path: Path, requested_format: str, min_status: float, max_hdop: float | None) -> tuple[list[GpsPoint], str]:
    points: list[GpsPoint] = []
    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        file.seek(0)
        delimiter = ";" if ";" in first_line else ","
        reader = csv.DictReader(file, delimiter=delimiter)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        data_format = detect_format(fieldnames, requested_format)
        reader.fieldnames = fieldnames
        for row in reader:
            row = normalize_row(row)
            if data_format == "dataflash":
                time_us = as_float(row.get("TimeUS"))
                lat = as_float(row.get("Lat"))
                lon = as_float(row.get("Lng") or row.get("Lon"))
                alt = as_float(row.get("Alt"))
                speed = as_float(row.get("Spd"))
                status = as_float(row.get("Status"))
                nsats = as_float(row.get("NSats"))
                hdop = as_float(row.get("HDop"))
            else:
                timestamp_ms = as_float(row.get("TimeStamp"))
                lat_minutes = as_float(row.get("LatMinutes"))
                lat_degrees = as_float(row.get("LatDegrees"))
                lon_minutes = as_float(row.get("LonMinutes"))
                lon_degrees = as_float(row.get("LonDegrees"))
                alt = as_float(row.get("Altitude, m"))
                time_us = timestamp_ms * 1000.0 if timestamp_ms is not None else None
                lat = (
                    decimal_from_degrees_minutes(lat_degrees, lat_minutes)
                    if lat_degrees is not None and lat_minutes is not None
                    else None
                )
                lon = (
                    decimal_from_degrees_minutes(lon_degrees, lon_minutes)
                    if lon_degrees is not None and lon_minutes is not None
                    else None
                )
                speed = None
                status = None
                nsats = None
                hdop = None

            if time_us is None or lat is None or lon is None or alt is None:
                continue
            if abs(lat) < 1e-12 and abs(lon) < 1e-12:
                continue
            if status is not None and status < min_status:
                continue
            if max_hdop is not None and hdop is not None and hdop > max_hdop:
                continue

            points.append(
                GpsPoint(
                    time_us=time_us,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    speed=speed,
                    status=status,
                    nsats=nsats,
                    hdop=hdop,
                )
            )

    return points, data_format


def gps_to_local_m(lat: float, lon: float, alt: float, origin: GpsPoint) -> tuple[float, float, float]:
    earth_radius_m = 6_371_000.0
    north = math.radians(lat - origin.lat) * earth_radius_m
    east = math.radians(lon - origin.lon) * earth_radius_m * math.cos(math.radians(origin.lat))
    up = alt - origin.alt
    return east, north, up


def horizontal_distance_m(first: GpsPoint, second: GpsPoint) -> float:
    earth_radius_m = 6_371_000.0
    north = math.radians(second.lat - first.lat) * earth_radius_m
    east = math.radians(second.lon - first.lon) * earth_radius_m * math.cos(math.radians(first.lat))
    return math.hypot(east, north)


def split_points(points: list[GpsPoint], max_gap_s: float, max_jump_m: float) -> list[list[GpsPoint]]:
    if not points:
        return []

    segments: list[list[GpsPoint]] = []
    current = [points[0]]
    for previous, point in zip(points, points[1:]):
        dt_s = (point.time_us - previous.time_us) / 1_000_000.0
        jump_m = horizontal_distance_m(previous, point)
        should_split = dt_s < 0.0 or dt_s > max_gap_s or jump_m > max_jump_m
        if should_split:
            segments.append(current)
            current = [point]
        else:
            current.append(point)
    segments.append(current)
    return segments


def build_local_track(points: list[GpsPoint]) -> list[LocalPoint]:
    if not points:
        return []

    origin = points[0]
    local: list[LocalPoint] = []
    previous_east = 0.0
    previous_north = 0.0
    distance = 0.0
    last_speed = origin.speed

    for index, point in enumerate(points):
        east, north, up = gps_to_local_m(point.lat, point.lon, point.alt, origin)
        speed = point.speed
        if index:
            step_distance = math.hypot(east - previous_east, north - previous_north)
            distance += step_distance
            dt = (point.time_us - points[index - 1].time_us) / 1_000_000.0
            if speed is None and dt >= 0.05:
                speed = step_distance / dt
            elif speed is None:
                speed = last_speed
        if speed is not None:
            last_speed = speed
        previous_east = east
        previous_north = north
        local.append(
            LocalPoint(
                time_s=(point.time_us - origin.time_us) / 1_000_000.0,
                east_m=east,
                north_m=north,
                up_m=up,
                distance_m=distance,
                speed=speed,
                lat=point.lat,
                lon=point.lon,
                alt=point.alt,
            )
        )
    return local


def summarize(
    points: list[GpsPoint],
    local: list[LocalPoint],
    source_segments: int = 1,
) -> dict[str, float | int | None]:
    speeds = [point.speed for point in local if point.speed is not None]
    nsats = [point.nsats for point in points if point.nsats is not None]
    hdops = [point.hdop for point in points if point.hdop is not None]
    duration_s = local[-1].time_s if local else 0.0
    return {
        "points": len(points),
        "duration_s": duration_s,
        "distance_m": local[-1].distance_m if local else 0.0,
        "max_speed_mps": max(speeds) if speeds else None,
        "median_speed_mps": median(speeds) if speeds else None,
        "min_alt_m": min(point.alt for point in points) if points else None,
        "max_alt_m": max(point.alt for point in points) if points else None,
        "median_nsats": median(nsats) if nsats else None,
        "median_hdop": median(hdops) if hdops else None,
        "start_lat": points[0].lat if points else None,
        "start_lon": points[0].lon if points else None,
        "end_lat": points[-1].lat if points else None,
        "end_lon": points[-1].lon if points else None,
        "max_gap_s": max_time_gap_s(points),
        "segments_by_gap_2s": count_segments_by_gap(points, 2.0),
        "source_segments": source_segments,
    }


def max_time_gap_s(points: list[GpsPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return max((points[idx].time_us - points[idx - 1].time_us) / 1_000_000.0 for idx in range(1, len(points)))


def count_segments_by_gap(points: list[GpsPoint], gap_s: float) -> int:
    if not points:
        return 0
    return 1 + sum(
        1
        for idx in range(1, len(points))
        if (points[idx].time_us - points[idx - 1].time_us) / 1_000_000.0 > gap_s
    )


def format_seconds(seconds: float) -> str:
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{rem:04.1f}"


def write_geojson(path: Path, points: list[GpsPoint], summary: dict[str, float | int | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "GPS flight track",
                    "points": summary["points"],
                    "duration_s": summary["duration_s"],
                    "distance_m": summary["distance_m"],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[point.lon, point.lat, point.alt] for point in points],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "start"},
                "geometry": {"type": "Point", "coordinates": [points[0].lon, points[0].lat, points[0].alt]},
            },
            {
                "type": "Feature",
                "properties": {"name": "finish"},
                "geometry": {"type": "Point", "coordinates": [points[-1].lon, points[-1].lat, points[-1].alt]},
            },
        ],
    }
    path.write_text(json.dumps(feature_collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_png(path: Path, local: list[LocalPoint]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    times = [point.time_s for point in local]
    east = [point.east_m for point in local]
    north = [point.north_m for point in local]
    up = [point.up_m for point in local]
    speeds = [point.speed if point.speed is not None else math.nan for point in local]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    path_axis, profile_axis = axes

    path_axis.plot(east, north, color="#1f77b4", linewidth=1.8)
    path_axis.scatter(east[0], north[0], color="#2ca02c", s=60, label="start", zorder=3)
    path_axis.scatter(east[-1], north[-1], color="#d62728", s=60, label="finish", zorder=3)
    path_axis.set_title("GPS flight path")
    path_axis.set_xlabel("East, m")
    path_axis.set_ylabel("North, m")
    path_axis.axis("equal")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend(loc="best")

    profile_axis.plot(times, up, color="#9467bd", linewidth=1.5, label="Altitude relative to start, m")
    speed_axis = profile_axis.twinx()
    speed_axis.plot(times, speeds, color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Speed, m/s")
    profile_axis.set_title("Altitude and speed")
    profile_axis.set_xlabel("Time, s")
    profile_axis.set_ylabel("Relative altitude, m")
    speed_axis.set_ylabel("Speed, m/s")
    profile_axis.grid(True, alpha=0.3)

    lines, labels = profile_axis.get_legend_handles_labels()
    speed_lines, speed_labels = speed_axis.get_legend_handles_labels()
    profile_axis.legend(lines + speed_lines, labels + speed_labels, loc="best")

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def scale_points(
    xs: list[float],
    ys: list[float],
    left: float,
    top: float,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    return [
        (
            left + (x - min_x) / span_x * width,
            top + height - (y - min_y) / span_y * height,
        )
        for x, y in zip(xs, ys)
    ]


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg(path: Path, local: list[LocalPoint], summary: dict[str, float | int | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    east = [point.east_m for point in local]
    north = [point.north_m for point in local]
    times = [point.time_s for point in local]
    up = [point.up_m for point in local]
    speeds = [point.speed if point.speed is not None else 0.0 for point in local]

    track = scale_points(east, north, 70, 75, 500, 430)
    altitude = scale_points(times, up, 680, 95, 420, 165)
    speed = scale_points(times, speeds, 680, 330, 420, 145)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="590" viewBox="0 0 1180 590">
  <rect width="1180" height="590" fill="#f7f8fa"/>
  <text x="45" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#17202a">GPS flight path</text>
  <text x="45" y="560" font-family="Arial, sans-serif" font-size="14" fill="#5d6b78">Distance: {float(summary['distance_m'] or 0.0):.1f} m | Duration: {format_seconds(float(summary['duration_s'] or 0.0))} | Points: {summary['points']}</text>

  <rect x="45" y="60" width="550" height="470" rx="6" fill="#ffffff" stroke="#d7dde3"/>
  <line x1="70" y1="505" x2="570" y2="505" stroke="#d7dde3"/>
  <line x1="70" y1="75" x2="70" y2="505" stroke="#d7dde3"/>
  <polyline points="{svg_polyline(track)}" fill="none" stroke="#1f77b4" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{track[0][0]:.2f}" cy="{track[0][1]:.2f}" r="7" fill="#2ca02c" stroke="#ffffff" stroke-width="2"/>
  <circle cx="{track[-1][0]:.2f}" cy="{track[-1][1]:.2f}" r="7" fill="#d62728" stroke="#ffffff" stroke-width="2"/>
  <text x="75" y="525" font-family="Arial, sans-serif" font-size="13" fill="#5d6b78">East/North, meters from first GPS point</text>

  <rect x="650" y="60" width="485" height="225" rx="6" fill="#ffffff" stroke="#d7dde3"/>
  <text x="680" y="88" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#17202a">Relative altitude</text>
  <line x1="680" y1="260" x2="1100" y2="260" stroke="#d7dde3"/>
  <polyline points="{svg_polyline(altitude)}" fill="none" stroke="#9467bd" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="680" y="277" font-family="Arial, sans-serif" font-size="13" fill="#5d6b78">{min(up):.1f}..{max(up):.1f} m relative to start</text>

  <rect x="650" y="305" width="485" height="200" rx="6" fill="#ffffff" stroke="#d7dde3"/>
  <text x="680" y="333" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#17202a">Speed</text>
  <line x1="680" y1="475" x2="1100" y2="475" stroke="#d7dde3"/>
  <polyline points="{svg_polyline(speed)}" fill="none" stroke="#ff7f0e" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="680" y="492" font-family="Arial, sans-serif" font-size="13" fill="#5d6b78">max {max(speeds):.2f} m/s</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def make_stat_rows(summary: dict[str, float | int | None], source_name: str, source_format: str) -> str:
    rows = [
        ("Source", source_name),
        ("Format", source_format),
        ("GPS points", f"{summary['points']}"),
        ("Duration", format_seconds(float(summary["duration_s"] or 0.0))),
        ("2D distance", f"{float(summary['distance_m'] or 0.0):.1f} m"),
        ("Max speed", value_with_unit(summary["max_speed_mps"], "m/s")),
        ("Altitude", f"{float(summary['min_alt_m'] or 0.0):.1f}..{float(summary['max_alt_m'] or 0.0):.1f} m"),
        ("Median satellites", value_plain(summary["median_nsats"])),
        ("Median HDOP", value_plain(summary["median_hdop"])),
        ("Max time gap", f"{float(summary['max_gap_s'] or 0.0):.3f} s"),
        ("Source segments", f"{summary['source_segments']}"),
    ]
    return "\n".join(
        f"<div class=\"stat\"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in rows
    )


def value_plain(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def value_with_unit(value: float | int | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f} {unit}"


def write_html(
    path: Path,
    points: list[GpsPoint],
    local: list[LocalPoint],
    summary: dict[str, float | int | None],
    source_name: str,
    source_format: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = [[point.lat, point.lon] for point in points]
    altitudes = [point.alt for point in points]
    times = [point.time_s for point in local]
    distances = [point.distance_m for point in local]
    center = [sum(point.lat for point in points) / len(points), sum(point.lon for point in points) / len(points)]
    bounds = [[min(point.lat for point in points), min(point.lon for point in points)], [max(point.lat for point in points), max(point.lon for point in points)]]
    html_text = HTML_TEMPLATE.replace("__COORDINATES__", json.dumps(coordinates))
    html_text = html_text.replace("__ALTITUDES__", json.dumps(altitudes))
    html_text = html_text.replace("__TIMES__", json.dumps(times))
    html_text = html_text.replace("__DISTANCES__", json.dumps(distances))
    html_text = html_text.replace("__CENTER__", json.dumps(center))
    html_text = html_text.replace("__BOUNDS__", json.dumps(bounds))
    html_text = html_text.replace("__STATS__", make_stat_rows(summary, source_name, source_format))
    html_text = html_text.replace("__START__", f"{points[0].lat:.7f}, {points[0].lon:.7f}")
    html_text = html_text.replace("__FINISH__", f"{points[-1].lat:.7f}, {points[-1].lon:.7f}")
    path.write_text(html_text, encoding="utf-8")


def write_simulation_html(
    path: Path,
    local: list[LocalPoint],
    summary: dict[str, float | int | None],
    source_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source_name,
        "summary": summary,
        "points": [
            {
                "t": point.time_s,
                "x": point.east_m,
                "y": point.north_m,
                "z": point.up_m,
                "speed": point.speed or 0.0,
                "distance": point.distance_m,
            }
            for point in local
        ],
    }
    html_text = SIMULATION_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    path.write_text(html_text, encoding="utf-8")


def write_manifest(
    path: Path,
    source_path: Path,
    source_name: str,
    source_format: str,
    summary: dict[str, float | int | None],
    outputs: dict[str, Path],
    segment_index: int | None,
    segment_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": {
            "path": str(source_path),
            "name": source_name,
            "format": source_format,
            "type": "original_input",
        },
        "generated": {key: str(value) for key, value in outputs.items()},
        "segment": {
            "index": segment_index,
            "count": segment_count,
        },
        "summary": summary,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPS flight map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body {
      height: 100%;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: #f3f5f7;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      height: 100%;
    }
    #map {
      min-height: 100%;
    }
    aside {
      padding: 20px;
      background: #ffffff;
      border-left: 1px solid #d7dde3;
      overflow: auto;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 22px;
      line-height: 1.2;
    }
    .stat {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 10px 0;
      border-bottom: 1px solid #edf0f2;
      font-size: 14px;
    }
    .stat span {
      color: #5d6b78;
    }
    .legend {
      margin-top: 18px;
      font-size: 13px;
      color: #5d6b78;
      line-height: 1.5;
    }
    .coord {
      margin-top: 14px;
      padding: 12px;
      background: #f3f5f7;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.5;
      word-break: break-word;
    }
    @media (max-width: 840px) {
      .layout {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(440px, 65vh) auto;
      }
      aside {
        border-left: 0;
        border-top: 1px solid #d7dde3;
      }
    }
  </style>
</head>
<body>
  <main class="layout">
    <div id="map"></div>
    <aside>
      <h1>GPS flight map</h1>
      __STATS__
      <div class="coord"><strong>Start</strong><br>__START__</div>
      <div class="coord"><strong>Finish</strong><br>__FINISH__</div>
      <div class="legend">
        The blue line is the filtered GPS track. Green and red markers show takeoff and finish points.
      </div>
    </aside>
  </main>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const coordinates = __COORDINATES__;
    const altitudes = __ALTITUDES__;
    const times = __TIMES__;
    const distances = __DISTANCES__;
    const center = __CENTER__;
    const bounds = __BOUNDS__;

    const map = L.map("map", { preferCanvas: true }).setView(center, 17);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 22,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    const track = L.polyline(coordinates, {
      color: "#1f77b4",
      weight: 4,
      opacity: 0.9
    }).addTo(map);
    map.fitBounds(bounds, { padding: [32, 32], maxZoom: 19 });

    L.circleMarker(coordinates[0], {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: "#2ca02c",
      fillOpacity: 1
    }).addTo(map).bindPopup("Start");

    L.circleMarker(coordinates[coordinates.length - 1], {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: "#d62728",
      fillOpacity: 1
    }).addTo(map).bindPopup("Finish");

    track.bindPopup(() => {
      const last = coordinates.length - 1;
      return `<strong>GPS track</strong><br>${coordinates.length} points<br>` +
        `Duration: ${times[last].toFixed(1)} s<br>` +
        `Distance: ${distances[last].toFixed(1)} m<br>` +
        `Altitude: ${Math.min(...altitudes).toFixed(1)}..${Math.max(...altitudes).toFixed(1)} m`;
    });
  </script>
</body>
</html>
"""


SIMULATION_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flight simulation replay</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef2f6;
      color: #17202a;
    }
    canvas {
      width: 100vw;
      height: 100vh;
      display: block;
    }
    .panel {
      position: fixed;
      left: 18px;
      right: 18px;
      bottom: 18px;
      display: grid;
      grid-template-columns: auto 1fr auto auto auto auto;
      align-items: center;
      gap: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #d7dde3;
      border-radius: 8px;
      box-shadow: 0 8px 28px rgba(24, 34, 46, 0.12);
      z-index: 3;
    }
    button {
      height: 34px;
      padding: 0 14px;
      border: 1px solid #b9c3cc;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
      font-weight: 700;
      cursor: pointer;
    }
    select {
      height: 34px;
      border: 1px solid #b9c3cc;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
      font-weight: 700;
    }
    input[type="range"] {
      width: 100%;
    }
    .metric {
      min-width: 90px;
      font-size: 13px;
      line-height: 1.2;
      color: #5d6b78;
    }
    .metric strong {
      display: block;
      color: #17202a;
      font-size: 15px;
    }
    .title {
      position: fixed;
      top: 18px;
      left: 18px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #d7dde3;
      border-radius: 8px;
      box-shadow: 0 8px 28px rgba(24, 34, 46, 0.10);
      z-index: 2;
    }
    .title h1 {
      margin: 0 0 6px;
      font-size: 18px;
    }
    .title div {
      color: #5d6b78;
      font-size: 13px;
    }
    @media (max-width: 760px) {
      .panel {
        grid-template-columns: auto 1fr;
      }
      .metric {
        display: none;
      }
    }
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <div class="title">
    <h1>Flight simulation replay</h1>
    <div id="source"></div>
  </div>
  <div class="panel">
    <button id="play" type="button">Play</button>
    <input id="time" type="range" min="0" max="1000" step="1" value="0">
    <select id="speed" aria-label="Playback speed">
      <option value="1">1x</option>
      <option value="5" selected>5x</option>
      <option value="10">10x</option>
      <option value="30">30x</option>
      <option value="60">60x</option>
    </select>
    <div class="metric">Time<strong id="timeValue">0.0 s</strong></div>
    <div class="metric">Speed<strong id="speedValue">0.0 m/s</strong></div>
    <div class="metric">Altitude<strong id="altValue">0.0 m</strong></div>
  </div>
  <script>
    const payload = __PAYLOAD__;
    const points = payload.points;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const playButton = document.getElementById("play");
    const slider = document.getElementById("time");
    const speedSelect = document.getElementById("speed");
    const timeValue = document.getElementById("timeValue");
    const speedValue = document.getElementById("speedValue");
    const altValue = document.getElementById("altValue");
    document.getElementById("source").textContent = payload.source;

    let playing = false;
    let currentIndex = 0;
    let lastFrame = performance.now();
    let playheadSeconds = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    const minX = Math.min(...points.map(p => p.x));
    const maxX = Math.max(...points.map(p => p.x));
    const minY = Math.min(...points.map(p => p.y));
    const maxY = Math.max(...points.map(p => p.y));
    const minZ = Math.min(...points.map(p => p.z));
    const maxZ = Math.max(...points.map(p => p.z));
    const spanX = Math.max(maxX - minX, 1);
    const spanY = Math.max(maxY - minY, 1);
    const spanZ = Math.max(maxZ - minZ, 1);
    const duration = points[points.length - 1].t;

    function project(p) {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const scale = Math.min((w - 140) / spanX, (h - 190) / spanY) * 0.86;
      const cx = w * 0.52;
      const cy = h * 0.48;
      const x = (p.x - (minX + maxX) / 2) * scale;
      const y = (p.y - (minY + maxY) / 2) * scale;
      const z = (p.z - minZ) / spanZ;
      return {
        x: cx + x - y * 0.24,
        y: cy + y * 0.58 - x * 0.08 - z * 95,
        shadowY: cy + y * 0.58 - x * 0.08
      };
    }

    function pointAtTime(t) {
      while (currentIndex < points.length - 2 && points[currentIndex + 1].t < t) currentIndex++;
      while (currentIndex > 0 && points[currentIndex].t > t) currentIndex--;
      return points[currentIndex];
    }

    function drawGrid() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.fillStyle = "#eef2f6";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#d7dde3";
      ctx.lineWidth = 1;
      for (let i = -10; i <= 10; i++) {
        ctx.beginPath();
        ctx.moveTo(w * 0.08, h * 0.5 + i * 28);
        ctx.lineTo(w * 0.92, h * 0.5 + i * 28);
        ctx.stroke();
      }
      for (let i = -12; i <= 12; i++) {
        ctx.beginPath();
        ctx.moveTo(w * 0.5 + i * 34, h * 0.12);
        ctx.lineTo(w * 0.5 + i * 34, h * 0.88);
        ctx.stroke();
      }
    }

    function drawPath() {
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#1f77b4";
      ctx.beginPath();
      points.forEach((p, index) => {
        const screen = project(p);
        if (index === 0) ctx.moveTo(screen.x, screen.y);
        else ctx.lineTo(screen.x, screen.y);
      });
      ctx.stroke();
    }

    function draw() {
      drawGrid();
      drawPath();
      const p = pointAtTime(playheadSeconds);
      const screen = project(p);
      ctx.strokeStyle = "rgba(23, 32, 42, 0.28)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(screen.x, screen.y);
      ctx.lineTo(screen.x, screen.shadowY);
      ctx.stroke();
      ctx.fillStyle = "rgba(23, 32, 42, 0.16)";
      ctx.beginPath();
      ctx.ellipse(screen.x, screen.shadowY, 15, 6, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#d62728";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      slider.value = String(Math.round((playheadSeconds / duration) * 1000));
      timeValue.textContent = `${p.t.toFixed(1)} s`;
      speedValue.textContent = `${p.speed.toFixed(2)} m/s`;
      altValue.textContent = `${p.z.toFixed(1)} m`;
    }

    function tick(now) {
      const delta = Math.min((now - lastFrame) / 1000, 0.25);
      lastFrame = now;
      if (playing) {
        const replaySpeed = Number(speedSelect.value);
        playheadSeconds = Math.min(duration, playheadSeconds + delta * replaySpeed);
        if (playheadSeconds >= duration) {
          playing = false;
          playButton.textContent = "Play";
        }
        draw();
      }
      requestAnimationFrame(tick);
    }

    playButton.addEventListener("click", () => {
      if (playheadSeconds >= duration) playheadSeconds = 0;
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
      lastFrame = performance.now();
      draw();
    });
    slider.addEventListener("input", () => {
      playing = false;
      playButton.textContent = "Play";
      playheadSeconds = duration * Number(slider.value) / 1000;
      draw();
    });
    window.addEventListener("resize", resize);
    resize();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def resolve_outputs(args: argparse.Namespace, segment_index: int | None = None) -> dict[str, Path]:
    source_name = args.name or args.gps.stem
    out_dir = args.out_dir or Path("artifacts/generated/gps/flights") / slugify(source_name)
    if segment_index is not None:
        out_dir = out_dir / f"segment_{segment_index:02d}"
    return {
        "map_html": args.out_html or out_dir / "map.html",
        "simulation_html": args.out_sim_html or out_dir / "simulation.html",
        "png": args.out_png or out_dir / "path.png",
        "svg": args.out_svg or out_dir / "path.svg",
        "geojson": args.out_geojson or out_dir / "track.geojson",
        "manifest": args.out_manifest or out_dir / "manifest.json",
    }


def generate_outputs(
    args: argparse.Namespace,
    source_name: str,
    source_format: str,
    points: list[GpsPoint],
    segment_index: int | None,
    segment_count: int,
) -> dict[str, float | int | None]:
    outputs = resolve_outputs(args, segment_index)
    local = build_local_track(points)
    summary = summarize(points, local, source_segments=segment_count)
    write_geojson(outputs["geojson"], points, summary)
    wrote_png = write_png(outputs["png"], local)
    write_svg(outputs["svg"], local, summary)
    display_name = source_name if segment_index is None else f"{source_name} segment {segment_index:02d}"
    write_html(outputs["map_html"], points, local, summary, display_name, source_format)
    write_simulation_html(outputs["simulation_html"], local, summary, display_name)
    write_manifest(outputs["manifest"], args.gps, source_name, source_format, summary, outputs, segment_index, segment_count)

    print(f"GPS points: {summary['points']}")
    print(f"Duration: {format_seconds(float(summary['duration_s'] or 0.0))}")
    print(f"2D distance: {float(summary['distance_m'] or 0.0):.1f} m")
    print(f"Altitude: {float(summary['min_alt_m'] or 0.0):.1f}..{float(summary['max_alt_m'] or 0.0):.1f} m")
    print(f"Wrote {outputs['map_html']}")
    print(f"Wrote {outputs['simulation_html']}")
    if wrote_png:
        print(f"Wrote {outputs['png']}")
    else:
        print("Skipped PNG: matplotlib is not installed")
    print(f"Wrote {outputs['svg']}")
    print(f"Wrote {outputs['geojson']}")
    print(f"Wrote {outputs['manifest']}")
    return summary


def main() -> None:
    args = parse_args()
    source_name = args.name or args.gps.stem
    points, source_format = read_gps(args.gps, args.format, args.min_status, args.max_hdop)
    if len(points) < 2:
        raise SystemExit(f"Need at least 2 valid GPS points in {args.gps}, got {len(points)}")

    segments = [segment for segment in split_points(points, args.max_gap_s, args.max_jump_m) if len(segment) >= 2]

    print(f"Source: {args.gps}")
    print(f"Format: {source_format}")
    print(f"Detected source segments: {len(segments)}")

    if args.split_all:
        for index, segment in enumerate(segments, start=1):
            print(f"\nSegment {index}/{len(segments)}")
            generate_outputs(args, source_name, source_format, segment, index, len(segments))
        return

    if args.segment_index is not None:
        if args.segment_index < 1 or args.segment_index > len(segments):
            raise SystemExit(f"segment-index must be between 1 and {len(segments)}")
        points = segments[args.segment_index - 1]
        segment_index = args.segment_index
    else:
        segment_index = None

    generate_outputs(args, source_name, source_format, points, segment_index, len(segments))


if __name__ == "__main__":
    main()
