#!/usr/bin/env python3
"""Build a single-page viewer for generated flight tracks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

from gps_flight_map import format_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-index", type=Path, default=Path("derived/datasets/flight_index.csv"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("derived/datasets/tracks"))
    parser.add_argument("--visual-dir", type=Path, default=Path("artifacts/generated/gps/flights"))
    parser.add_argument("--out-html", type=Path, default=Path("artifacts/generated/gps/flights/index.html"))
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def build_payload(records: list[dict[str, str]], tracks_dir: Path, visual_dir: Path, out_html: Path) -> list[dict[str, object]]:
    base_dir = out_html.parent
    payload: list[dict[str, object]] = []
    for record in records:
        flight_id = record["flight_id"]
        flight_dir = visual_dir / flight_id
        track_csv = tracks_dir / f"{flight_id}_track.csv"
        distance_m = as_float(record.get("gps_distance_2d_m"))
        duration_s = as_float(record.get("duration_s"))
        min_alt_m = as_float(record.get("min_alt_m"))
        max_alt_m = as_float(record.get("max_alt_m"))
        payload.append(
            {
                "flight_id": flight_id,
                "source_file": record["source_file"],
                "source_format": record["source_format"],
                "segment": f"{record['segment_index']}/{record['segment_count']}",
                "gps_points": int(record["gps_points"]),
                "sensor_rows": int(record["sensor_rows"]),
                "duration_s": duration_s,
                "duration_label": format_seconds(duration_s or 0.0),
                "distance_m": distance_m,
                "distance_label": f"{distance_m:.1f} m" if distance_m is not None else "n/a",
                "altitude_label": (
                    f"{min_alt_m:.1f}..{max_alt_m:.1f} m"
                    if min_alt_m is not None and max_alt_m is not None
                    else "n/a"
                ),
                "max_gap_s": as_float(record.get("max_internal_gap_s")),
                "paths": {
                    "map": relpath(flight_dir / "map.html", base_dir),
                    "replay": relpath(flight_dir / "simulation.html", base_dir),
                    "plot": relpath(flight_dir / "path.svg", base_dir),
                    "geojson": relpath(flight_dir / "track.geojson", base_dir),
                    "csv": relpath(track_csv, base_dir),
                    "manifest": relpath(flight_dir / "manifest.json", base_dir),
                },
            }
        )
    return payload


def write_viewer(path: Path, flights: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = "Flight Track Viewer"
    payload = json.dumps(flights, ensure_ascii=False)
    html_text = VIEWER_TEMPLATE.replace("__TITLE__", html.escape(title))
    html_text = html_text.replace("__FLIGHTS__", payload)
    path.write_text(html_text, encoding="utf-8")


VIEWER_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d9e0e6;
      --text: #17202a;
      --muted: #667482;
      --accent: #1769aa;
      --ok: #2f7d4f;
      --warn: #b86b15;
      --danger: #b63b3b;
    }
    * {
      box-sizing: border-box;
    }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    body {
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      height: 100vh;
    }
    aside {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-width: 0;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    header {
      padding: 18px 18px 12px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 21px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    .filters {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 116px;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    input, select {
      width: 100%;
      height: 36px;
      border: 1px solid #c5ced6;
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 14px;
      padding: 0 10px;
    }
    .flight-list {
      overflow: auto;
      padding: 8px;
    }
    .flight {
      width: 100%;
      display: grid;
      gap: 6px;
      padding: 10px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      text-align: left;
      cursor: pointer;
      color: var(--text);
      font: inherit;
    }
    .flight:hover {
      background: #f2f5f7;
    }
    .flight.active {
      border-color: #8bb8da;
      background: #eaf3fa;
    }
    .flight-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-weight: 800;
      font-size: 14px;
    }
    .badge {
      flex: 0 0 auto;
      padding: 2px 7px;
      border-radius: 999px;
      background: #e8ecef;
      color: #4e5965;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .flight.active .badge {
      background: #d6e8f5;
      color: #14527d;
    }
    .flight-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 12px;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: 100vh;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: center;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .selected-title {
      min-width: 0;
    }
    .selected-title strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 17px;
    }
    .selected-title span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    .stat {
      min-width: 92px;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      font-size: 12px;
      color: var(--muted);
    }
    .stat strong {
      display: block;
      margin-top: 2px;
      color: var(--text);
      font-size: 14px;
    }
    .workspace {
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .tabs {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .tabs button, .open-link {
      height: 34px;
      border: 1px solid #c5ced6;
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      padding: 0 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
    }
    .tabs button.active {
      border-color: #6da4cc;
      background: #e5f1f9;
      color: #114f79;
    }
    .open-link {
      margin-left: auto;
    }
    iframe {
      width: 100%;
      height: 100%;
      border: 0;
      background: #fff;
    }
    .empty {
      display: none;
      padding: 24px;
      color: var(--muted);
    }
    @media (max-width: 860px) {
      body {
        overflow: auto;
      }
      .app {
        grid-template-columns: 1fr;
        grid-template-rows: 44vh 80vh;
        height: auto;
        min-height: 100vh;
      }
      aside, main {
        height: auto;
        min-height: 0;
      }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .topbar {
        grid-template-columns: 1fr;
      }
      .stats {
        justify-content: flex-start;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Flight Track Viewer</h1>
        <div class="sub"><span id="count"></span></div>
      </header>
      <div class="filters">
        <input id="search" type="search" placeholder="flight_id или источник">
        <select id="format">
          <option value="all">Все</option>
          <option value="module">Module</option>
          <option value="dataflash">DataFlash</option>
        </select>
      </div>
      <div id="flightList" class="flight-list"></div>
    </aside>
    <main>
      <section class="topbar">
        <div class="selected-title">
          <strong id="selectedName"></strong>
          <span id="selectedSource"></span>
        </div>
        <div class="stats">
          <div class="stat">Длительность<strong id="statDuration"></strong></div>
          <div class="stat">Дистанция<strong id="statDistance"></strong></div>
          <div class="stat">Точки GPS<strong id="statPoints"></strong></div>
          <div class="stat">Высота<strong id="statAltitude"></strong></div>
          <div class="stat">Сегмент<strong id="statSegment"></strong></div>
        </div>
      </section>
      <section class="workspace">
        <nav class="tabs">
          <button data-view="map" class="active">Карта</button>
          <button data-view="replay">Replay</button>
          <button data-view="plot">График</button>
          <button data-view="csv">CSV</button>
          <button data-view="geojson">GeoJSON</button>
          <a id="openLink" class="open-link" href="#" target="_blank" rel="noreferrer">Открыть</a>
        </nav>
        <iframe id="viewer" title="Selected flight"></iframe>
        <div id="empty" class="empty">Нет треков по фильтру.</div>
      </section>
    </main>
  </div>
  <script>
    const flights = __FLIGHTS__;
    const state = {
      selectedId: flights[0]?.flight_id || "",
      view: "map",
      query: "",
      format: "all",
    };

    const els = {
      count: document.getElementById("count"),
      search: document.getElementById("search"),
      format: document.getElementById("format"),
      list: document.getElementById("flightList"),
      selectedName: document.getElementById("selectedName"),
      selectedSource: document.getElementById("selectedSource"),
      statDuration: document.getElementById("statDuration"),
      statDistance: document.getElementById("statDistance"),
      statPoints: document.getElementById("statPoints"),
      statAltitude: document.getElementById("statAltitude"),
      statSegment: document.getElementById("statSegment"),
      viewer: document.getElementById("viewer"),
      openLink: document.getElementById("openLink"),
      empty: document.getElementById("empty"),
      tabs: Array.from(document.querySelectorAll(".tabs button")),
    };

    function filteredFlights() {
      const query = state.query.trim().toLowerCase();
      return flights.filter((flight) => {
        const matchesFormat = state.format === "all" || flight.source_format === state.format;
        const haystack = `${flight.flight_id} ${flight.source_file}`.toLowerCase();
        return matchesFormat && (!query || haystack.includes(query));
      });
    }

    function chooseFallback(filtered) {
      if (!filtered.length) return null;
      return filtered.find((flight) => flight.flight_id === state.selectedId) || filtered[0];
    }

    function renderList(filtered, selected) {
      els.list.innerHTML = "";
      for (const flight of filtered) {
        const button = document.createElement("button");
        button.className = `flight${selected && flight.flight_id === selected.flight_id ? " active" : ""}`;
        button.type = "button";
        button.innerHTML = `
          <div class="flight-title">
            <span>${flight.flight_id}</span>
            <span class="badge">${flight.source_format}</span>
          </div>
          <div class="flight-meta">
            <span>${flight.duration_label}</span>
            <span>${flight.distance_label}</span>
            <span>${flight.gps_points} GPS</span>
            <span>${flight.segment}</span>
          </div>
        `;
        button.addEventListener("click", () => {
          state.selectedId = flight.flight_id;
          render();
        });
        els.list.appendChild(button);
      }
    }

    function renderDetails(selected) {
      if (!selected) {
        els.empty.style.display = "block";
        els.viewer.style.display = "none";
        els.openLink.style.pointerEvents = "none";
        els.selectedName.textContent = "Нет треков";
        els.selectedSource.textContent = "";
        for (const el of [els.statDuration, els.statDistance, els.statPoints, els.statAltitude, els.statSegment]) {
          el.textContent = "";
        }
        return;
      }
      els.empty.style.display = "none";
      els.viewer.style.display = "block";
      els.openLink.style.pointerEvents = "auto";
      els.selectedName.textContent = selected.flight_id;
      els.selectedSource.textContent = selected.source_file;
      els.statDuration.textContent = selected.duration_label;
      els.statDistance.textContent = selected.distance_label;
      els.statPoints.textContent = selected.gps_points;
      els.statAltitude.textContent = selected.altitude_label;
      els.statSegment.textContent = selected.segment;
      const path = selected.paths[state.view];
      els.viewer.src = path;
      els.openLink.href = path;
    }

    function renderTabs() {
      for (const tab of els.tabs) {
        tab.classList.toggle("active", tab.dataset.view === state.view);
      }
    }

    function render() {
      const filtered = filteredFlights();
      const selected = chooseFallback(filtered);
      if (selected) state.selectedId = selected.flight_id;
      els.count.textContent = `${filtered.length} из ${flights.length} треков`;
      renderList(filtered, selected);
      renderTabs();
      renderDetails(selected);
    }

    els.search.addEventListener("input", (event) => {
      state.query = event.target.value;
      render();
    });
    els.format.addEventListener("change", (event) => {
      state.format = event.target.value;
      render();
    });
    for (const tab of els.tabs) {
      tab.addEventListener("click", () => {
        state.view = tab.dataset.view;
        render();
      });
    }

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = read_index(args.flight_index)
    flights = build_payload(records, args.tracks_dir, args.visual_dir, args.out_html)
    write_viewer(args.out_html, flights)
    print(f"Wrote {args.out_html}")
    print(f"Indexed {len(flights)} tracks")


if __name__ == "__main__":
    main()
