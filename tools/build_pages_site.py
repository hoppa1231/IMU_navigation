#!/usr/bin/env python3
"""Build a small GitHub Pages site from selected generated HTML artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GENERATED = ROOT / "artifacts" / "generated"


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def page(title: str, heading: str, intro: str, groups: list[tuple[str, list[tuple[str, str]]]]) -> str:
    nav_parts = []
    for group_title, items in groups:
        links = "\n".join(
            f'<li><a href="{href}">{label}</a></li>' for label, href in items
        )
        nav_parts.append(
            f"""
            <section class="group">
              <h2>{group_title}</h2>
              <ul>
                {links}
              </ul>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #14202a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #f4f6f8;
    }}
    header {{
      padding: 28px 24px 18px;
      background: #ffffff;
      border-bottom: 1px solid #d8e0e8;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
    }}
    p {{
      margin: 10px 0 0;
      color: #5a6674;
      max-width: 900px;
    }}
    main {{
      padding: 22px 24px 32px;
      display: grid;
      gap: 18px;
      max-width: 1180px;
    }}
    .group {{
      background: #ffffff;
      border: 1px solid #dfe6ed;
      border-radius: 10px;
      padding: 18px 18px 14px;
    }}
    .group h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
    }}
    a {{
      color: #0f5d99;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .note {{
      color: #677483;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{heading}</h1>
    <p>{intro}</p>
  </header>
  <main>
    {"".join(nav_parts)}
  </main>
</body>
</html>
"""


def build() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    write_text(DOCS / ".nojekyll", "")

    copy_tree(GENERATED / "dataflash", DOCS / "dataflash")
    copy_tree(GENERATED / "navigation", DOCS / "navigation")
    copy_tree(GENERATED / "module_predictions", DOCS / "module_predictions")

    gps_dest = DOCS / "gps" / "flights"
    gps_dest.mkdir(parents=True, exist_ok=True)
    selected_flights = [
        "dataflash_2025_01_15",
        "linear_15_01_2025",
        "triangle_15_01_2025",
        "circle_07_02_2025",
        "square_07_02_2025",
    ]
    for flight_id in selected_flights:
        copy_tree(GENERATED / "gps" / "flights" / flight_id, gps_dest / flight_id)

    write_text(
        DOCS / "index.html",
        page(
            title="IMU Navigation Demo",
            heading="IMU Navigation Demo",
            intro="Curated static pages for GitHub Pages. Open the DataFlash reports, navigation comparisons, and a few GPS flight replays.",
            groups=[
                (
                    "DataFlash",
                    [
                        ("Neural comparison", "dataflash/neural_demo/index.html"),
                        ("Demo", "dataflash/demo/index.html"),
                        ("Final report", "dataflash/final_report/index.html"),
                        ("Diagnostics", "dataflash/diagnostics/index.html"),
                        ("Sequence rollout", "dataflash/rollouts/sequence/index.html"),
                        ("Ridge rollout", "dataflash/rollouts/ridge/index.html"),
                        ("Bias rollout", "dataflash/rollouts/sequence_fixed100_bias/index.html"),
                    ],
                ),
                (
                    "Navigation",
                    [
                        ("Comparison", "navigation/comparison/index.html"),
                        ("Trajectory overlay", "navigation/trajectory_overlay/index.html"),
                        ("IMU dead reckoning", "navigation/imu_dead_reckoning/index.html"),
                        ("POLI_NA rollout", "navigation/poli_na_rollout/index.html"),
                        ("All module route predictions", "module_predictions/index.html"),
                    ],
                ),
                (
                    "GPS flights",
                    [
                        ("DataFlash flight replay", "gps/flights/dataflash_2025_01_15/simulation.html"),
                        ("DataFlash flight map", "gps/flights/dataflash_2025_01_15/map.html"),
                        ("Linear flight replay", "gps/flights/linear_15_01_2025/simulation.html"),
                        ("Triangle flight replay", "gps/flights/triangle_15_01_2025/simulation.html"),
                        ("Circle flight replay", "gps/flights/circle_07_02_2025/simulation.html"),
                        ("Square flight replay", "gps/flights/square_07_02_2025/simulation.html"),
                    ],
                ),
            ],
        ),
    )

    write_text(
        DOCS / "dataflash" / "index.html",
        page(
            title="DataFlash",
            heading="DataFlash",
            intro="Final model, diagnostics, and rollout pages.",
            groups=[
                (
                    "Key pages",
                    [
                        ("Neural comparison", "neural_demo/index.html"),
                        ("Demo", "demo/index.html"),
                        ("Final report", "final_report/index.html"),
                        ("Diagnostics", "diagnostics/index.html"),
                    ],
                ),
                (
                    "Rollouts",
                    [
                        ("LSTM purged IMU+ATT+CRt", "rollouts/lstm_64_imu_att_crt_purged/index.html"),
                        ("GRU purged IMU+ATT+CRt", "rollouts/gru_64_imu_att_crt_purged/index.html"),
                        ("LSTM 64", "rollouts/lstm_64/index.html"),
                        ("GRU 64", "rollouts/gru_64/index.html"),
                        ("MLP 64", "rollouts/mlp_64/index.html"),
                        ("Zero baseline", "rollouts/zero/index.html"),
                        ("Train mean", "rollouts/train_mean/index.html"),
                        ("Ridge", "rollouts/ridge/index.html"),
                        ("Sequence", "rollouts/sequence/index.html"),
                        ("Bias corrected", "rollouts/sequence_fixed100_bias/index.html"),
                    ],
                ),
            ],
        ),
    )

    write_text(
        DOCS / "navigation" / "index.html",
        page(
            title="Navigation",
            heading="Navigation",
            intro="Direct comparisons between GPS reference and open-loop trajectory methods.",
            groups=[
                (
                    "Views",
                    [
                        ("Comparison", "comparison/index.html"),
                        ("Trajectory overlay", "trajectory_overlay/index.html"),
                        ("IMU dead reckoning", "imu_dead_reckoning/index.html"),
                        ("POLI_NA rollout", "poli_na_rollout/index.html"),
                    ],
                ),
            ],
        ),
    )

    write_text(
        DOCS / "gps" / "index.html",
        page(
            title="GPS Flights",
            heading="GPS Flights",
            intro="Curated GPS replays for the flights we used most often.",
            groups=[
                (
                    "Flights",
                    [
                        ("DataFlash replay", "flights/dataflash_2025_01_15/simulation.html"),
                        ("DataFlash map", "flights/dataflash_2025_01_15/map.html"),
                        ("Linear replay", "flights/linear_15_01_2025/simulation.html"),
                        ("Linear map", "flights/linear_15_01_2025/map.html"),
                        ("Triangle replay", "flights/triangle_15_01_2025/simulation.html"),
                        ("Triangle map", "flights/triangle_15_01_2025/map.html"),
                        ("Circle replay", "flights/circle_07_02_2025/simulation.html"),
                        ("Circle map", "flights/circle_07_02_2025/map.html"),
                        ("Square replay", "flights/square_07_02_2025/simulation.html"),
                        ("Square map", "flights/square_07_02_2025/map.html"),
                    ],
                ),
            ],
        ),
    )


if __name__ == "__main__":
    build()
