# Data Layout

This project keeps large flight inputs local and out of Git. Code, lightweight
documentation, and reproducible experiment definitions are versioned.

## Source Inputs

Local source inputs live under `artifacts/` and are intentionally ignored by
Git:

| path | role | notes |
| --- | --- | --- |
| `artifacts/linear_15_01_2025.csv` | original module flight | Single continuous flight with IMU, optical flow, barometer, lidar, and GPS columns. |
| `artifacts/triangle_15_01_2025.csv` | original module flight | Single continuous flight with the same module schema. |
| `artifacts/data.csv` | combined module capture | Contains multiple flights/segments. Use only through segment-aware scripts. Do not treat it as one flight. |
| `artifacts/2025-01-15 16-46-48.log` | original ArduPilot DataFlash text log | Source for `derived/dataflash/*.csv`. |
| `artifacts/2025-01-15 16-46-48.bin` | original ArduPilot DataFlash binary log | Kept as raw source; current extraction uses the text log. |
| `artifacts/POLI_NA.zip` | external pretrained model | Contains ONNX and TensorFlow exports, but not the original preprocessing spec. |
| `artifacts/photo_*` | project photos | Local supporting material, not required to run experiments. |

## Derived Data

`derived/` is generated and ignored by Git.

| path | role | regeneration |
| --- | --- | --- |
| `derived/dataflash/` | exported DataFlash message CSVs | `python3 src/dataflash_extract.py` |
| `derived/datasets/flight_index.*` | catalog of flights and source segments | `python3 src/build_flight_index.py` |
| `derived/datasets/tracks/` | GPS/POS tracks in local ENU meters | `python3 src/prepare_flight_tracks.py` |
| `derived/datasets/windows_*.npz` | supervised window datasets | `python3 src/build_window_dataset.py` |
| `derived/predictions/` | model predictions and rollout CSVs | individual `src/run_*` and `src/build_*` scripts |

## Generated Visuals

`artifacts/generated/` is generated and ignored by Git. It contains HTML/SVG
viewers, maps, and replay pages. Rebuild it from scripts when needed instead
of committing generated output.

Useful viewers:

- `python3 src/gps_flight_map.py`
- `python3 src/prepare_flight_tracks.py`
- `python3 src/build_track_viewer.py`
- `python3 src/build_trajectory_overlay.py`
- `python3 src/build_imu_dead_reckoning.py`
- `python3 src/build_flow_dead_reckoning.py`
- `PYTHONPATH=/tmp/poli_deps python3 src/run_poli_na_rollout.py`

## Versioned Files

Keep these in Git:

- `src/` experiment and data-processing scripts.
- `README.md`, `docs/`, and `requirements.txt`.
- `reports/` Markdown summaries when they contain interpretation worth keeping.
- `jupyter/` notebooks only when they are small and intentional.

Do not commit:

- raw telemetry files;
- `.venv/`;
- `__pycache__/`;
- generated HTML/SVG/map outputs;
- `.npz` datasets and prediction CSVs.

## Practical Rule

If a file is required to reproduce a result but is too large or external, keep
it under `artifacts/` and document it here. If a file can be recreated from a
script, keep it under `derived/` or `artifacts/generated/`.
