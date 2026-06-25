# Generated Output Layout

`artifacts/generated/` is ignored by Git and is only for local HTML/SVG/map
artifacts. The directory is grouped by purpose so new experiments do not create
another flat list of unrelated folders.

## Groups

| path | contents | typical producer |
| --- | --- | --- |
| `artifacts/generated/gps/flights/` | Per-flight GPS maps, replay pages, SVG tracks, GeoJSON, manifests, and `index.html`. | `src/gps_flight_map.py`, `src/prepare_flight_tracks.py`, `src/build_track_viewer.py` |
| `artifacts/generated/navigation/trajectory_overlay/` | Open-loop displacement rollouts overlaid with real GPS/POS tracks. | `src/build_trajectory_overlay.py` |
| `artifacts/generated/navigation/imu_dead_reckoning/` | Pure IMU dead-reckoning overlay from a shared start point. | `src/build_imu_dead_reckoning.py` |
| `artifacts/generated/navigation/flow_dead_reckoning/` | Optical-flow/IMU open-loop navigation overlay. | `src/build_flow_dead_reckoning.py` |
| `artifacts/generated/navigation/poli_na_rollout/` | POLI_NA open-loop rollout overlay with the real GPS track. | `src/run_poli_na_rollout.py` |
| `artifacts/generated/dataflash/predictions/sweep/` | Static viewer for DataFlash prediction sweep outputs. | `src/build_dataflash_prediction_viewer.py` |
| `artifacts/generated/dataflash/predictions/rolling/` | Static viewer for rolling-validation prediction outputs. | `src/build_dataflash_prediction_viewer.py --pred-dir derived/predictions/dataflash_rolling_validation` |
| `artifacts/generated/dataflash/predictions/sequence*/` | Static viewers for sequence baseline prediction variants. | `src/build_dataflash_prediction_viewer.py` |
| `artifacts/generated/dataflash/rollouts/*/` | Sparse rollout viewers for DataFlash model variants. | `src/build_dataflash_rollout.py` and experiment-specific commands |
| `artifacts/generated/dataflash/final_report/` | Final report generated figures/pages. | report-generation commands |
| `artifacts/generated/diagnostics/` | Exploratory one-off diagnostics such as barometer altitude and baseline trajectory plots. | ad hoc diagnostic scripts/notebooks |
| `artifacts/generated/legacy/` | First-run or pre-refactor generated outputs kept only for comparison. | historical commands |

## Rule For New Outputs

Put new generated visuals under one of the existing groups:

- GPS source inspection: `artifacts/generated/gps/...`
- navigation/drift overlays: `artifacts/generated/navigation/<experiment>/`
- DataFlash prediction viewers: `artifacts/generated/dataflash/predictions/<experiment>/`
- DataFlash rollout viewers: `artifacts/generated/dataflash/rollouts/<experiment>/`
- temporary diagnostics: `artifacts/generated/diagnostics/<topic>/`

If a new category is genuinely needed, add it here and to `docs/data_layout.md`
before adding more generated files.
