# GPS flight inventory

This report separates original input files from generated visualization artifacts.

Original GPS-capable files inspected:

- `derived/dataflash/GPS.csv`
- `derived/dataflash/POS.csv`
- `artifacts/linear_15_01_2025.csv`
- `artifacts/triangle_15_01_2025.csv`
- `artifacts/data.csv`

Generated files are written under grouped subdirectories in `artifacts/generated/`.

## Continuous tracks

| source | format | auto segments | points | duration sum | distance 2D sum | altitude | max internal gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `derived/dataflash/GPS.csv` | `dataflash` | 1 | 4394 | 14:39.2 | 2449.4 m | 53.7..94.9 m | 0.600 s |
| `derived/dataflash/POS.csv` | `dataflash` | 1 | 8779 | 14:39.2 | 2444.3 m | 57.1..99.5 m | 1.104 s |
| `artifacts/linear_15_01_2025.csv` | `module` | 1 | 9274 | 15:27.3 | 2328.4 m | 55.2..284.4 m | 0.123 s |
| `artifacts/triangle_15_01_2025.csv` | `module` | 1 | 9570 | 15:56.9 | 2521.0 m | 54.3..96.4 m | 0.124 s |
| `artifacts/data.csv` | `module` | 7 | 23941 | 39:52.9 | 3089.4 m | 49.2..105.0 m | 0.148 s |

## Interpretation

- `derived/dataflash/GPS.csv` and `derived/dataflash/POS.csv` come from one DataFlash log: `artifacts/2025-01-15 16-46-48.log`.
- `artifacts/linear_15_01_2025.csv`, `artifacts/triangle_15_01_2025.csv`, and `artifacts/data.csv` are separate original module CSV recordings.
- `artifacts/data.csv` contains 7 auto-detected segments caused by `TimeStamp` resets. This matches the chat note about 7 flights with coordinates.
- `artifacts/linear_15_01_2025.csv`, `artifacts/triangle_15_01_2025.csv`, and the DataFlash GPS/POS exports each look like one continuous track.
