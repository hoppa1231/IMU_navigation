# Baseline report

- sample step: `100` ms
- target horizon: `5000` ms
- train windows: `4736`
- test windows: `1185`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 5.803 | 3.786 | 1.721 | 8.480 | 12.545 | 8.266 | 2.972 | 15.315 |
| ridge baseline | 10.687 | 8.238 | 2.388 | 15.578 | 14.222 | 14.805 | 3.406 | 20.810 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Zmag2, uT` | 5.23320 |
| `Xmag2, uT` | 4.80636 |
| `Ymag2, uT` | 4.70349 |
| `Xflow` | 4.58485 |
| `mag2_norm` | 4.25469 |
| `Yflow` | 2.35299 |
| `Zgyro, DPS` | 1.82045 |
| `Lidar, sm` | 1.17969 |
| `lidar_m` | 1.17969 |
| `gyro_norm` | 1.16931 |
| `acc_norm` | 0.97770 |
| `flow_norm` | 0.83103 |
| `Xacc, g` | 0.75000 |
| `altbar_minus_lidar_m` | 0.68995 |
| `Zacc, g` | 0.49519 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `acc_norm` | 0.52385 |
| `gyro_norm` | 0.51791 |
| `Lidar, sm` | 0.45344 |
| `lidar_m` | 0.45344 |
| `Zmag2, uT` | 0.40092 |
| `mag2_norm` | 0.31611 |
| `altbar_minus_lidar_m` | 0.29804 |
| `Xmag2, uT` | 0.23040 |
| `Yflow` | 0.20357 |
| `Ymag2, uT` | 0.12589 |
| `Zacc, g` | 0.11056 |
| `Xflow` | 0.07351 |
| `flow_norm` | 0.05220 |
| `Yacc, g` | 0.05216 |
| `Xacc, g` | 0.01754 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
