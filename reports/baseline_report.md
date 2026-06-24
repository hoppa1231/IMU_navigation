# Baseline report

- sample step: `100` ms
- target horizon: `1000` ms
- train windows: `5791`
- test windows: `5945`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 2.144 | 1.312 | 0.353 | 3.086 | 4.127 | 2.813 | 1.100 | 5.115 |
| ridge baseline | 3.280 | 18.512 | 2.808 | 19.404 | 4.315 | 19.492 | 3.015 | 20.190 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `mag2_norm` | 5.91836 |
| `Ymag2, uT` | 5.50409 |
| `Zmag2, uT` | 4.96721 |
| `Xmag2, uT` | 0.72798 |
| `flow_norm` | 0.45009 |
| `Baro, bar` | 0.39326 |
| `lidar_m` | 0.29505 |
| `Lidar, sm` | 0.29505 |
| `AltBar, m` | 0.29126 |
| `Yflow` | 0.28900 |
| `gyro_norm` | 0.19352 |
| `acc_norm` | 0.18223 |
| `Zgyro, DPS` | 0.13552 |
| `Xgyro, DPS` | 0.11626 |
| `Xflow` | 0.10902 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `Lidar, sm` | 0.42034 |
| `lidar_m` | 0.42034 |
| `gyro_norm` | 0.39681 |
| `acc_norm` | 0.37410 |
| `altbar_minus_lidar_m` | 0.25826 |
| `Zmag2, uT` | 0.17627 |
| `Ymag2, uT` | 0.17118 |
| `Baro, bar` | 0.16776 |
| `mag2_norm` | 0.08480 |
| `Zacc, g` | 0.08251 |
| `AltBar, m` | 0.06671 |
| `Xmag2, uT` | 0.04031 |
| `flow_norm` | 0.03910 |
| `Yflow` | 0.03740 |
| `Xflow` | 0.03626 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
