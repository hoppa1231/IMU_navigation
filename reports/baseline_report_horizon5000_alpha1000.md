# Baseline report

- sample step: `100` ms
- target horizon: `5000` ms
- train windows: `5770`
- test windows: `5921`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 10.197 | 6.160 | 1.493 | 14.498 | 19.068 | 12.857 | 3.021 | 23.195 |
| ridge baseline | 32.486 | 115.296 | 15.168 | 122.031 | 36.900 | 119.235 | 15.634 | 125.790 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Zmag2, uT` | 8.73730 |
| `Ymag2, uT` | 4.67585 |
| `Xmag2, uT` | 3.26952 |
| `mag2_norm` | 1.69316 |
| `Lidar, sm` | 1.47530 |
| `lidar_m` | 1.47530 |
| `altbar_minus_lidar_m` | 0.79809 |
| `acc_norm` | 0.72487 |
| `gyro_norm` | 0.57824 |
| `Zgyro, DPS` | 0.38850 |
| `AltBar, m` | 0.37625 |
| `Xacc, g` | 0.34672 |
| `Baro, bar` | 0.32044 |
| `Xgyro, DPS` | 0.30793 |
| `flow_norm` | 0.27702 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `lidar_m` | 0.43884 |
| `Lidar, sm` | 0.43884 |
| `gyro_norm` | 0.40402 |
| `acc_norm` | 0.35102 |
| `altbar_minus_lidar_m` | 0.26904 |
| `Zmag2, uT` | 0.23441 |
| `Ymag2, uT` | 0.19305 |
| `Baro, bar` | 0.17596 |
| `Zacc, g` | 0.08158 |
| `mag2_norm` | 0.07184 |
| `AltBar, m` | 0.06981 |
| `Xmag2, uT` | 0.04189 |
| `Zgyro, DPS` | 0.03654 |
| `Xflow` | 0.02317 |
| `flow_norm` | 0.02276 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
