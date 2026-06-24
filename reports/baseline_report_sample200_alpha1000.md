# Baseline report

- sample step: `200` ms
- target horizon: `1000` ms
- train windows: `3967`
- test windows: `4080`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 3972 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 4085 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 2.214 | 1.365 | 0.380 | 3.202 | 4.272 | 2.924 | 1.287 | 5.334 |
| ridge baseline | 3.203 | 14.061 | 2.241 | 14.961 | 4.353 | 14.489 | 2.532 | 15.339 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Zmag2, uT` | 1.59712 |
| `Ymag2, uT` | 0.77808 |
| `Xmag2, uT` | 0.65356 |
| `mag2_norm` | 0.39428 |
| `altbar_minus_lidar_m` | 0.28137 |
| `Lidar, sm` | 0.24222 |
| `lidar_m` | 0.24222 |
| `acc_norm` | 0.16381 |
| `Yflow` | 0.13306 |
| `gyro_norm` | 0.11933 |
| `flow_norm` | 0.10946 |
| `Xgyro, DPS` | 0.08847 |
| `Zgyro, DPS` | 0.07983 |
| `Ygyro, DPS` | 0.07114 |
| `Xacc, g` | 0.06578 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `Lidar, sm` | 0.42689 |
| `lidar_m` | 0.42689 |
| `altbar_minus_lidar_m` | 0.42074 |
| `gyro_norm` | 0.38670 |
| `acc_norm` | 0.36906 |
| `Baro, bar` | 0.31570 |
| `AltBar, m` | 0.31530 |
| `Zmag2, uT` | 0.17981 |
| `Ymag2, uT` | 0.16442 |
| `Zacc, g` | 0.08946 |
| `mag2_norm` | 0.08051 |
| `flow_norm` | 0.04271 |
| `Yflow` | 0.04163 |
| `Xflow` | 0.04001 |
| `Xmag2, uT` | 0.03916 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
