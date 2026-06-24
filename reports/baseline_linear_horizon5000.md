# Baseline report

- sample step: `100` ms
- target horizon: `5000` ms
- train windows: `4616`
- test windows: `1154`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |
| test | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 7.358 | 6.663 | 2.550 | 12.253 | 15.323 | 13.846 | 3.653 | 20.973 |
| ridge baseline | 9.972 | 8.649 | 3.619 | 15.597 | 14.134 | 13.621 | 4.540 | 20.147 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Zmag2, uT` | 9.62553 |
| `Xmag2, uT` | 3.90537 |
| `Ymag2, uT` | 2.67655 |
| `mag2_norm` | 2.08940 |
| `acc_norm` | 1.67206 |
| `gyro_norm` | 1.36066 |
| `lidar_m` | 1.25847 |
| `Lidar, sm` | 1.25847 |
| `Baro, bar` | 0.69487 |
| `altbar_minus_lidar_m` | 0.67814 |
| `Zacc, g` | 0.58280 |
| `Xgyro, DPS` | 0.36853 |
| `AltBar, m` | 0.32283 |
| `Zgyro, DPS` | 0.26172 |
| `Xacc, g` | 0.24348 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `lidar_m` | 0.51850 |
| `Lidar, sm` | 0.51850 |
| `gyro_norm` | 0.41879 |
| `acc_norm` | 0.34980 |
| `Zmag2, uT` | 0.34870 |
| `altbar_minus_lidar_m` | 0.29291 |
| `Ymag2, uT` | 0.23031 |
| `Baro, bar` | 0.20811 |
| `AltBar, m` | 0.08190 |
| `Zacc, g` | 0.07479 |
| `Zgyro, DPS` | 0.03000 |
| `Ygyro, DPS` | 0.02722 |
| `mag2_norm` | 0.01785 |
| `Xgyro, DPS` | 0.01362 |
| `Yacc, g` | 0.00371 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
