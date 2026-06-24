# Baseline report

- sample step: `100` ms
- target horizon: `1000` ms
- train windows: `4756`
- test windows: `1189`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 1.213 | 0.798 | 0.387 | 1.791 | 2.729 | 1.796 | 0.683 | 3.337 |
| ridge baseline | 2.627 | 2.327 | 0.515 | 4.102 | 3.369 | 4.690 | 0.764 | 5.825 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Ymag2, uT` | 4.58394 |
| `mag2_norm` | 4.32138 |
| `Xflow` | 4.04799 |
| `Zmag2, uT` | 3.34051 |
| `flow_norm` | 2.30897 |
| `Yflow` | 2.01805 |
| `Baro, bar` | 1.86800 |
| `AltBar, m` | 1.17337 |
| `altbar_minus_lidar_m` | 0.90748 |
| `Xmag2, uT` | 0.68218 |
| `gyro_norm` | 0.45172 |
| `Zgyro, DPS` | 0.29213 |
| `Xacc, g` | 0.22312 |
| `Ygyro, DPS` | 0.17239 |
| `Zacc, g` | 0.16869 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `acc_norm` | 0.56490 |
| `gyro_norm` | 0.47357 |
| `Lidar, sm` | 0.42741 |
| `lidar_m` | 0.42741 |
| `Zmag2, uT` | 0.27907 |
| `altbar_minus_lidar_m` | 0.27745 |
| `mag2_norm` | 0.23380 |
| `Yflow` | 0.21532 |
| `Xmag2, uT` | 0.15929 |
| `Zacc, g` | 0.12515 |
| `Ymag2, uT` | 0.11246 |
| `flow_norm` | 0.09003 |
| `Xflow` | 0.06327 |
| `Yacc, g` | 0.05540 |
| `Xgyro, DPS` | 0.03378 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
