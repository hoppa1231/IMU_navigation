# Baseline report

- sample step: `100` ms
- target horizon: `1000` ms
- lookback window: `1000` ms
- train windows: `5781`
- test windows: `5935`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 2.146 | 1.309 | 0.324 | 3.061 | 4.131 | 2.811 | 0.639 | 5.037 |
| ridge baseline | 5.435 | 19.044 | 3.669 | 20.330 | 6.115 | 19.379 | 3.760 | 20.665 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Zmag2, uT_mean` | 1.48701 |
| `Zmag2, uT_last` | 1.07355 |
| `Ygyro, DPS_std` | 0.65388 |
| `Xgyro, DPS_std` | 0.65099 |
| `acc_norm_std` | 0.64386 |
| `Ymag2, uT_mean` | 0.60274 |
| `Ymag2, uT_last` | 0.51554 |
| `Zacc, g_std` | 0.51232 |
| `Zmag2, uT_delta` | 0.48701 |
| `Zgyro, DPS_std` | 0.40855 |
| `Xmag2, uT_mean` | 0.31882 |
| `Xmag2, uT_last` | 0.31321 |
| `acc_norm_mean` | 0.29839 |
| `mag2_norm_last` | 0.29674 |
| `Zmag2, uT_std` | 0.29363 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `Xgyro, DPS_std` | 0.64642 |
| `acc_norm_mean` | 0.63919 |
| `Yacc, g_std` | 0.63774 |
| `gyro_norm_mean` | 0.61140 |
| `Ygyro, DPS_std` | 0.59012 |
| `Zacc, g_std` | 0.52793 |
| `gyro_norm_std` | 0.51962 |
| `Xacc, g_std` | 0.51456 |
| `lidar_m_mean` | 0.42288 |
| `Lidar, sm_mean` | 0.42288 |
| `lidar_m_last` | 0.42187 |
| `Lidar, sm_last` | 0.42187 |
| `gyro_norm_last` | 0.39623 |
| `altbar_minus_lidar_m_mean` | 0.38846 |
| `acc_norm_last` | 0.37357 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
