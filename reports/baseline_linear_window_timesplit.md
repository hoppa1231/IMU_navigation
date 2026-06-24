# Baseline report

- sample step: `100` ms
- target horizon: `1000` ms
- lookback window: `1000` ms
- train windows: `4624`
- test windows: `1157`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |
| test | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 1.544 | 1.395 | 0.536 | 2.563 | 3.260 | 2.946 | 0.790 | 4.464 |
| ridge baseline | 2.578 | 1.941 | 0.980 | 3.857 | 3.497 | 3.506 | 1.168 | 5.088 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Zmag2, uT_mean` | 1.40550 |
| `Zmag2, uT_last` | 1.06899 |
| `Xgyro, DPS_std` | 0.73693 |
| `Ygyro, DPS_std` | 0.68720 |
| `acc_norm_std` | 0.61100 |
| `Zacc, g_std` | 0.53392 |
| `Zgyro, DPS_std` | 0.49011 |
| `Ymag2, uT_mean` | 0.45156 |
| `Zmag2, uT_delta` | 0.41058 |
| `Xmag2, uT_mean` | 0.39496 |
| `Ymag2, uT_last` | 0.38285 |
| `mag2_norm_last` | 0.36995 |
| `Xmag2, uT_last` | 0.35111 |
| `acc_norm_mean` | 0.34341 |
| `mag2_norm_mean` | 0.34013 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `Ygyro, DPS_std` | 0.65100 |
| `Xgyro, DPS_std` | 0.64952 |
| `gyro_norm_mean` | 0.63756 |
| `Yacc, g_std` | 0.63573 |
| `acc_norm_mean` | 0.63225 |
| `gyro_norm_std` | 0.53951 |
| `Zacc, g_std` | 0.51946 |
| `Lidar, sm_last` | 0.50429 |
| `lidar_m_last` | 0.50429 |
| `Lidar, sm_mean` | 0.50328 |
| `lidar_m_mean` | 0.50328 |
| `Xacc, g_std` | 0.50211 |
| `altbar_minus_lidar_m_mean` | 0.45886 |
| `gyro_norm_last` | 0.41547 |
| `Zgyro, DPS_std` | 0.37021 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
