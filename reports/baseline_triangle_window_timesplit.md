# Baseline report

- sample step: `100` ms
- target horizon: `1000` ms
- lookback window: `1000` ms
- train windows: `4748`
- test windows: `1187`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |
| test | `artifacts/triangle_15_01_2025.csv` | 988550 | 9570 | 5951 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 1.214 | 0.798 | 0.387 | 1.793 | 2.731 | 1.797 | 0.683 | 3.340 |
| ridge baseline | 2.647 | 1.973 | 0.666 | 3.854 | 3.370 | 4.268 | 0.955 | 5.522 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `Xflow_mean` | 0.95636 |
| `Xflow_last` | 0.88298 |
| `Zmag2, uT_mean` | 0.87686 |
| `Yflow_mean` | 0.68935 |
| `mag2_norm_mean` | 0.62596 |
| `Xmag2, uT_mean` | 0.61920 |
| `Zmag2, uT_last` | 0.61701 |
| `Ygyro, DPS_std` | 0.58754 |
| `Xmag2, uT_last` | 0.57539 |
| `Ymag2, uT_last` | 0.57120 |
| `mag2_norm_last` | 0.56202 |
| `Ymag2, uT_mean` | 0.54115 |
| `Yflow_last` | 0.50479 |
| `Zgyro, DPS_std` | 0.47467 |
| `flow_norm_std` | 0.45145 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `acc_norm_mean` | 0.83098 |
| `Yacc, g_std` | 0.81917 |
| `Xacc, g_std` | 0.79130 |
| `Xgyro, DPS_std` | 0.77424 |
| `Zacc, g_std` | 0.70566 |
| `gyro_norm_mean` | 0.69914 |
| `flow_norm_std` | 0.68656 |
| `Yflow_std` | 0.67329 |
| `gyro_norm_std` | 0.65122 |
| `Ygyro, DPS_std` | 0.63260 |
| `acc_norm_last` | 0.58346 |
| `Xflow_std` | 0.53013 |
| `gyro_norm_last` | 0.49299 |
| `Lidar, sm_mean` | 0.46194 |
| `lidar_m_mean` | 0.46194 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
