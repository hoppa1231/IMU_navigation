# Baseline report

- sample step: `100` ms
- target horizon: `1000` ms
- train windows: `4632`
- test windows: `1159`

## Files

| split | file | rows seen | rows with GPS | sampled GPS points |
| --- | --- | ---: | ---: | ---: |
| train | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |
| test | `artifacts/linear_15_01_2025.csv` | 954529 | 9274 | 5799 |

## Metrics

Error is measured for GPS displacement over the selected horizon, in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 1.542 | 1.393 | 0.537 | 2.561 | 3.257 | 2.943 | 0.792 | 4.461 |
| ridge baseline | 2.163 | 2.186 | 0.853 | 3.812 | 3.018 | 4.379 | 1.063 | 5.424 |

## Feature importance

Importance is the norm of standardized ridge weights over `dx/dy/dz`.

| feature | score |
| --- | ---: |
| `mag2_norm` | 7.06071 |
| `Zmag2, uT` | 6.49202 |
| `Ymag2, uT` | 4.90055 |
| `Xmag2, uT` | 0.85724 |
| `Baro, bar` | 0.61548 |
| `acc_norm` | 0.37773 |
| `AltBar, m` | 0.36314 |
| `gyro_norm` | 0.34533 |
| `lidar_m` | 0.33890 |
| `Lidar, sm` | 0.33890 |
| `altbar_minus_lidar_m` | 0.23881 |
| `Zacc, g` | 0.16375 |
| `Xgyro, DPS` | 0.11713 |
| `Zgyro, DPS` | 0.11462 |
| `Xacc, g` | 0.09598 |

## Simple correlations

Absolute Pearson correlation with the 3D target displacement length on train data.

| feature | abs corr |
| --- | ---: |
| `lidar_m` | 0.50200 |
| `Lidar, sm` | 0.50200 |
| `gyro_norm` | 0.41614 |
| `acc_norm` | 0.36950 |
| `Zmag2, uT` | 0.28883 |
| `altbar_minus_lidar_m` | 0.28441 |
| `Ymag2, uT` | 0.20757 |
| `Baro, bar` | 0.19985 |
| `AltBar, m` | 0.07876 |
| `Zacc, g` | 0.07618 |
| `mag2_norm` | 0.04122 |
| `Ygyro, DPS` | 0.02959 |
| `Zgyro, DPS` | 0.02446 |
| `Xgyro, DPS` | 0.00872 |
| `Xacc, g` | 0.00805 |

## Interpretation

- This is a first sanity-check, not the final navigation model.
- If ridge beats the zero-displacement baseline on a holdout flight, the input features contain useful movement signal.
- The next step is to add window features and motor features from the ArduPilot log, then compare LSTM/GRU/1D-CNN.
