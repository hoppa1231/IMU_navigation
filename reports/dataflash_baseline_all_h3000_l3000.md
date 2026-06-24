# DataFlash baseline

- feature set: `all`
- horizon: `3000` ms
- lookback: `3000` ms
- train windows: `6998`
- test windows: `1750`

## Sources

- `derived/dataflash/IMU.csv`
- `derived/dataflash/ATT.csv`
- `derived/dataflash/BARO.csv`
- `derived/dataflash/BAT.csv`
- `derived/dataflash/MOTB.csv`
- `derived/dataflash/RCOU_motor_features.csv`

## Metrics

Chronological split inside one DataFlash log. Error is future POS displacement in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 5.938 | 3.836 | 1.150 | 8.354 | 11.136 | 7.785 | 2.089 | 13.747 |
| train mean displacement | 6.722 | 4.082 | 1.185 | 8.884 | 11.472 | 7.862 | 2.107 | 14.066 |
| ridge baseline | 6.076 | 2.217 | 0.618 | 6.847 | 8.481 | 3.261 | 0.855 | 9.127 |

## Feature importance

| feature | score |
| --- | ---: |
| `ATT.att_sin_yaw_mean` | 2.76056 |
| `ATT.att_sin_yaw_last` | 2.73702 |
| `ATT.att_cos_yaw_mean` | 2.73204 |
| `ATT.att_cos_yaw_last` | 2.60928 |
| `RCOU_motor_features.motor_diff_c2_c4_mean` | 2.19523 |
| `IMU.AccZ_mean` | 2.04752 |
| `IMU.AccX_std` | 2.00773 |
| `IMU.AccX_mean` | 1.98875 |
| `BARO.Temp_delta` | 1.80454 |
| `IMU.AccX_last` | 1.75615 |
| `ATT.att_sin_yaw_delta` | 1.64484 |
| `IMU.AccY_mean` | 1.53740 |
| `ATT.Roll_last` | 1.41512 |
| `RCOU_motor_features.motor_range_last` | 1.40331 |
| `RCOU_motor_features.motor_range_mean` | 1.39164 |
| `IMU.AccZ_last` | 1.31265 |
| `IMU.GyrZ_last` | 1.24499 |
| `BAT.Res_last` | 1.23150 |
| `BAT.Res_mean` | 1.21476 |
| `ATT.Roll_mean` | 1.21331 |
