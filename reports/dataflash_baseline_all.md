# DataFlash baseline

- feature set: `all`
- horizon: `1000` ms
- lookback: `1000` ms
- train windows: `7013`
- test windows: `1754`

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
| zero displacement | 2.008 | 1.323 | 0.403 | 2.839 | 3.814 | 2.697 | 0.742 | 4.730 |
| train mean displacement | 2.281 | 1.401 | 0.415 | 3.030 | 3.936 | 2.723 | 0.748 | 4.844 |
| ridge baseline | 2.524 | 1.009 | 0.191 | 2.822 | 4.007 | 1.533 | 0.256 | 4.298 |

## Feature importance

| feature | score |
| --- | ---: |
| `ATT.att_sin_yaw_mean` | 1.07169 |
| `ATT.att_sin_yaw_last` | 1.06703 |
| `ATT.att_cos_yaw_mean` | 0.92644 |
| `ATT.att_cos_yaw_last` | 0.92106 |
| `RCOU_motor_features.motor_diff_c2_c4_mean` | 0.86766 |
| `IMU.AccY_mean` | 0.84559 |
| `IMU.AccX_mean` | 0.81325 |
| `IMU.AccX_last` | 0.75929 |
| `IMU.AccZ_mean` | 0.74182 |
| `IMU.AccX_std` | 0.63416 |
| `RCOU_motor_features.motor_range_mean` | 0.54783 |
| `IMU.AccZ_last` | 0.54518 |
| `IMU.GyrZ_delta` | 0.51212 |
| `BARO.Temp_delta` | 0.49539 |
| `RCOU_motor_features.motor_diff_c2_c4_last` | 0.47650 |
| `RCOU_motor_features.motor_range_last` | 0.45652 |
| `BAT.Res_mean` | 0.42886 |
| `BAT.Res_last` | 0.42776 |
| `BARO.Temp_std` | 0.41670 |
| `IMU.acc_norm_mean` | 0.41076 |
