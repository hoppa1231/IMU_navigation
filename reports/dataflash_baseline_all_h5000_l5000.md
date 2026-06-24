# DataFlash baseline

- feature set: `all`
- horizon: `5000` ms
- lookback: `5000` ms
- train windows: `6982`
- test windows: `1746`

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
| zero displacement | 9.977 | 6.334 | 1.876 | 13.970 | 18.259 | 12.615 | 3.337 | 22.442 |
| train mean displacement | 11.240 | 6.748 | 1.937 | 14.798 | 18.793 | 12.747 | 3.367 | 22.956 |
| ridge baseline | 8.310 | 3.971 | 1.231 | 10.041 | 12.098 | 6.109 | 1.776 | 13.668 |

## Feature importance

| feature | score |
| --- | ---: |
| `ATT.att_sin_yaw_delta` | 4.85951 |
| `ATT.att_cos_yaw_last` | 3.89692 |
| `ATT.att_cos_yaw_mean` | 3.85173 |
| `ATT.att_sin_yaw_last` | 3.60175 |
| `RCOU_motor_features.motor_diff_c2_c4_mean` | 3.12401 |
| `ATT.att_sin_yaw_mean` | 3.06130 |
| `RCOU_motor_features.motor_diff_c2_c4_std` | 2.89607 |
| `IMU.GyrZ_mean` | 2.58584 |
| `IMU.AccX_mean` | 2.53505 |
| `ATT.Roll_mean` | 2.51698 |
| `ATT.Roll_last` | 2.46856 |
| `IMU.T_delta` | 2.44640 |
| `IMU.AccX_last` | 2.24837 |
| `BARO.Temp_delta` | 2.07828 |
| `IMU.AccX_std` | 2.06319 |
| `IMU.AccZ_mean` | 2.06316 |
| `IMU.GyrZ_last` | 2.03031 |
| `ATT.DesPitch_std` | 2.01282 |
| `BAT.Res_mean` | 1.91345 |
| `BAT.Res_last` | 1.91260 |
