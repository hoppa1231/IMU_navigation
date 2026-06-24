# DataFlash baseline

- feature set: `imu_att`
- horizon: `5000` ms
- lookback: `5000` ms
- train windows: `6982`
- test windows: `1746`

## Sources

- `derived/dataflash/IMU.csv`
- `derived/dataflash/ATT.csv`
- `derived/dataflash/BARO.csv`

## Metrics

Chronological split inside one DataFlash log. Error is future POS displacement in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 9.977 | 6.334 | 1.876 | 13.970 | 18.259 | 12.615 | 3.337 | 22.442 |
| train mean displacement | 11.240 | 6.748 | 1.937 | 14.798 | 18.793 | 12.747 | 3.367 | 22.956 |
| ridge baseline | 9.605 | 4.102 | 1.199 | 11.345 | 12.879 | 6.239 | 1.744 | 14.417 |

## Feature importance

| feature | score |
| --- | ---: |
| `ATT.att_sin_yaw_delta` | 5.23889 |
| `ATT.att_sin_yaw_last` | 4.67825 |
| `ATT.att_sin_yaw_mean` | 3.98624 |
| `IMU.AccX_mean` | 3.93366 |
| `ATT.att_cos_yaw_last` | 3.73283 |
| `ATT.att_cos_yaw_mean` | 3.66253 |
| `IMU.AccX_last` | 3.21665 |
| `ATT.Roll_last` | 3.02991 |
| `ATT.Roll_mean` | 2.97464 |
| `IMU.GyrZ_mean` | 2.67494 |
| `IMU.T_delta` | 2.50218 |
| `ATT.DesPitch_mean` | 1.97041 |
| `IMU.GyrY_std` | 1.86595 |
| `IMU.GyrZ_last` | 1.82632 |
| `IMU.AccZ_mean` | 1.76002 |
| `BARO.Temp_std` | 1.75154 |
| `IMU.AccX_std` | 1.69816 |
| `IMU.acc_norm_mean` | 1.69808 |
| `ATT.Yaw_std` | 1.69372 |
| `ATT.DesRoll_last` | 1.69108 |
