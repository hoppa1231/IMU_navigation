# DataFlash baseline

- feature set: `imu_att`
- horizon: `1000` ms
- lookback: `1000` ms
- train windows: `7014`
- test windows: `1754`

## Sources

- `derived/dataflash/IMU.csv`
- `derived/dataflash/ATT.csv`
- `derived/dataflash/BARO.csv`

## Metrics

Chronological split inside one DataFlash log. Error is future POS displacement in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 2.008 | 1.323 | 0.403 | 2.839 | 3.814 | 2.697 | 0.742 | 4.730 |
| train mean displacement | 2.281 | 1.401 | 0.415 | 3.029 | 3.936 | 2.723 | 0.748 | 4.844 |
| ridge baseline | 2.640 | 1.138 | 0.205 | 3.007 | 3.846 | 1.724 | 0.277 | 4.223 |

## Feature importance

| feature | score |
| --- | ---: |
| `ATT.att_sin_yaw_last` | 1.27349 |
| `ATT.att_sin_yaw_mean` | 1.27138 |
| `IMU.AccX_mean` | 1.02224 |
| `IMU.AccX_last` | 0.93570 |
| `ATT.att_cos_yaw_mean` | 0.91742 |
| `ATT.att_cos_yaw_last` | 0.91036 |
| `IMU.AccZ_mean` | 0.76688 |
| `IMU.AccY_mean` | 0.70142 |
| `BARO.Temp_std` | 0.57567 |
| `IMU.AccZ_last` | 0.56752 |
| `IMU.acc_norm_mean` | 0.43681 |
| `IMU.acc_norm_last` | 0.41547 |
| `ATT.Roll_last` | 0.40548 |
| `ATT.Roll_mean` | 0.39980 |
| `IMU.AccX_std` | 0.39605 |
| `BARO.Temp_delta` | 0.32376 |
| `ATT.DesPitch_std` | 0.30663 |
| `IMU.AccY_last` | 0.30556 |
| `IMU.GyrX_std` | 0.26651 |
| `IMU.AccX_delta` | 0.26005 |
