# DataFlash baseline

- feature set: `imu`
- horizon: `1000` ms
- lookback: `1000` ms
- train windows: `7014`
- test windows: `1754`

## Sources

- `derived/dataflash/IMU.csv`
- `derived/dataflash/BARO.csv`

## Metrics

Chronological split inside one DataFlash log. Error is future POS displacement in meters.

| model | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero displacement | 2.008 | 1.323 | 0.403 | 2.839 | 3.814 | 2.697 | 0.742 | 4.730 |
| train mean displacement | 2.281 | 1.401 | 0.415 | 3.029 | 3.936 | 2.723 | 0.748 | 4.844 |
| ridge baseline | 2.960 | 1.276 | 0.206 | 3.430 | 3.753 | 2.144 | 0.278 | 4.331 |

## Feature importance

| feature | score |
| --- | ---: |
| `IMU.AccX_mean` | 0.85142 |
| `IMU.AccZ_mean` | 0.81700 |
| `IMU.AccX_last` | 0.78987 |
| `IMU.T_mean` | 0.78111 |
| `IMU.T_last` | 0.76545 |
| `BARO.Temp_std` | 0.65600 |
| `IMU.AccY_mean` | 0.64995 |
| `IMU.AccZ_last` | 0.60857 |
| `IMU.acc_norm_mean` | 0.47629 |
| `IMU.acc_norm_last` | 0.46507 |
| `IMU.AccX_std` | 0.46143 |
| `IMU.AccY_std` | 0.42133 |
| `BARO.CRt_std` | 0.36360 |
| `BARO.Temp_last` | 0.30009 |
| `BARO.Temp_mean` | 0.29800 |
| `IMU.AccY_last` | 0.29025 |
| `BARO.CRt_mean` | 0.24341 |
| `IMU.AccX_delta` | 0.23857 |
| `IMU.GyrY_std` | 0.23120 |
| `IMU.GyrZ_std` | 0.19289 |
