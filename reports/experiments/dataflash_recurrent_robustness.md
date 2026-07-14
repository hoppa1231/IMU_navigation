# DataFlash Recurrent Robustness

Strict within-flight check with a purged rolling split, five random seeds, and feature ablations.

- lookback/horizon: `5000/5000 ms`
- purge between split anchors: `10000 ms`
- seeds: `20260713, 20260714, 20260715, 20260716, 20260717`
- full physical channels: `32`
- no test-time bias correction, clipping, or state gating

The purge removes the last 10 seconds of train and validation blocks. Therefore the complete sensor/target interval of a retained window does not overlap the next split.

## LSTM feature ablation

| feature set | MAE 3D, m | rollout mean, m | rollout final, m | total test POS distance, m | mean fold final error per km, m/km |
| --- | ---: | ---: | ---: | ---: | ---: |
| `imu_raw` | 7.293 +/- 0.593 | 39.820 +/- 2.942 | 49.468 +/- 9.605 | 1335.5 | 115.7 +/- 30.0 |
| `imu_engineered` | 7.362 +/- 0.914 | 44.573 +/- 5.321 | 49.744 +/- 12.422 | 1335.5 | 109.6 +/- 28.7 |
| `imu_att` | 5.664 +/- 1.599 | 29.569 +/- 16.767 | 59.118 +/- 31.240 | 1335.5 | 125.7 +/- 60.1 |
| `imu_att_crt` | 4.510 +/- 0.296 | 23.796 +/- 2.968 | 37.512 +/- 2.561 | 1335.5 | 81.4 +/- 7.7 |
| `imu_att_baro` | 5.493 +/- 1.811 | 40.543 +/- 18.430 | 77.021 +/- 37.825 | 1335.5 | 155.7 +/- 66.8 |
| `imu_att_crt_motors` | 5.014 +/- 0.686 | 26.738 +/- 6.235 | 39.971 +/- 7.886 | 1335.5 | 87.6 +/- 20.1 |
| `all_direct` | 9.056 +/- 1.241 | 58.775 +/- 13.148 | 96.899 +/- 13.801 | 1335.5 | 218.1 +/- 18.6 |
| `all_physical` | 6.953 +/- 0.652 | 41.079 +/- 9.985 | 69.479 +/- 16.689 | 1335.5 | 157.3 +/- 45.7 |

## Recurrent comparison

| model/features | MAE 3D, m | rollout mean, m | rollout final, m | total test POS distance, m | mean fold final error per km, m/km |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lstm_64/imu_att_crt` | 4.510 +/- 0.296 | 23.796 +/- 2.968 | 37.512 +/- 2.561 | 1335.5 | 81.4 +/- 7.7 |
| `gru_64/imu_att_crt` | 4.665 +/- 0.190 | 26.921 +/- 4.851 | 45.329 +/- 10.735 | 1335.5 | 99.5 +/- 24.7 |
| `lstm_64/all_physical` | 6.953 +/- 0.652 | 41.079 +/- 9.985 | 69.479 +/- 16.689 | 1335.5 | 157.3 +/- 45.7 |
| `gru_64/all_physical` | 6.507 +/- 0.404 | 36.379 +/- 6.127 | 70.331 +/- 20.932 | 1335.5 | 164.2 +/- 62.4 |

Values are mean +/- sample standard deviation across seeds. Rollout values first aggregate the three independent folds within a seed.

## Purged split sizes

| fold | train rows | validation rows | test rows | test index range |
| ---: | ---: | ---: | ---: | --- |
| 1 | 2797 | 1348 | 1447 | 4343-5789 |
| 2 | 4244 | 1347 | 1447 | 5790-7236 |
| 3 | 5690 | 1348 | 1447 | 7237-8683 |

## Feature sets

- `imu_raw`: accelerometer and gyroscope axes only.
- `imu_engineered`: IMU plus norms, jerk norm, and angular-acceleration norm.
- `imu_att`: engineered IMU plus attitude and gravity-compensated ENU acceleration.
- `imu_att_crt`: adds barometric climb rate without absolute altitude.
- `imu_att_baro`: adds barometric altitude and climb rate.
- `imu_att_crt_motors`: adds climb rate and selected motor channels/interactions, without battery or absolute altitude.
- `all_direct`: adds direct battery and motor channels, without interaction features.
- `all_physical`: adds thrust interactions and all derived channels.

Distance is the dense POS path length over the three test blocks. It can include EKF/GPS jitter and is not directly comparable to a separately defined 3 km benchmark.

This is still validation inside one flight. A holdout DataFlash flight remains required before claiming cross-flight generalization.
