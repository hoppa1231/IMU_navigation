# Flight index

This report is generated from original GPS-capable telemetry and DataFlash exports.

Original / source-like inputs:

- `artifacts/data.csv` - original module CSV, split into 7 flights by `TimeStamp` resets.
- `artifacts/linear_15_01_2025.csv` - original module CSV, one flight.
- `artifacts/triangle_15_01_2025.csv` - original module CSV, one flight.
- `derived/dataflash/*.csv` - exported tables from the original DataFlash log `artifacts/2025-01-15 16-46-48.log`.

Generated outputs from this step:

- `derived/datasets/flight_index.csv`
- `derived/datasets/flight_index.json`
- `reports/navigation/flight_index.md`

No new files are written to the root of `artifacts/`.

## Flights

| flight_id | source | segment | format | GPS points | sensor rows | duration | 2D distance | altitude | rates | feature groups |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `circle_07_02_2025` | `artifacts/initial/circle_07_02_2025.csv` | 1/1 | `module` | 6689 | 668594 | 11:08.8 | 3770.2 m | 56.2..89.0 m | module_rows=999.67 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `dataflash_2025_01_15` | `derived/dataflash` | 1/1 | `dataflash` | 4394 | 87819 | 14:39.2 | 2449.4 m | 53.7..94.9 m | IMU=24.97 Hz<br>ATT=9.99 Hz<br>BARO=9.99 Hz<br>BAT=9.99 Hz<br>MOTB=9.98 Hz<br>RCOU=9.98 Hz<br>RCOU_motor_features=9.98 Hz<br>GPS=5.00 Hz<br>POS=9.98 Hz | IMU:imu_acc,imu_gyro,temperature, ATT:attitude, BARO:barometer, BAT:battery, MOTB:motor_telemetry, RCOU:motor_outputs, RCOU_motor_features:motor_output_features, GPS:gps_target, POS:position_target |
| `linear_15_01_2025` | `artifacts/initial/linear_15_01_2025.csv` | 1/1 | `module` | 9274 | 927041 | 15:27.3 | 2328.4 m | 55.2..284.4 m | module_rows=999.72 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s01` | `artifacts/initial/data.csv` | 1/7 | `module` | 1675 | 167335 | 02:47.4 | 31.0 m | 76.6..88.8 m | module_rows=999.67 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s02` | `artifacts/initial/data.csv` | 2/7 | `module` | 2340 | 233823 | 03:53.9 | 28.3 m | 65.2..77.2 m | module_rows=999.72 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s03` | `artifacts/initial/data.csv` | 3/7 | `module` | 1104 | 110266 | 01:50.3 | 39.2 m | 73.0..105.0 m | module_rows=999.68 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s04` | `artifacts/initial/data.csv` | 4/7 | `module` | 2141 | 213757 | 03:33.8 | 39.4 m | 49.2..70.5 m | module_rows=999.69 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s05` | `artifacts/initial/data.csv` | 5/7 | `module` | 387 | 38538 | 00:38.5 | 4.4 m | 60.8..63.7 m | module_rows=999.74 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s06` | `artifacts/initial/data.csv` | 6/7 | `module` | 10653 | 1064552 | 17:44.9 | 1120.0 m | 57.9..98.8 m | module_rows=999.65 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `module_data_s07` | `artifacts/initial/data.csv` | 7/7 | `module` | 5641 | 563859 | 09:24.0 | 1827.1 m | 59.1..99.0 m | module_rows=999.67 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `square_07_02_2025` | `artifacts/initial/square_07_02_2025.csv` | 1/1 | `module` | 5570 | 556765 | 09:16.9 | 3163.0 m | 54.5..77.8 m | module_rows=999.71 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |
| `triangle_15_01_2025` | `artifacts/initial/triangle_15_01_2025.csv` | 1/1 | `module` | 9570 | 956636 | 15:56.9 | 2521.0 m | 54.3..96.4 m | module_rows=999.69 Hz | imu_acc, imu_gyro, mag1, mag2, optical_flow, barometer, lidar, gps |

## Notes

- `flight_id` is stable and should be used by later dataset, split, model, and prediction scripts.
- GPS/GNSS columns are cataloged as target/reference data, not as input features for GNSS-free navigation.
- `module_data_s01` ... `module_data_s07` are separate flights from `artifacts/data.csv`; they must not be joined into one trajectory.
- `dataflash_2025_01_15` uses `derived/dataflash/GPS.csv` as the GPS reference and catalogs sensor rates from the other exported DataFlash tables.
