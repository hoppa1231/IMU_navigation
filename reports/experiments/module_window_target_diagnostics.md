# Module Window Target Diagnostics

This report checks whether the GPS-derived targets contain obvious jumps or startup transients.

## Track-Level Checks

| flight_id | duration | distance | up range | first 5s up delta | max horizontal step | max vertical step | max h speed | max v speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dataflash_2025_01_15` | 879.2 s | 2449.4 m | -4.8..36.5 m | -0.1 m | 2.02 m | 0.63 m | 12.3 m/s | 3.1 m/s |
| `linear_15_01_2025` | 927.3 s | 2328.4 m | -3.8..225.4 m | 1.7 m | 1.04 m | 0.50 m | 10.8 m/s | 5.0 m/s |
| `module_data_s01` | 167.4 s | 31.0 m | -12.2..0.0 m | -0.8 m | 0.39 m | 0.30 m | 3.8 m/s | 3.1 m/s |
| `module_data_s02` | 233.9 s | 28.3 m | -2.6..9.4 m | 0.6 m | 0.12 m | 0.10 m | 1.2 m/s | 1.1 m/s |
| `module_data_s03` | 110.3 s | 39.2 m | -32.0..0.0 m | -0.7 m | 0.61 m | 0.50 m | 8.3 m/s | 5.3 m/s |
| `module_data_s04` | 213.8 s | 39.4 m | 0.0..21.3 m | 18.5 m | 4.56 m | 14.90 m | 253.3 m/s | 827.8 m/s |
| `module_data_s05` | 38.5 s | 4.4 m | -2.7..0.2 m | -0.7 m | 0.06 m | 0.10 m | 0.8 m/s | 1.1 m/s |
| `module_data_s06` | 1064.9 s | 1120.0 m | -32.9..8.0 m | -23.6 m | 3.09 m | 11.10 m | 154.7 m/s | 555.0 m/s |
| `module_data_s07` | 564.0 s | 1827.1 m | -5.1..34.8 m | -0.5 m | 1.27 m | 0.40 m | 14.4 m/s | 4.1 m/s |
| `triangle_15_01_2025` | 956.9 s | 2521.0 m | -36.7..5.4 m | -35.3 m | 3.51 m | 19.50 m | 33.5 m/s | 185.7 m/s |

## Window Target Checks

| dataset | flight_id | windows | target 3D median | target 3D p95 | target 3D max | abs dz p95 | abs dz max | dz>5m | dz>10m | dz>20m | first 5s windows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `windows_module_h1000_l1000` | `linear_15_01_2025` | 9255 | 1.50 m | 10.81 m | 11.12 m | 2.40 m | 3.30 m | 0 | 0 | 0 | 43 |
| `windows_module_h1000_l1000` | `module_data_s01` | 1656 | 0.12 m | 0.61 m | 3.90 m | 0.30 m | 2.50 m | 0 | 0 | 0 | 43 |
| `windows_module_h1000_l1000` | `module_data_s02` | 2322 | 0.12 m | 0.37 m | 0.69 m | 0.20 m | 0.60 m | 0 | 0 | 0 | 43 |
| `windows_module_h1000_l1000` | `module_data_s03` | 1086 | 0.39 m | 1.18 m | 4.90 m | 1.10 m | 4.60 m | 0 | 0 | 0 | 43 |
| `windows_module_h1000_l1000` | `module_data_s04` | 2121 | 0.11 m | 0.52 m | 3.71 m | 0.20 m | 1.70 m | 0 | 0 | 0 | 42 |
| `windows_module_h1000_l1000` | `module_data_s05` | 367 | 0.13 m | 0.40 m | 0.54 m | 0.30 m | 0.50 m | 0 | 0 | 0 | 42 |
| `windows_module_h1000_l1000` | `module_data_s06` | 10632 | 0.14 m | 9.89 m | 11.45 m | 1.20 m | 2.90 m | 0 | 0 | 0 | 42 |
| `windows_module_h1000_l1000` | `module_data_s07` | 5623 | 0.32 m | 10.98 m | 13.92 m | 2.00 m | 4.30 m | 0 | 0 | 0 | 43 |
| `windows_module_h1000_l1000` | `triangle_15_01_2025` | 9551 | 0.83 m | 10.57 m | 34.48 m | 1.60 m | 34.00 m | 5 | 3 | 2 | 42 |
| `windows_module_h3000_l3000` | `linear_15_01_2025` | 9215 | 4.30 m | 30.44 m | 31.30 m | 7.00 m | 7.60 m | 1254 | 0 | 0 | 23 |
| `windows_module_h3000_l3000` | `module_data_s01` | 1616 | 0.33 m | 2.08 m | 7.14 m | 0.90 m | 4.70 m | 0 | 0 | 0 | 23 |
| `windows_module_h3000_l3000` | `module_data_s02` | 2282 | 0.38 m | 1.01 m | 1.39 m | 0.70 m | 1.00 m | 0 | 0 | 0 | 23 |
| `windows_module_h3000_l3000` | `module_data_s03` | 1046 | 1.15 m | 4.05 m | 10.67 m | 3.75 m | 9.90 m | 40 | 0 | 0 | 23 |
| `windows_module_h3000_l3000` | `module_data_s04` | 2081 | 0.23 m | 0.86 m | 2.94 m | 0.50 m | 1.70 m | 0 | 0 | 0 | 22 |
| `windows_module_h3000_l3000` | `module_data_s05` | 327 | 0.43 m | 1.02 m | 1.14 m | 0.80 m | 0.90 m | 0 | 0 | 0 | 22 |
| `windows_module_h3000_l3000` | `module_data_s06` | 10592 | 0.36 m | 29.52 m | 31.99 m | 3.20 m | 7.70 m | 123 | 0 | 0 | 22 |
| `windows_module_h3000_l3000` | `module_data_s07` | 5582 | 0.91 m | 30.94 m | 37.95 m | 5.50 m | 11.70 m | 330 | 43 | 0 | 23 |
| `windows_module_h3000_l3000` | `triangle_15_01_2025` | 9512 | 2.70 m | 30.00 m | 31.08 m | 4.45 m | 7.40 m | 380 | 0 | 0 | 22 |
| `windows_module_h5000_l5000` | `linear_15_01_2025` | 9176 | 7.11 m | 50.18 m | 51.40 m | 11.60 m | 12.30 m | 3488 | 1128 | 0 | 3 |
| `windows_module_h5000_l5000` | `module_data_s01` | 1577 | 0.54 m | 2.79 m | 7.86 m | 1.40 m | 5.20 m | 8 | 0 | 0 | 3 |
| `windows_module_h5000_l5000` | `module_data_s02` | 2241 | 0.60 m | 1.71 m | 1.96 m | 1.00 m | 1.30 m | 0 | 0 | 0 | 3 |
| `windows_module_h5000_l5000` | `module_data_s03` | 1006 | 1.88 m | 5.00 m | 13.83 m | 4.27 m | 12.70 m | 43 | 19 | 0 | 3 |
| `windows_module_h5000_l5000` | `module_data_s04` | 2041 | 0.36 m | 1.08 m | 4.34 m | 0.70 m | 1.30 m | 0 | 0 | 0 | 2 |
| `windows_module_h5000_l5000` | `module_data_s05` | 287 | 0.66 m | 1.41 m | 1.54 m | 1.10 m | 1.20 m | 0 | 0 | 0 | 2 |
| `windows_module_h5000_l5000` | `module_data_s06` | 10552 | 0.52 m | 48.89 m | 52.21 m | 5.00 m | 12.40 m | 522 | 74 | 0 | 2 |
| `windows_module_h5000_l5000` | `module_data_s07` | 5542 | 1.78 m | 50.91 m | 58.91 m | 8.00 m | 18.60 m | 712 | 137 | 0 | 3 |
| `windows_module_h5000_l5000` | `triangle_15_01_2025` | 9472 | 5.10 m | 49.68 m | 51.06 m | 6.90 m | 12.10 m | 948 | 235 | 0 | 2 |

## Interpretation

Suspicious startup/altitude behavior was found:

- `module_data_s04`: first 5s up delta 18.5 m, max vertical speed 827.8 m/s.
- `module_data_s06`: first 5s up delta -23.6 m, max vertical speed 555.0 m/s.
- `triangle_15_01_2025`: first 5s up delta -35.3 m, max vertical speed 185.7 m/s.

Next practical check: rebuild module window datasets with the first 5 seconds of each flight excluded, then rerun baselines.
