# DataFlash Neural and Physical Feature Experiment

No test-time bias correction or state gating is applied. GPS/POS is used only as the displacement target.

- windows: `8684`
- sequence: `20` samples over `5000` ms
- raw features: `1080`
- curated physical features: `640` (32 channels)
- evaluation: three rolling folds inside one DataFlash flight

- random seed: `20260712`
- maximum MLP epochs: `250`

## Overall test metrics

| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_ridge` | all | 38.514 | 48.399 | 94.138 | 461.186 |
| `physical_ridge` | all | 16.191 | 18.338 | 32.675 | 185.584 |
| `mlp_64` | all | 12.224 | 14.714 | 27.552 | 77.680 |
| `mlp_128_64` | all | 10.579 | 12.707 | 24.177 | 84.280 |

## Per-fold test metrics

| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_ridge` | 1 | 75.122 | 77.277 | 100.136 | 958.017 |
| `physical_ridge` | 1 | 17.365 | 18.313 | 26.160 | 211.663 |
| `mlp_64` | 1 | 11.979 | 14.034 | 26.456 | 68.555 |
| `mlp_128_64` | 1 | 12.538 | 14.169 | 24.393 | 100.407 |
| `raw_ridge` | 2 | 26.222 | 27.954 | 45.799 | 292.746 |
| `physical_ridge` | 2 | 20.449 | 22.568 | 35.505 | 299.063 |
| `mlp_64` | 2 | 16.022 | 18.406 | 38.984 | 96.557 |
| `mlp_128_64` | 2 | 12.136 | 14.518 | 28.448 | 82.212 |
| `raw_ridge` | 3 | 14.196 | 16.557 | 35.349 | 132.795 |
| `physical_ridge` | 3 | 10.760 | 12.812 | 24.303 | 46.026 |
| `mlp_64` | 3 | 8.673 | 10.666 | 24.904 | 67.926 |
| `mlp_128_64` | 3 | 7.062 | 8.536 | 16.048 | 70.221 |

## Validation details

- fold 1 raw_ridge: validation alpha=1000, MAE=31.577
- fold 1 physical_ridge: validation alpha=1000, MAE=18.345
- fold 1 mlp_64: validation MAE=18.700, epochs=185
- fold 1 mlp_128_64: validation MAE=11.236, epochs=181
- fold 2 raw_ridge: validation alpha=1000, MAE=19.507
- fold 2 physical_ridge: validation alpha=1000, MAE=11.323
- fold 2 mlp_64: validation MAE=12.112, epochs=159
- fold 2 mlp_128_64: validation MAE=11.540, epochs=158
- fold 3 raw_ridge: validation alpha=10, MAE=15.048
- fold 3 physical_ridge: validation alpha=10, MAE=12.845
- fold 3 mlp_64: validation MAE=9.059, epochs=214
- fold 3 mlp_128_64: validation MAE=6.782, epochs=250

## Interpretation

`raw_ridge` uses the original flattened all-sensor sequence. `physical_ridge` uses the same samples after channel selection and explicit physical transforms. The MLP variants use the physical features and predict displacement directly; no residual bias is added afterwards.

The rotation-based linear acceleration assumes ArduPilot body FRD and NED attitude conventions. This is a testable feature hypothesis, not a calibrated INS mechanization.
