# DataFlash Rollout Summary

This summary compares sparse non-overlapping 5-second rollouts built from `imu_att h5000_l5000` rolling-validation predictions.

Source prediction directory:

- `derived/predictions/dataflash_rolling_validation/imu_att_h5000_l5000/`

Generated reports:

- `reports/experiments/dataflash_rollout_imu_att_h5000_zero.md`
- `reports/experiments/dataflash_rollout_imu_att_h5000_train_mean.md`
- `reports/experiments/dataflash_rollout_imu_att_h5000.md`
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000.md`
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100.md`
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_bias.md`
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink.md`
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink_rollout.md`

Generated HTML:

- `artifacts/generated/dataflash_rollout_zero/index.html`
- `artifacts/generated/dataflash_rollout_train_mean/index.html`
- `artifacts/generated/dataflash_rollout/index.html`
- `artifacts/generated/dataflash_rollout_sequence/index.html`
- `artifacts/generated/dataflash_rollout_sequence_fixed100/index.html`
- `artifacts/generated/dataflash_rollout_sequence_fixed100_bias/index.html`
- `artifacts/generated/dataflash_rollout_sequence_fixed100_shrink/index.html`
- `artifacts/generated/dataflash_rollout_sequence_fixed100_shrink_rolloutmetric/index.html`

## Overall

| model | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: |
| zero | 87 | 210.141 | 103.898 | 284.380 |
| train_mean | 87 | 210.890 | 97.994 | 235.566 |
| aggregate_ridge | 87 | 80.832 | 116.900 | 354.817 |
| sequence_ridge_per_fold_alpha | 87 | 284.451 | 155.636 | 432.895 |
| sequence_ridge_fixed_alpha_100 | 87 | 284.451 | 95.471 | 284.451 |
| sequence_ridge_fixed_alpha_100_bias_corrected | 87 | 52.818 | 106.143 | 234.332 |
| sequence_ridge_fixed_alpha_100_bias_tuned | 87 | 52.818 | 93.538 | 187.322 |
| sequence_ridge_fixed_alpha_100_bias_tuned_rollout_metric | 87 | 52.818 | 97.233 | 234.332 |

## By Fold

| model | fold | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | 1 | 29 | 171.868 | 67.209 | 171.868 |
| zero | 2 | 29 | 199.676 | 110.147 | 284.380 |
| zero | 3 | 29 | 210.141 | 134.339 | 210.322 |
| train_mean | 1 | 29 | 172.265 | 66.917 | 172.265 |
| train_mean | 2 | 29 | 141.616 | 92.568 | 235.566 |
| train_mean | 3 | 29 | 210.890 | 134.497 | 211.037 |
| aggregate_ridge | 1 | 29 | 115.262 | 57.201 | 132.034 |
| aggregate_ridge | 2 | 29 | 340.972 | 221.168 | 354.817 |
| aggregate_ridge | 3 | 29 | 80.832 | 72.331 | 121.972 |
| sequence_ridge_per_fold_alpha | 1 | 29 | 97.413 | 90.172 | 147.602 |
| sequence_ridge_per_fold_alpha | 2 | 29 | 383.001 | 271.554 | 432.895 |
| sequence_ridge_per_fold_alpha | 3 | 29 | 284.451 | 105.181 | 284.451 |
| sequence_ridge_fixed_alpha_100 | 1 | 29 | 142.498 | 86.454 | 145.696 |
| sequence_ridge_fixed_alpha_100 | 2 | 29 | 113.910 | 94.777 | 148.061 |
| sequence_ridge_fixed_alpha_100 | 3 | 29 | 284.451 | 105.181 | 284.451 |
| sequence_ridge_fixed_alpha_100_bias_corrected | 1 | 29 | 159.646 | 78.312 | 159.646 |
| sequence_ridge_fixed_alpha_100_bias_corrected | 2 | 29 | 190.459 | 159.060 | 234.332 |
| sequence_ridge_fixed_alpha_100_bias_corrected | 3 | 29 | 52.818 | 81.058 | 140.646 |
| sequence_ridge_fixed_alpha_100_bias_tuned | 1 | 29 | 159.646 | 78.312 | 159.646 |
| sequence_ridge_fixed_alpha_100_bias_tuned | 2 | 29 | 126.784 | 121.245 | 187.322 |
| sequence_ridge_fixed_alpha_100_bias_tuned | 3 | 29 | 52.818 | 81.058 | 140.646 |
| sequence_ridge_fixed_alpha_100_bias_tuned_rollout_metric | 1 | 29 | 99.724 | 51.580 | 102.925 |
| sequence_ridge_fixed_alpha_100_bias_tuned_rollout_metric | 2 | 29 | 190.459 | 159.060 | 234.332 |
| sequence_ridge_fixed_alpha_100_bias_tuned_rollout_metric | 3 | 29 | 52.818 | 81.058 | 140.646 |

## Interpretation

The aggregate ridge model improves final rollout error on folds 1 and 3, but fails badly on fold 2. Its overall mean rollout error is worse than `zero` and `train_mean`, even though its rolling-window RMSE/P95 are better for `imu_att h5000_l5000`.

The sequence ridge with per-fold alpha selection is unstable. Using fixed `alpha=100`, chosen from mean rolling-validation sensitivity, improves the overall mean rollout error to `95.471 m`, slightly better than `train_mean`, but fold 3 still drifts badly and final error remains worse than aggregate ridge.

Validation-residual bias correction improves local displacement metrics and final rollout error. It reduces final rollout error to `52.818 m` and max rollout error to `234.332 m`, but mean rollout error is `106.143 m`, still worse than `train_mean`. Fold 2 remains the main weakness.

Validation-tuned bias shrinkage is the best rollout variant so far. It keeps final error at `52.818 m`, improves mean rollout error to `93.538 m`, and reduces max error to `187.322 m`. The selected shrink factors were `1.0` for folds 1 and 3 and `0.5` for fold 2.

Choosing shrink by validation rollout mean improves fold 1 but fails to protect fold 2. Its overall mean rollout error is `97.233 m`, so the current best remains shrink selection by validation MAE.

Conclusion: window-level displacement metrics are not sufficient. Future models must be selected with rollout/drift metrics, not only with `dx/dy/dz` error.
