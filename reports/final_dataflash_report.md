# DataFlash Final Candidate Report

Дата: 2026-06-21.

Этот отчет фиксирует текущий лучший DataFlash-вариант для восстановления будущего смещения по sensor features без использования GPS/POS во входных признаках.

## Данные

Исходный источник:

- `artifacts/2025-01-15 16-46-48.log`
- `artifacts/2025-01-15 16-46-48.bin`

Производный экспорт DataFlash:

- `derived/dataflash/IMU.csv`
- `derived/dataflash/ATT.csv`
- `derived/dataflash/BARO.csv`
- `derived/dataflash/POS.csv`

Важно: `POS.csv` используется только для target и оценки. Во вход модели входят только `IMU`, `ATT`, `BARO`.

## Best Model

Текущий лучший кандидат:

- model: `sequence_ridge_bias_tuned`
- source: DataFlash only
- feature set: `imu_att`
- horizon: `5000 ms`
- lookback: `5000 ms`
- sequence length: `20`
- ridge alpha: `100`
- bias correction: validation residual
- shrink selection: validation MAE
- selected shrink factors: fold 1 = `1.0`, fold 2 = `0.5`, fold 3 = `1.0`

Команда воспроизведения:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --tune-bias-shrink \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100_shrink
```

## Local Displacement Metrics

Rolling validation, 3 folds, 4344 test windows.

| model | MAE 3D | RMSE 3D | P95 3D |
| --- | ---: | ---: | ---: |
| zero | 16.251 | 24.523 | 49.658 |
| train_mean | 16.305 | 24.374 | 49.540 |
| sequence_ridge | 15.316 | 17.307 | 30.754 |
| sequence_ridge_bias_corrected | 14.295 | 16.684 | 29.954 |
| sequence_ridge_bias_tuned | 13.990 | 16.353 | 30.245 |

Вывод: best model улучшает MAE 3D относительно `zero` на `2.261 m`, а RMSE 3D на `8.170 m`.

## Sparse Rollout Metrics

Sparse rollout использует непересекающиеся 5-секундные окна внутри каждого fold. Это не IMU-rate inertial integration, а проверка накопления предсказанных displacement.

| model | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: |
| zero | 87 | 210.141 | 103.898 | 284.380 |
| train_mean | 87 | 210.890 | 97.994 | 235.566 |
| aggregate_ridge | 87 | 80.832 | 116.900 | 354.817 |
| sequence_ridge_fixed_alpha_100 | 87 | 284.451 | 95.471 | 284.451 |
| sequence_ridge_fixed_alpha_100_bias_corrected | 87 | 52.818 | 106.143 | 234.332 |
| sequence_ridge_fixed_alpha_100_bias_tuned | 87 | 52.818 | 93.538 | 187.322 |

Вывод: best model дает лучший final error, mean error и max error среди проверенных rollout-вариантов.

## Fold Rollout

| fold | final error m | mean error m | max error m |
| ---: | ---: | ---: | ---: |
| 1 | 159.646 | 78.312 | 159.646 |
| 2 | 126.784 | 121.245 | 187.322 |
| 3 | 52.818 | 81.058 | 140.646 |

Fold 2 остается самым слабым по mean error. Validation-MAE shrink выбрал `0.5` для fold 2 и снизил max rollout error относительно full bias correction.

## Artifacts

Main reports:

- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md`
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink.md`
- `reports/experiments/dataflash_rollout_summary.md`
- `reports/experiments/dataflash_fold_residuals_sequence_fixed100.md`

Prediction CSV:

- `derived/predictions/dataflash_sequence_fixed100_shrink/imu_att_h5000_l5000_s20/sequence_ridge_bias_tuned_pred.csv`
- `derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv`

HTML viewers:

- `artifacts/generated/dataflash_sequence_fixed100_shrink_predictions/index.html`
- `artifacts/generated/dataflash_rollout_sequence_fixed100_shrink/index.html`

## Limitations

- Оценка сделана на одном DataFlash log.
- Rolling folds находятся внутри одного полета, а не между независимыми полетами.
- Rollout sparse 5-second, не полноценная навигационная интеграция по IMU-rate.
- Bias/shrink correction переносится не одинаково между folds; fold 2 остается зоной риска.

## Next Step

Для финального отчета текущий best model уже можно использовать как основной результат. Следующий технический шаг - сделать одну воспроизводимую команду/pipeline, которая строит:

1. sequence predictions;
2. rollout CSV;
3. rollout HTML;
4. final summary tables.
