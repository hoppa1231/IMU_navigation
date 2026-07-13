# DataFlash Recurrent Models

GRU and LSTM predict 5-second ENU displacement directly. No bias correction or state gating is applied.

- input: `20 x 32` physical feature sequence
- windows: `8684`
- evaluation: three rolling folds inside one DataFlash flight
- seed: `20260713`
- maximum epochs: `100`
- early-stopping patience: `12`

## Overall test metrics

| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |
| --- | --- | ---: | ---: | ---: | ---: |
| `gru_64` | all | 6.862 | 8.524 | 17.516 | 38.650 |
| `lstm_64` | all | 6.462 | 8.713 | 18.562 | 37.761 |

## Per-fold test metrics

| model | fold | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |
| --- | --- | ---: | ---: | ---: | ---: |
| `gru_64` | 1 | 10.022 | 11.079 | 18.994 | 49.160 |
| `lstm_64` | 1 | 7.980 | 10.570 | 24.424 | 63.076 |
| `gru_64` | 2 | 7.191 | 8.963 | 18.966 | 44.422 |
| `lstm_64` | 2 | 7.378 | 9.545 | 21.014 | 20.574 |
| `gru_64` | 3 | 3.371 | 3.860 | 6.845 | 22.369 |
| `lstm_64` | 3 | 4.030 | 4.991 | 9.274 | 29.634 |

## Validation details

- fold 1 gru_64: validation MAE=9.770, best epoch=4, trained epochs=16
- fold 1 lstm_64: validation MAE=7.554, best epoch=2, trained epochs=14
- fold 2 gru_64: validation MAE=5.631, best epoch=14, trained epochs=26
- fold 2 lstm_64: validation MAE=6.246, best epoch=5, trained epochs=17
- fold 3 gru_64: validation MAE=4.098, best epoch=25, trained epochs=37
- fold 3 lstm_64: validation MAE=4.298, best epoch=19, trained epochs=31

GPS/POS is used only as the supervised displacement target and for evaluation. Validation blocks control early stopping; test blocks are not used during training.

## Seed stability check

The complete experiment was repeated with `--seed 20260714`; all other parameters and folds were unchanged.

| model | seed | MAE 3D | RMSE 3D | P95 3D | sparse rollout mean error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gru_64` | 20260713 | 6.862 | 8.524 | 17.516 | 38.650 |
| `gru_64` | 20260714 | 6.318 | 8.008 | 15.972 | 45.721 |
| `lstm_64` | 20260713 | 6.462 | 8.713 | 18.562 | 37.761 |
| `lstm_64` | 20260714 | 6.546 | 8.519 | 18.057 | 28.687 |

The recurrent improvement over the dense and ridge baselines survives the seed change. Rollout remains seed-sensitive, especially on fold 1, so results still need validation on independent flights.
