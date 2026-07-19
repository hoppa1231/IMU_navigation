# Module recurrent route holdout

GRU and LSTM were trained on chronological sequences of module sensor features and evaluated on routes entirely excluded from training. GPS/GNSS columns are not part of `X`; they are used only to form the ENU displacement target and compute metrics.

- dataset: `derived/datasets/windows_module_h1000_l1000.npz`;
- input features: the `last` and `mean` sensor-window aggregates, 48 channels;
- validation route: `module_data_s07`, excluded from train;
- circle GRU/LSTM: 20 time steps, 6,000 train sequences, early stopping;
- square GRU: 20 time steps, 6,000 train sequences, early stopping;
- square LSTM: a short 10-step / 1,000-sequence run to fit the available CPU execution budget.

| test route | model | sequence length | train sequences | epochs | MAE 3D m | RMSE 3D m | P95 3D m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `circle_07_02_2025` | GRU 64 | 20 | 6,000 | 7 | 5.860 | 7.502 | 15.714 |
| `circle_07_02_2025` | LSTM 64 | 20 | 6,000 | 6 | 5.704 | 7.570 | 16.785 |
| `square_07_02_2025` | GRU 64 | 20 | 6,000 | 8 | 3.573 | 4.765 | 11.051 |
| `square_07_02_2025` | LSTM 64 | 10 | 1,000 | 1 | 6.000 | 7.448 | 11.269 |

## Comparison with the previous ridge baseline

At the same 1-second horizon, ridge had MAE 3D `6.079 m` on circle and `5.827 m` on square. The GRU improves both held-out routes, especially square. The short LSTM run is a valid execution check but not yet a tuned comparison; it needs a longer CPU/GPU run before model selection.
