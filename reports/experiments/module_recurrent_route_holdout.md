# Module recurrent route holdout

GRU and LSTM were trained on chronological sequences of module sensor features and evaluated on routes entirely excluded from training. GPS/GNSS columns are not part of `X`; they are used only to form the ENU displacement target and compute metrics.

- dataset: `derived/datasets/windows_module_h1000_l1000.npz`;
- input features: the `last` and `mean` sensor-window aggregates, 48 channels;
- validation route: `module_data_s07`, excluded from train;
- circle GRU/LSTM: 20 time steps, 6,000 train sequences, early stopping;
- square GRU: 20 time steps, 6,000 train sequences, early stopping;
- square LSTM: a short 10-step / 1,000-sequence run to fit the available CPU execution budget.

| test route | model | sequence length | train sequences | epochs | local MAE 3D m | rollout final horizontal m | rollout mean horizontal m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `circle_07_02_2025` | GRU 64 | 20 | 6,000 | 7 | 5.860 | 1086.794 | 466.887 |
| `circle_07_02_2025` | LSTM 64 | 20 | 6,000 | 6 | 5.704 | 1570.739 | 689.906 |
| `square_07_02_2025` | GRU 64 | 20 | 6,000 | 8 | 3.573 | 363.698 | 207.659 |
| `square_07_02_2025` | LSTM 64 | 10 | 1,000 | 1 | 6.000 | 250.950 | 401.027 |

## Comparison with the previous ridge baseline

At the same 1-second horizon, ridge had local MAE 3D `6.079 m` on circle and `5.827 m` on square. GRU improves that **local** metric, but the accumulated rollout drifts by hundreds of metres to more than a kilometre. Therefore neither GRU nor LSTM is a usable GNSS-free navigation candidate on these route holdouts. The short LSTM run is only an execution check, not a model-selection result.
