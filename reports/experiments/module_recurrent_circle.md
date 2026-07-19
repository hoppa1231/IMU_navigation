# Module recurrent route holdout

GRU/LSTM use chronological sensor-window vectors. GPS is absent from X and used only as ENU target.

- dataset: `derived/datasets/windows_module_h1000_l1000.npz`
- sequence length: 20
- feature mode: `last_mean` (48 channels)
- validation route: `module_data_s07` (excluded from train)

| test route | model | train sequences | validation | test | epochs | MAE 3D m | RMSE 3D m | P95 3D m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `circle_07_02_2025` | `gru_64` | 6000 | 5604 | 6652 | 7 | 5.860 | 7.502 | 15.714 |
| `circle_07_02_2025` | `lstm_64` | 6000 | 5604 | 6652 | 6 | 5.704 | 7.570 | 16.789 |
