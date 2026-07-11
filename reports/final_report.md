# Final Navigation Analysis

Date: 2026-07-11.

This report summarizes the current state of the project after the GPS track
preparation, open-loop navigation experiments, and DataFlash model search.

## Scope

The project currently contains four different navigation-style experiment
families:

- pure IMU dead reckoning from a shared start point;
- module-CSV optical-flow dead reckoning from a shared start point;
- POLI_NA rollout from a shared start point;
- DataFlash displacement prediction with sparse rollout accumulation.

The real GPS or POS trajectory is always kept as the reference trajectory for
comparison. None of the evaluated models uses GPS/POS as an online correction
signal during rollout.

## Important Comparison Note

These experiments are not perfectly identical:

- IMU dead reckoning uses DataFlash `IMU + ATT + POS` and integrates
  acceleration at IMU rate.
- Flow and POLI_NA use separate module CSV flights with GPS as the target.
- DataFlash best model predicts sparse 5-second displacements on rolling folds
  inside one DataFlash log. It is the strongest model result in the repo, but
  it is not yet a full IMU-rate inertial navigation pipeline.

Because of that, the table below should be read as a practical engineering
comparison, not as a strict benchmark.

## Summary Table

| method | source | reference flights | final error m | mean error m | max error m | key note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pure IMU dead reckoning | DataFlash `IMU + ATT` | one DataFlash flight | 37687.296 | 12384.697 | 37687.296 | classic double-integration drift |
| flow dead reckoning, best case | module CSV, `flow_only` | `linear_15_01_2025` | 40.762 | 308.016 | 468.326 | best final point among module open-loop runs |
| flow dead reckoning, second best | module CSV, `flow_only` | `triangle_15_01_2025` | 106.120 | 119.176 | 293.163 | lower mean drift than linear, worse final error |
| POLI_NA, best case | module CSV, `imu_flow_mag10_raw` | `linear_15_01_2025` | 286.447 | 236.889 | 466.818 | likely limited by missing original preprocessing |
| POLI_NA, second best | module CSV, `imu_flow_mag10_raw` | `triangle_15_01_2025` | 365.935 | 181.344 | 365.935 | same preprocessing uncertainty |
| DataFlash best rollout | DataFlash `IMU + ATT + BARO` | 3 rolling folds, sparse 5 s rollout | 52.818 | 93.538 | 187.322 | strongest overall learned displacement model |

## Best Current Result

The current best model in the repository is:

- model: `sequence_ridge_bias_tuned`
- source: DataFlash only
- features: `imu_att`
- horizon: `5000 ms`
- lookback: `5000 ms`
- sequence length: `20`
- ridge alpha: `100`
- bias correction: validation residual
- shrink selection: validation MAE

Its current metrics are:

- local displacement MAE 3D: `13.990 m`
- local displacement RMSE 3D: `16.353 m`
- sparse rollout final error: `52.818 m`
- sparse rollout mean error: `93.538 m`
- sparse rollout max error: `187.322 m`

This is the best practical result because it improves both local window
accuracy and accumulated rollout drift over the checked DataFlash alternatives.

## Method-Level Conclusions

### Pure IMU

Pure IMU dead reckoning is not viable in the current setup. Even after gravity
compensation and initial bias calibration, the position drift grows to tens of
kilometers. This confirms that low-cost IMU double integration without strong
external constraints is unusable for the target task.

### Optical Flow

Optical flow gives a meaningful constraint compared with pure IMU, but the
result is unstable across flights. The best final-point result is good enough
to show that short-term motion information exists in the sensors, but the large
mean error shows that open-loop drift still accumulates strongly over time.

### POLI_NA

POLI_NA does not beat the best flow baseline. The main limitation is not
necessarily the network itself, but the missing preprocessing specification in
`artifacts/POLI_NA.zip`. Without the original channel order and normalization,
its evaluation remains provisional.

### DataFlash Best Model

The DataFlash sequence model is the most convincing result in the project. It
beats naive baselines on local displacement and gives the best sparse-rollout
behavior among the checked DataFlash variants. Its main weakness is that the
evaluation still happens inside one log with rolling folds, not across multiple
independent flights.

## What Is Proven Now

The current repository already supports these claims:

- real GPS/POS reference trajectories are prepared and plotted next to the
  predicted trajectories;
- pure inertial double integration fails badly on this hardware/data quality;
- adding learned motion constraints from flow or DataFlash features is
  necessary;
- the best current candidate is not POLI_NA, but the DataFlash
  `sequence_ridge_bias_tuned` pipeline.

## What Is Still Missing

The current repository does not yet prove these stronger claims:

- generalization across many independent flights;
- a full GNSS-free navigation stack that integrates at sensor rate for the
  whole flight;
- fair apples-to-apples comparison between module CSV experiments and DataFlash
  experiments on the exact same raw trajectory source.

## Next Practical Step

The reproducible best-DataFlash pipeline now exists:

```bash
python3 src/run_best_dataflash_pipeline.py
```

The next analysis task is to build a single comparison page that places
GPS/POS, flow, POLI_NA, and DataFlash trajectories side by side for
presentation.
