# Cleanup Candidates

This list separates files that are safe to delete from files that need a manual
decision. Nothing here should be removed automatically unless the raw inputs are
already archived or the experiment is intentionally being discarded.

## Safe To Delete Anytime

These are caches or superseded test outputs. They can be recreated or are no
longer referenced by the default commands.

| path | reason |
| --- | --- |
| `src/__pycache__/` | Python bytecode cache. |
| `reports/poli_na_rollout_seq100.md` | Old sequence-length test report; default POLI_NA report is `reports/poli_na_rollout.md`. |
| `derived/predictions/poli_na_rollout/poli_na_rollout_seq100.csv` | Superseded POLI_NA test output; default output is `poli_na_rollout.csv`. |
| `artifacts/generated/navigation/poli_na_rollout/index_seq100.html` | Superseded POLI_NA HTML from the same sequence-length test. |

## Regenerable Generated Visuals

Delete these when disk space matters or before a clean rerun. They are ignored
by Git and should be reproducible from the corresponding scripts and reports.

| path | reason |
| --- | --- |
| `artifacts/generated/gps/flights/` | GPS maps/replays generated from `derived/datasets/flight_index.csv` and source GPS files. |
| `artifacts/generated/navigation/` | Navigation overlays generated from prediction CSVs and tracks. |
| `artifacts/generated/dataflash/predictions/` | Static viewers generated from `derived/predictions/...`. |
| `artifacts/generated/dataflash/rollouts/` | Rollout viewers generated from rollout CSVs. |
| `artifacts/generated/dataflash/final_report/` | Report visuals; keep only if the rendered report depends on local links. |

## Legacy Or Exploratory Outputs

These look useful for historical comparison only. They are good deletion
candidates once the current reports are accepted.

| path | reason |
| --- | --- |
| `artifacts/generated/legacy/first_run/` | Pre-refactor first-pass outputs with old layout and naming. |
| `artifacts/generated/diagnostics/barometer_altitude/` | One-off diagnostic output, not part of the current navigation pipeline. |
| `artifacts/generated/diagnostics/baseline_trajectory/` | One-off baseline plot output, superseded by grouped navigation overlays. |

## Derived Prediction Sweep Candidates

These are larger generated datasets. Delete only after deciding which
experiment variants are still needed for comparison.

| path | reason |
| --- | --- |
| `derived/predictions/module_window_baselines_debug/` | Debug run output. |
| `derived/predictions/module_window_baselines_xy_base/` | Likely superseded by the current `module_window_baselines_xy` run. |
| `derived/predictions/dataflash_sequence_fixed100_shrink/` | Intermediate sequence variant; compare with the rollout-metric variant before deleting. |
| `derived/predictions/dataflash_sequence_fixed100_shrink_rollout/` | Candidate final variant, but generated and reproducible. Keep only if it is the selected report baseline. |
| `derived/predictions/dataflash_sequence_fixed100_bias/` | Intermediate bias-correction variant; keep only if still referenced in a report. |

## Needs Manual Decision

These are not junk by default.

| path | reason |
| --- | --- |
| `artifacts/data.csv` | Combined raw capture with multiple flights. It is risky as a modeling input, but it is still raw source data. Archive before deleting. |
| `artifacts/2025-01-15 16-46-48.log` and `.bin` | Raw DataFlash sources. Keep at least one canonical raw source for reproducibility. |
| `artifacts/POLI_NA.zip` | External pretrained model; required for POLI_NA tests and not reproducible from repo code. |
| `.venv/` | Rebuildable local environment, but useful while iterating. Delete only when dependencies can be reinstalled. |
| `derived/datasets/windows_*.npz` | Rebuildable, but expensive enough that deletion should be a deliberate cleanup step. |

## Keep

Keep source scripts, raw single-flight CSVs, reports with interpretation, and
the data-layout docs. They are the pieces needed to understand and reproduce
the experiments without relying on old generated HTML.
