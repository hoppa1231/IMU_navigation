#!/usr/bin/env python3
"""Run the reproducible best DataFlash pipeline and build summary artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BEST_CASE = "imu_att_h5000_l5000_s20"
BEST_MODEL = "sequence_ridge_bias_tuned"
ROLLUP_MODELS = [
    "zero",
    "train_mean",
    "sequence_ridge",
    "sequence_ridge_bias_corrected",
    "sequence_ridge_bias_tuned",
]
MODEL_LABELS = {
    "zero": "zero",
    "train_mean": "train_mean",
    "sequence_ridge": "sequence_ridge_fixed_alpha_100",
    "sequence_ridge_bias_corrected": "sequence_ridge_fixed_alpha_100_bias_corrected",
    "sequence_ridge_bias_tuned": "sequence_ridge_fixed_alpha_100_bias_tuned",
}
MODEL_REPORT_NAMES = {
    "zero": "dataflash_rollout_imu_att_h5000_zero.md",
    "train_mean": "dataflash_rollout_imu_att_h5000_train_mean.md",
    "sequence_ridge": "dataflash_rollout_sequence_imu_att_h5000_fixed100.md",
    "sequence_ridge_bias_corrected": "dataflash_rollout_sequence_imu_att_h5000_fixed100_bias.md",
    "sequence_ridge_bias_tuned": "dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink.md",
}
MODEL_ROLLOUT_NAMES = {
    "zero": "imu_att_h5000_l5000_zero_rollout.csv",
    "train_mean": "imu_att_h5000_l5000_train_mean_rollout.csv",
    "sequence_ridge": "imu_att_h5000_l5000_sequence_fixed100_rollout.csv",
    "sequence_ridge_bias_corrected": "imu_att_h5000_l5000_sequence_fixed100_bias_rollout.csv",
    "sequence_ridge_bias_tuned": "imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv",
}
MODEL_HTML_DIRS = {
    "zero": "zero",
    "train_mean": "train_mean",
    "sequence_ridge": "sequence_fixed100",
    "sequence_ridge_bias_corrected": "sequence_fixed100_bias",
    "sequence_ridge_bias_tuned": "sequence_fixed100_shrink",
}


@dataclass
class LocalMetrics:
    mae_3d: float
    rmse_3d: float
    p95_3d: float
    count: int


@dataclass
class RolloutMetrics:
    steps: int
    final_error_m: float
    mean_error_m: float
    max_error_m: float
    by_fold: dict[str, tuple[int, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("derived/dataflash"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/dataflash_sequence_fixed100_shrink"))
    parser.add_argument(
        "--sequence-report",
        type=Path,
        default=Path("reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md"),
    )
    parser.add_argument("--rollout-dir", type=Path, default=Path("derived/predictions/dataflash_rollout"))
    parser.add_argument("--rollout-summary", type=Path, default=Path("reports/experiments/dataflash_rollout_summary.md"))
    parser.add_argument("--final-report", type=Path, default=Path("reports/final_dataflash_report.md"))
    parser.add_argument(
        "--prediction-viewer",
        type=Path,
        default=Path("artifacts/generated/dataflash/predictions/sequence_fixed100_shrink/index.html"),
    )
    parser.add_argument("--final-html", type=Path, default=Path("artifacts/generated/dataflash/final_report/index.html"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def run_step(python_bin: Path, script_name: str, *args: str) -> None:
    script = Path(__file__).with_name(script_name)
    cmd = [str(python_bin), str(script), *args]
    subprocess.run(cmd, check=True)


def as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    index = int(fraction * (len(sorted_values) - 1))
    return sorted_values[index]


def compute_local_metrics(path: Path) -> LocalMetrics:
    errors: list[float] = []
    for row in read_csv_rows(path):
        dx = as_float(row["pred_dx_east_m"]) - as_float(row["true_dx_east_m"])
        dy = as_float(row["pred_dy_north_m"]) - as_float(row["true_dy_north_m"])
        dz = as_float(row["pred_dz_up_m"]) - as_float(row["true_dz_up_m"])
        errors.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    if not errors:
        raise ValueError(f"No rows in {path}")
    mean_sq = sum(value * value for value in errors) / len(errors)
    return LocalMetrics(
        mae_3d=sum(errors) / len(errors),
        rmse_3d=math.sqrt(mean_sq),
        p95_3d=percentile(sorted(errors), 0.95),
        count=len(errors),
    )


def compute_rollout_metrics(path: Path) -> RolloutMetrics:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")

    all_errors = [as_float(row["err_3d_m"]) for row in rows]
    by_fold_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_fold_rows.setdefault(row["fold_id"], []).append(row)

    by_fold: dict[str, tuple[int, float, float, float]] = {}
    for fold_id, fold_rows in sorted(by_fold_rows.items(), key=lambda item: int(item[0])):
        fold_errors = [as_float(row["err_3d_m"]) for row in fold_rows]
        by_fold[fold_id] = (
            len(fold_errors),
            fold_errors[-1],
            sum(fold_errors) / len(fold_errors),
            max(fold_errors),
        )

    return RolloutMetrics(
        steps=len(rows),
        final_error_m=all_errors[-1],
        mean_error_m=sum(all_errors) / len(all_errors),
        max_error_m=max(all_errors),
        by_fold=by_fold,
    )


def write_rollout_summary(
    path: Path,
    model_rollouts: dict[str, RolloutMetrics],
    rollout_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash Rollout Summary",
        "",
        "This summary is generated by `src/run_best_dataflash_pipeline.py`.",
        "",
        f"Rollout directory: `{rollout_dir}`",
        "",
        "## Overall",
        "",
        "| model | steps | final error m | mean error m | max error m |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ROLLUP_MODELS:
        metrics = model_rollouts[model_name]
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {metrics.steps} | {metrics.final_error_m:.3f} | "
            f"{metrics.mean_error_m:.3f} | {metrics.max_error_m:.3f} |"
        )
    lines.extend(["", "## By Fold", "", "| model | fold | steps | final error m | mean error m | max error m |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for model_name in ROLLUP_MODELS:
        for fold_id, values in model_rollouts[model_name].by_fold.items():
            steps, final_error, mean_error, max_error = values
            lines.append(
                f"| {MODEL_LABELS[model_name]} | {fold_id} | {steps} | {final_error:.3f} | {mean_error:.3f} | {max_error:.3f} |"
            )
    best = model_rollouts[BEST_MODEL]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current best rollout variant remains `sequence_ridge_bias_tuned` with fixed `alpha=100` and validation-tuned bias shrinkage.",
            f"It keeps final error at `{best.final_error_m:.3f} m`, mean rollout error at `{best.mean_error_m:.3f} m`, and max rollout error at `{best.max_error_m:.3f} m`.",
            "Conclusion: local displacement metrics alone are not enough; rollout drift must stay in the model-selection loop.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    path: Path,
    pred_dir: Path,
    sequence_report: Path,
    rollout_summary: Path,
    final_html: Path,
    local_metrics: dict[str, LocalMetrics],
    model_rollouts: dict[str, RolloutMetrics],
) -> None:
    best_local = local_metrics[BEST_MODEL]
    best_rollout = model_rollouts[BEST_MODEL]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataFlash Final Candidate Report",
        "",
        "Date: 2026-07-11.",
        "",
        "This report is generated by `src/run_best_dataflash_pipeline.py`.",
        "",
        "## Best Model",
        "",
        "Current best candidate:",
        "",
        "- model: `sequence_ridge_bias_tuned`",
        "- source: DataFlash only",
        "- feature set: `imu_att`",
        "- horizon: `5000 ms`",
        "- lookback: `5000 ms`",
        "- sequence length: `20`",
        "- ridge alpha: `100`",
        "- bias correction: validation residual",
        "- shrink selection: validation MAE",
        "",
        "Reproduction command:",
        "",
        "```bash",
        "python3 src/run_best_dataflash_pipeline.py",
        "```",
        "",
        "## Local Displacement Metrics",
        "",
        f"Prediction directory: `{pred_dir}`",
        f"Sequence report: `{sequence_report}`",
        "",
        "| model | windows | MAE 3D | RMSE 3D | P95 3D |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ROLLUP_MODELS:
        metrics = local_metrics[model_name]
        lines.append(
            f"| `{model_name}` | {metrics.count} | {metrics.mae_3d:.3f} | {metrics.rmse_3d:.3f} | {metrics.p95_3d:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Sparse Rollout Metrics",
            "",
            f"Rollout summary: `{rollout_summary}`",
            f"Final HTML dashboard: `{final_html}`",
            "",
            "| model | steps | final error m | mean error m | max error m |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name in ROLLUP_MODELS:
        metrics = model_rollouts[model_name]
        lines.append(
            f"| `{MODEL_LABELS[model_name]}` | {metrics.steps} | {metrics.final_error_m:.3f} | "
            f"{metrics.mean_error_m:.3f} | {metrics.max_error_m:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best Result",
            "",
            f"- local displacement MAE 3D: `{best_local.mae_3d:.3f} m`",
            f"- local displacement RMSE 3D: `{best_local.rmse_3d:.3f} m`",
            f"- sparse rollout final error: `{best_rollout.final_error_m:.3f} m`",
            f"- sparse rollout mean error: `{best_rollout.mean_error_m:.3f} m`",
            f"- sparse rollout max error: `{best_rollout.max_error_m:.3f} m`",
            "",
            "## Limitations",
            "",
            "- Evaluation still uses one DataFlash log.",
            "- Rolling folds are inside one flight, not across independent flights.",
            "- Sparse rollout is based on non-overlapping 5-second displacement windows, not full IMU-rate integration.",
            "",
            "## Generated Artifacts",
            "",
            f"- `{sequence_report}`",
            f"- `{rollout_summary}`",
            f"- `{path}`",
            f"- `{final_html}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_html(
    path: Path,
    prediction_viewer: Path,
    best_rollout_html: Path,
    local_metrics: dict[str, LocalMetrics],
    model_rollouts: dict[str, RolloutMetrics],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_local = local_metrics[BEST_MODEL]
    best_rollout = model_rollouts[BEST_MODEL]
    pred_link = html.escape(os.path.relpath(prediction_viewer, start=path.parent).replace(os.sep, "/"))
    rollout_link = html.escape(os.path.relpath(best_rollout_html, start=path.parent).replace(os.sep, "/"))
    rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(MODEL_LABELS[model_name])}</td>"
        f"<td>{local_metrics[model_name].mae_3d:.3f}</td>"
        f"<td>{local_metrics[model_name].rmse_3d:.3f}</td>"
        f"<td>{model_rollouts[model_name].final_error_m:.3f}</td>"
        f"<td>{model_rollouts[model_name].mean_error_m:.3f}</td>"
        f"<td>{model_rollouts[model_name].max_error_m:.3f}</td>"
        "</tr>"
        for model_name in ROLLUP_MODELS
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFlash Final Report</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #16202a;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: #f4f6f8;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d7dee8;
      border-radius: 8px;
      padding: 18px 20px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid #e5e9ef;
      border-radius: 6px;
      padding: 12px;
      background: #fafbfd;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
      margin-top: 4px;
    }}
    a {{
      color: #0f62fe;
      text-decoration: none;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid #e8edf3;
    }}
    th {{
      font-weight: 650;
      background: #fafbfd;
    }}
    @media (max-width: 720px) {{
      body {{ padding: 12px; }}
      section {{ padding: 14px; }}
      table {{ font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>DataFlash Final Candidate</h1>
      <p>Best model: <code>{BEST_MODEL}</code> on <code>{BEST_CASE}</code> with fixed <code>alpha=100</code> and validation-tuned bias shrinkage.</p>
      <div class="grid">
        <div class="metric"><span>Local MAE 3D</span><strong>{best_local.mae_3d:.3f} m</strong></div>
        <div class="metric"><span>Local RMSE 3D</span><strong>{best_local.rmse_3d:.3f} m</strong></div>
        <div class="metric"><span>Rollout Final</span><strong>{best_rollout.final_error_m:.3f} m</strong></div>
        <div class="metric"><span>Rollout Mean</span><strong>{best_rollout.mean_error_m:.3f} m</strong></div>
        <div class="metric"><span>Rollout Max</span><strong>{best_rollout.max_error_m:.3f} m</strong></div>
      </div>
    </section>
    <section>
      <h2>Viewers</h2>
      <p><a href="{pred_link}">Prediction viewer</a></p>
      <p><a href="{rollout_link}">Best rollout viewer</a></p>
    </section>
    <section>
      <h2>Model Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>model</th>
            <th>MAE 3D</th>
            <th>RMSE 3D</th>
            <th>final error</th>
            <th>mean error</th>
            <th>max error</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    run_step(
        args.python,
        "run_dataflash_sequence_baseline.py",
        "--data-dir",
        str(args.data_dir),
        "--fixed-alpha",
        "100",
        "--tune-bias-shrink",
        "--report",
        str(args.sequence_report),
        "--pred-dir",
        str(args.pred_dir),
    )

    run_step(
        args.python,
        "build_dataflash_prediction_viewer.py",
        "--pred-dir",
        str(args.pred_dir),
        "--output",
        str(args.prediction_viewer),
    )

    model_rollouts: dict[str, RolloutMetrics] = {}
    local_metrics: dict[str, LocalMetrics] = {}

    case_dir = args.pred_dir / BEST_CASE
    for model_name in ROLLUP_MODELS:
        pred_csv = case_dir / f"{model_name}_pred.csv"
        report_path = Path("reports/experiments") / MODEL_REPORT_NAMES[model_name]
        rollout_csv = args.rollout_dir / MODEL_ROLLOUT_NAMES[model_name]
        html_path = Path("artifacts/generated/dataflash/rollouts") / MODEL_HTML_DIRS[model_name] / "index.html"
        run_step(
            args.python,
            "build_dataflash_rollout.py",
            "--pred-csv",
            str(pred_csv),
            "--report",
            str(report_path),
            "--rollout-csv",
            str(rollout_csv),
            "--html",
            str(html_path),
        )
        local_metrics[model_name] = compute_local_metrics(pred_csv)
        model_rollouts[model_name] = compute_rollout_metrics(rollout_csv)

    write_rollout_summary(args.rollout_summary, model_rollouts, args.rollout_dir)
    write_final_report(
        args.final_report,
        args.pred_dir,
        args.sequence_report,
        args.rollout_summary,
        args.final_html,
        local_metrics,
        model_rollouts,
    )
    best_rollout_html = Path("artifacts/generated/dataflash/rollouts") / MODEL_HTML_DIRS[BEST_MODEL] / "index.html"
    write_final_html(args.final_html, args.prediction_viewer, best_rollout_html, local_metrics, model_rollouts)

    print(f"Wrote {args.final_report}")
    print(f"Wrote {args.final_html}")


if __name__ == "__main__":
    main()
