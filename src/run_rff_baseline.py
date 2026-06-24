#!/usr/bin/env python3
"""Run a pure-NumPy nonlinear random Fourier feature baseline."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from run_window_baselines import (
    ALPHAS,
    Dataset,
    Split,
    default_splits,
    fit_ridge,
    mask_for,
    metrics,
    predict_ridge,
    read_dataset,
    write_predictions,
)


GAMMAS = [0.005, 0.01, 0.02, 0.05]


@dataclass
class RffParams:
    x_mean: np.ndarray
    x_std: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    ridge_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=Path,
        default=sorted(Path("derived/datasets").glob("windows_module_h*_l*_move1.npz")),
    )
    parser.add_argument("--components", type=int, default=256)
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/module_window_rff_move1.md"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/module_window_rff_move1"))
    parser.add_argument("--seed", type=int, default=20260620)
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: str) -> int:
    key = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def rff_features(x: np.ndarray, params: RffParams) -> np.ndarray:
    z = (x - params.x_mean) / params.x_std
    return math.sqrt(2.0 / params.weights.shape[1]) * np.cos(z @ params.weights + params.bias)


def fit_rff(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    components: int,
    seed: int,
) -> tuple[float, float, RffParams, float]:
    x_mean, x_std = fit_standardizer(x_train)
    z_train = (x_train - x_mean) / x_std
    z_val = (x_val - x_mean) / x_std

    best_score = math.inf
    best_alpha = ALPHAS[0]
    best_gamma = GAMMAS[0]
    best_params: RffParams | None = None

    for gamma in GAMMAS:
        rng = np.random.default_rng(stable_seed(seed, f"gamma={gamma:g}", f"components={components}"))
        weights = rng.normal(0.0, math.sqrt(2.0 * gamma), size=(x_train.shape[1], components))
        bias = rng.uniform(0.0, 2.0 * math.pi, size=components)
        phi_train = math.sqrt(2.0 / components) * np.cos(z_train @ weights + bias)
        phi_val = math.sqrt(2.0 / components) * np.cos(z_val @ weights + bias)
        for alpha in ALPHAS:
            ridge_params = fit_ridge(phi_train, y_train, alpha)
            pred_val = predict_ridge(phi_val, ridge_params)
            score = float(metrics(y_val, pred_val)["mae_3d"])
            if score < best_score:
                best_score = score
                best_alpha = alpha
                best_gamma = gamma
                best_params = RffParams(
                    x_mean=x_mean,
                    x_std=x_std,
                    weights=weights,
                    bias=bias,
                    ridge_params=ridge_params,
                )

    assert best_params is not None
    return best_alpha, best_gamma, best_params, best_score


def predict_rff(x: np.ndarray, params: RffParams) -> np.ndarray:
    return predict_ridge(rff_features(x, params), params.ridge_params)


def metric_row(dataset: str, split: str, subset: str, model: str, params: str, values: dict[str, float | list[float]]) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, list)
    assert isinstance(rmse, list)
    return (
        f"| `{dataset}` | `{split}` | `{subset}` | `{model}` | {params} | "
        f"{mae[0]:.3f} | {mae[1]:.3f} | {mae[2]:.3f} | {values['mae_3d']:.3f} | "
        f"{rmse[0]:.3f} | {rmse[1]:.3f} | {rmse[2]:.3f} | {values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def run_split(
    dataset: Dataset,
    split: Split,
    components: int,
    seed: int,
    pred_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    train_mask = mask_for(dataset.flight_id, split.train)
    val_mask = mask_for(dataset.flight_id, split.val)
    test_mask = mask_for(dataset.flight_id, split.test)
    x_train = dataset.x[train_mask]
    y_train = dataset.y[train_mask]
    x_val = dataset.x[val_mask]
    y_val = dataset.y[val_mask]
    x_test = dataset.x[test_mask]
    y_test = dataset.y[test_mask]

    best_ridge_alpha = ALPHAS[0]
    best_ridge_params = fit_ridge(x_train, y_train, best_ridge_alpha)
    best_ridge_score = math.inf
    for alpha in ALPHAS:
        params = fit_ridge(x_train, y_train, alpha)
        score = float(metrics(y_val, predict_ridge(x_val, params))["mae_3d"])
        if score < best_ridge_score:
            best_ridge_score = score
            best_ridge_alpha = alpha
            best_ridge_params = params

    split_seed = stable_seed(seed, dataset.name, split.name)
    rff_alpha, rff_gamma, rff_params, rff_val_score = fit_rff(
        x_train,
        y_train,
        x_val,
        y_val,
        components,
        split_seed,
    )

    rows: list[str] = []
    best_rows: list[str] = []
    notes: list[str] = []
    for subset_name, mask, x_current, y_current in [
        ("val", val_mask, x_val, y_val),
        ("test", test_mask, x_test, y_test),
    ]:
        predictions = {
            "zero": (np.zeros_like(y_current), ""),
            "train_mean": (np.repeat(y_train.mean(axis=0)[None, :], len(y_current), axis=0), ""),
            "ridge": (predict_ridge(x_current, best_ridge_params), f"alpha={best_ridge_alpha:g}"),
            "rff_ridge": (predict_rff(x_current, rff_params), f"alpha={rff_alpha:g}, gamma={rff_gamma:g}"),
        }
        scored: list[tuple[str, dict[str, float | list[float]]]] = []
        for model_name, (pred, params_text) in predictions.items():
            values = metrics(y_current, pred)
            rows.append(metric_row(dataset.name, split.name, subset_name, model_name, params_text, values))
            scored.append((model_name, values))
            if subset_name == "test":
                pred_path = pred_dir / dataset.name / split.name / f"{model_name}_pred.csv"
                write_predictions(pred_path, dataset, mask, pred, model_name, split.name)
        if subset_name == "test":
            best_model, best_values = min(scored, key=lambda item: float(item[1]["mae_3d"]))
            best_rows.append(
                f"| `{dataset.name}` | `{split.name}` | `{best_model}` | "
                f"{float(best_values['mae_3d']):.3f} | {float(best_values['rmse_3d']):.3f} |"
            )

    notes.append(
        f"`{dataset.name}/{split.name}` train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"test={int(test_mask.sum())} ridge_alpha={best_ridge_alpha:g} "
        f"rff_alpha={rff_alpha:g} rff_gamma={rff_gamma:g} rff_val_mae_3d={rff_val_score:.3f}"
    )
    return rows, best_rows, notes


def write_report(
    path: Path,
    datasets: list[Dataset],
    rows: list[str],
    best_rows: list[str],
    notes: list[str],
    pred_dir: Path,
    components: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Module Moving-Only RFF Baseline",
        "",
        "This report is generated by `src/run_rff_baseline.py`.",
        "",
        "Scope: moving-only module datasets (`*_move1.npz`). GPS is used only as target.",
        "",
        f"Random Fourier components: `{components}`.",
        f"Predictions: `{pred_dir}`",
        "",
        "Input datasets:",
        "",
    ]
    for dataset in datasets:
        lines.append(f"- `{dataset.path}`: X={dataset.x.shape}, y={dataset.y.shape}")
    lines.extend(
        [
            "",
            "Models:",
            "",
            "- `zero`: always predicts no displacement.",
            "- `train_mean`: predicts mean train displacement.",
            "- `ridge`: linear ridge on original window features.",
            "- `rff_ridge`: ridge on random Fourier nonlinear features.",
            "",
            "## Metrics",
            "",
            "| dataset | split | subset | model | params | MAE east | MAE north | MAE up | MAE 3D | RMSE east | RMSE north | RMSE up | RMSE 3D | P95 3D |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(rows)
    lines.extend(
        [
            "",
            "## Best Test Baseline",
            "",
            "| dataset | split | model | MAE 3D | RMSE 3D |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    lines.extend(best_rows)
    lines.extend(["", "## Split Details", ""])
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = [read_dataset(path) for path in args.datasets]
    rows: list[str] = []
    best_rows: list[str] = []
    notes: list[str] = []
    for dataset in datasets:
        for split in default_splits(set(dataset.flight_id.tolist())):
            split_rows, split_best, split_notes = run_split(dataset, split, args.components, args.seed, args.pred_dir)
            rows.extend(split_rows)
            best_rows.extend(split_best)
            notes.extend(split_notes)
            print(f"Ran {dataset.name}/{split.name}")
    write_report(args.report, datasets, rows, best_rows, notes, args.pred_dir, args.components)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
