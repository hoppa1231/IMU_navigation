#!/usr/bin/env python3
"""Run baseline models on prepared window datasets with flight-level splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


@dataclass
class Split:
    name: str
    train: list[str]
    val: list[str]
    test: list[str]


@dataclass
class Dataset:
    path: Path
    name: str
    x: np.ndarray
    y: np.ndarray
    flight_id: np.ndarray
    time_s: np.ndarray
    future_time_s: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=Path,
        default=sorted(
            path
            for path in Path("derived/datasets").glob("windows_module_h*_l*.npz")
            if "_trim" not in path.stem
        ),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/module_window_baselines.md"))
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/module_window_baselines"))
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--target-mode", choices=["xyz", "xy"], default="xyz")
    return parser.parse_args()


def dataset_name(path: Path) -> str:
    return path.stem


def read_dataset(path: Path) -> Dataset:
    data = np.load(path, allow_pickle=False)
    target_names = [str(name) for name in data["target_names"].tolist()] if "target_names" in data.files else [
        "dx_east_m",
        "dy_north_m",
        "dz_up_m",
    ]
    return Dataset(
        path=path,
        name=dataset_name(path),
        x=data["x"].astype(np.float64),
        y=data["y"].astype(np.float64),
        flight_id=data["flight_id"].astype(str),
        time_s=data["time_s"].astype(np.float64),
        future_time_s=data["future_time_s"].astype(np.float64),
        feature_names=[str(name) for name in data["feature_names"].tolist()],
        target_names=target_names,
    )


def apply_target_mode(dataset: Dataset, target_mode: str) -> Dataset:
    if target_mode == "xyz":
        return dataset
    if target_mode == "xy":
        return Dataset(
            path=dataset.path,
            name=f"{dataset.name}_xy",
            x=dataset.x,
            y=dataset.y[:, :2],
            flight_id=dataset.flight_id,
            time_s=dataset.time_s,
            future_time_s=dataset.future_time_s,
            feature_names=dataset.feature_names,
            target_names=["dx_east_m", "dy_north_m"],
        )
    raise ValueError(f"Unsupported target mode: {target_mode}")


def default_splits(flights: set[str]) -> list[Split]:
    splits = [
        Split(
            name="module_data_holdout",
            train=[f"module_data_s{idx:02d}" for idx in range(1, 6)],
            val=["module_data_s06"],
            test=["module_data_s07"],
        ),
        Split(
            name="route_holdout_triangle",
            train=[f"module_data_s{idx:02d}" for idx in range(1, 7)] + ["linear_15_01_2025"],
            val=["module_data_s07"],
            test=["triangle_15_01_2025"],
        ),
        Split(
            name="route_holdout_linear",
            train=[f"module_data_s{idx:02d}" for idx in range(1, 7)] + ["triangle_15_01_2025"],
            val=["module_data_s07"],
            test=["linear_15_01_2025"],
        ),
    ]
    valid: list[Split] = []
    for split in splits:
        train = [flight for flight in split.train if flight in flights]
        val = [flight for flight in split.val if flight in flights]
        test = [flight for flight in split.test if flight in flights]
        if train and len(val) == len(split.val) and len(test) == len(split.test):
            valid.append(Split(name=split.name, train=train, val=val, test=test))
    return valid


def mask_for(flight_ids: np.ndarray, selected: list[str]) -> np.ndarray:
    return np.isin(flight_ids, np.asarray(selected))


def maybe_subsample(mask: np.ndarray, max_count: int) -> np.ndarray:
    indices = np.nonzero(mask)[0]
    if max_count and len(indices) > max_count:
        indices = indices[:max_count]
    result = np.zeros_like(mask, dtype=bool)
    result[indices] = True
    return result


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean, x_std = fit_standardizer(x)
    y_mean = y.mean(axis=0)
    xz = (x - x_mean) / x_std
    yc = y - y_mean
    reg = alpha * np.eye(xz.shape[1])
    weights = np.linalg.solve(xz.T @ xz + reg, xz.T @ yc)
    return weights, x_mean, x_std, y_mean


def predict_ridge(x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, x_mean, x_std, y_mean = params
    return ((x - x_mean) / x_std) @ weights + y_mean


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    err = y_pred - y_true
    mae_axis = np.mean(np.abs(err), axis=0)
    rmse_axis = np.sqrt(np.mean(err * err, axis=0))
    dist_err = np.linalg.norm(err, axis=1)
    return {
        "mae_axis": [float(value) for value in mae_axis],
        "rmse_axis": [float(value) for value in rmse_axis],
        "mae_3d": float(np.mean(dist_err)),
        "rmse_3d": float(np.sqrt(np.mean(dist_err * dist_err))),
        "p95_3d": float(np.percentile(dist_err, 95)),
    }


def metric_row(dataset: str, split: str, subset: str, model: str, values: dict[str, float | list[float]], alpha: float | None) -> str:
    mae = values["mae_axis"]
    rmse = values["rmse_axis"]
    assert isinstance(mae, list)
    assert isinstance(rmse, list)
    alpha_text = "" if alpha is None else f"{alpha:g}"
    up_mae = f"{mae[2]:.3f}" if len(mae) > 2 else "n/a"
    up_rmse = f"{rmse[2]:.3f}" if len(rmse) > 2 else "n/a"
    return (
        f"| `{dataset}` | `{split}` | `{subset}` | `{model}` | {alpha_text} | "
        f"{mae[0]:.3f} | {mae[1]:.3f} | {up_mae} | {values['mae_3d']:.3f} | "
        f"{rmse[0]:.3f} | {rmse[1]:.3f} | {up_rmse} | {values['rmse_3d']:.3f} | {values['p95_3d']:.3f} |"
    )


def write_predictions(
    path: Path,
    dataset: Dataset,
    mask: np.ndarray,
    y_pred: np.ndarray,
    model: str,
    split_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.nonzero(mask)[0]
    include_z = y_pred.shape[1] >= 3
    fieldnames = [
        "dataset",
        "split",
        "model",
        "flight_id",
        "time_s",
        "future_time_s",
        "true_dx_east_m",
        "true_dy_north_m",
        "pred_dx_east_m",
        "pred_dy_north_m",
    ]
    if include_z:
        fieldnames.insert(fieldnames.index("pred_dx_east_m"), "true_dz_up_m")
        fieldnames.append("pred_dz_up_m")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, data_idx in enumerate(indices):
            row = {
                "dataset": dataset.name,
                "split": split_name,
                "model": model,
                "flight_id": dataset.flight_id[data_idx],
                "time_s": f"{dataset.time_s[data_idx]:.6f}",
                "future_time_s": f"{dataset.future_time_s[data_idx]:.6f}",
                "true_dx_east_m": f"{dataset.y[data_idx, 0]:.6f}",
                "true_dy_north_m": f"{dataset.y[data_idx, 1]:.6f}",
                "pred_dx_east_m": f"{y_pred[row_idx, 0]:.6f}",
                "pred_dy_north_m": f"{y_pred[row_idx, 1]:.6f}",
            }
            if include_z:
                row["true_dz_up_m"] = f"{dataset.y[data_idx, 2]:.6f}"
                row["pred_dz_up_m"] = f"{y_pred[row_idx, 2]:.6f}"
            writer.writerow(row)


def top_features(feature_names: list[str], weights: np.ndarray, limit: int = 15) -> list[tuple[str, float]]:
    scores = np.linalg.norm(weights, axis=1)
    order = np.argsort(scores)[::-1][:limit]
    return [(feature_names[idx], float(scores[idx])) for idx in order]


def run_dataset_split(
    dataset: Dataset,
    split: Split,
    pred_dir: Path,
    max_train_windows: int,
) -> tuple[list[dict[str, object]], list[str], list[tuple[str, float]]]:
    train_mask = maybe_subsample(mask_for(dataset.flight_id, split.train), max_train_windows)
    val_mask = mask_for(dataset.flight_id, split.val)
    test_mask = mask_for(dataset.flight_id, split.test)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(f"{dataset.name}/{split.name}: empty train/val/test mask")

    x_train = dataset.x[train_mask]
    y_train = dataset.y[train_mask]
    x_val = dataset.x[val_mask]
    y_val = dataset.y[val_mask]
    x_test = dataset.x[test_mask]
    y_test = dataset.y[test_mask]

    train_mean = np.repeat(y_train.mean(axis=0)[None, :], len(y_test), axis=0)
    zero_test = np.zeros_like(y_test)
    mean_test = train_mean

    best_alpha = ALPHAS[0]
    best_params = fit_ridge(x_train, y_train, best_alpha)
    best_score = math.inf
    for alpha in ALPHAS:
        params = fit_ridge(x_train, y_train, alpha)
        val_pred = predict_ridge(x_val, params)
        score = float(metrics(y_val, val_pred)["mae_3d"])
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_params = params

    ridge_test = predict_ridge(x_test, best_params)
    ridge_val = predict_ridge(x_val, best_params)

    result_rows: list[dict[str, object]] = []
    for subset_name, mask, y_true, predictions in [
        ("val", val_mask, y_val, {
            "zero": np.zeros_like(y_val),
            "train_mean": np.repeat(y_train.mean(axis=0)[None, :], len(y_val), axis=0),
            "ridge": ridge_val,
        }),
        ("test", test_mask, y_test, {
            "zero": zero_test,
            "train_mean": mean_test,
            "ridge": ridge_test,
        }),
    ]:
        for model_name, y_pred in predictions.items():
            alpha = best_alpha if model_name == "ridge" else None
            values = metrics(y_true, y_pred)
            result_rows.append(
                {
                    "dataset": dataset.name,
                    "split": split.name,
                    "subset": subset_name,
                    "model": model_name,
                    "alpha": alpha,
                    "metrics": values,
                    "count": len(y_true),
                }
            )
            if subset_name == "test":
                pred_path = pred_dir / dataset.name / split.name / f"{model_name}_pred.csv"
                write_predictions(pred_path, dataset, mask, y_pred, model_name, split.name)

    weights = best_params[0]
    feature_scores = top_features(dataset.feature_names, weights)
    notes = [
        f"`{dataset.name}/{split.name}` train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())} best_alpha={best_alpha:g}",
    ]
    return result_rows, notes, feature_scores


def write_report(
    path: Path,
    datasets: list[Dataset],
    rows: list[dict[str, object]],
    notes: list[str],
    feature_sections: dict[str, list[tuple[str, float]]],
    pred_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Module window baselines",
        "",
        "This report is generated by `src/run_window_baselines.py`.",
        "",
        "Input datasets:",
        "",
    ]
    for dataset in datasets:
        lines.append(f"- `{dataset.path}`: X={dataset.x.shape}, y={dataset.y.shape}")
    lines.extend(
        [
            "",
        "Splits are flight-level holdouts. Windows from the same holdout flight are not used for training.",
            "",
            "Predictions are written under:",
            "",
            f"- `{pred_dir}`",
            "",
            "## Metrics",
            "",
        "Errors are in meters for one target displacement over the dataset horizon. In `xy` reports, the 3D columns represent horizontal 2D norm and `up` is `n/a`.",
            "",
            f"| dataset | split | subset | model | alpha | MAE {datasets[0].target_names[0]} | MAE {datasets[0].target_names[1]} | MAE {datasets[0].target_names[2] if len(datasets[0].target_names) > 2 else 'axis_3'} | MAE norm | RMSE {datasets[0].target_names[0]} | RMSE {datasets[0].target_names[1]} | RMSE {datasets[0].target_names[2] if len(datasets[0].target_names) > 2 else 'axis_3'} | RMSE norm | P95 norm |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            metric_row(
                str(row["dataset"]),
                str(row["split"]),
                str(row["subset"]),
                str(row["model"]),
                row["metrics"],  # type: ignore[arg-type]
                row["alpha"] if isinstance(row["alpha"], float) else None,
            )
        )
    best_test: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if row["subset"] != "test":
            continue
        key = (str(row["dataset"]), str(row["split"]))
        values = row["metrics"]
        assert isinstance(values, dict)
        score = float(values["mae_3d"])
        if key not in best_test or score < float(best_test[key]["mae_3d"]):
            best_test[key] = {
                "model": row["model"],
                "mae_3d": score,
                "rmse_3d": float(values["rmse_3d"]),
            }
    lines.extend(
        [
            "",
            "## Best Test Baseline",
            "",
            "| dataset | split | model | MAE 3D | RMSE 3D |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for (dataset_name, split_name), values in sorted(best_test.items()):
        lines.append(
            f"| `{dataset_name}` | `{split_name}` | `{values['model']}` | "
            f"{float(values['mae_3d']):.3f} | {float(values['rmse_3d']):.3f} |"
        )
    lines.extend(["", "## Split Details", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Ridge Feature Scores", ""])
    for section, scores in feature_sections.items():
        lines.extend([f"### `{section}`", "", "| feature | score |", "| --- | ---: |"])
        for name, score in scores:
            lines.append(f"| `{name}` | {score:.5f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = [apply_target_mode(read_dataset(path), args.target_mode) for path in args.datasets]
    all_rows: list[dict[str, object]] = []
    all_notes: list[str] = []
    feature_sections: dict[str, list[tuple[str, float]]] = {}
    for dataset in datasets:
        splits = default_splits(set(dataset.flight_id.tolist()))
        if not splits:
            raise ValueError(f"No default splits apply to {dataset.name}")
        for split in splits:
            rows, notes, scores = run_dataset_split(dataset, split, args.pred_dir, args.max_train_windows)
            all_rows.extend(rows)
            all_notes.extend(notes)
            feature_sections[f"{dataset.name}/{split.name}"] = scores
            print(f"Ran {dataset.name}/{split.name}")
    write_report(args.report, datasets, all_rows, all_notes, feature_sections, args.pred_dir)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
