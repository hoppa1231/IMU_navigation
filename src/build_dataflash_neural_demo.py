#!/usr/bin/env python3
"""Build one animated comparison page for DataFlash neural rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_dataflash_demo import build_payload, html_template, read_rows


DEFAULT_CASES = {
    "LSTM 64": Path("derived/predictions/dataflash_recurrent_rollout/lstm_64.csv"),
    "GRU 64": Path("derived/predictions/dataflash_recurrent_rollout/gru_64.csv"),
    "MLP 64": Path("derived/predictions/dataflash_neural_rollout/mlp_64.csv"),
    "MLP 128 -> 64": Path("derived/predictions/dataflash_neural_rollout/mlp_128_64.csv"),
    "Ridge + bias tuned": Path(
        "derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/dataflash/neural_demo/index.html"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: list[dict[str, object]] = []
    for model_id, (model_label, path) in enumerate(DEFAULT_CASES.items(), start=1):
        if not path.exists():
            raise FileNotFoundError(f"Missing rollout for {model_label}: {path}")
        for fold in build_payload(read_rows(path)):
            fold_id = str(fold["id"])
            fold["id"] = f"{model_id}:{fold_id}"
            fold["label"] = f"{model_label} / fold {fold_id}"
            payload.append(fold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        html_template(payload, "Recurrent and dense models compared with the previous Ridge baseline"),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
