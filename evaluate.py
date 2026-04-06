#!/usr/bin/env python
"""Evaluate saved MLP models on the test set.

Usage
-----
    python evaluate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from river_impairment.data import load_and_preprocess, spatiotemporal_split
from river_impairment.metrics import compute_classification_metrics
from river_impairment.model import MLPImpairment
from river_impairment.trainer import predict_mlp, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = seed_everything(cfg["seed"])

    pred_vars = cfg["variables"]["predictors"]
    dep_vars = cfg["variables"]["targets"]

    df, _, _ = load_and_preprocess(
        cfg["data"]["data_path"],
        pred_vars,
        dep_vars,
        encoding=cfg["data"].get("encoding", "cp949"),
    )

    split = spatiotemporal_split(
        df, pred_vars, dep_vars,
        train_years=cfg["split"]["train_years"],
        test_years=cfg["split"]["test_years"],
    )

    model_dir = Path(cfg["output"]["model_dir"])
    results: list[dict] = []

    for dv in dep_vars:
        pkl_path = model_dir / f"{dv}_best.pkl"
        if not pkl_path.exists():
            logger.warning("Model not found: %s — skipping.", pkl_path)
            continue

        data = pickle.load(open(pkl_path, "rb"))
        params = data["params"]
        scaler = data["scaler"]

        model = MLPImpairment(
            len(pred_vars), 2,
            params["hidden_dim"], params["num_layer"],
            params["act"], params["ratio"],
        )
        model.load_state_dict(data["model_state_dict"])
        model = model.to(device).eval()

        # Train set
        X_tr = scaler.transform(split["x_train"][pred_vars].values)
        y_tr = split["y_train"][dv].values
        tr_pred, tr_prob = predict_mlp(model, X_tr, device)
        tr_m = compute_classification_metrics(y_tr, tr_pred, tr_prob)

        # Test set
        X_te = scaler.transform(split["x_test"][pred_vars].values)
        y_te = split["y_test"][dv].values
        te_pred, te_prob = predict_mlp(model, X_te, device)
        te_m = compute_classification_metrics(y_te, te_pred, te_prob)

        logger.info("%s  Train — %s", dv, {k: f"{v:.4f}" for k, v in tr_m.items()})
        logger.info("%s  Test  — %s", dv, {k: f"{v:.4f}" for k, v in te_m.items()})

        results.append({"target": dv, "split": "train", **{k: round(v, 4) for k, v in tr_m.items()}})
        results.append({"target": dv, "split": "test", **{k: round(v, 4) for k, v in te_m.items()}})

    if results:
        df_out = pd.DataFrame(results)
        out_path = Path(cfg["output"]["results_dir"]) / "evaluation.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print("\n" + df_out.to_string(index=False))
        logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved MLP models")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
