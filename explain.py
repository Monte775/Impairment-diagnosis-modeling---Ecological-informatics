#!/usr/bin/env python
"""Generate SHAP explanations (summary & waterfall plots) for saved models.

Usage
-----
    python explain.py --config configs/default.yaml
    python explain.py --config configs/default.yaml --target BMI_훼손
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

from river_impairment.data import fit_scaler, load_and_preprocess, spatiotemporal_split
from river_impairment.explainer import shap_summary, shap_waterfall
from river_impairment.model import MLPImpairment
from river_impairment.trainer import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(cfg_path: str, target: str | None = None) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = seed_everything(cfg["seed"])

    pred_vars = cfg["variables"]["predictors"]
    pred_labels = cfg["variables"].get("predictor_labels", pred_vars)
    dep_vars = cfg["variables"]["targets"]
    if target:
        dep_vars = [target]

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
    shap_dir = Path(cfg["output"]["shap_dir"])
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Background: scaled full dataset (using training-set scaler)
    scaler_bg, X_bg = fit_scaler(split["x_train"][pred_vars].values)
    X_all_scaled = scaler_bg.transform(df[pred_vars].values)

    for dv in dep_vars:
        pkl_path = model_dir / f"{dv}_best.pkl"
        if not pkl_path.exists():
            logger.warning("Model not found: %s — skipping.", pkl_path)
            continue

        data = pickle.load(open(pkl_path, "rb"))
        params = data["params"]

        model = MLPImpairment(
            len(pred_vars), 2,
            params["hidden_dim"], params["num_layer"],
            params["act"], params["ratio"],
        )
        model.load_state_dict(data["model_state_dict"])

        logger.info("Generating SHAP summary for %s ...", dv)
        shap_summary(
            model,
            X_background=X_all_scaled,
            X_explain=X_all_scaled,
            feature_names=pred_labels,
            device=device,
            save_path=shap_dir / f"{dv}_summary.png",
            title=f"SHAP Summary — {dv}",
        )

        logger.info("Generating SHAP waterfall (first test sample) for %s ...", dv)
        X_te_scaled = scaler_bg.transform(split["x_test"][pred_vars].values)
        shap_waterfall(
            model,
            X_background=X_all_scaled,
            x_instance=X_te_scaled[0],
            feature_names=pred_labels,
            device=device,
            save_path=shap_dir / f"{dv}_waterfall_sample0.png",
            title=f"SHAP Waterfall — {dv} (test sample 0)",
        )

    logger.info("All SHAP plots saved in %s", shap_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explanation for MLP models")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--target", "-t", default=None, help="Single target variable")
    args = parser.parse_args()
    main(args.config, args.target)
