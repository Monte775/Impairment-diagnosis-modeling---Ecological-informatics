#!/usr/bin/env python
"""Train MLP models for river impairment diagnosis.

Usage
-----
    python train.py --config configs/default.yaml

The script performs, *for each target variable*:

1. Spatiotemporal train/test split.
2. Hyperopt TPE optimisation (stratified K-fold CV on the training set).
3. Re-training the best model on the full training set.
4. Evaluation on both train and test sets.
5. Saving model weights, scaler, and metrics.
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from river_impairment.data import (
    fit_scaler,
    load_and_preprocess,
    spatiotemporal_split,
)
from river_impairment.metrics import compute_classification_metrics
from river_impairment.model import MLPImpairment
from river_impairment.trainer import (
    ACT_CHOICES,
    optimize_mlp,
    predict_mlp,
    seed_everything,
    train_mlp,
)

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

    # ---- Load & preprocess -------------------------------------------
    df, le_conn, le_embed = load_and_preprocess(
        cfg["data"]["data_path"],
        pred_vars,
        dep_vars,
        encoding=cfg["data"].get("encoding", "cp949"),
    )

    # ---- Optional site-river map for spatial overlap check -----------
    site_river_map = None
    srp = cfg["data"].get("site_river_path")
    if srp and Path(srp).exists():
        ref = pd.read_csv(srp, encoding=cfg["data"].get("encoding", "cp949"))
        site_river_map = (
            ref.drop_duplicates(subset=["지점명"])
            .set_index("지점명")["하천명"]
            .to_dict()
        )

    # ---- Spatiotemporal split ----------------------------------------
    split = spatiotemporal_split(
        df,
        pred_vars,
        dep_vars,
        train_years=cfg["split"]["train_years"],
        test_years=cfg["split"]["test_years"],
        site_col=cfg["split"].get("site_col", "조사구간명"),
        year_col=cfg["split"].get("year_col", "연도"),
        site_river_map=site_river_map,
    )

    x_train = split["x_train"]
    y_train = split["y_train"]
    x_test = split["x_test"]
    y_test = split["y_test"]

    results_dir = Path(cfg["output"]["results_dir"])
    model_dir = Path(cfg["output"]["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for dv in dep_vars:
        logger.info("=" * 60)
        logger.info("  Target: %s", dv)
        logger.info("=" * 60)

        X_cv = x_train[pred_vars].values.copy()
        y_cv = y_train[dv].values.copy()
        X_te = x_test[pred_vars].values.copy()
        y_te = y_test[dv].values.copy()

        # ---- Hyperopt TPE optimisation -------------------------------
        trial_dir = model_dir / dv
        opt_result = optimize_mlp(
            X_cv,
            y_cv,
            n_folds=cfg["training"]["n_folds"],
            epochs=cfg["training"]["epochs"],
            max_evals=cfg["training"]["max_evals"],
            device=device,
            save_dir=trial_dir,
            seed=cfg["seed"],
        )

        best_params = opt_result["best_params"]
        logger.info("Best params for %s: %s", dv, best_params)
        logger.info("Best CV accuracy: %.4f", opt_result["best_cv_accuracy"])

        # ---- Retrain best model on full train set --------------------
        scaler, X_train_scaled = fit_scaler(X_cv)
        X_test_scaled = scaler.transform(X_te)

        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed(cfg["seed"])
        best_model = MLPImpairment(
            X_cv.shape[1],
            2,
            best_params["hidden_dim"],
            best_params["num_layer"],
            best_params["act"],
            best_params["ratio"],
        )
        best_model = train_mlp(
            best_model,
            X_train_scaled,
            y_cv,
            epochs=cfg["training"]["epochs"],
            lr=best_params["lr"],
            weight_decay=best_params["wd"],
            device=device,
        )

        # ---- Evaluate ------------------------------------------------
        tr_pred, tr_prob = predict_mlp(best_model, X_train_scaled, device)
        te_pred, te_prob = predict_mlp(best_model, X_test_scaled, device)
        tr_metrics = compute_classification_metrics(y_cv, tr_pred, tr_prob)
        te_metrics = compute_classification_metrics(y_te, te_pred, te_prob)

        logger.info(
            "  Train — AUC=%.4f  ACC=%.4f  Recall=%.4f  F1=%.4f",
            *tr_metrics.values(),
        )
        logger.info(
            "  Test  — AUC=%.4f  ACC=%.4f  Recall=%.4f  F1=%.4f",
            *te_metrics.values(),
        )

        all_results.append(
            {"target": dv, "split": "train", **{k: round(v, 4) for k, v in tr_metrics.items()}}
        )
        all_results.append(
            {"target": dv, "split": "test", **{k: round(v, 4) for k, v in te_metrics.items()}}
        )

        # ---- Save best model -----------------------------------------
        save_path = model_dir / f"{dv}_best.pkl"
        pickle.dump(
            {
                "model_state_dict": best_model.cpu().state_dict(),
                "params": best_params,
                "scaler": scaler,
                "train_metrics": tr_metrics,
                "test_metrics": te_metrics,
            },
            open(save_path, "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        logger.info("Saved best model → %s", save_path)

        # ---- Save trial records --------------------------------------
        trial_df = pd.DataFrame(opt_result["trial_records"])
        trial_df.to_csv(results_dir / f"{dv}_trials.csv", index=False)

        del best_model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Save aggregated results -------------------------------------
    results_df = pd.DataFrame(all_results)
    out_path = results_dir / "performance.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Performance summary saved → %s", out_path)
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP for river impairment diagnosis")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    main(args.config)
