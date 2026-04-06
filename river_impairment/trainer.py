"""Hyperopt TPE optimisation and MLP training utilities."""

from __future__ import annotations

import gc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hyperopt import Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from river_impairment.metrics import compute_classification_metrics
from river_impairment.model import MLPImpairment

logger = logging.getLogger(__name__)

ACT_CHOICES = ["leaky_relu", "elu", "linear"]

# ------------------------------------------------------------------
# Default search space
# ------------------------------------------------------------------

DEFAULT_SEARCH_SPACE = [
    hp.quniform("hidden_dim", 30, 100, q=1),
    hp.quniform("num_layer", 3, 5, q=1),
    hp.choice("act", ACT_CHOICES),
    hp.quniform("ratio", 0.5, 1.0, q=0.1),
    hp.uniform("lr", 0.0001, 0.005),
    hp.uniform("wd", 1e-7, 1e-3),
]


# ------------------------------------------------------------------
# Deterministic seeding
# ------------------------------------------------------------------

def seed_everything(seed: int = 42) -> torch.device:
    """Set random seeds and return the available torch device."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  |  seed: %d", device, seed)
    return device


# ------------------------------------------------------------------
# Single-model training
# ------------------------------------------------------------------

def train_mlp(
    model: MLPImpairment,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
) -> MLPImpairment:
    """Train an MLP model and return the trained model (in-place).

    Parameters
    ----------
    model : MLPImpairment
        Initialised (untrained) model.
    X_train : ndarray of shape (n, d)
        Scaled training features.
    y_train : ndarray of shape (n,)
        Binary labels (0/1).
    epochs : int
    lr : float
    weight_decay : float
    device : torch.device, optional

    Returns
    -------
    model : MLPImpairment
        Trained model (eval mode).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    xx = torch.FloatTensor(X_train).to(device)
    yy = F.one_hot(
        torch.LongTensor(y_train.astype(int)), num_classes=2,
    ).float().to(device)

    model.train()
    for _ in range(epochs):
        logits = model(xx).double()
        loss = loss_fn(logits, yy.double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_mlp(
    model: MLPImpairment,
    X: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return ``(predicted_labels, positive_class_proba)``."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        proba = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    return np.argmax(proba, axis=1), proba[:, 1]


# ------------------------------------------------------------------
# Hyperopt TPE optimisation
# ------------------------------------------------------------------

def _mlp_cv_objective(
    args,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    n_folds: int,
    epochs: int,
    device: torch.device,
) -> float:
    """Stratified K-fold CV objective (returns negative accuracy)."""
    hidden_dim, num_layer, act, ratio, lr, wd = args
    hidden_dim, num_layer = int(hidden_dim), int(num_layer)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs: list[float] = []

    for tr_idx, val_idx in skf.split(X_cv, y_cv):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_cv[tr_idx])
        X_val = scaler.transform(X_cv[val_idx])

        train_xx = torch.FloatTensor(X_tr).to(device)
        train_yy = F.one_hot(
            torch.LongTensor(y_cv[tr_idx].astype(int)), num_classes=2,
        ).float().to(device)
        val_xx = torch.FloatTensor(X_val).to(device)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = MLPImpairment(
            X_tr.shape[1], 2, hidden_dim, num_layer, act, ratio,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5, weight_decay=wd)
        loss_fn = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            loss = loss_fn(model(train_xx).double(), train_yy.double())
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_xx).cpu().numpy()

        from sklearn.metrics import accuracy_score
        accs.append(accuracy_score(y_cv[val_idx], np.argmax(val_pred, axis=1)))

        del model, train_xx, train_yy, val_xx
        gc.collect()
        torch.cuda.empty_cache()

    return -np.mean(accs)


def optimize_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_folds: int = 5,
    epochs: int = 200,
    max_evals: int = 100,
    search_space: Optional[list] = None,
    device: Optional[torch.device] = None,
    save_dir: Optional[str | Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run TPE hyperparameter optimisation for the MLP model.

    Parameters
    ----------
    X_train : ndarray (n, d)
        Raw (un-scaled) training features.
    y_train : ndarray (n,)
        Binary labels.
    n_folds : int
        Number of CV folds.
    epochs : int
        Training epochs per trial.
    max_evals : int
        Number of TPE evaluations.
    search_space : list, optional
        Hyperopt search space.  Uses :data:`DEFAULT_SEARCH_SPACE` if *None*.
    device : torch.device, optional
    save_dir : path-like, optional
        If given, each trial model is saved as a ``.pkl`` file.
    seed : int

    Returns
    -------
    dict
        ``best_params``, ``best_cv_accuracy``, ``trials`` (hyperopt Trials object).
    """
    if device is None:
        device = seed_everything(seed)
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    trial_records: list[dict] = []

    def _tracking_objective(args):
        cv_loss = _mlp_cv_objective(
            args, X_train, y_train, n_folds, epochs, device,
        )
        cv_acc = -cv_loss
        hidden_dim, num_layer, act, ratio, lr, wd = args
        params = {
            "hidden_dim": int(hidden_dim),
            "num_layer": int(num_layer),
            "act": act,
            "ratio": ratio,
            "lr": lr,
            "wd": wd,
        }

        trial_num = len(trial_records) + 1
        trial_records.append({"trial": trial_num, "cv_accuracy": cv_acc, **params})

        if save_dir is not None:
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_train)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            model = MLPImpairment(
                X_train.shape[1], 2, params["hidden_dim"],
                params["num_layer"], params["act"], params["ratio"],
            ).to(device)
            model = train_mlp(
                model, X_scaled, y_train,
                epochs=epochs, lr=params["lr"],
                weight_decay=params["wd"], device=device,
            )
            pkl_path = save_dir / f"trial_{trial_num:03d}.pkl"
            pickle.dump(
                {
                    "model_state_dict": model.cpu().state_dict(),
                    "params": params,
                    "scaler": scaler,
                    "cv_accuracy": cv_acc,
                },
                open(pkl_path, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()

        return cv_loss

    trials = Trials()
    best = fmin(
        fn=_tracking_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=True,
    )

    best_cv_acc = -min(trials.losses())
    act_idx = best.get("act", 0)
    best_params = {
        "hidden_dim": int(best["hidden_dim"]),
        "num_layer": int(best["num_layer"]),
        "act": ACT_CHOICES[int(act_idx)] if isinstance(act_idx, (int, float)) else act_idx,
        "ratio": best["ratio"],
        "lr": best["lr"],
        "wd": best["wd"],
    }

    logger.info("Best CV accuracy: %.4f  |  params: %s", best_cv_acc, best_params)

    return {
        "best_params": best_params,
        "best_cv_accuracy": best_cv_acc,
        "trials": trials,
        "trial_records": trial_records,
    }
