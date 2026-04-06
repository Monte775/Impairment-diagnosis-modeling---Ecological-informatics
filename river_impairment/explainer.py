"""SHAP-based model explanation utilities for MLP impairment models.

Uses ``shap.DeepExplainer`` (DeepLIFT) for efficient gradient-based
attribution on PyTorch models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from river_impairment.model import MLPImpairment

logger = logging.getLogger(__name__)


def _to_tensor(
    x: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor on *device*."""
    return torch.tensor(x, dtype=torch.float32).to(device)


def shap_summary(
    model: MLPImpairment,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    *,
    device: Optional[torch.device] = None,
    max_display: int = 15,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> np.ndarray:
    """Compute SHAP values via DeepExplainer and generate a summary plot.

    Parameters
    ----------
    model : MLPImpairment
        Trained model (eval mode).
    X_background : ndarray
        Background (reference) dataset for DeepExplainer.
    X_explain : ndarray
        Instances to explain.
    feature_names : list of str
        Human-readable feature names (same order as columns).
    device : torch.device, optional
    max_display : int
        Maximum number of features to display.
    save_path : path-like, optional
        If given, save the figure instead of showing.
    title : str, optional

    Returns
    -------
    shap_values : ndarray of shape (n_samples, n_features, n_classes)
        Stacked SHAP values for all output classes.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bg_tensor = _to_tensor(X_background, device)
    ex_tensor = _to_tensor(X_explain, device)

    explainer = shap.DeepExplainer(model, bg_tensor)
    shap_values_list = explainer.shap_values(ex_tensor)
    # shap_values_list: list of 2 arrays, each (n_samples, n_features)
    shap_values = np.stack(shap_values_list, axis=-1)

    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values_list[1],
        X_explain,
        feature_names=feature_names,
        max_display=max_display,
        color_bar=True,
        show=False,
        plot_size=None,
    )
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)
        logger.info("SHAP summary plot saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)

    return shap_values


def shap_waterfall(
    model: MLPImpairment,
    X_background: np.ndarray,
    x_instance: np.ndarray,
    feature_names: List[str],
    *,
    device: Optional[torch.device] = None,
    max_display: int = 8,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> None:
    """Generate a SHAP waterfall plot for a single instance via DeepExplainer.

    Parameters
    ----------
    model : MLPImpairment
        Trained model.
    X_background : ndarray
        Background data for DeepExplainer.
    x_instance : ndarray of shape (1, d) or (d,)
        The single instance to explain.
    feature_names : list of str
    device : torch.device, optional
    max_display : int
    save_path : path-like, optional
    title : str, optional
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bg_tensor = _to_tensor(X_background, device)

    x_instance = np.atleast_2d(x_instance)
    inst_tensor = _to_tensor(x_instance, device)

    explainer = shap.DeepExplainer(model, bg_tensor)
    sv = explainer.shap_values(inst_tensor)
    # sv: list of 2 arrays, each (1, n_features)

    exp = shap.Explanation(
        values=sv[1][0],
        base_values=explainer.expected_value[1].item()
        if isinstance(explainer.expected_value[1], torch.Tensor)
        else float(explainer.expected_value[1]),
        data=x_instance[0],
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)
        logger.info("SHAP waterfall plot saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
"""SHAP-based model explanation utilities for MLP impairment models.

Uses ``shap.DeepExplainer`` (DeepLIFT) for efficient gradient-based
attribution on PyTorch models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from river_impairment.model import MLPImpairment

logger = logging.getLogger(__name__)


def _to_tensor(
    x: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor on *device*."""
    return torch.tensor(x, dtype=torch.float32).to(device)


def shap_summary(
    model: MLPImpairment,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    *,
    device: Optional[torch.device] = None,
    max_display: int = 15,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> np.ndarray:
    """Compute SHAP values via DeepExplainer and generate a summary plot.

    Parameters
    ----------
    model : MLPImpairment
        Trained model (eval mode).
    X_background : ndarray
        Background (reference) dataset for DeepExplainer.
    X_explain : ndarray
        Instances to explain.
    feature_names : list of str
        Human-readable feature names (same order as columns).
    device : torch.device, optional
    max_display : int
        Maximum number of features to display.
    save_path : path-like, optional
        If given, save the figure instead of showing.
    title : str, optional

    Returns
    -------
    shap_values : ndarray of shape (n_samples, n_features, n_classes)
        Stacked SHAP values for all output classes.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bg_tensor = _to_tensor(X_background, device)
    ex_tensor = _to_tensor(X_explain, device)

    explainer = shap.DeepExplainer(model, bg_tensor)
    shap_values_list = explainer.shap_values(ex_tensor)
    # shap_values_list: list of 2 arrays, each (n_samples, n_features)
    shap_values = np.stack(shap_values_list, axis=-1)

    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values_list[1],
        X_explain,
        feature_names=feature_names,
        max_display=max_display,
        color_bar=True,
        show=False,
        plot_size=None,
    )
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)
        logger.info("SHAP summary plot saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)

    return shap_values


def shap_waterfall(
    model: MLPImpairment,
    X_background: np.ndarray,
    x_instance: np.ndarray,
    feature_names: List[str],
    *,
    device: Optional[torch.device] = None,
    max_display: int = 8,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> None:
    """Generate a SHAP waterfall plot for a single instance via DeepExplainer.

    Parameters
    ----------
    model : MLPImpairment
        Trained model.
    X_background : ndarray
        Background data for DeepExplainer.
    x_instance : ndarray of shape (1, d) or (d,)
        The single instance to explain.
    feature_names : list of str
    device : torch.device, optional
    max_display : int
    save_path : path-like, optional
    title : str, optional
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bg_tensor = _to_tensor(X_background, device)

    x_instance = np.atleast_2d(x_instance)
    inst_tensor = _to_tensor(x_instance, device)

    explainer = shap.DeepExplainer(model, bg_tensor)
    sv = explainer.shap_values(inst_tensor)
    # sv: list of 2 arrays, each (1, n_features)

    exp = shap.Explanation(
        values=sv[1][0],
        base_values=explainer.expected_value[1].item()
        if isinstance(explainer.expected_value[1], torch.Tensor)
        else float(explainer.expected_value[1]),
        data=x_instance[0],
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)
        logger.info("SHAP waterfall plot saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
