"""Data loading, preprocessing, and spatiotemporal train–test splitting.

Expected CSV columns
--------------------
- Predictor variables (continuous & categorical).
- Dependent variables (binary impairment labels: 0/1).
- ``연도`` (year) and ``회차`` (survey round) for temporal splitting.
- ``조사구간명`` (site name) for spatial-independence verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import column_or_1d

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Ordinal label encoder (preserves user-supplied class order)
# ------------------------------------------------------------------

class OrdinalLabelEncoder(LabelEncoder):
    """Label encoder that preserves the order of classes given at ``fit``."""

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


# ------------------------------------------------------------------
# Default categorical encoding maps
# ------------------------------------------------------------------

CONNECTIVITY_CLASSES = [
    "Natural", "Channel", "Compound", "Dam", "Construction",
    "Natural/Dam", "Channel/Dam", "Compound/Dam",
]

EMBEDDING_CLASSES = ["None", "10-20%", "20-50%", "50-80%", ">80%"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_and_preprocess(
    data_path: str | Path,
    pred_vars: List[str],
    dep_vars: List[str],
    connectivity_col: str = "하천흐름의 단순화",
    embedding_col: str = "기질매몰도(%)",
    encoding: str = "cp949",
) -> Tuple[pd.DataFrame, OrdinalLabelEncoder, OrdinalLabelEncoder]:
    """Load a CSV file and apply ordinal encoding to categorical predictors.

    Parameters
    ----------
    data_path : path-like
        Path to the CSV file.
    pred_vars : list of str
        Predictor column names.
    dep_vars : list of str
        Dependent (target) column names.
    connectivity_col : str
        Column name for river-connectivity categorical variable.
    embedding_col : str
        Column name for substrate-embedding categorical variable.
    encoding : str
        File encoding (default ``"cp949"`` for Korean).

    Returns
    -------
    df : DataFrame
        Preprocessed dataframe (NaN rows dropped, categoricals encoded).
    le_conn : OrdinalLabelEncoder
        Fitted encoder for *connectivity_col*.
    le_embed : OrdinalLabelEncoder
        Fitted encoder for *embedding_col*.
    """
    df = pd.read_csv(data_path, encoding=encoding)
    df = df.dropna().copy()
    logger.info("Loaded %d rows from %s (after dropping NaN)", len(df), data_path)

    if embedding_col in df.columns:
        df[embedding_col] = df[embedding_col].replace("거의없음", "None")

    le_conn = OrdinalLabelEncoder()
    le_conn.fit(CONNECTIVITY_CLASSES)
    if connectivity_col in df.columns:
        df[connectivity_col] = le_conn.transform(df[connectivity_col])

    le_embed = OrdinalLabelEncoder()
    le_embed.fit(EMBEDDING_CLASSES)
    if embedding_col in df.columns:
        df[embedding_col] = le_embed.transform(df[embedding_col])

    return df, le_conn, le_embed


def spatiotemporal_split(
    df: pd.DataFrame,
    pred_vars: List[str],
    dep_vars: List[str],
    train_years: List[int],
    test_years: List[int],
    site_col: str = "조사구간명",
    year_col: str = "연도",
    site_river_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """Split data by year and optionally verify spatial independence.

    Parameters
    ----------
    df : DataFrame
        Preprocessed data (output of :func:`load_and_preprocess`).
    pred_vars, dep_vars : list of str
        Column name lists.
    train_years : list of int
        Years assigned to the training set.
    test_years : list of int
        Years assigned to the test set.
    site_col : str
        Column that identifies the monitoring site.
    year_col : str
        Column that stores the survey year.
    site_river_map : dict, optional
        ``{site_name: river_name}`` mapping for spatial overlap check.

    Returns
    -------
    dict with keys:
        ``x_train``, ``y_train``, ``x_test``, ``y_test``,
        ``train_df``, ``test_df``, ``spatial_overlap``.
    """
    train_df = df[df[year_col].isin(train_years)].copy()
    test_df = df[df[year_col].isin(test_years)].copy()

    x_train = train_df[pred_vars]
    y_train = train_df[dep_vars]
    x_test = test_df[pred_vars]
    y_test = test_df[dep_vars]

    logger.info(
        "Train (%s): %s  |  Test (%s): %s",
        train_years, x_train.shape, test_years, x_test.shape,
    )

    spatial_overlap = None
    if site_river_map is not None and site_col in df.columns:
        train_rivers = {site_river_map.get(s, s) for s in train_df[site_col].unique()}
        test_rivers = {site_river_map.get(s, s) for s in test_df[site_col].unique()}
        spatial_overlap = sorted(train_rivers & test_rivers) or None
        if spatial_overlap:
            logger.warning("Spatial overlap detected: %s", spatial_overlap)
        else:
            logger.info("No spatial overlap — spatially independent split.")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "spatial_overlap": spatial_overlap,
    }


def fit_scaler(
    x_train: np.ndarray | pd.DataFrame,
) -> Tuple[MinMaxScaler, np.ndarray]:
    """Fit a MinMaxScaler on training data and return (scaler, scaled array)."""
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_train)
    return scaler, x_scaled
