"""
pca_split_utils.py
==================
Reusable PCA-directed train/test split utility for the HypatiaX benchmarks.

Drop this file into hypatiax/experiments/ and import with:
    from hypatiax.utils.pca_split_utils import pca_directed_split
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def pca_directed_split(X, y, test_size=0.6, random_state=None):
    """
    PCA-directed OOD split.

    Samples are projected onto PC1 and sorted by their PC1 score.

    Training uses the first train_frac of samples along PC1.
    Testing uses the remainder.

    If multiplier is specified, the test region is clipped to:

        pc1_train_max + multiplier * train_range

    mimicking the extrapolation semantics used by
    build_extrap_split().

        Data is sorted along the first principal component and split at the
    (1 - test_size) quantile, creating an 'aggressive extrapolation'
    scenario where the test set lies beyond the training set along the
    primary axis of variance.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or pd.Series
        Target vector of length n_samples.
    test_size : float, default 0.6
        Fraction of samples to include in the test split (0 < test_size < 1).
    random_state : int or None, default None
        Passed to PCA for reproducibility (PCA is deterministic for full SVD
        but some solvers use randomisation).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split arrays.

    Raises
    ------
    ValueError
        If X has zero samples or zero features.
    """

    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be between 0 and 1, got {test_size}"
        )

    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_series = (
        y.copy()
        if isinstance(y, pd.Series)
        else pd.Series(y, name="target")
    )

    n_samples, n_features = X_df.shape

    if n_samples < 2:
        raise ValueError(
            "Need at least 2 samples for PCA split."
        )

    if n_features < 1:
        raise ValueError(
            "Need at least 1 feature."
        )

    # PCA projection
    pca = PCA(
        n_components=1,
        random_state=random_state,
    )

    pc1_scores = pca.fit_transform(X_df).ravel()

    # Sort along PC1
    order = np.argsort(pc1_scores)

    # Safe split
    split_point = int(
        n_samples * (1.0 - test_size)
    )

    split_point = max(1, split_point)
    split_point = min(split_point, n_samples - 1)

    train_idx = order[:split_point]
    test_idx = order[split_point:]

    if isinstance(X, pd.DataFrame):
        X_train = X_df.loc[train_idx].values
        X_test = X_df.loc[test_idx].values
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    if isinstance(y, pd.Series):
        y_train = y_series.loc[train_idx].values
        y_test = y_series.loc[test_idx].values
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]

    return X_train, X_test, y_train, y_test
