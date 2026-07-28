"""
pca_split_utils.py
==================
Reusable PCA-directed train/test split utility for the HypatiaX benchmarks.

Drop this file into hypatiax/utils/ and import with:
    from hypatiax.utils.pca_split_utils import pca_directed_split
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_directed_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.6,
    random_state: int | None = 42,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_test, y_train, y_test) split along PC-1.

    Samples are sorted by their projection onto the first principal component.
    The test set contains the top ``test_size`` fraction (largest PC-1 scores),
    creating an aggressive extrapolation scenario where the test set lies at
    the extrapolation frontier of the dominant axis of variance rather than
    randomly inside the convex hull of the training set.

    This is the FIX-C3 protocol used by run_comparative_suite_benchmark_pca.py
    and described in the paper §4.4.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Feature matrix.  1-D inputs are reshaped to (n, 1).
    y : array of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    test_size : float, default 0.6
        Fraction of samples assigned to the test (extrapolation) set.
        Must be in (0, 1).  The FIX-C3 protocol uses 0.6.
    random_state : int or None, default 42
        Passed to PCA for reproducibility (affects SVD solver for degenerate
        covariance matrices; the split itself is deterministic).
    scale : bool, default True
        Whether to standardise X before PCA.  Recommended when features have
        different scales; disable only for pre-scaled inputs.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray

    Raises
    ------
    ValueError
        If inputs are malformed or test_size is out of range.

    Notes
    -----
    *  The split is deterministic for a given (X, test_size): no stochastic
       element is introduced.  random_state only affects the PCA SVD solver.
    *  For univariate X the PC-1 projection is trivially the feature itself,
       so the split reduces to a max-value extrapolation split.
    *  The test set contains the top-``test_size`` fraction of samples sorted
       by PC-1 score (largest values → hardest extrapolation).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {X.shape}")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples "
            f"(got {len(X)} vs {len(y)})"
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    n = len(X)
    n_test = max(1, int(np.round(n * test_size)))
    n_train = n - n_test

    if n_train < 1:
        raise ValueError(
            f"test_size={test_size} leaves no training samples for n={n}"
        )

    # Project onto PC-1 (optionally after standardising)
    X_work = X
    if scale and X.shape[1] > 1:
        X_work = StandardScaler().fit_transform(X)

    pca = PCA(n_components=1, random_state=random_state)
    scores = pca.fit_transform(X_work).ravel()  # shape (n,)

    # Sort ascending; test set = top test_size fraction (hardest extrapolation)
    order = np.argsort(scores)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_syn = rng.standard_normal((100, 2))
    y_syn = X_syn[:, 0] * 2 + X_syn[:, 1]

    Xtr, Xte, ytr, yte = pca_directed_split(X_syn, y_syn, test_size=0.6)
    assert Xtr.shape == (40, 2), f"Expected (40,2), got {Xtr.shape}"
    assert Xte.shape == (60, 2), f"Expected (60,2), got {Xte.shape}"
    print(f"Train: {Xtr.shape}  Test: {Xte.shape}")
    print("Split sizes OK")
