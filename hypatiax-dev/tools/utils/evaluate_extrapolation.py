import numpy as np
from sklearn.metrics import r2_score


def evaluate_model(func_true, func_pred, data):
    y_true = []
    y_pred = []

    for d in data:
        try:
            y_true.append(func_true(**d))
            y_pred.append(func_pred(**d))
        except Exception:
            continue

    if len(y_true) < 5:
        return np.nan

    return r2_score(y_true, y_pred)
