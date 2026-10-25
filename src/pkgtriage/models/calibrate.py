"""Probability calibration on the validation split.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from pkgtriage.models.dataset import SplitData

METHODS = ("sigmoid", "isotonic")


def _calibrator(model, method: str) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(FrozenEstimator(model), method=method)


def compare_methods(model, validation: SplitData, seed: int, folds: int = 5) -> pd.DataFrame:
    X, y = validation.X[model.feature_names_in_], validation.y
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    split_idx = list(cv.split(X, y, groups=validation.families))

    raw = model.predict_proba(X)[:, 1]
    rows = [{"method": "none", "brier": brier_score_loss(y, raw), "log_loss": log_loss(y, raw)}]
    for method in METHODS:
        oof = np.zeros(len(y))
        for fit_idx, held_idx in split_idx:
            cal = _calibrator(model, method).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            oof[held_idx] = cal.predict_proba(X.iloc[held_idx])[:, 1]
        rows.append(
            {"method": method, "brier": brier_score_loss(y, oof), "log_loss": log_loss(y, oof)}
        )
    return pd.DataFrame(rows)


def calibrate(model, validation: SplitData, seed: int):
    """Returns (calibrated model or the raw model, comparison table)."""
    table = compare_methods(model, validation, seed)
    best = table.sort_values("brier").iloc[0].method
    if best == "none":
        return model, table
    X = validation.X[model.feature_names_in_]
    return _calibrator(model, best).fit(X, validation.y), table
