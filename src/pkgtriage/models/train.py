from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from pkgtriage.models.dataset import SplitData

log = logging.getLogger(__name__)

# Small on purpose. The point of the project isn't squeezing out the last 0.1% of AUC, and a
# big search on 10k rows mostly tunes to noise.
PARAM_GRID = {
    "learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],    "l2_regularization": [0.0, 1.0]
}

FIXED_PARAMS = {
    "max_iter": 300,
    # early stopping would carve its own random validation set out of the training data and
    # put package siblings on both sides of it, so the iteration count is fixed instead
    "early_stopping": False
}

CV_FOLDS = 5


def make_model(seed: int, **params) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **FIXED_PARAMS, **params)


def _param_sets():
    keys = list(PARAM_GRID)
    for values in itertools.product(*PARAM_GRID.values()):
        yield dict(zip(keys, values, strict=True))


def tune(
    train: SplitData, seed: int, columns: list[str] | None = None
) -> tuple[dict, pd.DataFrame]:
    """Grid search scored by average precision, with folds grouped by package family."""
    X = train.X[columns] if columns else train.X
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    folds = list(cv.split(X, train.y, groups=train.families))

    rows = []
    for params in _param_sets():
        scores = []
        for fit_idx, val_idx in folds:
            model = make_model(seed, **params).fit(X.iloc[fit_idx], train.y.iloc[fit_idx])
            p = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(average_precision_score(train.y.iloc[val_idx], p))
        rows.append({**params, "cv_pr_auc_mean": np.mean(scores), "cv_pr_auc_std": np.std(scores)})
        log.info("%s -> %.4f", params, rows[-1]["cv_pr_auc_mean"])

    results = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False)
    best = results.iloc[0][list(PARAM_GRID)].to_dict()
    # pandas hands back floats; tree sizes have to be ints
    for key in ("max_leaf_nodes", "min_samples_leaf"):
        best[key] = int(best[key])
    return best, results.reset_index(drop=True)


def fit(train: SplitData, params: dict, seed: int, columns: list[str] | None = None):
    X = train.X[columns] if columns else train.X
    return make_model(seed, **params).fit(X, train.y)
