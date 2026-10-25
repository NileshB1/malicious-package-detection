from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    roc_auc_score,  roc_curve
)


def recall_at_fpr(y, score, max_fpr: float) -> float:
    """Share of malicious packages caught while wrongly flagging at most max_fpr of benign ones."""
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def score_metrics(y, score, probabilities: bool = True) -> dict[str, float]:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    out = {
        "roc_auc": roc_auc_score(y, score),  "pr_auc": average_precision_score(y, score),
        "recall_at_1pct_fpr": recall_at_fpr(y, score, 0.01),   "recall_at_5pct_fpr": recall_at_fpr(y, score, 0.05)
    }
    if probabilities:
        out["brier"] = brier_score_loss(y, score)
    return out


def bootstrap_ci(
    y, score, probabilities: bool = True, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05
) -> pd.DataFrame:
    """Point estimate and percentile interval for each metric, resampling test packages."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    point = score_metrics(y, score, probabilities)

    samples = {k: [] for k in point}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        for k, v in score_metrics(y[idx], score[idx], probabilities).items():
            samples[k].append(v)

    rows = []
    for k, v in point.items():
        lo, hi = np.percentile(samples[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
