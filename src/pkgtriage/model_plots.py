"""Model evaluation figures. Same style and colours as the EDA figures in plots.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from pkgtriage.plots import BENIGN, DOUBLE_COL, INK_SOFT, MALICIOUS, NEUTRAL, SINGLE_COL

MODEL_COLOUR = BENIGN  # blue line for the model throughout
BASELINE_COLOUR = MALICIOUS  # orange for the rule baseline


def _legend_above(ax, **kw):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.5, **kw)


def pr_curve_figure(y, model_score, baseline_score):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    for score, label, colour in (
        (model_score, "Gradient boosting (calibrated)", MODEL_COLOUR),
        (baseline_score, "Rule baseline", BASELINE_COLOUR),
    ):
        precision, recall, _ = precision_recall_curve(y, score)
        ax.step(recall, precision, where="post", color=colour, label=label)
    ax.axhline(np.mean(y), color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(0.02, np.mean(y) - 0.06, "share malicious in test set", fontsize=6.5, color=INK_SOFT)
    ax.set_xlabel("Recall (malicious packages caught)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.4, 1.01)
    _legend_above(ax)
    ax.set_title("Precision-recall on the grouped test set", pad=18)
    return fig


def reliability_figure(y, raw_score, calibrated_score, bins: int = 10):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1)
    for score, label, colour in (
        (raw_score, "Before calibration", NEUTRAL),
        (calibrated_score, "After calibration", MODEL_COLOUR),
    ):
        frac, mean_pred = calibration_curve(y, score, n_bins=bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", markersize=3.5, color=colour, label=label)
    ax.set_xlabel("Predicted probability of malicious")
    ax.set_ylabel("Observed share malicious")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _legend_above(ax)
    ax.set_title("Reliability on the grouped test set", pad=18)
    return fig


METRIC_LABELS = {
    "pr_auc": "PR-AUC on the test set", "roc_auc": "ROC-AUC on the test set",
    "recall_at_1pct_fpr": "Malicious caught at 1% false-alarm rate",
    "recall_at_5pct_fpr": "Malicious caught at 5% false-alarm rate",
}


# PR-AUC sits near 0.99 for most settings and hides the differences; recall at a fixed,
# low false-alarm rate is what the triage policy actually depends on
def ablation_figure(ablation: pd.DataFrame, metric: str = "recall_at_1pct_fpr"):
    d = ablation.copy()
    d["label"] = np.where(d.split == "random", d.setting + " (random split)", d.setting)
    d = d.sort_values(metric)
    colours = [NEUTRAL if s == "random" else MODEL_COLOUR for s in d.split]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    bars = ax.barh(d.label, d[metric], color=colours, height=0.6)
    for b, v in zip(bars, d[metric], strict=True):
        ax.text(
            v + 0.003,
            b.get_y() + b.get_height() / 2,
            f"{v:.3f}",   va="center",
            fontsize=6.5,   color=INK_SOFT
        )
    lo = max(0.0, d[metric].min() - 0.08)
    ax.set_xlim(lo, 1.03)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", visible=False)
    ax.set_title("Which inputs carry the signal")
    return fig


def shap_importance_figure(importance: pd.DataFrame, top: int = 12):
    d = importance.head(top).iloc[::-1]
    group_colour = {
        "network": MALICIOUS,
        "dependency": MALICIOUS,
        "missingness": MALICIOUS,
    }
    colours = [group_colour.get(g, MODEL_COLOUR) for g in d.group]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.barh(d.description.str.slice(0, 42), d.mean_abs_shap, color=colours, height=0.65)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.grid(axis="y", visible=False)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=MODEL_COLOUR, label="Behaviour (filesystem, syscall, process)"),
            Patch(color=MALICIOUS, label="Collection-sensitive"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.35, 1.0),
        ncol=1,
        handlelength=1.2,
    )
    ax.set_title("What drives the model's scores", pad=28)
    return fig


def score_distribution_figure(y, score):
    y = np.asarray(y)
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL / 2, 2.2))
    ax.hist(score[y == 0], bins=bins, color=BENIGN, alpha=0.85, label="Benign")
    ax.hist(score[y == 1], bins=bins, color=MALICIOUS, alpha=0.85, label="Malicious")
    ax.set_yscale("log")
    ax.set_xlabel("Calibrated probability of malicious")
    ax.set_ylabel("Packages (log scale)")
    ax.grid(axis="x", visible=False)
    _legend_above(ax)
    ax.set_title("Score distribution on the test set", pad=18)
    return fig
