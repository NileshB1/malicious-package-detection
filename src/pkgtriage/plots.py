"""Matplotlib figures for the report and the app"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts run headless; Streamlit passes figures explicitly anyway

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkgtriage.config import load_config

SINGLE_COL = 3.5
DOUBLE_COL = 7.16

# benign / malicious keep the same colours in every figure
BENIGN = "#2a78d6"
MALICIOUS = "#eb6834"
NEUTRAL = "#8a8984"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"


def use_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.edgecolor": INK_SOFT,
            "axes.labelcolor": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.5,
        }
    )


def save(fig, name: str, images_dir: Path | None = None) -> Path:
    images_dir = images_dir or load_config().paths.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _class_legend(ax):
    from matplotlib.patches import Patch

    # above the axes so it never sits on top of a bar
    ax.legend(
        handles=[Patch(color=BENIGN, label="Benign"), Patch(color=MALICIOUS, label="Malicious")],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.2,
    )


def missingness_figure(miss: pd.DataFrame):
    # one row per pattern family is enough; Pattern_2..10 are almost identical
    keep = miss[~miss.column.str.match(r"Pattern_([3-9]|10)$")].copy()
    keep["label"] = keep.column.str.replace("_", " ")
    keep = keep.sort_values("missing_malicious_pct")
    y = np.arange(len(keep))
    h = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    ax.barh(y + h / 2, keep.missing_benign_pct, height=h, color=BENIGN)
    ax.barh(y - h / 2, keep.missing_malicious_pct, height=h, color=MALICIOUS)
    ax.set_yticks(y, keep.label)
    ax.set_xlabel("Rows with the field missing (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="y", visible=False)
    _class_legend(ax)
    ax.set_title("Missing text fields by class", pad=16)
    return fig


def count_distributions_figure(df: pd.DataFrame, features: list[str]):
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL, 1.75 * rows))
    for ax, feat in zip(axes.flat, features, strict=False):
        data = [np.log10(df.loc[df.Level == k, feat] + 1) for k in (0, 1)]
        parts = ax.boxplot(
            data,
            vert=False,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1},
        )
        for patch, colour in zip(parts["boxes"], (BENIGN, MALICIOUS), strict=True):
            patch.set_facecolor(colour)
            patch.set_edgecolor(colour)
        for key in ("whiskers", "caps"):
            for line in parts[key]:
                line.set_color(INK_SOFT)
        ax.set_yticks([1, 2], ["Benign", "Malicious"])
        ax.set_title(feat.replace("_", " "))
        ax.set_xlabel("log10(count + 1)")
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def single_feature_auc_figure(auc: pd.DataFrame):
    d = auc.sort_values("auc")
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.barh(d.feature.str.replace("_", " "), d.auc, color=BENIGN, height=0.65)
    ax.axvline(0.5, color=NEUTRAL, linewidth=1, linestyle="--")
    ax.text(0.505, -0.9, "chance", color=INK_SOFT, fontsize=6.5)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("ROC-AUC of the feature on its own")
    ax.grid(axis="y", visible=False)
    ax.set_title("No single count separates the classes")
    return fig


def process_tokens_figure(tokens: pd.DataFrame, top: int = 12):
    d = tokens[tokens.group != "package"].head(top).iloc[::-1]
    y = np.arange(len(d))
    h = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.barh(y + h / 2, d.benign_pct, height=h, color=BENIGN)
    ax.barh(y - h / 2, d.malicious_pct, height=h, color=MALICIOUS)
    ax.set_yticks(y, [f"{t} ({g})" for t, g in zip(d.token, d.group, strict=True)])
    ax.set_xlabel("Packages whose trace lists the process (%)")
    ax.grid(axis="y", visible=False)
    _class_legend(ax)
    ax.set_title("Background processes differ by class", pad=16)
    return fig


def family_size_figure(per_family: pd.DataFrame):
    bins = [1, 2, 3, 5, 10, 20, 50]
    labels = ["1", "2", "3-4", "5-9", "10-19", "20+"]
    sized = pd.cut(per_family["size"], bins=bins, right=False, labels=labels)
    rows = per_family.assign(bucket=sized).groupby("bucket", observed=False)["size"].sum()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    ax.bar(rows.index.astype(str), rows.values, color=BENIGN, width=0.65)
    for x, v in enumerate(rows.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=6.5, color=INK_SOFT)
    ax.set_yscale("log")
    ax.set_xlabel("Packages in the same name family")
    ax.set_ylabel("Rows (log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Near-duplicate package families")
    return fig


def correlation_figure(df: pd.DataFrame, features: list[str]):
    corr = df[features].corr(method="spearman")
    labels = [f.replace("_", " ") for f in features]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, DOUBLE_COL * 0.55))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Correlation between count features")
    return fig


def split_overlap_figure(summary: pd.DataFrame):
    """Rows in each split whose package family also appears in another split."""
    order = ["train", "validation", "test"]
    random = summary[summary.split_type == "random"].set_index("split").reindex(order)
    grouped = summary[summary.split_type == "grouped"].set_index("split").reindex(order)
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.1))
    bars = [
        ax.bar(x - w / 2, random.families_shared_with_other_splits, w, color=NEUTRAL),
        ax.bar(x + w / 2, grouped.families_shared_with_other_splits, w, color=BENIGN),
    ]
    for group in bars:
        for b in group:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{int(b.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK_SOFT,
            )
    ax.set_xticks(x, ["Train", "Validation", "Test"])
    ax.set_ylabel("Rows with a sibling\nin another split")
    ax.grid(axis="x", visible=False)
    ax.legend(
        bars,
        ["Random split", "Family-grouped split"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.2,
    )
    ax.set_title("Package variants leaking across splits", pad=16)
    return fig
