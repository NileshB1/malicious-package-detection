"""Figures for the decision policy and the business case."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkgtriage.plots import BENIGN, DOUBLE_COL, INK, INK_SOFT, MALICIOUS, NEUTRAL, SINGLE_COL

# model and rule keep the colours they have in the model figures
POLICY_COLOURS = {
    "allow everything": NEUTRAL,  "review everything": "#1baf7a",
    "rule screen (flag to review)": MALICIOUS,
    "model triage": BENIGN
}
POLICY_LABELS = {
    "allow everything": "Allow everything", "review everything": "Review everything",
    "rule screen (flag to review)": "Rule screen",
    "model triage": "Model triage"
}
POLICY_ORDER = list(POLICY_COLOURS)


def _money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def cost_by_scenario_figure(scenarios: pd.DataFrame):
    names = list(dict.fromkeys(scenarios.scenario))
    fig, axes = plt.subplots(1, len(names), figsize=(DOUBLE_COL, 2.3), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        d = scenarios[scenarios.scenario == name].set_index("policy").reindex(POLICY_ORDER)
        y = np.arange(len(d))
        for i, (policy, row) in enumerate(d.iterrows()):
            ax.barh(
                i,
                row.cost_per_10k,   color=POLICY_COLOURS[policy],
                height=0.62,  hatch=None if row.feasible else "///",
                edgecolor="white"
            )
            note = "" if row.feasible else "  over capacity"
            ax.text(
                row.cost_per_10k * 1.15,
                i,
                _money(row.cost_per_10k) + note,
                va="center",   fontsize=6.5,  color=INK_SOFT
            )
        ax.set_xscale("log")
        ax.set_xlim(d.cost_per_10k.min() / 3, d.cost_per_10k.max() * 40)
        ax.set_yticks(y, [POLICY_LABELS[p] for p in d.index])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        prevalence = d.prevalence.iloc[0]
        ax.set_title(f"{name}: {prevalence:.1%} malicious")
        ax.set_xlabel("Expected cost per 10k requests (log)")
    fig.tight_layout()
    return fig


def workload_figure(base: pd.DataFrame, capacity: float):
    d = base.set_index("policy").reindex(POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    x = np.arange(len(d))
    ax.bar(x, d.reviews_per_day, color=[POLICY_COLOURS[p] for p in d.index], width=0.6)
    ax.axhline(capacity, color=INK, linestyle="--", linewidth=1)
    # left end of the line, above the empty "allow everything" slot, so it can't hit a bar
    ax.text(
        -0.45, capacity * 1.08,
        f"analyst capacity ({capacity:.0f}/day)",
        ha="left",  va="bottom",  fontsize=6.5,  color=INK
    )
    for xi, v in zip(x, d.reviews_per_day, strict=True):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK_SOFT)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(x, [POLICY_LABELS[p].replace(" ", "\n", 1) for p in d.index])
    ax.set_ylabel("Packages to review per day")
    ax.grid(axis="x", visible=False)
    ax.set_title("Analyst workload, base scenario")
    return fig


ASSUMPTION_LABELS = {
    "breach_probability": "Breach chance per malicious install",
    "breach_cost": "Cost of a breach", "review_minutes": "Minutes per review",
    "analyst_hourly_rate": "Analyst hourly cost",  "analyst_detection_rate": "Analyst detection rate",
    "developer_hours_per_false_block": "Developer hours lost per false block",
    "requests_per_day": "Package requests per day",  "analysts": "Number of analysts",
}


def tornado_figure(sens: pd.DataFrame):
    d = sens.iloc[::-1]
    base = d.saving_at_base.iloc[0]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    y = np.arange(len(d))
    lows = d.saving_at_low.to_numpy()
    highs = d.saving_at_high.to_numpy()
    ax.barh(y, lows - base, left=base, color=NEUTRAL, height=0.6, label="Low end of range")
    ax.barh(y, highs - base, left=base, color=BENIGN, height=0.6, label="High end of range")
    ax.axvline(base, color=INK, linewidth=1)
    values = np.concatenate([lows, highs])
    if values.min() > 0:
        ax.set_xscale("log")
        ax.set_xlim(values.min() / 2, values.max() * 2)
        scale_note = "log"
    else:
        # some setting makes the model policy the more expensive one; show where zero is
        ax.axvline(0, color=MALICIOUS, linewidth=1, linestyle="--")
        ax.set_xscale("symlog", linthresh=1000)
        scale_note = "symlog"
    ax.set_yticks(y, [ASSUMPTION_LABELS.get(a, a) for a in d.assumption])
    ax.set_xlabel(f"Saving vs reviewing everything\nper 10k requests (USD, {scale_note})")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2)
    ax.set_title("Which assumptions move the answer", pad=16)
    return fig
