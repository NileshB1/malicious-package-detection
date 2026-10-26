"""Shared state, cached loaders and small display helpers for the Streamlit pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pkgtriage import plots
from pkgtriage.business.costs import load_assumptions
from pkgtriage.business.policy import ThresholdPolicy
from pkgtriage.config import load_config
from pkgtriage.data.download import load_raw
from pkgtriage.features.build import LABEL, NAME
from pkgtriage.models.dataset import load_splits
from pkgtriage.services.scoring import ScoringService

DECISION_TEXT = {"allow": "Allow", "review": "Send to analyst", "block": "Block"}


@st.cache_resource(show_spinner="Loading the model...")
def service() -> ScoringService:
    return ScoringService()


@st.cache_data(show_spinner=False)
def assumptions():
    return load_assumptions()


@st.cache_data(show_spinner="Loading held-out packages...")
def test_pool() -> pd.DataFrame:
    """Raw rows of the grouped test split - packages the model never saw while training."""
    cfg = load_config()
    raw = load_raw(cfg)
    split = pd.read_csv(cfg.paths.splits_dir / "grouped.csv")
    test_names = set(split.loc[split.split == "test", NAME])
    return raw[raw[NAME].isin(test_names)].reset_index(drop=True)


@st.cache_data(show_spinner="Scoring the held-out packages...")
def test_scores() -> dict[str, np.ndarray]:
    """Validation and test probabilities, labels and rule flags for the simulator."""
    svc = service()
    splits = load_splits("grouped")
    val, test = splits["validation"], splits["test"]
    return {
        "val_p": svc.bundle.predict_proba(val.X),
        "val_y": val.y.to_numpy(),
        "test_p": svc.bundle.predict_proba(test.X),
        "test_y": test.y.to_numpy(),
        "rule_flags": svc.bundle.baseline.flag(test.X),
    }


def simulate_day(pool: pd.DataFrame, n: int, prevalence: float, seed: int) -> pd.DataFrame:
    """A day's worth of requests drawn from the held-out packages at a chosen malicious share."""
    rng = np.random.default_rng(seed)
    n_mal = int(rng.binomial(n, prevalence))
    mal = pool[pool[LABEL] == 1].sample(n_mal, random_state=seed)
    ben = pool[pool[LABEL] == 0].sample(n - n_mal, random_state=seed)
    return pd.concat([mal, ben]).sample(frac=1, random_state=seed).reset_index(drop=True)


def current_policy() -> ThresholdPolicy:
    if "t_low" not in st.session_state:
        default = service().default_policy
        st.session_state.t_low = default.t_low
        st.session_state.t_high = default.t_high
    return ThresholdPolicy(st.session_state.t_low, st.session_state.t_high)


def decisions_log() -> list[dict]:
    return st.session_state.setdefault("decisions", [])


def money(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.1f}k"
    return f"${v:,.0f}"


def decision_counts(decisions: pd.Series) -> dict[str, int]:
    return {k: int((decisions == k).sum()) for k in DECISION_TEXT}


def sidebar():
    a, _ = assumptions()
    policy = current_policy()
    with st.sidebar:
        st.markdown("**Current policy**")
        st.markdown(
            f"- Allow below **{policy.t_low:.3f}**\n"
            f"- Block at or above **{policy.t_high:.3f}**\n"
            f"- Otherwise send to an analyst\n\n"
            f"Analyst capacity: **{a.review_capacity_per_day:.0f}** reviews/day"
        )
        if st.button("Reset to saved thresholds"):
            default = service().default_policy
            st.session_state.t_low, st.session_state.t_high = default.t_low, default.t_high
            st.rerun()
        st.caption(
            "Scores are offline estimates from the QUT-DV25 dataset. "
            "A block is a recommendation; an analyst can override it."
        )


def pyplot(fig):
    st.pyplot(fig, width="stretch")


def report_style():
    plots.use_report_style()
