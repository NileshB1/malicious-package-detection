import pandas as pd
import streamlit as st

import components as ui
from pkgtriage.config import load_config

ui.sidebar()
cfg = load_config()
a, scenarios = ui.assumptions()

st.title("Package admission triage")
st.write(
    "New open-source packages are installed once in a sandbox and their behaviour is recorded. "
    "This console scores each one and recommends **allow**, **send to an analyst** or "
    "**block**, with the reasons behind each recommendation, so a small security team can "
    "spend its review time where it matters."
)

scen = pd.read_csv(cfg.paths.tables_dir / "business_scenarios.csv")
base = scen[scen.scenario == "base"].set_index("policy")
model, review_all = base.loc["model triage"], base.loc["review everything"]

st.subheader("Base case at a glance")
st.caption(
    f"{a.requests_per_day:.0f} requests a day, {scenarios['base']:.0%} malicious, "
    f"{a.review_capacity_per_day:.0f} analyst reviews a day. "
    "Offline estimate, not measured savings."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews per day", f"{model.reviews_per_day:.1f}",
          f"{model.reviews_per_day - review_all.reviews_per_day:.0f} vs reviewing everything",
          delta_color="inverse")
c2.metric("Malicious let through per 10k", f"{model.missed_malicious_per_10k:.2f}",
          f"{model.missed_malicious_per_10k - review_all.missed_malicious_per_10k:.1f}",
          delta_color="inverse")
c3.metric("Safe packages blocked per 10k", f"{model.false_blocks_per_10k:.0f}")
c4.metric("Expected cost per 10k", ui.money(model.cost_per_10k),
          f"{ui.money(model.cost_per_10k - review_all.cost_per_10k)}", delta_color="inverse")

st.subheader("How to use it")
st.markdown(
    "1. **Review queue** - load a day of requests (drawn from held-out packages) or upload a "
    "sandbox export, and see what goes where.\n"
    "2. **Package detail** - open one package, read why it scored the way it did, and record "
    "your decision.\n"
    "3. **Policy simulator** - change capacity, costs and thresholds and see the effect on "
    "workload, missed attacks and cost.\n"
    "4. **Model evidence** - test results, robustness checks and known limitations."
)

st.subheader("Before you trust a score")
st.markdown(
    "- The model was trained on QUT-DV25, where malicious packages were installed from local "
    "archives and benign ones from PyPI. Reasons marked *collection-sensitive* partly reflect "
    "that difference rather than what the package did.\n"
    "- Malware that waits until it is imported, rather than installed, won't show up here.\n"
    "- Every block is a recommendation. Overrides are expected and recorded."
)
