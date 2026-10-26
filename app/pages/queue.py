import pandas as pd
import streamlit as st

import components as ui
from pkgtriage.services.scoring import InputError

ui.sidebar()
svc = ui.service()
a, scenarios = ui.assumptions()

st.title("Review queue")

source = st.radio(
    "Requests", ["Simulate a day from held-out packages", "Upload a sandbox export (CSV)"],
    horizontal=True,
)

raw = None
if source.startswith("Simulate"):
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("Requests in the day", 20, 1000, int(a.requests_per_day), step=10)
    prevalence = c2.select_slider(
        "Share malicious", options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
        value=scenarios["base"], format_func=lambda v: f"{v:.1%}",
    )
    seed = c3.number_input("Day (random seed)", 1, 9999, 1)
    raw = ui.simulate_day(ui.test_pool(), int(n), float(prevalence), int(seed))
    st.caption(
        "Drawn from the grouped test split, so the model has never seen these packages or their "
        "variants. Known labels are shown only so you can check the outcome."
    )
else:
    upload = st.file_uploader("CSV with the QUT-DV25 processed columns", type="csv")
    st.caption("A small example is in data/sample/sample_requests.csv.")
    if upload is not None:
        raw = pd.read_csv(upload)

if raw is None:
    st.stop()

try:
    batch = svc.score(raw, ui.current_policy())
except InputError as err:
    st.error(str(err))
    st.stop()

st.session_state.batch = batch
table = batch.table
counts = ui.decision_counts(table.decision)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Allowed automatically", counts["allow"])
c2.metric("Sent to an analyst", counts["review"])
c3.metric("Blocked", counts["block"])
used = counts["review"] / a.review_capacity_per_day
c4.metric("Analyst capacity used", f"{used:.0%}")
if used > 1:
    st.warning(
        f"{counts['review']} packages need review but the team can do "
        f"{a.review_capacity_per_day:.0f} a day. Raise the allow threshold or add capacity in the "
        "policy simulator."
    )

if "known_label" in table:
    missed = ((table.decision == "allow") & (table.known_label == "malicious")).sum()
    wrong_block = ((table.decision == "block") & (table.known_label == "benign")).sum()
    st.caption(
        f"Checked against known labels: {missed} malicious package(s) allowed, "
        f"{wrong_block} safe package(s) blocked."
    )

show = st.multiselect(
    "Show", list(ui.DECISION_TEXT), default=["review", "block"],
    format_func=lambda k: ui.DECISION_TEXT[k],
)
view = table[table.decision.isin(show)].copy()
view["decision"] = view.decision.map(ui.DECISION_TEXT)
view["direction"] = view.reason_raises_risk.map({True: "raises risk", False: "lowers risk"})
cols = ["package", "probability", "decision", "main_reason", "direction"]
if "known_label" in view:
    cols.append("known_label")

st.dataframe(
    view[cols],
    width="stretch",
    hide_index=True,
    column_config={
        "probability": st.column_config.ProgressColumn(
            "Probability malicious", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "main_reason": "Strongest reason",
        "direction": "Effect",
        "known_label": "Known label",
    },
)

export = table.assign(decision=table.decision.map(ui.DECISION_TEXT))
st.download_button(
    "Download decisions (CSV)", export.to_csv(index=False), "triage_decisions.csv", "text/csv"
)
st.caption("Open a package on the Package detail page to see all of its reasons.")
