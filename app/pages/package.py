from datetime import datetime

import pandas as pd
import streamlit as st

import components as ui
from pkgtriage import model_plots

ui.sidebar()
svc = ui.service()

st.title("Package detail")

batch = st.session_state.get("batch")
if batch is None:
    st.info("Load a queue on the Review queue page first.")
    st.stop()

table = batch.table
# review and block first, since those are the ones that need a person
rank = table.decision.map({"block": 0, "review": 1, "allow": 2})
order = table.assign(rank=rank).sort_values(["rank", "probability"], ascending=[True, False])
labels = {
    i: f"{row.package}  |  {ui.DECISION_TEXT[row.decision]}  |  {row.probability:.3f}"
    for i, row in order.iterrows()
}
idx = st.selectbox("Package", list(labels), format_func=labels.get)
row = table.loc[idx]

c1, c2, c3 = st.columns(3)
c1.metric("Probability malicious", f"{row.probability:.3f}")
c2.metric("Recommendation", ui.DECISION_TEXT[row.decision])
if "known_label" in table:
    c3.metric("Known label (held-out data)", row.known_label)

drivers = svc.explain(batch.features.loc[idx])
st.subheader("Why it scored this way")
ui.report_style()
ui.pyplot(model_plots.drivers_figure(drivers))
st.caption(
    "Orange pushes towards malicious, blue away from it. Hatched bars come from features that "
    "partly reflect how the training data was collected, so give them less weight."
)

for d in drivers:
    effect = "raises risk" if d.raises_risk else "lowers risk"
    note = " *(collection-sensitive)*" if d.collection_sensitive else ""
    st.markdown(f"- {d.sentence()} - {effect}{note}")

st.subheader("Before deciding")
st.markdown(
    "- Does the package name look like a misspelling of a popular package?\n"
    "- Is there a public source repository, and does it match what was installed?\n"
    "- Do the file accesses above (/root, /etc, home directories) make sense for what the "
    "package claims to do?\n"
    "- Did it start shells, compilers or unexpected processes during install?"
)

st.subheader("Record your decision")
with st.form("decision"):
    choice = st.radio("Decision", ["Allow", "Block", "Escalate"], horizontal=True)
    note = st.text_input("Note (optional)")
    if st.form_submit_button("Save"):
        ui.decisions_log().append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "package": row.package,
                "model_probability": round(float(row.probability), 4),
                "recommended": ui.DECISION_TEXT[row.decision],
                "analyst_decision": choice,
                "override": choice != ui.DECISION_TEXT[row.decision],
                "note": note,
            }
        )
        st.success("Saved.")

log = ui.decisions_log()
if log:
    st.subheader("Decisions this session")
    df = pd.DataFrame(log)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download log (CSV)", df.to_csv(index=False), "analyst_decisions.csv")
