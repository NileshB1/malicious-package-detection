import pandas as pd
import streamlit as st

import components as ui
from pkgtriage.config import load_config

ui.sidebar()
cfg = load_config()
tables, images = cfg.paths.tables_dir, cfg.paths.images_dir

st.title("Model evidence")
st.write(
    "How well the model does on packages it never saw, how much of that holds up without the "
    "features that reflect how the data was collected, and what it can't do."
)

metrics = pd.read_csv(tables / "test_metrics.csv")
pivot = metrics.pivot(index="model", columns="metric", values="value")
st.subheader("Grouped test set (2,141 packages, no family overlap with training)")
st.dataframe(
    pivot[["pr_auc", "roc_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr", "brier"]].round(4),
    width="stretch",
)

c1, c2 = st.columns(2)
c1.image(str(images / "model_pr_curve.png"), width="stretch")
c2.image(str(images / "model_reliability.png"), width="stretch")

st.subheader("Robustness checks")
ablation = pd.read_csv(tables / "ablation.csv")
c1, c2 = st.columns([3, 2])
c1.image(str(images / "model_ablation.png"), width="stretch")
c2.dataframe(
    ablation[["setting", "split", "n_features", "recall_at_1pct_fpr", "pr_auc"]].round(4),
    width="stretch", hide_index=True,
)
st.markdown(
    "With only behaviour features (no network, dependency or missingness groups) the model "
    "still catches 95.6% of malicious packages at a 1% false-alarm rate, against 99.3% with "
    "everything. The difference is the part most likely to come from collection artefacts."
)

st.subheader("What drives the scores")
c1, c2 = st.columns([3, 2])
c1.image(str(images / "model_shap_importance.png"), width="stretch")
share = pd.read_csv(tables / "shap_group_share.csv", index_col=0)
c2.dataframe(share.round(3), width="stretch")

st.subheader("Known limitations")
st.markdown(
    "- Malicious samples were installed from local files and benign ones from PyPI, which "
    "leaves traces in network, dependency and background-process data.\n"
    "- Benign labels mean 'not known to be malicious', not independently verified.\n"
    "- One sandbox, one ecosystem (PyPI), no timestamps, so drift over time is untested.\n"
    "- Install-time traces can't see malware that only acts when imported.\n"
    "- Business figures are offline estimates under the assumptions in configs/business.yaml."
)
