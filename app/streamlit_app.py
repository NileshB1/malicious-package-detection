"""Package admission triage console.

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the script's own folder on the path, not the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Package Triage", layout="wide")

pages = [
    st.Page("pages/home.py", title="Overview", default=True),
    st.Page("pages/queue.py", title="Review queue"),
    st.Page("pages/package.py", title="Package detail"),
    st.Page("pages/simulator.py", title="Policy simulator"),
    st.Page("pages/evidence.py", title="Model evidence"),
]
st.navigation(pages).run()
