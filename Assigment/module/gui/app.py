import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Nucleus Segmentation",
    layout="wide",
)

st.title("Nucleus Segmentation — MoNuSeg 2018")
st.markdown(
    """
    **Two pages available via the sidebar:**

    | Page | Description |
    |------|-------------|
    | **Pipelines** | Pre-tuned pipelines for 4 tissue samples in a 2×2 grid. Click *View Image* to see the 4-panel overlay. |
    | **Random Search** | Brute-force random search over preprocessing/segmentation parameter space. Configure ranges, run trials, and find the best pipeline per sample. |
    """
)
