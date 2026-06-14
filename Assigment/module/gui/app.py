import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Nucleus Segmentation",
    page_icon="🔬",
    layout="wide",
)

st.title("Nucleus Segmentation Pipeline")
st.markdown(
    """
    This app demonstrates nucleus segmentation on **MoNuSeg 2018** H&E-stained tissue images.

    Each page runs a different preprocessing/segmentation pipeline optimized for a specific
    tissue sample:

    | Page | Sample | Key Operations |
    |------|--------|---------------|
    | **AR** | TCGA-AR-A1AS | median → bilateral → otsu → filter → morph → dilate |
    | **AY** | TCGA-AY-A8YK | bilateral → clahe → percentile thresh → dilate |
    | **E2** | TCGA-E2-A1B5 | median → otsu → morph_clean |
    | **RD** | TCGA-RD-A8N9 | clahe → otsu → watershed → dilate |

    Use the sidebar to navigate between samples.
    """
)
