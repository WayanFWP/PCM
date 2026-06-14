import sys
from pathlib import Path

import streamlit as st
import time
from gui.nucleus_utils import (
    load, bilateral_filter, clahe, percentile_threshold,
    dilate, xml_to_mask, dice_iou, plot_pipeline,
    TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="AY Pipeline", page_icon="🔬", layout="wide")
st.title("AY — TCGA-AY-A8YK-01A-01-TS1")

fname = "TCGA-AY-A8YK-01A-01-TS1"

with st.spinner("Running AY pipeline..."):
    img, h_ch = load(TISSUE_DIR, fname)
    gt = xml_to_mask(ANN_DIR / f"{fname}.xml", img.shape)

    t0 = time.time()
    blurred = bilateral_filter(h_ch, d=7, sigma_color=30, sigma_space=50)
    eq = clahe(blurred, clip_limit=0.5, tile_grid=(4, 4))
    result = dilate(percentile_threshold(eq, pct=69), kernel_size=3, iterations=1)
    elapsed = time.time() - t0

    dice, iou = dice_iou(result, gt)

st.metric("DICE", f"{dice:.4f}")
st.metric("IoU", f"{iou:.4f}")
st.metric("Time", f"{elapsed:.2f}s")

fig = plot_pipeline(img, gt, result, dice, iou, title=f"AY — DICE: {dice:.4f}")
st.pyplot(fig)
