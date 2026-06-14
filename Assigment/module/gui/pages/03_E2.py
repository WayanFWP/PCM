import sys
from pathlib import Path

import streamlit as st
import time
from gui.nucleus_utils import (
    load, median_blur, otsu_threshold, morph_clean,
    xml_to_mask, dice_iou, plot_pipeline,
    TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="E2 Pipeline", page_icon="🔬", layout="wide")
st.title("E2 — TCGA-E2-A1B5-01Z-00-DX1")

fname = "TCGA-E2-A1B5-01Z-00-DX1"

with st.spinner("Running E2 pipeline..."):
    img, h_ch = load(TISSUE_DIR, fname)
    gt = xml_to_mask(ANN_DIR / f"{fname}.xml", img.shape)

    t0 = time.time()
    blurred = median_blur(h_ch, ksize=7)
    result = morph_clean(otsu_threshold(blurred), kernel_size=3, open_iter=3, close_iter=1)
    elapsed = time.time() - t0

    dice, iou = dice_iou(result, gt)

st.metric("DICE", f"{dice:.4f}")
st.metric("IoU", f"{iou:.4f}")
st.metric("Time", f"{elapsed:.2f}s")

fig = plot_pipeline(img, gt, result, dice, iou, title=f"E2 — DICE: {dice:.4f}")
st.pyplot(fig)
