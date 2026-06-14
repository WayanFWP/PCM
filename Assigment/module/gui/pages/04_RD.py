import sys
from pathlib import Path

import streamlit as st
import time
from gui.nucleus_utils import (
    load, clahe, otsu_threshold, watershed_nuclei, dilate,
    xml_to_mask, dice_iou, plot_pipeline,
    TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="RD Pipeline", page_icon="🔬", layout="wide")
st.title("RD — TCGA-RD-A8N9-01A-01-TS1")

fname = "TCGA-RD-A8N9-01A-01-TS1"

with st.spinner("Running RD pipeline..."):
    img, h_ch = load(TISSUE_DIR, fname)
    gt = xml_to_mask(ANN_DIR / f"{fname}.xml", img.shape)

    t0 = time.time()
    eq = clahe(h_ch, clip_limit=0.3, tile_grid=(16, 16))
    result = dilate(watershed_nuclei(otsu_threshold(eq), dist_thresh=0.15), kernel_size=3, iterations=1)
    elapsed = time.time() - t0

    dice, iou = dice_iou(result, gt)

st.metric("DICE", f"{dice:.4f}")
st.metric("IoU", f"{iou:.4f}")
st.metric("Time", f"{elapsed:.2f}s")

fig = plot_pipeline(img, gt, result, dice, iou, title=f"RD — DICE: {dice:.4f}")
st.pyplot(fig)
