import sys
from pathlib import Path

import streamlit as st
import time
from gui.nucleus_utils import (
    load, median_blur, bilateral_filter, otsu_threshold,
    filter_components, morph_clean, dilate,
    xml_to_mask, dice_iou, plot_pipeline,
    TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="AR Pipeline", page_icon="🔬", layout="wide")
st.title("AR — TCGA-AR-A1AS-01Z-00-DX1")

fname = "TCGA-AR-A1AS-01Z-00-DX1"

with st.spinner("Running AR pipeline..."):
    img, h_ch = load(TISSUE_DIR, fname)
    gt = xml_to_mask(ANN_DIR / f"{fname}.xml", img.shape)

    t0 = time.time()
    blurred = median_blur(h_ch, ksize=9)
    blurred = bilateral_filter(blurred, d=5, sigma_color=50, sigma_space=100)
    th = otsu_threshold(blurred)
    filtered = filter_components(th, min_area=100, max_area=8000, max_aspect=5.0)
    segment = morph_clean(filtered, kernel_size=3, open_iter=2, close_iter=2)
    result = dilate(segment, kernel_size=3, iterations=1)
    elapsed = time.time() - t0

    dice, iou = dice_iou(result, gt)

st.metric("DICE", f"{dice:.4f}")
st.metric("IoU", f"{iou:.4f}")
st.metric("Time", f"{elapsed:.2f}s")

fig = plot_pipeline(img, gt, result, dice, iou, title=f"AR — DICE: {dice:.4f}")
st.pyplot(fig)
