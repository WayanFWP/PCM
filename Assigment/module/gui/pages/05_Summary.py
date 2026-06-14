import sys
from pathlib import Path

import streamlit as st
import numpy as np
import time

from gui.nucleus_utils import (
    load, median_blur, bilateral_filter, clahe,
    otsu_threshold, percentile_threshold, morph_clean, dilate,
    filter_components, watershed_nuclei,
    xml_to_mask, dice_iou,
    TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="Summary", page_icon="📊", layout="wide")
st.title("Summary — All Samples")

SAMPLES = {
    "AR": {
        "fname": "TCGA-AR-A1AS-01Z-00-DX1",
        "desc": "median → bilateral → otsu → filter → morph → dilate",
    },
    "AY": {
        "fname": "TCGA-AY-A8YK-01A-01-TS1",
        "desc": "bilateral → clahe → percentile 69 → dilate",
    },
    "E2": {
        "fname": "TCGA-E2-A1B5-01Z-00-DX1",
        "desc": "median → otsu → morph_clean",
    },
    "RD": {
        "fname": "TCGA-RD-A8N9-01A-01-TS1",
        "desc": "clahe → otsu → watershed → dilate",
    },
}


def run_pipeline(key):
    info = SAMPLES[key]
    img, h_ch = load(TISSUE_DIR, info["fname"])
    gt = xml_to_mask(ANN_DIR / f"{info['fname']}.xml", img.shape)

    t0 = time.time()

    if key == "AR":
        blurred = median_blur(h_ch, ksize=9)
        blurred = bilateral_filter(blurred, d=5, sigma_color=50, sigma_space=100)
        th = otsu_threshold(blurred)
        filtered = filter_components(th, min_area=100, max_area=8000, max_aspect=5.0)
        segment = morph_clean(filtered, kernel_size=3, open_iter=2, close_iter=2)
        result = dilate(segment, kernel_size=3, iterations=1)
    elif key == "AY":
        blurred = bilateral_filter(h_ch, d=7, sigma_color=30, sigma_space=50)
        eq = clahe(blurred, clip_limit=0.5, tile_grid=(4, 4))
        result = dilate(percentile_threshold(eq, pct=69), kernel_size=3, iterations=1)
    elif key == "E2":
        blurred = median_blur(h_ch, ksize=7)
        result = morph_clean(otsu_threshold(blurred), kernel_size=3, open_iter=3, close_iter=1)
    elif key == "RD":
        eq = clahe(h_ch, clip_limit=0.3, tile_grid=(16, 16))
        result = dilate(watershed_nuclei(otsu_threshold(eq), dist_thresh=0.15), kernel_size=3, iterations=1)

    elapsed = time.time() - t0
    dice, iou = dice_iou(result, gt)
    return dice, iou, elapsed


with st.spinner("Running all pipelines..."):
    results = {}
    for key in SAMPLES:
        dice, iou, t = run_pipeline(key)
        results[key] = {"dice": dice, "iou": iou, "time": t}

avg_dice = np.mean([r["dice"] for r in results.values()])
avg_iou = np.mean([r["iou"] for r in results.values()])
avg_time = np.mean([r["time"] for r in results.values()])

col1, col2, col3 = st.columns(3)
col1.metric("Avg DICE", f"{avg_dice:.4f}")
col2.metric("Avg IoU", f"{avg_iou:.4f}")
col3.metric("Avg Time", f"{avg_time:.2f}s")

st.divider()

data = []
for key, r in results.items():
    data.append({
        "Sample": key,
        "DICE": f"{r['dice']:.4f}",
        "IoU": f"{r['iou']:.4f}",
        "Time (s)": f"{r['time']:.2f}",
        "Pipeline": SAMPLES[key]["desc"],
    })

st.table(data)
