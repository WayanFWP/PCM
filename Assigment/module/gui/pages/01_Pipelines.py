import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import time
import concurrent.futures

from gui.nucleus_utils import (
    load, median_blur, bilateral_filter, clahe,
    otsu_threshold, percentile_threshold, morph_clean, dilate,
    filter_components, watershed_nuclei,
    xml_to_mask, dice_iou, plot_pipeline,
    TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="Pipelines", page_icon="🔬", layout="wide")
st.title("Pre-tuned Pipelines")

SAMPLES = {
    "AR": {
        "fname": "TCGA-AR-A1AS-01Z-00-DX1",
        "desc": "median(9) → bilateral(5,50,100) → otsu → filter(100,8000,5) → morph(3,2,2) → dilate(3)",
    },
    "AY": {
        "fname": "TCGA-AY-A8YK-01A-01-TS1",
        "desc": "bilateral(7,30,50) → clahe(0.5,4) → percentile(69) → dilate(3)",
    },
    "E2": {
        "fname": "TCGA-E2-A1B5-01Z-00-DX1",
        "desc": "median(7) → otsu → morph_clean(3,3,1)",
    },
    "RD": {
        "fname": "TCGA-RD-A8N9-01A-01-TS1",
        "desc": "clahe(0.3,16) → otsu → watershed(0.15) → dilate(3)",
    },
}


def _exec_pipeline(key):
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
    return key, {"img": img, "gt": gt, "result": result, "dice": dice, "iou": iou, "time": elapsed}


# ── Run (or restore from session state) ──
if st.button("Run All Pipelines", type="primary", use_container_width=True):
    results = {}
    status = st.status("Running 4 pipelines in parallel ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_exec_pipeline, k): k for k in SAMPLES}
        for f in concurrent.futures.as_completed(futures):
            key, data = f.result()
            results[key] = data
    status.update(label="Done!", state="complete")
    st.session_state.pipeline_results = results

results = st.session_state.get("pipeline_results")
if results is None:
    st.info("Click **Run All Pipelines** above to execute all 4 segmentation pipelines.")
    st.stop()

# ── Display ──
avg_dice = sum(r["dice"] for r in results.values()) / 4
avg_iou = sum(r["iou"] for r in results.values()) / 4
avg_time = sum(r["time"] for r in results.values()) / 4

cols = st.columns(3)
cols[0].metric("Avg DICE", f"{avg_dice:.4f}")
cols[1].metric("Avg IoU", f"{avg_iou:.4f}")
cols[2].metric("Avg Time", f"{avg_time:.2f}s")

st.divider()

row1 = st.columns(2)
row2 = st.columns(2)

for idx, key in enumerate(["AR", "AY", "E2", "RD"]):
    col = row1[idx] if idx < 2 else row2[idx - 2]
    r = results[key]
    with col:
        st.subheader(f":blue[{key}]")
        st.caption(SAMPLES[key]["desc"])
        mc = st.columns(3)
        mc[0].metric("DICE", f"{r['dice']:.4f}")
        mc[1].metric("IoU", f"{r['iou']:.4f}")
        mc[2].metric("Time", f"{r['time']:.2f}s")
        with st.popover("View Image", use_container_width=True):
            fig = plot_pipeline(r["img"], r["gt"], r["result"],
                                r["dice"], r["iou"],
                                title=f"{key} — DICE: {r['dice']:.4f}")
            st.pyplot(fig)
