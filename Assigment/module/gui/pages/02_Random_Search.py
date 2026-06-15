import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import random
import numpy as np
import cv2
import concurrent.futures
import io
import time

from gui.nucleus_utils import (
    hed_convert, otsu_threshold, percentile_threshold,
    adaptive_threshold, bilateral_filter, median_blur, clahe,
    gaussian_blur, morph_clean, dilate, filter_components,
    watershed_nuclei, xml_to_mask, dice_iou, apply_pipeline,
    plot_pipeline, TISSUE_DIR, ANN_DIR,
)

st.set_page_config(page_title="Random Search", page_icon="🎲", layout="wide")
st.title("Random Parameter Search")

# ── Input source ──
source = st.radio("Image source", ["Upload custom", "From dataset"], horizontal=True)

targets = []  # list of {stem, tif_data/xml_data or path}

if source == "Upload custom":
    tif_file = st.file_uploader("Upload tissue image (.tif)", type=["tif", "tiff"])
    xml_file = st.file_uploader("Upload annotation (.xml)", type=["xml"])
    if tif_file is not None and xml_file is not None:
        stem = Path(tif_file.name).stem
        tif_bytes = np.frombuffer(tif_file.read(), np.uint8)
        img_bgr = cv2.imdecode(tif_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Could not decode the uploaded .tif file.")
            st.stop()
        xml_content = xml_file.read()
        targets.append({"stem": stem, "img_bgr": img_bgr, "xml_bytes": xml_content, "tif_path": None})
        st.success(f"Loaded: {tif_file.name} + {xml_file.name}")
    if not targets:
        st.info("Upload a .tif image and its matching .xml annotation.")
        st.stop()
else:
    tif_files = sorted(TISSUE_DIR.glob("*.tif"))
    available = []
    for t in tif_files:
        stem = t.stem
        xml_path = ANN_DIR / f"{stem}.xml"
        if xml_path.exists():
            available.append(stem)
    if not available:
        st.error("No matching .tif/.xml pairs found in dataset.")
        st.stop()
    selected = st.multiselect("Choose images to process", available, default=available[:2])
    if not selected:
        st.info("Pick at least one image.")
        st.stop()
    for stem in selected:
        targets.append({"stem": stem, "img_bgr": None, "xml_bytes": None,
                        "tif_path": str(TISSUE_DIR / f"{stem}.tif"),
                        "xml_path": str(ANN_DIR / f"{stem}.xml")})

# ── Parameter controls ──
with st.expander("Parameter Ranges", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Enhancement")
        pct_range = st.slider("Percentile range", 30, 99, (45, 90))
        d_range = st.slider("Bilateral d", 3, 15, (5, 13))
        sigma_range = st.slider("Bilateral sigma", 10, 120, (25, 100))
        clip_range = st.slider("CLAHE clip limit", 0.1, 10.0, (0.3, 5.0), 0.1)
        tile_opts = st.multiselect("CLAHE tile sizes", [2, 4, 8, 16, 32], [4, 8, 16])
        ksize_range = st.slider("Median / Gauss kernel", 3, 11, (3, 9), 2)
    with col2:
        st.subheader("Threshold")
        block_range = st.slider("Adaptive block size", 11, 99, (21, 81), 2)
        c_range = st.slider("Adaptive C", 1, 15, (2, 10))
        dist_range = st.slider("Watershed dist thresh", 0.05, 0.6, (0.15, 0.45), 0.01)
    with col3:
        st.subheader("Cleanup")
        min_area_range = st.slider("Min area", 10, 200, (20, 100))
        max_area_range = st.slider("Max area", 1000, 20000, (3000, 10000), 100)
        aspect_range = st.slider("Max aspect ratio", 2.0, 6.0, (2.5, 4.0), 0.1)
        kern_range = st.slider("Morph kernel", 3, 7, (3, 5), 2)
        open_range = st.slider("Open iterations", 1, 4, (1, 3))
        close_range = st.slider("Close iterations", 1, 4, (1, 3))
        dilate_check = st.checkbox("Include dilate", value=True)

n_iter = st.slider("Iterations per image", 50, 2000, 200, 50)


def _search_single(target, n_iter):
    if target["img_bgr"] is not None:
        img_bgr = target["img_bgr"]
        gt = xml_to_mask(io.BytesIO(target["xml_bytes"]), img_bgr.shape)
    else:
        img_bgr = cv2.imread(target["tif_path"])
        gt = xml_to_mask(target["xml_path"], img_bgr.shape)

    best = {"dice": 0, "iou": 0, "pipeline": None}

    for _ in range(n_iter):
        stages = []
        enhancers = [
            ("clahe", clahe, {"clip_limit": random.uniform(clip_range[0], clip_range[1]),
                               "tile_grid": (random.choice(tile_opts), random.choice(tile_opts))}),
            ("bilateral", bilateral_filter, {"d": random.randrange(d_range[0], d_range[1]+1, 2),
                                              "sigma_color": random.randint(sigma_range[0], sigma_range[1]),
                                              "sigma_space": random.randint(sigma_range[0], sigma_range[1])}),
            ("median", median_blur, {"ksize": random.randrange(ksize_range[0], ksize_range[1]+1, 2)}),
            ("gaussian", gaussian_blur, {"ksize": random.randrange(ksize_range[0], ksize_range[1]+1, 2)}),
        ]
        n_enh = random.randint(0, 2)
        if n_enh > 0:
            for idx in random.sample(range(len(enhancers)), n_enh):
                stages.append(enhancers[idx])

        thresh_opts = [
            ("otsu", otsu_threshold, {}),
            ("percentile", percentile_threshold, {"pct": random.randint(pct_range[0], pct_range[1])}),
            ("adaptive", adaptive_threshold, {"block_size": random.randrange(block_range[0], block_range[1]+1, 2),
                                               "c": random.randint(c_range[0], c_range[1])}),
        ]
        stages.append(random.choice(thresh_opts))

        cleaners = [
            ("morph_clean", morph_clean, {"kernel_size": random.randrange(kern_range[0], kern_range[1]+1, 2),
                                           "open_iter": random.randint(open_range[0], open_range[1]),
                                           "close_iter": random.randint(close_range[0], close_range[1])}),
            ("filter", filter_components, {"min_area": random.randint(min_area_range[0], min_area_range[1]),
                                            "max_area": random.randint(max_area_range[0], max_area_range[1]),
                                            "max_aspect": random.uniform(aspect_range[0], aspect_range[1])}),
            ("watershed", watershed_nuclei, {"dist_thresh": random.uniform(dist_range[0], dist_range[1])}),
        ]
        if dilate_check:
            cleaners.append(("dilate", dilate, {"kernel_size": random.randrange(kern_range[0], kern_range[1]+1, 2),
                                                 "iterations": random.randint(1, 2)}))
        n_clean = random.randint(0, len(cleaners))
        if n_clean > 0:
            for idx in random.sample(range(len(cleaners)), n_clean):
                stages.append(cleaners[idx])

        try:
            t0 = time.time()
            pred = apply_pipeline(img_bgr, stages)
            elapsed = time.time() - t0
        except Exception:
            continue
        dice, iou = dice_iou(pred, gt)
        if dice > best["dice"]:
            best = {"dice": dice, "iou": iou, "time": elapsed, "pipeline": stages}

    return target["stem"], best, img_bgr, gt


# ── Run ──
if st.button("Run Random Search", type="primary", use_container_width=True,
             disabled=len(tile_opts) == 0):

    results = {}
    _images = {}
    progress = st.progress(0, text="Starting...")

    n_workers = min(4, len(targets))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_search_single, t, n_iter): t["stem"] for t in targets}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            stem, best, img_bgr, gt = f.result()
            results[stem] = best
            _images[stem] = (img_bgr, gt)
            done += 1
            progress.progress(done / len(targets),
                              text=f"{stem}  (DICE={best['dice']:.4f}, {best.get('time',0):.3f}s)")

    progress.empty()
    st.session_state.search_results = results
    st.session_state.search_images = _images
    st.session_state.search_targets = targets

# ── Restore ──
results = st.session_state.get("search_results")
_images = st.session_state.get("search_images")

if results is None:
    st.info("Configure parameters, pick images, then click **Run Random Search**.")
    st.stop()

st.success(f"Results ({n_iter} iterations per image, {len(results)} image(s))")

st.subheader("Best Pipeline per Image")
rows = []
for label, best in results.items():
    stage_names = " → ".join(s[0] for s in best["pipeline"])
    t = best.get("time", 0)
    rows.append({"Image": label, "DICE": f"{best['dice']:.4f}",
                 "IoU": f"{best['iou']:.4f}", "Time (s)": f"{t:.3f}",
                 "Pipeline": stage_names})
st.table(rows)

avg_dice = np.mean([r["dice"] for r in results.values()])
avg_iou = np.mean([r["iou"] for r in results.values()])
mc = st.columns(2)
mc[0].metric("Avg DICE", f"{avg_dice:.4f}")
mc[1].metric("Avg IoU", f"{avg_iou:.4f}")

st.subheader("View Best Result")
view_img = st.selectbox("Choose image", list(results.keys()), key="view_img")
if view_img:
    best = results[view_img]
    if _images and view_img in _images:
        img_rgb, gt = _images[view_img]
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    elif source == "From dataset":
        entry = next(t for t in targets if t["stem"] == view_img)
        img_bgr = cv2.imread(entry["tif_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = xml_to_mask(entry["xml_path"], img_bgr.shape)

    pred = apply_pipeline(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), best["pipeline"])
    fig = plot_pipeline(img_rgb, gt, pred, best["dice"], best["iou"],
                        title=f"Best for {view_img}")
    st.pyplot(fig)

    st.caption(f"Processing time: **{best.get('time', 0):.3f} seconds**")

    with st.expander("Show full pipeline details"):
        for name, fn, kwargs in best["pipeline"]:
            kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            st.code(f"{name}({kw_str})")
