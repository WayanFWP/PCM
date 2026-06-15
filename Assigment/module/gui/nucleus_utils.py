import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.color import rgb2hed


def extract_nuclei(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    microns_per_px = float(root.get("MicronsPerPixel", 0.2468))
    nuclei = []
    for region in root.findall(".//Region"):
        vertices = []
        for vertex in region.findall(".//Vertex"):
            vertices.append((float(vertex.get("X")), float(vertex.get("Y"))))
        if vertices:
            nuclei.append({
                "id": region.get("Id"),
                "display_id": region.get("DisplayId"),
                "area_px": float(region.get("Area", 0)),
                "length_px": float(region.get("Length", 0)),
                "area_microns": float(region.get("AreaMicrons", 0)),
                "vertices": np.array(vertices),
                "file": str(xml_path),
            })
    return nuclei


def extract_all_nuclei(annotation_dir):
    xml_dir = Path(annotation_dir)
    all_nuclei = {}
    for xml_file in sorted(xml_dir.glob("*.xml")):
        nuclei = extract_nuclei(xml_file)
        all_nuclei[xml_file.stem] = nuclei
    return all_nuclei


_BASE = Path(__file__).resolve().parent.parent.parent.parent
TISSUE_DIR = _BASE / "dataset/lecture/MoNuSeg2018/Tissue Images"
ANN_DIR = _BASE / "dataset/lecture/MoNuSeg2018/Annotations"


def load(folder, image_path):
    image = cv2.imread(str(Path(folder) / f"{image_path}.tif"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hed = rgb2hed(image)
    h = hed[:, :, 0]
    h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image, h_norm


def hed_convert(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hed = rgb2hed(img_rgb.astype(np.float32) / 255.0)
    h = img_hed[:, :, 0]
    return cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def median_blur(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(img)


def otsu_threshold(img):
    _, t = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def percentile_threshold(img, pct):
    val = np.percentile(img, pct)
    _, t = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
    return t


def morph_clean(img, kernel_size=3, open_iter=2, close_iter=2):
    k = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return closed


def dilate(img, kernel_size=3, iterations=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(img, k, iterations=iterations)


def xml_to_mask(xml_source, image_shape):
    tree = ET.parse(xml_source)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for region in tree.findall(".//Region"):
        pts = []
        for vertex in region.findall(".//Vertex"):
            pts.append([float(vertex.get("X")), float(vertex.get("Y"))])
        if len(pts) >= 3:
            pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
    return mask


def dice_iou(pred, gt, eps=1e-7):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou


def adaptive_threshold(img, block_size=61, c=5):
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def filter_components(binary_mask, min_area=30, max_area=5000, max_aspect=3.0):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    result = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue
        result[labels == i] = 255
    return result


def watershed_nuclei(binary_mask, dist_thresh=0.35):
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist, int(dist_thresh * 255), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    inv_mask = (255 - binary_mask).astype(np.uint8)
    markers = cv2.watershed(
        cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR), markers
    )
    result = np.zeros_like(binary_mask)
    result[markers > 1] = 255
    return result


def gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_pipeline(img_bgr, stages):
    img = hed_convert(img_bgr)
    for _name, fn, kwargs in stages:
        img = fn(img, **kwargs)
    return img


def plot_pipeline(img_rgb, gt_mask, pred_mask, dice, iou, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    overlay = img_rgb.copy()
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    tp = (pred_bin & gt_bin)
    fp = (pred_bin & ~gt_bin)
    fn = (~pred_bin & gt_bin)

    overlay[~tp.astype(bool)] = overlay[~tp.astype(bool)] * 0.4
    overlay[tp.astype(bool)] = [0, 255, 0]
    overlay[fp.astype(bool)] = [255, 0, 0]
    overlay[fn.astype(bool)] = [0, 0, 255]

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gt_mask, cmap='nipy_spectral')
    axes[0, 1].set_title(f"Ground Truth — {gt_bin.sum():,} px", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(pred_mask, cmap='nipy_spectral')
    axes[1, 0].set_title(f"Prediction — {pred_bin.sum():,} px", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        f"DICE: {dice:.4f}  |  IoU: {iou:.4f}\nTP(green) FP(red) FN(blue)",
        fontsize=11, fontweight='bold'
    )
    axes[1, 1].axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
