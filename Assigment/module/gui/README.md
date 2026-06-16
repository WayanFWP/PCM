# Nucleus Segmentation GUI

Streamlit-based GUI for nucleus segmentation on H&E-stained tissue images (MoNuSeg 2018 dataset).

## Quick Start

```bash
# from project root (PCM/)
pip install streamlit
streamlit run Assigment/module/gui/app.py
```

## Pages

### 1. Pipelines

Four pre-tuned segmentation pipelines, one per tissue sample. Click **Run All Pipelines** to execute all four in parallel.

Each cell in the 2×2 grid shows:
- Sample name and pipeline description
- **DICE** and **IoU** scores vs ground truth
- **Processing time**
- A **View Image** popover with the 4-panel plot (original, ground truth, prediction, overlay)

| Cell | Sample | Pipeline |
|------|--------|----------|
| AR | TCGA-AR-A1AS | median(9) → bilateral(5,50,100) → otsu → filter(100,8000,5) → morph(3,2,2) → dilate(3) |
| AY | TCGA-AY-A8YK | bilateral(7,30,50) → clahe(0.5,4) → percentile(69) → dilate(3) |
| E2 | TCGA-E2-A1B5 | median(7) → otsu → morph_clean(3,3,1) |
| RD | TCGA-RD-A8N9 | clahe(0.3,16) → otsu → watershed(0.15) → dilate(3) |

Results are cached in session state — switch pages and come back without re-running.

### 2. Random Search

Brute-force search for optimal preprocessing/segmentation parameters. Two input modes:

**Upload custom**: upload your own `.tif` image and `.xml` annotation. The search runs on your data.

**From dataset**: pick from available MoNuSeg samples.

#### Parameter ranges (can add more method)

| Section | Parameters |
|---------|-----------|
| Enhancement | Percentile range, bilateral d/sigma, CLAHE clip/tile, median/gauss kernel |
| Threshold | Adaptive block size/C, watershed distance threshold |
| Cleanup | Min/max area, aspect ratio, morph kernel/iterations, dilate toggle |

Adjust sliders, set iterations per image (50–2000), and click **Run Random Search**. All selected images run in parallel with a progress bar.

#### Results

- Table per image: DICE, IoU, **processing time**, pipeline stages
- Average DICE/IoU across images
- Select any image to view the 4-panel plot and full pipeline details

Results persist in session state across page navigations. Change parameters and re-run to overwrite.

## Color Legend (Overlay Plot)

| Color | Meaning |
|-------|---------|
| Green | True Positive (correct) |
| Red | False Positive |
| Blue | False Negative (missed) |

## Project Structure

```
Assigment/module/gui/
├── __init__.py
├── app.py                 # Main entry point
├── nucleus_utils.py       # All nucleus functions
├── pages/
   ├── 01_Pipelines.py     # Pre-tuned pipelines page
   └── 02_Random_Search.py # Random parameter search page
```

## Dependencies

- streamlit
- opencv-python
- numpy
- matplotlib
- scikit-image
- pandas
