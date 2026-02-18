# Scripts

Ad-hoc analysis and utility scripts. These are **not** part of the core pipeline — they live here to keep the main `src/` clean.

## Size Bias Analysis

Motivated by a user report that spheroids in 1.1 µm/px images appeared larger than expected after segmentation.

### `analyze_size_bias.py`

Compares ground truth (hand-labeled) and predicted spheroid sizes across both imaging scales (1.10 and 2.76 µm/px). Runs inference on all 9 training images with the production model and measures per-object area and effective diameter.

```bash
source .venv/bin/activate
python scripts/analyze_size_bias.py
```

**Outputs:**
- `_size_analysis_results.csv` — per-object measurements (source, scale, area, diameter) for both GT and predicted masks
- `_size_analysis_preds/` — predicted masks generated during analysis

**Findings (2026-02-18):**
- GT spheroids at 1.10 µm/px are genuinely larger than at 2.76 µm/px (mean area 36,251 vs 20,980 µm²; p < 0.0001). This reflects the biological samples, not a model artifact.
- Predicted sizes match GT at both scales: pred/gt area ratio = 0.977 (2.76) and 1.001 (1.10), neither statistically significant.
- No systematic size bias in predictions at either scale.

### `visualize_gt_vs_pred.py`

Generates overlay images comparing GT masks (green) and predicted masks (red) on top of the original image. Overlap regions appear yellow. Each object is labeled with its effective diameter in µm.

```bash
source .venv/bin/activate
python scripts/visualize_gt_vs_pred.py
```

**Outputs:**
- `gt_vs_pred_overlay.png` — full image, 3-panel view (GT, Pred, overlay)
- `gt_vs_pred_overlay_crop.png` — zoomed center crop with diameter labels

Currently configured for GFR_1_1 (1.10 µm/px). Edit the constants at the top of the script to change the target image.

## Utility Scripts

### `test_model_loading.py`

Demonstrates loading a production model checkpoint and running single-image inference. Useful for verifying a checkpoint works.

```bash
python scripts/test_model_loading.py \
    --checkpoint runs/train_20260216_173233/checkpoints/best_model.pth \
    --image data/working_276/images/Matri_1_1.tif
```

### `run_loocv_validation.sh`

Runs leave-one-out cross-validation and prints a comparison against the baseline. Used during development to validate multi-scale training.

```bash
bash scripts/run_loocv_validation.sh
```
