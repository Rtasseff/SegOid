# SegOid — Spheroid Segmentation Pipeline

A PyTorch-based semantic segmentation pipeline for identifying spheroids in microscopy images. Trained on well-plate images, outputs binary masks and morphology metrics.

## Overview

SegOid uses a U-Net architecture with a ResNet18 encoder to segment spheroids from microscopy images. The pipeline:

1. Takes full-resolution microscopy images as input
2. Applies tiled inference (256×256 patches with 25% overlap)
3. Outputs binary segmentation masks
4. Optionally computes morphology metrics (area, diameter, circularity)

**Production Model Performance:**

| Metric | Value |
|--------|-------|
| Training | 6 labeled images, 100 epochs |
| Validation Dice | 0.91+ |
| Cross-validation (6-fold LOOCV) | 0.917 ± 0.023 |

---

## Windows Executable (No Python Required)

A standalone Windows executable is available for users who don't want to install Python.

**Download:** `SegOid.exe` (~150MB)

**Usage:**
1. Double-click `SegOid.exe`
2. Select input folder containing TIFF images
3. Select output folder
4. Click "Run Inference"

The executable includes a bundled ONNX model and produces binary masks and morphology metrics.

**For developers:** See [docs/WINDOWS_DESKTOP_BUNDLE.md](docs/WINDOWS_DESKTOP_BUNDLE.md) for build instructions.

---

## Installation (Python)

**Requirements:**
- Python 3.11+
- ~4GB disk space for dependencies
- Mac M1/M2 or Linux with NVIDIA GPU recommended

```bash
# Clone repository
git clone <repository>
cd segoid

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# Verify installation
pytest
```

**Key Dependencies:**
- PyTorch
- segmentation-models-pytorch
- albumentations
- tifffile, imagecodecs
- scikit-image, scipy, pandas

---

## Running Inference

### Step 1: Prepare Your Images

Create a manifest CSV listing your images (e.g., `my_images.csv`):

```csv
basename,image_path,mask_path
sample_001,path/to/sample_001.tif,
sample_002,path/to/sample_002.tif,
sample_003,path/to/sample_003.tif,
```

Notes:
- `mask_path` is empty for unlabeled images
- Paths can be relative to `--data-root` or absolute
- Images should be TIFF format (RGB or grayscale)

**Quick manifest generation from a directory:**

```bash
python -c "
from pathlib import Path

image_dir = 'path/to/your/images'
output_csv = 'my_images.csv'

with open(output_csv, 'w') as f:
    f.write('basename,image_path,mask_path\n')
    for img in Path(image_dir).glob('*.tif'):
        if '_mask' not in img.stem:
            f.write(f'{img.stem},{img},\n')
print(f'Created {output_csv}')
"
```

### Step 2: Run Prediction

```bash
source .venv/bin/activate

predict_full \
    --checkpoint runs/train_20251229_194116/checkpoints/best_model.pth \
    --manifest my_images.csv \
    --output-dir inference/my_batch/ \
    --data-root .
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint |
| `--manifest` | (required) | CSV listing images to process |
| `--output-dir` | `inference/` | Where to save predictions |
| `--data-root` | `data/` | Root for relative paths in manifest |
| `--tile-size` | 256 | Tile size for inference (must match training) |
| `--overlap` | 0.25 | Tile overlap fraction (0.25 = 25%) |
| `--threshold` | 0.5 | Probability threshold for binary mask |
| `--min-object-area` | 100 | Remove objects smaller than this (px²) |
| `--pixel-size` | None | Pixel size of input images (µm/px). See Multi-Resolution section below. |

**Output:**

```
inference/my_batch/
├── sample_001_pred_mask.tif    # Binary mask (0/255)
├── sample_001_pred_prob.tif    # Probability map (float32)
├── sample_002_pred_mask.tif
├── sample_002_pred_prob.tif
├── ...
└── pixel_metrics.csv           # Dice/IoU if ground truth available
```

**Adjusting Threshold:**

```bash
# More sensitive (catches more, but more false positives)
predict_full ... --threshold 0.3

# More conservative (fewer false positives, may miss faint spheroids)
predict_full ... --threshold 0.7
```

---

### Multi-Resolution Inference

The production model was trained on images at **2.76 µm/pixel**. If your images are captured at a different magnification, use `--pixel-size` for automatic rescaling:

```bash
# High-magnification images (1.1 µm/pixel)
predict_full \
    --checkpoint runs/train_20251229_194116/checkpoints/best_model.pth \
    --manifest high_mag_images.csv \
    --output-dir inference/high_mag/ \
    --pixel-size 1.1

# Low-magnification images (5.0 µm/pixel)
predict_full \
    --checkpoint model.pth \
    --manifest low_mag_images.csv \
    --output-dir inference/low_mag/ \
    --pixel-size 5.0
```

**How it works:**
1. Input image is automatically rescaled to match training resolution (2.76 µm/px)
2. Inference runs on rescaled image
3. Probability map is rescaled back to original resolution
4. Output masks are at your original image resolution

**Performance benefits:**
- High-resolution images (e.g., 1.1 µm/px) are downsampled → **faster inference**
- Example: 4000×4000 image → rescaled to ~1594×1594 → ~6× fewer tiles

**Quality Guidelines:**

| Input Pixel Size | Scale Factor | Quality | Notes |
|------------------|--------------|---------|-------|
| 2.3 - 3.5 µm/px | 0.8 - 1.2× | ✓ Excellent | Safe to use |
| 1.4 - 2.3 or 3.5 - 5.5 µm/px | 0.5 - 0.8 or 1.2 - 2.0× | ✓ Good | Should work well |
| 0.9 - 1.4 or 5.5 - 9.2 µm/px | 0.3 - 0.5 or 2.0 - 3.0× | ⚠ Fair | Expect some degradation |
| <0.9 or >9.2 µm/px | <0.3 or >3.0× | ✗ Poor | Retrain at target resolution |

**Limitations:**
- **Very high magnification (<0.3× scale)**: Model sees overly magnified texture, loses whole-spheroid context
- **Very low magnification (>3.0× scale)**: Model sees blurry, undersampled spheroids, loses fine details
- **Memory**: Peak usage ~2.5× original image size during processing
- **Interpolation artifacts**: 1-2 pixel smoothing at boundaries (acceptable for most use cases)

For regular processing at extreme magnifications, consider retraining the model at that target resolution.

---

### Step 3: Interactive Prediction Review

Review predictions visually and flag those needing correction:

```bash
review_predictions \
    --image-dir path/to/your/images/ \
    --pred-mask-dir inference/my_batch/ \
    --output-flagged flagged_images.txt
```

**Controls:**

| Key | Action |
|-----|--------|
| **LEFT CLICK** | Flag/unflag current image |
| **SPACE** | Pause/resume slideshow |
| **LEFT ARROW** | Previous image |
| **RIGHT ARROW** | Next image |
| **ESC** | Exit and save flagged list |

**Display Cycle:**
1. Original image (3 seconds)
2. Predicted mask overlay (3 seconds)
3. Next image...

**Output:**

`flagged_images.txt` contains one filename per line:
```
sample_003.tif
sample_007.tif
sample_012.tif
```

---

#### Alternative: Video Export (for WSL/Headless)

If you're running on WSL or a headless server where the interactive display doesn't work, export a video instead:

```bash
review_predictions \
    --image-dir path/to/your/images/ \
    --pred-mask-dir inference/my_batch/ \
    --video-export review.mp4
```

**Optional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--display-duration` | 3.0 | Seconds to show each image |
| `--overlay-alpha` | 0.5 | Mask transparency (0.0-1.0) |
| `--fps` | 30 | Video frames per second |

**Example with custom settings:**

```bash
review_predictions \
    --image-dir path/to/images/ \
    --pred-mask-dir inference/my_batch/ \
    --video-export review.mp4 \
    --display-duration 5.0 \
    --overlay-alpha 0.6 \
    --fps 30
```

The output video shows: original image → mask overlay → next image...

This is useful for:
- WSL environments (display issues)
- Remote/headless servers
- Sharing predictions with collaborators
- Offline review

---

### Step 4: Compute Morphology Metrics (Optional)

Extract quantitative measurements from segmented spheroids:

```bash
quantify_objects \
    --pred-mask-dir inference/my_batch/ \
    --gt-manifest my_images.csv \
    --output-dir metrics/my_batch/
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pred-mask-dir` | (required) | Directory with predicted masks |
| `--gt-manifest` | (required) | Manifest CSV (for image paths) |
| `--output-dir` | `metrics/` | Where to save results |
| `--min-object-area` | 100 | Minimum object size (px²) |
| `--iou-threshold` | 0.5 | IoU threshold for instance matching |
| `--pixel-size` | None | µm per pixel (for physical units) |

**Output:**

```
metrics/my_batch/
├── per_image/
│   ├── sample_001_objects.csv    # Per-object measurements
│   └── sample_002_objects.csv
├── all_objects.csv               # Combined object table
├── instance_eval.csv             # Detection metrics (if GT available)
└── summary.csv                   # Dataset statistics
```

**Morphology Metrics Per Object:**

When `--pixel-size` is provided, both pixel and physical measurements are output:

| Metric (pixel) | Metric (physical) | Description |
|----------------|-------------------|-------------|
| `area_px` | `area_um2` | Object area in pixels² / µm² |
| `perimeter_px` | `perimeter_um` | Boundary length in pixels / µm |
| `equivalent_diameter_px` | `equivalent_diameter_um` | Diameter of equal-area circle |
| `major_axis_length_px` | `major_axis_length_um` | Major axis of fitted ellipse |
| `minor_axis_length_px` | `minor_axis_length_um` | Minor axis of fitted ellipse |
| `eccentricity` | — | Ellipse eccentricity (0=circle, 1=line) |
| `circularity` | — | 4πA/P² (1=perfect circle) |
| `centroid_x`, `centroid_y` | — | Object center coordinates (pixels) |

When `--pixel-size` is **not** provided, only pixel measurements are output (e.g., `area_px`, `perimeter_px`).

**Example with physical units:**

```bash
quantify_objects \
    --pred-mask-dir inference/my_batch/ \
    --gt-manifest my_images.csv \
    --output-dir metrics/my_batch/ \
    --pixel-size 2.76
```

Output CSV will include columns like: `area_px`, `area_um2`, `equivalent_diameter_px`, `equivalent_diameter_um`, etc.

---

## Complete Example Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Create manifest for new images
cat > batch_001.csv << EOF
basename,image_path,mask_path
img_001,/data/new_experiment/img_001.tif,
img_002,/data/new_experiment/img_002.tif,
img_003,/data/new_experiment/img_003.tif,
EOF

# 3. Run inference
predict_full \
    --checkpoint runs/train_20251229_194116/checkpoints/best_model.pth \
    --manifest batch_001.csv \
    --output-dir inference/batch_001/ \
    --data-root /

# 4. Review predictions interactively (or use --video-export for WSL/headless)
review_predictions \
    --image-dir /data/new_experiment/ \
    --pred-mask-dir inference/batch_001/ \
    --output-flagged batch_001_flagged.txt

# Alternative: Export video for offline review
# review_predictions \
#     --image-dir /data/new_experiment/ \
#     --pred-mask-dir inference/batch_001/ \
#     --video-export batch_001_review.mp4

# 5. Check which images were flagged
cat batch_001_flagged.txt

# 6. Compute morphology metrics
quantify_objects \
    --pred-mask-dir inference/batch_001/ \
    --gt-manifest batch_001.csv \
    --output-dir metrics/batch_001/
```

---

## Correcting Predictions and Retraining

If predictions need correction, you can fix them and retrain with expanded data.

### 1. Correct Flagged Predictions

Open flagged images in annotation software (e.g., Fiji/ImageJ):
- Load original image
- Load predicted mask as overlay
- Edit mask using ROI tools
- Save corrected mask as `<basename>_mask.tif`

### 2. Add Corrected Data to Training Set

```bash
# Copy corrected images and masks
cp corrected_images/*.tif data/working/images/
cp corrected_images/*_mask.tif data/working/masks/

# Update dataset manifest
validate_dataset --input-dir data/working/ --output-dir data/splits/
```

### 3. Retrain Model

```bash
train --config configs/production_train.yaml
```

New model saved to: `runs/train_<timestamp>/checkpoints/best_model.pth`

### Data Flywheel

Each iteration improves the model:

```
TRAIN → INFER → REVIEW → CORRECT → RETRAIN
  ↑                                    ↓
  └────────────────────────────────────┘
```

- Start: 6 images → Model Dice 0.91
- After corrections: 10+ images → Improved performance
- Repeat until predictions need minimal correction

---

## Training Commands

### Validate Dataset

Check image/mask pairing and compute statistics:

```bash
validate_dataset \
    --input-dir data/working/ \
    --output-dir data/splits/
```

### Create Train/Val/Test Splits

```bash
make_splits \
    --manifest data/splits/all.csv \
    --seed 42 \
    --output-dir data/splits/
```

### Train Model

```bash
train --config configs/production_train.yaml
```

**Parameters (via config file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `batch_size` | 4 | Batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `patch_size` | 256 | Training patch size |
| `patches_per_image` | 30 | Patches sampled per image per epoch |
| `early_stopping.enabled` | false | Stop early if no improvement |

### Monitor Training

```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

### Cross-Validation

Run leave-one-out cross-validation for performance estimation:

```bash
run_cv --config configs/cv_config.yaml
```

**Results location:** `runs/cv_<timestamp>/results/`
- `fold_metrics.csv` — Per-fold performance
- `summary.yaml` — Aggregated statistics
- `REPORT.md` — Human-readable summary

---

## Model Performance Details

### Cross-Validation Results (6-fold Leave-One-Out)

| Fold | Val Image | Best Val Dice | Best Epoch |
|------|-----------|---------------|------------|
| 0 | dECM_1_1 | ? | ? |
| 1 | dECM_1_2 | ? | ? |
| 2 | dECM_2_1 | ? | ? |
| 3 | dECM_2_2 | ? | ? |
| 4 | Matri_1_1 | ? | ? |
| 5 | Matri_1_2 | ? | ? |
| **Mean** | | **0.917 ± 0.023** | |

**CV Run:** `runs/cv_20251228_*/`
**CV Config:** `configs/cv_config.yaml`

### Production Model

| Property | Value |
|----------|-------|
| Checkpoint | `runs/train_20251229_194116/checkpoints/best_model.pth` |
| Config | `configs/production_train.yaml` |
| Training Data | All 6 labeled images |
| Epochs | 100 |
| Validation Dice | 0.91+ |

---

## Project Structure

```
segoid/
├── data/
│   ├── working/
│   │   ├── images/          # Training images (*.tif)
│   │   └── masks/           # Binary masks (*_mask.tif)
│   └── splits/              # CSV manifests
├── runs/                    # Training outputs
│   ├── train_20251229_*/    # Production model
│   └── cv_20251228_*/       # Cross-validation results
├── inference/               # Prediction outputs
├── metrics/                 # Quantification outputs
├── configs/                 # YAML configurations
│   ├── production_train.yaml
│   ├── train.yaml
│   └── cv_config.yaml
├── src/                     # Source code
│   ├── data/                # Dataset, validation
│   ├── training/            # Training, cross-validation
│   ├── inference/           # Prediction
│   └── analysis/            # Quantification
├── tests/                   # Unit tests
└── docs/                    # Additional documentation
```

---

## Data Format

### Images
- **Format:** TIFF (LZW compression supported)
- **Color:** RGB (converted to grayscale internally) or grayscale
- **Resolution:** Any (tiled inference handles large images)

### Masks
- **Format:** TIFF
- **Values:** Binary (0 = background, 255 = spheroid)
- **Naming:** `<basename>_mask.tif` for image `<basename>.tif`
- **Dimensions:** Must match corresponding image

### Manifest CSV

```csv
basename,image_path,mask_path,mask_coverage,object_count,empty_confirmed
image_001,working/images/image_001.tif,working/masks/image_001_mask.tif,0.042,12,
image_002,working/images/image_002.tif,working/masks/image_002_mask.tif,0.038,10,
unlabeled_001,path/to/unlabeled_001.tif,,,
```

| Column | Required | Description |
|--------|----------|-------------|
| `basename` | Yes | Unique identifier (filename without extension) |
| `image_path` | Yes | Path to image file |
| `mask_path` | No | Path to mask (empty for unlabeled) |
| `mask_coverage` | No | Fraction of foreground pixels |
| `object_count` | No | Number of objects in mask |
| `empty_confirmed` | No | True if image confirmed to have no objects |

---

## Troubleshooting

### Installation Issues

**"No module named 'src'"**
```bash
pip install -e .
```

**Missing imagecodecs**
```bash
pip install imagecodecs
```

### Inference Issues

**Out of memory**
- Reduce `--tile-size` (try 128)
- Process fewer images at once

**Predictions look wrong**
1. Check image format matches training data (TIFF, similar resolution)
2. Try different threshold values (0.3–0.7)
3. View probability maps (`*_pred_prob.tif`) to see model confidence
4. Verify images are similar to training data (well-plate spheroids)

**Slow inference**
- CPU inference is usually fast enough
- For GPU: ensure CUDA is properly installed

### Training Issues

**TensorBoard not showing data**
```bash
tensorboard --logdir runs/ --reload_multifile=true
```

**Training loss not decreasing**
- Check data paths in config
- Verify masks are binary (0/255)
- Try lower learning rate

### Review Interface Issues

**Window not appearing**
- Ensure display is connected
- Check for error messages in terminal

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `predict_full` | Run inference on images |
| `review_predictions` | Interactive prediction review (or video export with `--video-export`) |
| `quantify_objects` | Compute morphology metrics |
| `validate_dataset` | Check dataset integrity |
| `make_splits` | Create train/val/test splits |
| `train` | Train segmentation model |
| `run_cv` | Run cross-validation |
| `sanity_check` | Quick pipeline validation |
| `export_onnx` | Export PyTorch model to ONNX format |
| `segoid_gui` | Launch graphical interface |

---

## License

[Your license here]

## Citation

[Citation information if applicable]
