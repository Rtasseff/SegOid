# SegOid — Spheroid Segmentation Pipeline

A PyTorch-based semantic segmentation pipeline for identifying spheroids in microscopy images. Trained on well-plate images, outputs binary masks and morphology metrics.

> **Are you a researcher running SegOid (not modifying it)?** Read the friendly walkthrough at [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) instead. This README is the technical reference for developers.

## Overview

SegOid uses a U-Net architecture with a ResNet18 encoder to segment spheroids from microscopy images. The pipeline:

1. Takes full-resolution microscopy images as input
2. Applies tiled inference (256×256 patches with 25% overlap)
3. Outputs binary segmentation masks
4. Optionally computes morphology metrics (area, diameter, circularity)

**Production Model Performance:**

| Metric | Value |
|--------|-------|
| Training | 9 labeled images (2 resolutions), 100 epochs |
| Validation Dice | 0.94 |
| Cross-validation (9-fold LOOCV) | 0.934 ± 0.026 |

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
    --checkpoint models/production_v2.0/checkpoints/best_model.pth \
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
    --checkpoint models/production_v2.0/checkpoints/best_model.pth \
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

**Important: Resolution is not preserved.** All inference is performed at the training resolution (2.76 µm/px). Higher-resolution input images (e.g., 1.1 µm/px) are downsampled before the model sees them. The output mask is upsampled back to the original image dimensions, but the segmentation detail is limited to what 2.76 µm/px can resolve. To segment at finer resolution would require retraining with larger tile sizes or a different architecture — see [Multi-Scale Training](#multi-scale-training) below.

**Histogram Matching:**

If images at a different magnification also have different intensity characteristics (e.g., darker background, different contrast due to microscopy settings), the model may produce poor results even with rescaling. Use `--histogram-match` to transform input intensities to match a reference image representative of the training data:

```bash
predict_full \
    --checkpoint models/production_v2.0/checkpoints/best_model.pth \
    --manifest high_mag_images.csv \
    --output-dir inference/high_mag/ \
    --pixel-size 1.1 \
    --histogram-match path/to/training_reference_image.tif
```

The reference image should be a typical image from the training set (e.g., one of the original 2.76 µm/px images).

**Quality Guidelines:**

| Input Pixel Size | Scale Factor | Quality | Notes |
|------------------|--------------|---------|-------|
| 2.3 - 3.5 µm/px | 0.8 - 1.2× | ✓ Excellent | Safe to use |
| 1.4 - 2.3 or 3.5 - 5.5 µm/px | 0.5 - 0.8 or 1.2 - 2.0× | ✓ Good | Should work well |
| 0.9 - 1.4 or 5.5 - 9.2 µm/px | 0.3 - 0.5 or 2.0 - 3.0× | ⚠ Fair | Expect some degradation |
| <0.9 or >9.2 µm/px | <0.3 or >3.0× | ✗ Poor | Retrain at target resolution |

**Limitations:**
- **No enhanced resolution**: High-magnification input is downsampled to 2.76 µm/px for inference — finer detail is discarded
- **Very high magnification (<0.3× scale)**: Model sees overly magnified texture, loses whole-spheroid context
- **Very low magnification (>3.0× scale)**: Model sees blurry, undersampled spheroids, loses fine details
- **Memory**: Peak usage ~2.5× original image size during processing
- **Interpolation artifacts**: 1-2 pixel smoothing at boundaries (acceptable for most use cases)
- **Intensity mismatch**: Different magnifications often have different brightness/contrast — use `--histogram-match` if results are poor

### Multi-Scale Training

Train a single model on images captured at different magnifications. All images are rescaled to a common training resolution (2.76 µm/px) during patch extraction. The model architecture and tile size remain unchanged.

#### Manifest Setup

Add a `pixel_size` column to your training manifest:

```csv
basename,image_path,mask_path,pixel_size
sample_001,working/images/sample_001.tif,working/masks/sample_001_mask.tif,2.76
highres_001,highres/images/highres_001.tif,highres/masks/highres_001_mask.tif,1.1
```

**Backward compatible:** If the `pixel_size` column is missing, all images are assumed to be at the training resolution (no rescaling). Existing manifests work unchanged.

#### How It Works

During dataset loading, each image is rescaled to the target training resolution based on its `pixel_size` value:

- **Images** are rescaled with bilinear interpolation (anti-aliasing when downsampling)
- **Masks** are rescaled with nearest-neighbor interpolation to preserve binary 0/255 values
- Rescaling happens once at load time, not per-patch

#### Area-Proportional Patch Sampling

After rescaling, images from different magnifications may have very different pixel areas (e.g., a 1.1 µm/px image covers ~4x the pixel area of a 2.76 µm/px image after downscaling). To ensure uniform sampling density:

- Each image's patch count is scaled proportionally to its area relative to the mean image area
- A `max_patches_per_image` cap prevents any single large image from dominating an epoch (default: 4x `patches_per_image`)

#### Dead-Patch Rejection

Large field-of-view images (e.g., full-well captures) often include black borders after rescaling. Negative patches (background samples) that are >95% zero-valued are automatically rejected, so the model doesn't waste training on pure-black tiles.

#### Config Parameters

Multi-scale training uses the existing training config with these relevant parameters:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `model.training_pixel_size` | Config YAML | 2.76 | Target resolution for rescaling (µm/px) |
| `dataset.patches_per_image` | Config YAML | 30 | Base patch count (for reference-sized images) |
| `dataset.max_patches_per_image` | Config YAML | 4x base | Cap on patches per image |

#### Example Training Command

```bash
# 1. Prepare combined manifest with pixel_size column
# 2. Train as usual
train --config configs/production_train_multiscale.yaml
```

#### Limitations

- **No enhanced resolution:** Higher-resolution inputs (e.g., 1.1 µm/px) are downsampled to 2.76 µm/px — finer detail is discarded. This is acceptable when 2.76 µm/px is sufficient for segmentation. Preserving higher resolution would require larger tiles or architecture changes.
- **Histogram matching not needed:** The model learns both intensity distributions from the training data directly. Existing brightness/contrast augmentation provides additional robustness.

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
    --checkpoint models/production_v2.0/checkpoints/best_model.pth \
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
train --config configs/production_train_multiscale.yaml
```

New model saved to: `runs/train_<timestamp>/checkpoints/best_model.pth`

### Data Flywheel

Each iteration improves the model:

```
TRAIN → INFER → REVIEW → CORRECT → RETRAIN
  ↑                                    ↓
  └────────────────────────────────────┘
```

- Start: 9 images (2 resolutions) → Model Dice 0.934
- After corrections: 12+ images → Improved performance
- Repeat until predictions need minimal correction

> **Want to contribute corrected masks?** See [`docs/CONTRIBUTING_TRAINING_DATA.md`](docs/CONTRIBUTING_TRAINING_DATA.md) — currently a stub. Email the maintainer in the meantime.

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
train --config configs/production_train_multiscale.yaml
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
run_cv --config configs/cv_multiscale.yaml
```

**Results location:** `runs/cv_<timestamp>/results/`
- `fold_metrics.csv` — Per-fold performance
- `summary.yaml` — Aggregated statistics
- `REPORT.md` — Human-readable summary

---

## Model Performance Details

### Cross-Validation Results (9-fold Leave-One-Out, Multi-Scale)

| Fold | Val Image | Resolution | Best Val Dice | Best Epoch |
|------|-----------|-----------|---------------|------------|
| 0 | dECM_1_1 | 2.76 µm/px | 0.9206 | 28 |
| 1 | dECM_1_2 | 2.76 µm/px | 0.9304 | 20 |
| 2 | dECM_2_1 | 2.76 µm/px | 0.9156 | 5 |
| 3 | dECM_2_2 | 2.76 µm/px | 0.9296 | 5 |
| 4 | Matri_1_1 | 2.76 µm/px | 0.8839 | 9 |
| 5 | Matri_1_2 | 2.76 µm/px | 0.9407 | 26 |
| 6 | GFR_1_1 | 1.10 µm/px | 0.9554 | 7 |
| 7 | Normal_1_1 | 1.10 µm/px | 0.9616 | 12 |
| 8 | Normal_1_2 | 1.10 µm/px | 0.9665 | 18 |
| **Mean** | | | **0.934 ± 0.026** | |

**CV Run:** `runs/cv_20260216_171900/`
**CV Config:** `configs/cv_multiscale.yaml`

### Production Model

| Property | Value |
|----------|-------|
| Checkpoint | `models/production_v2.0/checkpoints/best_model.pth` |
| Config | `configs/production_train_multiscale.yaml` |
| Training Data | All 9 labeled images (6 at 2.76 µm/px + 3 at 1.10 µm/px) |
| Epochs | 100 |
| Best Validation Dice | 0.940 (epoch 62) |

#### `models/` vs `runs/` Convention

Curated/blessed model artifacts live in `models/<version>/`, kept separate from experimental training output in `runs/`. Each `models/<version>/` folder is **self-describing** so the run can be reproduced or audited without external lookup:

```
models/production_v2.0/
├── checkpoints/best_model.pth   # production weights
├── config.yaml                   # training config snapshot
├── manifest.csv                  # dataset manifest snapshot (which images, masks, pixel sizes)
└── tensorboard/                  # training logs (optional)
```

`runs/` is treated as ephemeral/experimental scratch — it may be gitignored and may live on a non-backed-up drive. Anything blessed for production gets promoted into `models/` so it has a stable path and a backup story decoupled from training scratch space. Both `runs/` and `models/` are gitignored (large `.pth` files), but the convention is universal — every checkout of this repo should adopt it.

To promote a run to a production model:

```bash
mkdir -p models/<version>
cp -r runs/<run_name>/checkpoints models/<version>/
cp runs/<run_name>/config.yaml models/<version>/
cp data/splits/<manifest_used>.csv models/<version>/manifest.csv
cp -r runs/<run_name>/tensorboard models/<version>/   # optional
```

---

## Project Structure

```
segoid/
├── data/
│   ├── working_276/         # Training images at 2.76 µm/px
│   │   ├── images/
│   │   └── masks/
│   ├── working_110/         # Training images at 1.10 µm/px
│   │   ├── images/
│   │   └── masks/
│   └── splits/              # CSV manifests (all.csv with pixel_size)
├── runs/                    # Training outputs (experimental/scratch — gitignored)
│   ├── train_20260216_*/    # Source training run for production_v2.0
│   └── cv_20260216_*/       # Cross-validation results
├── models/                  # Curated/blessed production checkpoints (gitignored)
│   └── production_v2.0/     # Self-describing: checkpoints + config + manifest
├── inference/               # Prediction outputs
├── metrics/                 # Quantification outputs
├── configs/                 # YAML configurations
│   ├── production_train_multiscale.yaml
│   ├── cv_multiscale.yaml
│   ├── production_train.yaml
│   └── cv_config.yaml
├── src/                     # Source code
│   ├── data/                # Dataset, validation
│   ├── training/            # Training, cross-validation
│   ├── inference/           # Prediction
│   └── analysis/            # Quantification
├── tests/                   # Unit tests
└── docs/                    # Additional documentation
    ├── PRODUCTION_MODEL.md
    ├── MULTI_SCALE_TRAINING_RESULTS.md
    └── WINDOWS_DESKTOP_BUNDLE.md
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

MIT License. See [LICENSE](LICENSE) for details.
