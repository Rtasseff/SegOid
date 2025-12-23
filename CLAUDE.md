# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SegOid** is a semantic segmentation pipeline for identifying spheroids in microscopy images. The pipeline trains a PyTorch U-Net model on patches extracted from well-plate images with binary masks (created in Fiji or ilastik), then performs tiled inference to produce full-resolution segmentations and quantitative morphology metrics.

**Strategic context:** This is designed as general-purpose infrastructure for cell microscopy segmentation, not just spheroids. The architecture supports future extension to 2D cell cultures, organoids, and variable imaging conditions.

**Current status:** Phase 0 (Project bootstrap) - Complete. Package structure (src/) and CLI placeholders implemented per SDD.

## Development Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS/Linux

# Install dependencies (after pyproject.toml or requirements.txt is created)
pip install -e .
```

### Pipeline Commands (once implemented)

Phase 1 - Data validation and splits:
```bash
validate_dataset --input-dir working/ --output-dir splits/
make_splits --manifest splits/all.csv --seed 42 --output-dir splits/
```

Phase 1.5 - Sanity check:
```bash
sanity_check --config configs/sanity_check.yaml
```

Phase 2 is integrated into Phase 3 (patch extraction happens during training).

Phase 3 - Model training:
```bash
train --config configs/train.yaml
```

Phase 4 - Full-image inference:
```bash
predict_full --config configs/predict.yaml --checkpoint runs/<run_id>/checkpoints/best.pth
```

Phase 5 - Object quantification:
```bash
quantify_objects --config configs/quantify.yaml
```

### Testing

```bash
# Run all unit tests
pytest

# Run specific test file
pytest tests/test_dataset.py

# Run with coverage
pytest --cov=src tests/
```

## Architecture

### Directory Structure

```
project/
  raw/                      # Original microscope exports (never edited)
  working/
    images/                 # Standardized 8-bit images
    masks/                  # Binary masks (0/255) aligned to images
    rois/                   # Optional Fiji ROI exports
  splits/                   # train/val/test manifests (CSV)
  runs/                     # Training outputs, checkpoints, TensorBoard logs
  inference/                # Full-image predictions
  metrics/                  # Object tables and summary plots
  src/                      # Python package
  configs/                  # YAML configuration files
  docs/                     # Documentation (SDD.md)
  notebooks/                # Optional exploratory notebooks
```

### File Naming Convention

**Critical constraint:** Image/mask pairing is based on filename stems.

- Image: `<basename>.tif`
- Mask: `<basename>_mask.tif`
- ROI: `<basename>_rois.zip`

For any image `X.tif`, the mask MUST be `X_mask.tif`.

### Code Organization

The `src/` package is organized by pipeline phase:

- `src/data/`
  - `validate.py` - Image/mask pairing validation, QC metrics
  - `dataset.py` - `PatchDataset` for training (dynamic patch sampling)
- `src/training/`
  - `train.py` - Model training with early stopping, checkpointing
- `src/inference/`
  - `predict.py` - Tiled inference for full-resolution images
- `src/analysis/`
  - `quantify.py` - Object extraction, instance matching, morphology metrics

### Data Pipeline Architecture

**Critical design decisions:**

1. **Patch-based training without pre-cropping:** `PatchDataset` dynamically samples patches from full images during training. Patches are NOT saved to disk.

2. **Sampling strategy (per epoch):**
   - 70% positive-centered: Random foreground pixel + jitter (up to 25% patch size)
   - 30% negative: Random location with <5% mask coverage
   - Default: 20 patches per image

3. **Patch size formula:** `patch_size ≈ 2.5 × estimated_spheroid_diameter`
   - Measured from masks during Phase 1 QC
   - Round to power of 2 for GPU efficiency (e.g., 256, 512)
   - Rationale: With jittered sampling, spheroid may be off-center; 2.5× ensures full object + context fits

4. **Augmentation strategy (conservative):**
   - Horizontal/vertical flip (p=0.5)
   - 90° rotation (p=0.5)
   - Brightness/contrast adjustment (±10%)
   - **Intentionally excluded:** Aggressive augmentation, elastic deformation
   - **Rationale:** Well-plate images have consistent dark ring artifacts at well edges. Aggressive augmentation might teach the model to over-rely on these features, hurting generalization to non-well-plate data.

5. **Tiled inference:**
   - Tile size = training patch size (critical for consistent receptive field)
   - 25% overlap (sufficient for round objects)
   - Average probabilities in overlaps, then threshold

### Model Architecture

- **Framework:** PyTorch with `segmentation-models-pytorch` (SMP)
- **Architecture:** U-Net
- **Encoder:** ResNet18 (baseline) or EfficientNet-B0, pretrained on ImageNet
- **Input:** Grayscale (1 channel) - requires adapter or channel replication
- **Loss:** `0.5 × BCE + 0.5 × DiceLoss`
- **Metrics:** Dice coefficient (primary), IoU, precision/recall

### Instance-level Evaluation

Beyond pixel-level segmentation metrics, the pipeline performs instance-level object detection evaluation:

1. Extract objects via connected components from both predicted and GT masks
2. Compute IoU matrix between all predicted/GT object pairs
3. Hungarian algorithm for optimal matching (maximize total IoU)
4. Matches with IoU < 0.5 are rejected
5. Compute TP/FP/FN, precision/recall/F1, mean matched IoU

This separates "did we find the right objects?" (detection) from "how accurate are the boundaries?" (segmentation).

## Platform Considerations

**MacBook Air M1 (2020):** Development and smoke tests only
- Use CPU or MPS backend
- Expect 10-20× slower than GPU
- Sanity check (5 epochs, 10% data): ~10-15 minutes
- Full training NOT recommended

**Linux GPU workstation (target):** Full training and inference
- NVIDIA RTX 5060 Ti, 16 GB GDDR7
- Mixed precision training (AMP with GradScaler)
- Full training (100 epochs): ~1-2 hours (estimate)
- Single image inference: ~5-15 seconds

## Configuration Management

All pipeline stages use YAML configs in `configs/`:

- `dataset.yaml` - Patch sampling and augmentation parameters
- `sanity_check.yaml` - Reduced training for pipeline validation
- `train.yaml` - Model architecture, training hyperparameters
- `predict.yaml` - Tiled inference parameters
- `quantify.yaml` - Analysis and matching parameters

**Reproducibility:** Each training run snapshots its config to `runs/<run_id>/config.yaml`.

## Key Implementation Details

### Data Manifests

All dataset splits are defined by CSV manifests in `splits/`:

Required columns:
- `basename` - Filename stem
- `image_path` - Relative path to image
- `mask_path` - Relative path to mask
- `mask_coverage` - Fraction of foreground pixels
- `object_count` - Connected components in mask
- `empty_confirmed` - Boolean for confirmed empty images

**Critical:** Split by image, NOT by patches, to prevent data leakage.

### Empty Image Handling

Images with zero mask coverage require explicit confirmation:
- If `mask_coverage == 0` and `empty_confirmed == False`: flag for review
- Confirmed empty images are valid negative examples, not errors
- Phase 1 validation produces QC report with flagged images

### Post-processing

After tiled inference:
1. Remove small objects (< 100 px² by default)
2. Binary fill holes within objects
3. Optional morphological smoothing

### Morphology Metrics

Extracted per object via `skimage.measure.regionprops`:
- area, perimeter, equivalent_diameter
- major/minor axis length, eccentricity
- circularity (4πA/P²)
- centroid, bounding box

**Optional:** If pixel size is known, convert to physical units (µm).

## Tech Stack

- **Python:** 3.11+
- **Core:** PyTorch, segmentation-models-pytorch, albumentations
- **Image I/O:** tifffile, imageio
- **Analysis:** numpy, pandas, scikit-image, scipy
- **Logging:** TensorBoard
- **Testing:** pytest

## Development Phases

The project follows a 6-phase development plan (see SDD.md for full details):

- **Phase 0:** Project bootstrap (current)
- **Phase 1:** Image curation, dataset validation, train/val/test splits
- **Phase 1.5:** Sanity check - validate pipeline before full training
- **Phase 2:** Patch extraction (integrated into training)
- **Phase 3:** Model training on patches
- **Phase 4:** Tiled inference on full images
- **Phase 5:** Object quantification and instance evaluation

**Phase 1.5 exit criteria:**
- Loss decreases over epochs
- Predictions are spatially coherent
- Predicted masks align with actual spheroid locations (no systematic offset)
- Visual overlays confirm no data pipeline bugs

## Future Considerations

**Well-edge artifact sensitivity:** Current training data has dark rings at well edges. Monitor for:
- Model relying on well edges as segmentation cues
- Poor generalization to non-well-plate images
- If detected, consider augmentations that mask/vary ring appearance

**Potential extensions:**
- Per-well localization preprocessing (if accuracy insufficient)
- Touching objects (watershed or instance segmentation)
- Multi-class segmentation (spheroid vs diffuse plating)
- 2D cell cultures, 3D organoids

**POC success criteria:**
- Validation Dice > 0.8
- Detection F1 > 0.9 on well-separated spheroids
- Pipeline runs end-to-end without manual intervention
