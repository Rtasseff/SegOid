# SegOid - Spheroid Segmentation Pipeline

A reproducible, researcher-friendly pipeline for training semantic segmentation models to identify spheroids in microscopy images using PyTorch U-Net.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install in development mode
pip install -e ".[dev]"

# Verify installation
pytest
```

## Project Status

**🎉 PROOF-OF-CONCEPT COMPLETE 🎉**

All pipeline phases implemented and functional end-to-end.

**Completed Phases:**
- ✅ Phase 1: Dataset validation and train/val/test splits
- ✅ Phase 1.5: Sanity check (pipeline validation)
- ✅ Phase 3: Production training infrastructure (val Dice: 0.799)
- ✅ Phase 4: Tiled inference on full-resolution images (test Dice: 0.794)
- ✅ Phase 5: Object quantification and instance evaluation (F1: 0.682)

**Results:**
- Pixel-level segmentation: **79.4% Dice** on test set
- Instance detection: **68.2% F1** (high recall 94.8%, lower precision 53.8%)
- Pipeline runs fully automated from raw images to quantified objects

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Operational guide for Claude Code
- **[docs/SDD.md](docs/SDD.md)** - Complete Software Design Document with detailed specifications
- **[CURRENT_TASK.md](CURRENT_TASK.md)** - Current task context and session notes

## Pipeline Workflow

### Phase 1: Dataset Preparation

**1.1. Validate Dataset**

Validate image/mask pairing and compute quality metrics:

```bash
validate_dataset --input-dir data/working/ --output-dir data/splits/
```

This generates:
- `data/splits/all.csv` - Complete manifest with QC metrics
- `data/splits/qc_summary.csv` - Quality control summary

**1.2. Create Train/Val/Test Splits**

Generate stratified splits by mask coverage:

```bash
make_splits --manifest data/splits/all.csv \
            --seed 42 \
            --output-dir data/splits/ \
            --train-ratio 0.6 \
            --val-ratio 0.2 \
            --test-ratio 0.2
```

This generates:
- `data/splits/train.csv` - Training set manifest
- `data/splits/val.csv` - Validation set manifest
- `data/splits/test.csv` - Test set manifest

### Phase 1.5: Sanity Check

Validate the pipeline with minimal training (5 epochs):

```bash
sanity_check --config configs/sanity_check.yaml
```

This runs a quick training loop to verify:
- Data loading works correctly
- Model trains (loss decreases)
- Predictions are spatially coherent
- No systematic offset between predictions and ground truth

Output directory: `runs/sanity_check/`

**Optional CLI overrides:**

```bash
sanity_check --config configs/sanity_check.yaml \
             --epochs 3 \
             --patches-per-image 5 \
             --batch-size 2
```

### Phase 3: Full Model Training

Train the production model with checkpointing, early stopping, and TensorBoard:

```bash
train --config configs/train.yaml
```

**Features:**
- **Config-driven:** All parameters in `configs/train.yaml`
- **Checkpointing:** Best model (by val Dice), periodic checkpoints (every 10 epochs), final model
- **Early stopping:** Stops if val Dice doesn't improve for 10 epochs
- **LR scheduling:** ReduceLROnPlateau (reduces LR by 0.5 if val loss plateaus)
- **TensorBoard logging:** Metrics + sample prediction visualizations
- **Resume support:** Continue from checkpoint with `--resume`

**Training outputs:**

```
runs/train_<timestamp>/
├── config.yaml              # Config snapshot for reproducibility
├── checkpoints/
│   ├── best_model.pth       # Best model by validation Dice
│   ├── checkpoint_epoch_010.pth
│   ├── checkpoint_epoch_020.pth
│   └── final_model.pth      # Final model after training
└── tensorboard/             # TensorBoard logs
```

**Monitor training with TensorBoard:**

```bash
# In a separate terminal
tensorboard --logdir runs/train_<timestamp>/tensorboard
# Open http://localhost:6006
```

**Resume training from checkpoint:**

```bash
train --config configs/train.yaml \
      --resume runs/train_<timestamp>/checkpoints/checkpoint_epoch_020.pth
```

**Expected training behavior:**
- Epochs: 50 (or until early stopping)
- Training time: ~30-40 minutes on M1 Mac, faster on GPU
- Target validation Dice: 0.75-0.90
- Early stopping typically triggers around epoch 25-35

### Phase 4: Tiled Inference

Apply trained model to full-resolution images using sliding window approach:

```bash
predict_full --checkpoint runs/train_<timestamp>/checkpoints/best_model.pth \
             --manifest data/splits/test.csv \
             --output-dir inference/test_predictions/
```

**Parameters:**
- `--checkpoint` (required): Path to trained model checkpoint
- `--manifest` (required): CSV manifest with image/mask paths
- `--output-dir`: Output directory for predictions (default: `inference/`)
- `--tile-size`: Tile size for sliding window (default: 256)
- `--overlap`: Overlap fraction between tiles (default: 0.25)
- `--threshold`: Probability threshold for binarization (default: 0.5)
- `--min-object-area`: Minimum object area for post-processing (default: 100 px)
- `--data-root`: Root directory for relative paths (default: `data/`)

**Outputs:**
```
inference/test_predictions/
├── <image>_pred_mask.tif   # Binary mask (0/255)
├── <image>_pred_prob.tif   # Probability map (float32)
└── pixel_metrics.csv       # Per-image Dice/IoU scores
```

**Features:**
- Sliding window with configurable overlap for smooth predictions
- Post-processing: small object removal, hole filling
- Pixel-level evaluation when ground truth available

### Phase 5: Object Quantification

Extract individual spheroids, match to ground truth, and compute morphology metrics:

```bash
quantify_objects --pred-mask-dir inference/test_predictions/ \
                 --gt-manifest data/splits/test.csv \
                 --output-dir metrics/
```

**Parameters:**
- `--pred-mask-dir` (required): Directory with predicted masks
- `--gt-manifest` (required): Path to ground truth manifest CSV
- `--output-dir`: Output directory (default: `metrics/`)
- `--min-object-area`: Minimum object area in pixels (default: 100)
- `--iou-threshold`: IoU threshold for valid match (default: 0.5)
- `--pixel-size`: Pixel size in µm for physical units (optional)
- `--data-root`: Root directory for relative paths (default: `data/`)

**Outputs:**
```
metrics/
├── all_objects.csv           # All detected objects with morphology
├── instance_eval.csv         # Per-image instance metrics (TP/FP/FN)
├── summary.csv               # Dataset-level summary statistics
├── per_image/
│   └── <image>_objects.csv   # Per-object morphology for each image
└── plots/
    └── summary_plots.png     # Visualization (4 plots)
```

**Morphology Metrics (per object):**
- `object_id`: Unique identifier
- `area`: Pixel count (or µm² if pixel_size provided)
- `perimeter`: Boundary length
- `equivalent_diameter`: Diameter of equal-area circle
- `major_axis_length`, `minor_axis_length`: Fitted ellipse axes
- `eccentricity`: 0=circle, 1=line
- `circularity`: 4πA/P² (1=perfect circle)
- `centroid_x`, `centroid_y`: Object center
- `bbox_*`: Bounding box coordinates

**Instance Metrics:**
- True Positives (TP), False Positives (FP), False Negatives (FN)
- Precision, Recall, F1 Score
- Mean Matched IoU

**Visualizations:**
- Histogram of spheroid areas
- Histogram of equivalent diameters
- Histogram of circularity
- Scatter plot: predicted vs ground truth object count

## Complete End-to-End Example

Here's a complete workflow from raw data to quantified objects:

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Validate dataset and create splits
validate_dataset --input-dir data/working/ --output-dir data/splits/
make_splits --manifest data/splits/all.csv --seed 42 --output-dir data/splits/

# 3. Run sanity check (optional but recommended)
sanity_check --config configs/sanity_check.yaml

# 4. Train full model
train --config configs/train.yaml

# 5. Monitor training (in separate terminal)
tensorboard --logdir runs/

# 6. Run inference on test set
predict_full --checkpoint runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
             --manifest data/splits/test.csv \
             --output-dir inference/test_predictions/

# 7. Quantify objects and generate metrics
quantify_objects --pred-mask-dir inference/test_predictions/ \
                 --gt-manifest data/splits/test.csv \
                 --output-dir metrics/

# 8. Review results
cat metrics/summary.csv
open metrics/plots/summary_plots.png  # macOS
```

**Expected timing (M1 Mac):**
- Phase 1 (validation + splits): < 1 minute
- Phase 1.5 (sanity check): 5-10 minutes
- Phase 3 (full training): 30-40 minutes
- Phase 4 (inference): < 1 minute
- Phase 5 (quantification): < 1 minute

**Total:** ~45 minutes from data to results

## Configuration Files

All pipeline stages use YAML configuration files in `configs/`:

- **`dataset.yaml`** - Patch extraction and augmentation parameters
- **`sanity_check.yaml`** - Sanity check configuration
- **`train.yaml`** - Full training configuration

Edit these files to customize behavior without changing code.

## Testing

Run the full test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_training.py -v
```

Current test coverage: 73 tests across all modules (dataset, training, inference, quantification).

## Directory Structure

```
segoid/
├── configs/              # YAML configuration files
├── data/
│   ├── raw/              # Original exports (never edit)
│   ├── working/
│   │   ├── images/       # 8-bit TIFFs for ML
│   │   └── masks/        # Binary masks (0/255)
│   └── splits/           # CSV manifests (train/val/test)
├── runs/                 # Training outputs (checkpoints, logs)
├── inference/            # Prediction outputs (Phase 4)
├── metrics/              # Analysis outputs (Phase 5)
├── src/                  # Package code
│   ├── data/             # Dataset and validation
│   ├── training/         # Training infrastructure
│   ├── inference/        # Tiled inference (Phase 4)
│   └── analysis/         # Object quantification (Phase 5)
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Tech Stack

- **PyTorch** - Deep learning framework
- **segmentation_models_pytorch** - U-Net architecture with ResNet18 encoder
- **albumentations** - Data augmentation
- **TensorBoard** - Training visualization
- **tifffile** + **imagecodecs** - TIFF I/O with LZW compression
- **scikit-image** - Image processing and morphology analysis
- **scipy** - Hungarian algorithm for instance matching
- **matplotlib** - Visualization and plotting
- **pandas** - Data management
- **pytest** - Testing

## Platform Notes

- **Mac M1/M2:** Development and testing on MPS (Metal Performance Shaders)
- **Linux GPU:** Full training on CUDA-enabled GPUs (recommended for production)
- **CPU:** Supported but ~10-20× slower than GPU

## License

MIT License - see [LICENSE](LICENSE)
