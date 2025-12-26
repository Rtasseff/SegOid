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

**Current Phase:** Phase 3 (Full Model Training) - ✅ Complete

**Completed Phases:**
- ✅ Phase 1: Dataset validation and train/val/test splits
- ✅ Phase 1.5: Sanity check (pipeline validation)
- ✅ Phase 3: Production training infrastructure

**Next:** Phase 4 (Tiled inference on full-resolution images)

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

### Phase 4: Tiled Inference (Not Yet Implemented)

Apply trained model to full-resolution images:

```bash
predict_full --config configs/predict.yaml \
             --checkpoint runs/train_<timestamp>/checkpoints/best_model.pth
```

### Phase 5: Object Quantification (Not Yet Implemented)

Extract individual spheroids and compute morphology metrics:

```bash
quantify_objects --config configs/quantify.yaml
```

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

Current test coverage: 45 tests across dataset, training, and validation.

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
- **scikit-image** - Image processing
- **pandas** - Data management
- **pytest** - Testing

## Platform Notes

- **Mac M1/M2:** Development and testing on MPS (Metal Performance Shaders)
- **Linux GPU:** Full training on CUDA-enabled GPUs (recommended for production)
- **CPU:** Supported but ~10-20× slower than GPU

## License

MIT License - see [LICENSE](LICENSE)
