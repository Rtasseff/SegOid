# CLAUDE.md

Operational guide for Claude Code working in this repository.

## Quick Start

1. Read `README.md` for project overview
2. Check current phase below
3. Read relevant task/plan file if doing development work

## Project Status

**SegOid** — PyTorch U-Net pipeline for spheroid segmentation in microscopy images.

**POC (Phases 0-5):** ✅ Complete  
**Data Flywheel (Phases 6-10):** 🔄 In progress

| Phase | Status | Description |
|-------|--------|-------------|
| 0-5 | ✅ Done | POC pipeline (see `docs/SDD.md`) |
| 6 | 🔄 Active | Cross-validation infrastructure |
| 7 | ⏳ Next | Batch inference pipeline |
| 8 | ⏳ Planned | Visual review interface |
| 9 | ⏳ Planned | Annotation workflow |
| 10 | ⏳ Planned | Iteration & retraining |

## Key Documents

| Document | Purpose | When to read |
|----------|---------|--------------|
| `README.md` | Project overview, setup, usage | First time, onboarding |
| `docs/SDD.md` | POC design (Phases 0-5) | Reference only, do not reimplement |
| `FLYWHEEL_MASTER_PLAN.md` | Phases 6-10 roadmap | Planning new features |
| `AUTONOMOUS_SESSION.md` | Unattended execution instructions | Only for autonomous runs |

## Commands

```bash
# Environment
source .venv/bin/activate
pip install -e .

# Testing
pytest
pytest --cov=src tests/

# POC Pipeline
validate_dataset --input-dir data/working/ --output-dir data/splits/
make_splits --manifest data/splits/all.csv --seed 42 --output-dir data/splits/
train --config configs/train.yaml
predict_full --checkpoint runs/<run_id>/checkpoints/best_model.pth --manifest data/splits/test.csv
quantify_objects --pred-mask-dir inference/test_predictions/ --gt-manifest data/splits/test.csv

# Interactive Review
review_predictions --image-dir data/working/images --pred-mask-dir inference/full_dataset_review

# Cross-validation (Phase 6)
run_cv --config configs/cv_config.yaml

# TensorBoard
tensorboard --logdir runs/
```

## Directory Structure

```
segoid/
  data/
    working/images/, masks/    # Training data (6 labeled images)
    splits/                    # CSV manifests
  runs/                        # Training outputs, CV experiments
  inference/                   # Predictions  
  metrics/                     # Analysis outputs
  src/                         # Package code
  configs/                     # YAML configs
  docs/                        # SDD.md, other documentation
```

## Critical Conventions

- **File naming:** Image `X.tif` → Mask `X_mask.tif`
- **Image format:** RGB images (convert to grayscale), grayscale masks, LZW compression
- **Splits:** Always by image, never by patch
- **Masks:** Binary 0/255
- **Patch size:** 256 pixels
- **Commits:** Small, focused, with clear messages

## Tech Stack

Python 3.11+, PyTorch, segmentation-models-pytorch, albumentations, tifffile, imagecodecs, scikit-image, scipy, pandas, PyYAML, TensorBoard

## Code Style

- Type hints on public functions
- Docstrings for modules and classes
- Use `logging` module, not print statements
- Config via YAML files in `configs/`
- Match existing patterns in `src/`

## POC Results (Reference)

- Val Dice: 0.799
- Test Dice: 0.794
- Model: U-Net with ResNet18 encoder
- Training: 34 epochs (early stopping), ~37 min on M1
