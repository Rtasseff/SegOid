# CLAUDE.md

Operational guide for Claude Code working in this repository.

## Project Summary

**SegOid** — PyTorch U-Net pipeline for spheroid segmentation in microscopy images.

## Current Work

**Active Phase:** 6 (Cross-validation infrastructure)

**What to read:**
- `AUTONOMOUS_SESSION.md` — If running autonomously, follow these instructions
- `CURRENT_TASK.md` — Current phase tasks and completion criteria
- `FLYWHEEL_MASTER_PLAN.md` — Roadmap for Phases 6-10

## Completed Work (Do Not Reimplement)

**POC (Phases 0-5) is complete and working.** The design is documented in `docs/SDD.md` for reference only. Do not modify core POC code unless fixing bugs.

Implemented commands:
```bash
validate_dataset --input-dir data/working/ --output-dir data/splits/
make_splits --manifest data/splits/all.csv --seed 42 --output-dir data/splits/
sanity_check --patches-per-image 10 --epochs 5 --output-dir runs/sanity_check/
train --config configs/train.yaml
predict_full --checkpoint runs/<run_id>/checkpoints/best_model.pth --manifest data/splits/test.csv
quantify_objects --pred-mask-dir inference/test_predictions/ --gt-manifest data/splits/test.csv
```

POC results: Val Dice 0.799, Test Dice 0.794

## Commands

```bash
# Environment
source .venv/bin/activate
pip install -e .

# Testing
pytest
pytest --cov=src tests/

# TensorBoard
tensorboard --logdir runs/
```

## Directory Structure

```
segoid/
  data/
    working/images/, masks/    # Training data
    splits/                    # CSV manifests
  runs/                        # Training outputs
  inference/                   # Predictions  
  metrics/                     # Analysis outputs
  src/                         # Package code
  configs/                     # YAML configs
  docs/                        # SDD.md (POC design, reference only)
```

## Critical Conventions

- **File naming:** Image `X.tif` → Mask `X_mask.tif`
- **Image format:** RGB images (convert to grayscale), grayscale masks, LZW compression
- **Splits:** Always by image, never by patch
- **Masks:** Binary 0/255
- **Patch size:** 256 pixels

## Tech Stack

Python 3.11+, PyTorch, segmentation-models-pytorch, albumentations, tifffile, imagecodecs, scikit-image, pandas, TensorBoard

## Code Style

- Type hints on public functions
- Docstrings for modules and classes
- Use `logging` module, not print statements
- Config via YAML files in `configs/`
- Match existing patterns in `src/`