# CLAUDE.md

Operational guide for Claude Code working in this repository.

## Project Summary

**SegOid** — PyTorch U-Net pipeline for spheroid segmentation in microscopy images. Trains on patches, infers on full images via tiling, outputs binary masks and morphology metrics.

**Current Phase:** 1 (Image curation and dataset splits)  
**Last Completed:** Phase 0 (Project bootstrap)

For design rationale, phase details, and architectural decisions, see `docs/SDD.md`.

## Commands

```bash
# Environment
source .venv/bin/activate
pip install -e .

# Testing
pytest
pytest tests/test_dataset.py
pytest --cov=src tests/

# Pipeline (as implemented)
validate_dataset --input-dir data/working/ --output-dir data/splits/
make_splits --manifest data/splits/all.csv --seed 42 --output-dir data/splits/
# sanity_check, train, predict_full, quantify_objects — not yet implemented
```

## Directory Structure

```
segoid/
  data/
    raw/                    # Original exports (never edit)
    working/
      images/               # 8-bit TIFFs for ML
      masks/                # Binary masks (0/255)
    splits/                 # CSV manifests
  runs/                     # Training outputs
  inference/                # Predictions
  metrics/                  # Analysis outputs
  src/                      # Package code
    data/                   # validate.py, dataset.py
    training/               # train.py
    inference/              # predict.py
    analysis/               # quantify.py
  configs/                  # YAML configs
  docs/                     # SDD.md
```

## Critical Conventions

**File naming:** Image `X.tif` → Mask `X_mask.tif` (enforced pairing)

**Splits:** Always by image, never by patch (prevents data leakage)

**Masks:** Binary 0/255, same dimensions as corresponding image

**Patch size:** 2.5× spheroid diameter, rounded to power of 2

**Empty images:** `mask_coverage == 0` requires `empty_confirmed == True` or flagged for review

## Tech Stack

Python 3.11+, PyTorch, segmentation-models-pytorch, albumentations, tifffile, scikit-image, pandas, TensorBoard

## Platform Notes

- **Mac M1:** Dev/smoke tests only (CPU or MPS, 10-20× slower)
- **Linux GPU:** Full training (RTX 5060 Ti, mixed precision)

## Session Workflow

1. Ask user if we need to git pull origin to sync project before starting
2. Read `CURRENT_TASK.md` for session goals and scope
3. Consult `docs/SDD.md` sections as referenced in task file
4. Update task file completion checkboxes as you progress
5. Note any decisions or issues in the task file's log section

## Code Style

- Type hints on public functions
- Docstrings for modules and classes
- Use `logging` module, not print statements
- Config via YAML files in `configs/`
- CLI via `click` or `argparse` (check existing patterns in `src/cli.py`)