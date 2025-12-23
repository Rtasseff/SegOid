# SegOid - Spheroid Segmentation Pipeline

A reproducible, researcher-friendly pipeline for training semantic segmentation models to identify spheroids in microscopy images using PyTorch U-Net.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install in development mode
pip install -e ".[dev]"
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Architecture overview and development guide for Claude Code
- **[docs/SDD.md](docs/SDD.md)** - Complete Software Design Document with detailed specifications

## Project Status

**Current Phase:** Phase 0 (Project bootstrap) - ✅ Complete

See `docs/SDD.md` for the full development roadmap.

## Pipeline Commands

Once implemented, the pipeline will provide these commands:

```bash
# Phase 1: Data validation and splits
validate_dataset --input-dir working/ --output-dir splits/
make_splits --manifest splits/all.csv --seed 42 --output-dir splits/

# Phase 1.5: Sanity check
sanity_check --config configs/sanity_check.yaml

# Phase 3: Training
train --config configs/train.yaml

# Phase 4: Inference
predict_full --config configs/predict.yaml --checkpoint runs/<run_id>/checkpoints/best.pth

# Phase 5: Quantification
quantify_objects --config configs/quantify.yaml
```

## License

MIT License - see [LICENSE](LICENSE)
