# CLAUDE.md

## Project

**SegOid** — Spheroid segmentation pipeline. Production-ready tool for microscopy image analysis.

## Documentation

**See `README.md`** for complete documentation including:
- Installation
- Running inference
- Interactive prediction review
- Morphology metrics
- Retraining workflow
- All command parameters
- Troubleshooting

## Typical Usage

Run the production model on new images:

```bash
source .venv/bin/activate

# Run inference
predict_full \
    --checkpoint models/production_v2.0/checkpoints/best_model.pth \
    --manifest <your_images.csv> \
    --output-dir inference/<batch_name>/

# Review predictions
review_predictions \
    --image-dir <path/to/images/> \
    --pred-mask-dir inference/<batch_name>/ \
    --output-flagged flagged.txt
```

## Quick Command Reference

```bash
# Inference
predict_full --checkpoint <model.pth> --manifest <images.csv> --output-dir <output/>
review_predictions --image-dir <images/> --pred-mask-dir <preds/> --output-flagged <flagged.txt>
quantify_objects --pred-mask-dir <preds/> --gt-manifest <manifest.csv> --output-dir <metrics/>

# Training (if retraining)
validate_dataset --input-dir data/working/ --output-dir data/splits/
train --config configs/production_train_multiscale.yaml
```

## Key Paths

| Path | Description |
|------|-------------|
| `models/production_v2.0/checkpoints/best_model.pth` | Production model (multi-scale, 9 images) — backed up on SSD |
| `models/production_v2.0/config.yaml` | Training config snapshot used to produce the production model |
| `models/production_v2.0/manifest.csv` | Manifest snapshot used to produce the production model |
| `configs/production_train_multiscale.yaml` | Production training config |
| `configs/cv_multiscale.yaml` | Cross-validation config |
| `data/working_276/`, `data/working_110/` | Training data (2 resolutions) |
| `data/splits/all.csv` | Current dataset manifest (9 images, with pixel_size) |

## Production Model Convention

Curated/blessed model artifacts live in `models/<version>/`, kept separate from experimental training output in `runs/`. Each `models/<version>/` folder is self-describing — it contains everything needed to reproduce the model:

- `checkpoints/best_model.pth` — the production weights
- `config.yaml` — training config snapshot
- `manifest.csv` — dataset manifest snapshot (which images, masks, pixel sizes)
- `tensorboard/` — training logs (optional)

Why separate from `runs/`: `runs/` is treated as ephemeral/experimental output (often gitignored, sometimes stored on non-backed-up scratch drives). Anything blessed for production gets promoted into `models/` so it has a stable path and a backup story decoupled from training scratch space.

`models/` is gitignored (large `.pth` files), but the convention is universal — every checkout of this repo should adopt it.

## Local Setup

If a `LOCAL_SETUP.md` file exists in the project root (gitignored, machine-specific), it documents that machine's storage arrangement — for example, whether `data/`, `runs/`, or `inference/` are symlinks to a different drive, and any local conventions for backups or scratch space. Read it when working on a machine that has one; it captures things that aren't true universally for the repo.

## Conventions

- **Images:** TIFF, RGB or grayscale
- **Masks:** Binary 0/255, named `<basename>_mask.tif`
- **Manifests:** CSV with `basename,image_path,mask_path,pixel_size`
