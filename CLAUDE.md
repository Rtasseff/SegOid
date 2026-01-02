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
    --checkpoint runs/train_20251229_194116/checkpoints/best_model.pth \
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
train --config configs/production_train.yaml
```

## Key Paths

| Path | Description |
|------|-------------|
| `runs/train_20251229_194116/checkpoints/best_model.pth` | Production model |
| `configs/production_train.yaml` | Training config |
| `data/working/images/`, `masks/` | Training data |
| `data/splits/all.csv` | Dataset manifest |

## Conventions

- **Images:** TIFF, RGB or grayscale
- **Masks:** Binary 0/255, named `<basename>_mask.tif`
- **Manifests:** CSV with `basename,image_path,mask_path`
