# Production Model - Quick Start Guide

## Train Production Model (All 6 Images)

```bash
# Activate environment
source .venv/bin/activate

# Train on all labeled data
train --config configs/production_train.yaml

# Monitor training (separate terminal)
tensorboard --logdir runs/
```

**Output**: `runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth`

**Training time**: ~60-90 minutes (100 epochs on M1 Mac)

## Run Inference on New Images

### 1. Create manifest for unlabeled images

Create `unlabeled_images.csv`:
```csv
basename,image_path,mask_path
new_image_1,path/to/new_image_1.tif,
new_image_2,path/to/new_image_2.tif,
```

### 2. Run inference

```bash
predict_full --checkpoint runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
             --manifest unlabeled_images.csv \
             --output-dir inference/batch_001/ \
             --data-root .
```

**Output**: Binary masks in `inference/batch_001/`

## Review and Flag for Correction

```bash
review_predictions --image-dir path/to/images \
                   --pred-mask-dir inference/batch_001 \
                   --output-flagged needs_correction.txt
```

**Controls:**
- LEFT CLICK: Flag/unflag image
- SPACE: Pause/resume
- ESC: Exit

## Correct and Retrain

### 1. Manually correct flagged predictions
- Use annotation tool to fix errors in predicted masks
- Save corrected masks with `_mask.tif` suffix

### 2. Add to training dataset
```bash
# Copy corrected data
cp corrected/*.tif data/working/images/
cp corrected/*_mask.tif data/working/masks/

# Update manifest
validate_dataset --input-dir data/working/ --output-dir data/splits/
```

### 3. Retrain with more data
```bash
train --config configs/production_train.yaml
```

**Result**: New model in `runs/train_<timestamp>/`

## Data Flywheel Loop

```
TRAIN → INFER → REVIEW → CORRECT → RETRAIN
  ↑                                    ↓
  └────────────────────────────────────┘
```

Each iteration improves the model with real-world corrections!

## Key Files

- **Config**: `configs/production_train.yaml`
- **Best model**: `runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth`
- **All training data**: `data/splits/all.csv`
- **Documentation**: `docs/PRODUCTION_MODEL.md` (detailed guide)

## Tips

1. **Use best_model.pth** for inference (highest performance)
2. **Adjust threshold** if predictions too conservative/aggressive:
   ```bash
   predict_full ... --threshold 0.3  # More sensitive
   predict_full ... --threshold 0.7  # More conservative
   ```
3. **Batch corrections**: Collect 5-10 before retraining
4. **Save everything**: Keep runs/ directory for production model
5. **Version control**: Track data/splits/all.csv changes

## Example Full Workflow

```bash
# 1. Train initial model
train --config configs/production_train.yaml

# 2. Infer on batch of new images
predict_full --checkpoint runs/train_20251229_194116/checkpoints/best_model.pth \
             --manifest unlabeled_batch_001.csv \
             --output-dir inference/batch_001/

# 3. Review predictions
review_predictions --image-dir unlabeled/ \
                   --pred-mask-dir inference/batch_001/ \
                   --output-flagged batch_001_flagged.txt

# 4. Correct 5-10 worst predictions manually
# ... use annotation tool ...

# 5. Add corrected data
cp corrected_batch_001/*.tif data/working/images/
cp corrected_batch_001/*_mask.tif data/working/masks/
validate_dataset --input-dir data/working/ --output-dir data/splits/

# 6. Retrain (now with 11-16 images instead of 6)
train --config configs/production_train.yaml

# 7. Repeat with next batch!
```

---

**See `docs/PRODUCTION_MODEL.md` for detailed documentation.**
