# Production Model Training and Usage Guide

## Overview

This guide documents the production model trained on all 6 labeled images for maximum data utilization. This model is designed for use in a human-in-the-loop data flywheel:

1. **Train** on all available labeled data (6 images)
2. **Infer** on unlabeled images
3. **Review** predictions and correct errors
4. **Retrain** with expanded dataset

## Training the Production Model

### Configuration

The production training configuration is in `configs/production_train.yaml`:

- **Training data**: All 6 labeled images (data/splits/all.csv)
- **Validation data**: Same 6 images (monitors training performance, not generalization)
- **Epochs**: 100 (no early stopping)
- **Patches per image**: 30 (increased sampling for better coverage)
- **Architecture**: U-Net with ResNet18 encoder
- **Learning rate**: 1e-4 with ReduceLROnPlateau scheduling

### Training Command

```bash
# Activate environment
source .venv/bin/activate

# Train production model
train --config configs/production_train.yaml
```

### Training Outputs

Training creates a timestamped directory in `runs/`:

```
runs/train_YYYYMMDD_HHMMSS/
├── config.yaml                    # Config snapshot for reproducibility
├── checkpoints/
│   ├── best_model.pth             # Best model by validation Dice
│   ├── checkpoint_epoch_020.pth   # Periodic checkpoints
│   ├── checkpoint_epoch_040.pth
│   ├── ...
│   └── final_model.pth            # Final model after all epochs
└── tensorboard/                   # TensorBoard logs
```

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
# In a separate terminal
tensorboard --logdir runs/train_YYYYMMDD_HHMMSS/tensorboard

# Open browser to http://localhost:6006
```

**Expected training behavior:**
- Training time: ~60-90 minutes on M1 Mac (100 epochs)
- Validation Dice: Should reach 0.95+ (training on validation data)
- Learning rate will decrease automatically when loss plateaus

### Model Files

Each checkpoint contains:
- `model_state_dict`: Model weights (recommended for loading)
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `epoch`: Training epoch number
- `best_val_dice`: Best validation Dice achieved
- `history`: Training metrics history

**For production inference, use `best_model.pth`** - it has the highest validation Dice.

## Using the Production Model for Inference

### Quick Start

Run inference on a single image or batch of images:

```bash
# Create a manifest CSV for your unlabeled images
# Format: basename,image_path,mask_path (mask_path can be empty)
# Example manifest (unlabeled_images.csv):
#   basename,image_path,mask_path
#   new_image_1,unlabeled/new_image_1.tif,
#   new_image_2,unlabeled/new_image_2.tif,

# Run inference
predict_full --checkpoint runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
             --manifest unlabeled_images.csv \
             --output-dir inference/production_predictions/ \
             --data-root .
```

### Inference Parameters

- `--checkpoint`: Path to trained model (use best_model.pth)
- `--manifest`: CSV with images to process
- `--output-dir`: Where to save predictions
- `--tile-size`: Tile size for sliding window (default: 256)
- `--overlap`: Overlap fraction between tiles (default: 0.25)
- `--threshold`: Probability threshold for binarization (default: 0.5)
- `--min-object-area`: Minimum object size in pixels (default: 100)
- `--data-root`: Root directory for resolving paths

### Inference Outputs

For each input image, inference generates:

```
inference/production_predictions/
├── <image>_pred_mask.tif   # Binary mask (0/255), ready for review
├── <image>_pred_prob.tif   # Probability map (float32), for threshold tuning
└── pixel_metrics.csv       # Per-image metrics (if ground truth available)
```

## Interactive Review Workflow

Review predictions and flag images for correction:

```bash
review_predictions --image-dir unlabeled/ \
                   --pred-mask-dir inference/production_predictions/ \
                   --display-duration 3.0 \
                   --output-flagged flagged_for_correction.txt
```

**Controls:**
- **LEFT CLICK**: Flag/unflag image for manual correction
- **SPACE**: Pause/resume slideshow
- **ESC**: Exit and save flagged list

**Output**: List of flagged images needing manual correction

## Manual Correction and Retraining

### 1. Correct Flagged Predictions

For flagged images, manually correct the predictions in your annotation tool:
- Use the predicted mask as a starting point
- Fix false positives and false negatives
- Save corrected masks with `_mask.tif` suffix

### 2. Expand Training Dataset

Add corrected images to your training data:

```bash
# Copy corrected images and masks to working directory
cp corrected_images/*.tif data/working/images/
cp corrected_masks/*_mask.tif data/working/masks/

# Regenerate manifest with new data
validate_dataset --input-dir data/working/ --output-dir data/splits/

# Check: you should now have more than 6 images
wc -l data/splits/all.csv
```

### 3. Retrain with Expanded Dataset

```bash
# Retrain production model with expanded dataset
train --config configs/production_train.yaml

# New model will be saved to runs/train_<new_timestamp>/
```

### 4. Compare Performance

```bash
# Run inference with new model
predict_full --checkpoint runs/train_<new_timestamp>/checkpoints/best_model.pth \
             --manifest data/splits/all.csv \
             --output-dir inference/iteration_2/

# Quantify improvement
quantify_objects --pred-mask-dir inference/iteration_2/ \
                 --gt-manifest data/splits/all.csv \
                 --output-dir metrics/iteration_2/

# Compare with previous iteration
cat metrics/iteration_2/summary.csv
```

## Loading Models Programmatically

For custom inference pipelines, load the model in Python:

```python
import torch
import segmentation_models_pytorch as smp

# Load model architecture
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,  # We'll load trained weights
    in_channels=1,         # Grayscale
    classes=1              # Binary segmentation
)

# Load trained weights
checkpoint_path = "runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load state dict
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()

# Move to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Now you can use model for inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
```

### Accessing Checkpoint Metadata

```python
checkpoint = torch.load(checkpoint_path)

print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
print(f"Training history keys: {checkpoint['history'].keys()}")

# Access training curves
train_dice = checkpoint['history']['train_dice']
val_dice = checkpoint['history']['val_dice']
```

## Model Performance Notes

**Important**: This production model is trained and validated on the same 6 images, so validation metrics reflect **training performance**, not generalization to new data.

**Expected metrics:**
- Training Dice: 0.95-0.99 (very high, fitting training data)
- Validation Dice: 0.95-0.99 (same data, not independent)
- True performance: Will be revealed when applying to new unlabeled images

**Why this is OK:**
- We're using ALL available labeled data for maximum learning
- True validation happens through human review of predictions
- Incorrect predictions will be manually corrected and added to training
- Each iteration improves the model with real-world corrections

## Data Flywheel Workflow Summary

```
┌─────────────────────────────────────────────────────┐
│  1. TRAIN on all labeled data                       │
│     → train --config configs/production_train.yaml  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  2. INFER on unlabeled images                       │
│     → predict_full --checkpoint best_model.pth      │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  3. REVIEW predictions                               │
│     → review_predictions --output-flagged list.txt  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  4. CORRECT errors manually                          │
│     → Edit masks in annotation tool                 │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  5. EXPAND dataset                                   │
│     → Copy to data/working/, validate_dataset       │
└────────────────┬────────────────────────────────────┘
                 │
                 └──────────► REPEAT (go to step 1)
```

## Troubleshooting

### Out of Memory During Training

Reduce batch size in config:
```yaml
training:
  batch_size: 2  # Reduce from 4
```

### Predictions Too Conservative/Aggressive

Adjust probability threshold:
```bash
predict_full ... --threshold 0.3  # More sensitive (lower threshold)
predict_full ... --threshold 0.7  # More conservative (higher threshold)
```

### Model Not Improving

- Check TensorBoard for loss curves
- Ensure data augmentation is enabled
- Consider reducing learning rate
- Verify input images are correctly normalized

### Loading Checkpoint Errors

Make sure model architecture matches:
- encoder: resnet18
- in_channels: 1
- classes: 1

If you changed architecture, you'll need to retrain from scratch.

## Best Practices

1. **Always use best_model.pth** for production inference
2. **Save the entire runs/ directory** for the production model
3. **Keep training config** (`config.yaml`) with the model
4. **Document each iteration**: Note which images were added/corrected
5. **Version control manifests**: Track data/splits/all.csv changes
6. **Monitor TensorBoard**: Ensure model isn't oscillating or diverging
7. **Test on diverse images**: Don't just correct worst cases
8. **Batch corrections**: Collect 5-10 corrections before retraining

## Next Steps

After training your first production model:

1. **Run inference** on a batch of unlabeled images
2. **Review predictions** systematically
3. **Correct 5-10 images** with diverse errors
4. **Retrain** with expanded dataset
5. **Measure improvement** using quantify_objects
6. **Repeat** until performance is satisfactory

Remember: Each correction makes the model smarter for similar cases!
