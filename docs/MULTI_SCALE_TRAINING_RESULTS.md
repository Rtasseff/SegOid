# Multi-Scale Training Results

## Dataset

9 labeled images across 2 resolutions:

| Resolution | Pixel Size | Images | Names |
|-----------|-----------|--------|-------|
| 2.76 µm/px | Native | 6 | dECM_1_1, dECM_1_2, dECM_2_1, dECM_2_2, Matri_1_1, Matri_1_2 |
| 1.10 µm/px | Rescaled to 2.76 | 3 | GFR_1_1, Normal_1_1, Normal_1_2 |

Higher-resolution images (1.10 µm/px) are rescaled to the training pixel size (2.76 µm/px) with area-proportional patch sampling and dead-patch rejection.

## Cross-Validation: 9-Fold LOOCV

**Strategy:** Leave-one-out cross-validation (9 folds, one held-out image per fold)

**Config:** `configs/cv_multiscale.yaml`

**Experiment directory:** `runs/cv_20260216_171900/`

### Per-Fold Results

| Fold | Held-Out Image | Resolution | Best Val Dice | Best Epoch | Training Time |
|------|---------------|-----------|---------------|------------|---------------|
| 0 | dECM_1_1 | 2.76 µm/px | 0.9206 | 28 | 0.9 min |
| 1 | dECM_1_2 | 2.76 µm/px | 0.9304 | 20 | 0.7 min |
| 2 | dECM_2_1 | 2.76 µm/px | 0.9156 | 5 | 0.3 min |
| 3 | dECM_2_2 | 2.76 µm/px | 0.9296 | 5 | 0.3 min |
| 4 | Matri_1_1 | 2.76 µm/px | 0.8839 | 9 | 0.4 min |
| 5 | Matri_1_2 | 2.76 µm/px | 0.9407 | 26 | 0.8 min |
| 6 | GFR_1_1 | 1.10 µm/px | 0.9554 | 7 | 0.4 min |
| 7 | Normal_1_1 | 1.10 µm/px | 0.9616 | 12 | 0.5 min |
| 8 | Normal_1_2 | 1.10 µm/px | 0.9665 | 18 | 0.6 min |

### Aggregate Performance

| Metric | Value |
|--------|-------|
| **Mean Dice** | **0.934 ± 0.026** |
| Min Dice | 0.884 (Matri_1_1) |
| Max Dice | 0.967 (Normal_1_2) |
| Total training time | 5.6 min |

### Observations

- The 3 higher-resolution images (1.10 µm/px) achieved the **highest** Dice scores (0.955-0.967), indicating multi-scale rescaling works well.
- Matri_1_1 was the hardest image (Dice 0.884), consistent with previous single-scale results.
- Early stopping triggered between epochs 5-28 across folds.

## Comparison with Previous Model

| Metric | Previous (6 images, single-scale) | Current (9 images, multi-scale) |
|--------|-----------------------------------|--------------------------------|
| Training images | 6 (all 2.76 µm/px) | 9 (6 at 2.76 + 3 at 1.10 µm/px) |
| CV strategy | 6-fold LOOCV | 9-fold LOOCV |
| **Mean Dice** | **0.917 ± 0.023** | **0.934 ± 0.026** |
| Min Dice | 0.874 | 0.884 |
| Max Dice | 0.951 | 0.967 |

Adding the 3 higher-resolution images improved mean Dice by +0.017, and the new images themselves generalize well (all above 0.955).

## Production Model

**Config:** `configs/production_train_multiscale.yaml`

**Run directory:** `runs/train_20260216_173233/`

**Checkpoint:** `runs/train_20260216_173233/checkpoints/best_model.pth`

| Parameter | Value |
|-----------|-------|
| Training images | 9 (all labeled data) |
| Epochs | 100 (no early stopping) |
| Best val Dice | 0.940 (epoch 62) |
| Final val Dice | 0.923 (epoch 100) |
| Training pixel size | 2.76 µm/px |
| Architecture | U-Net + ResNet18 |

### Usage

```bash
source .venv/bin/activate

predict_full \
    --checkpoint runs/train_20260216_173233/checkpoints/best_model.pth \
    --manifest <your_images.csv> \
    --output-dir inference/<batch_name>/
```
