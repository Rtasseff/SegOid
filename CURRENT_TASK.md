# Current Task

**Project:** SegOid (Spheroid Segmentation Pipeline)  
**Date:** 2025-12-26  
**Session:** Phase 1.5 — Sanity Check

---

## Status

| Item | Value |
|------|-------|
| **Active Phase** | 1.5 |
| **Last Completed** | Phase 1 — Validation and splits (6 images: 3 train, 2 val, 1 test) |
| **Blocking Issues** | None |

---

## Context from Phase 1

- Images are RGB (converted to grayscale during loading); masks are grayscale
- LZW-compressed TIFFs require `imagecodecs` (already added to dependencies)
- Recommended patch size: **256 pixels** (based on mean spheroid diameter ~58 px)
- Small dataset: only 3 training images, 2 validation image

---

## Session Goal

Implement minimal training infrastructure and run a quick sanity check (5 epochs, reduced data) to validate the full pipeline—data loading, model forward pass, loss computation, mask alignment—before committing to full training.

---

## Tasks

### 1. Implement `PatchDataset`

- [ ] Create `src/data/dataset.py` module
- [ ] Load image (RGB→grayscale) and mask pairs from manifest CSV
- [ ] Implement patch sampling:
  - 70% positive-centered (random foreground pixel + jitter up to 25% patch size)
  - 30% negative (random location with <5% mask coverage)
  - Configurable `patches_per_image` (default: 20, reduce for sanity check)
- [ ] Implement augmentations via albumentations:
  - Horizontal/vertical flip (p=0.5)
  - 90° rotation (p=0.5)
  - Brightness/contrast (±10%)
- [ ] Normalize images to [0, 1] float, masks to binary {0, 1}
- [ ] Return dict: `{"image": tensor, "mask": tensor}`

### 2. Implement basic training loop

- [ ] Create `src/training/train.py` module
- [ ] Load model from segmentation-models-pytorch:
  - U-Net with ResNet18 encoder, pretrained ImageNet weights
  - Adapt for 1-channel grayscale input
- [ ] Implement combined loss: `0.5 × BCE + 0.5 × DiceLoss`
- [ ] Implement validation Dice metric
- [ ] Basic training loop with:
  - Configurable epochs, batch size, learning rate
  - Validation after each epoch
  - Print loss and Dice per epoch
- [ ] Save checkpoint at end

### 3. Implement prediction overlay visualization

- [ ] Create function to generate visual overlays:
  - Original image with GT mask contour (green)
  - Original image with predicted mask contour (red)
  - Or blended/side-by-side comparison
- [ ] Save overlay images to `runs/sanity_check/overlays/`

### 4. Implement `sanity_check` CLI command

- [ ] Wire up in `src/cli.py`
- [ ] Parameters:
  - `--patches-per-image` (default: 10 for sanity check)
  - `--epochs` (default: 5)
  - `--batch-size` (default: 4)
  - `--output-dir` (default: `runs/sanity_check/`)
- [ ] Run training on full train set (only 3 images, so no subsetting needed)
- [ ] Run prediction on validation image(s)
- [ ] Generate and save overlay visualizations
- [ ] Print summary: final loss, final val Dice

### 5. Unit tests

- [ ] Test `PatchDataset` returns correct shapes (256×256 for both image and mask)
- [ ] Test patch sampling produces expected positive/negative ratio (approximately)
- [ ] Test augmentations apply identically to image and mask
- [ ] Test model forward pass produces correct output shape

---

## Reference Sections (in docs/SDD.md)

- **Section 8:** Phase 1.5 specification (procedure, exit criteria, visual inspection)
- **Section 9:** Phase 2 — Patch extraction details (sampling policy, augmentation)
- **Section 10:** Phase 3 — Model training (architecture, loss, metrics)

---

## Files to Create/Modify

| File | Action | Notes |
|------|--------|-------|
| `src/data/dataset.py` | Create | `PatchDataset` class |
| `src/training/train.py` | Create | Training loop, loss, metrics |
| `src/training/visualize.py` | Create | Overlay generation (or include in train.py) |
| `src/cli.py` | Modify | Add `sanity_check` command |
| `tests/test_dataset.py` | Create | Unit tests for PatchDataset |
| `runs/sanity_check/` | Create dir | Output location |

---

## What NOT to Do This Session

- Do not implement full checkpointing strategy (best model, periodic saves)
- Do not implement early stopping or learning rate scheduling
- Do not implement tiled inference (that's Phase 4)
- Do not implement config YAML loading (use CLI args for now)
- Do not optimize for performance (this is validation, not production)
- Do not worry about the small dataset size—we're checking pipeline correctness, not model quality

---

## Completion Criteria (Exit Criteria for Phase 1.5)

This session is complete when:

1. `sanity_check` command runs without errors
2. **Loss decreases** over the 5 epochs (model is learning)
3. **Predictions are spatially coherent** (not random noise or uniform output)
4. **Overlay images** show predicted masks correspond to actual spheroid locations
5. **No systematic offset** between predictions and ground truth (no data pipeline bugs)
6. Unit tests pass: `pytest tests/test_dataset.py`
7. **Go/no-go decision documented** in Notes section below

If any exit criterion fails, investigate and fix before proceeding to Phase 3.

---

## Expected Output Example

```
Epoch 1/5 - Loss: 0.682 - Val Dice: 0.23
Epoch 2/5 - Loss: 0.534 - Val Dice: 0.41
Epoch 3/5 - Loss: 0.421 - Val Dice: 0.58
Epoch 4/5 - Loss: 0.356 - Val Dice: 0.67
Epoch 5/5 - Loss: 0.312 - Val Dice: 0.72

Sanity check complete. Overlays saved to runs/sanity_check/overlays/
Visual inspection required before proceeding to full training.
```

---

## Notes / Decisions Log

_Update during session:_

- 

---

## Next Session Preview

**Phase 3 (Full Training):** If sanity check passes, implement full training with checkpointing, early stopping, TensorBoard logging, and config YAML support. Train for 100 epochs on the complete dataset.