# Multi-Scale Training Specification

## Problem

The production model is trained exclusively on 2.76 µm/px images. When presented with 1.1 µm/px images at inference, the model produces poor results even with rescaling, primarily due to intensity distribution differences between the two image sources. Histogram matching partially addresses this, but training on both scales directly will produce a more robust model.

## Goal

Train a single model on images from multiple pixel sizes (currently 2.76 and 1.1 µm/px) by rescaling all images to a common training resolution during patch extraction. The model architecture and tile size remain unchanged.

## Approach

**Rescale before tiling.** During training data loading, each image is rescaled to the target training resolution (2.76 µm/px) based on a per-image `pixel_size` column in the manifest. This is the same rescaling operation already used at inference time.

**Resolution trade-off:** 1.1 µm/px images are downsampled to 2.76 µm/px. The finer detail is discarded. This is acceptable because the experimentalist has confirmed that 2.76 µm/px resolution is sufficient for spheroid segmentation. Preserving the higher resolution would require larger tiles and/or architecture changes — out of scope for this work.

## Changes Required

### 1. Manifest Format

Add an optional `pixel_size` column to training manifests:

```csv
basename,image_path,mask_path,pixel_size
"Well 1, dECM-Image Export-01","working/images/Well 1, dECM-Image Export-01.tif","working/masks/Well 1, dECM-Image Export-01_mask.tif",2.76
"24h, normal Matrigel-Stitching_s1","highres/images/24h, normal Matrigel-Stitching_s1.tif","highres/masks/24h, normal Matrigel-Stitching_s1_mask.tif",1.1
```

**Backward compatibility:** If the `pixel_size` column is missing, all images are assumed to be at the training resolution (no rescaling). Existing manifests continue to work unchanged.

### 2. PatchDataset Changes (`src/data/dataset.py`)

#### `__init__`

Add parameter:
```python
training_pixel_size: float = 2.76
```

Read from config, matches the value stored in checkpoints.

#### `_load_images()`

After loading each image and mask, check for pixel_size in the manifest row. If present and different from `training_pixel_size`, rescale both image and mask:

```python
pixel_size = row.get("pixel_size", None)

if pixel_size is not None and abs(pixel_size - self.training_pixel_size) > 1e-6:
    scale_factor = self.training_pixel_size / pixel_size

    # Rescale image (bilinear, anti-alias if downsampling)
    image = rescale(
        image, scale_factor,
        order=1,
        anti_aliasing=(scale_factor < 1.0),
        preserve_range=True,
    ).astype(image.dtype)

    # Rescale mask (nearest-neighbor to preserve binary values)
    mask = rescale(
        mask, scale_factor,
        order=0,  # Nearest-neighbor for binary masks
        anti_aliasing=False,
        preserve_range=True,
    ).astype(mask.dtype)
```

**Mask interpolation:** Use `order=0` (nearest-neighbor) for masks to preserve binary 0/255 values. Bilinear interpolation would create non-binary edge values.

**When rescaling happens:** At dataset initialization (`_load_images`), not per-patch. Since all images are preloaded into memory, rescaling once at load time is efficient.

#### No changes to `__getitem__`

Patch extraction, augmentation, and normalization remain the same. By the time patches are sampled, all images are already at the common resolution.

### 3. Training Config Changes

Add `training_pixel_size` to the dataset section (it's already in the model section, but the dataset needs it too):

```yaml
dataset:
  patch_size: 256
  patches_per_image: 30
  positive_ratio: 0.7
  training_pixel_size: 2.76    # Target resolution for rescaling
  # ... existing fields
```

Alternatively, just read `model.training_pixel_size` from the existing config — avoid duplicating the value.

### 4. CLI / Training Entry Point Changes (`src/cli.py`)

Pass `training_pixel_size` from config to `PatchDataset`:

```python
train_dataset = PatchDataset(
    manifest_csv=train_manifest,
    patch_size=config['dataset']['patch_size'],
    # ... existing params
    training_pixel_size=config['model'].get('training_pixel_size', 2.76),
)
```

### 5. Preparing the Training Data

#### Labeling high-res images

The 1.1 µm/px images need ground truth masks. Options:

1. **Run inference with histogram matching** on the high-res images to get initial masks, then manually correct using a tool like QuPath, napari, or GIMP
2. **Label from scratch** on the original high-res images
3. **Label on downscaled versions** at 2.76 µm/px, then upscale the masks (easiest, since labeling at 2.76 µm/px is what was already done)

Recommendation: Option 1 (model-assisted labeling) if the histogram-matched inference produces reasonable starting masks. Otherwise option 3.

#### Manifest preparation

Create a combined manifest that includes both the existing 2.76 µm/px training images and the new 1.1 µm/px images with their masks and pixel sizes.

### 6. Histogram Matching During Training

**Not needed.** The purpose of multi-scale training is for the model to learn both intensity distributions natively. The existing brightness/contrast augmentation (±10%) provides some robustness, but seeing real examples from both microscopy setups is the proper solution.

If the intensity difference between scales is large, consider increasing the brightness/contrast augmentation limits:

```yaml
augmentation:
  brightness_limit: 0.2    # Was 0.1
  contrast_limit: 0.2      # Was 0.1
```

### 7. Validation Strategy

For validation, use a mixed manifest containing images from both resolutions. This ensures the model is evaluated on both scales.

Consider stratified splitting so each fold has images from both pixel sizes, rather than all 1.1 µm/px images ending up in one split.

## Files to Modify

| File | Change |
|------|--------|
| `src/data/dataset.py` | Add `training_pixel_size` param, rescale in `_load_images()` |
| `src/cli.py` | Pass `training_pixel_size` to PatchDataset in `train()` and `sanity_check()` |
| `configs/production_train.yaml` | No change needed (already has `model.training_pixel_size`) |
| Training manifest | Add `pixel_size` column |

## Files NOT Modified

| File | Reason |
|------|--------|
| `src/inference/predict.py` | Already handles multi-resolution via `--pixel-size` |
| `src/training/train.py` | No changes needed — receives DataLoader, architecture unchanged |
| Model architecture | Stays U-Net with ResNet-18, 256×256 tiles |

## Implementation Order

1. Prepare labeled masks for 1.1 µm/px images
2. Create combined training manifest with `pixel_size` column
3. Modify `PatchDataset._load_images()` to rescale per-image
4. Add `training_pixel_size` parameter to `PatchDataset.__init__`
5. Update CLI to pass `training_pixel_size` to dataset
6. Run sanity check with mixed-resolution manifest
7. Full training run
8. Evaluate on both resolutions (with and without `--pixel-size`)

## Verification

After training, the model should:

1. Produce good segmentation on 2.76 µm/px images **without** `--pixel-size` (no regression)
2. Produce good segmentation on 1.1 µm/px images **with** `--pixel-size 1.1` (new capability)
3. Not require `--histogram-match` for either resolution (intensity robustness learned from data)
4. Store `training_pixel_size: 2.76` in the checkpoint (unchanged)
