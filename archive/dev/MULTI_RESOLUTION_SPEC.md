# Multi-Resolution Inference Support

**Document Type:** Technical Specification  
**Date:** 2025-01-30  
**Status:** Ready for Implementation

---

## Problem Statement

The SegOid production model was trained on microscopy images with a pixel size of **2.76 µm/pixel**. Users now want to process images captured at **1.1 µm/pixel** (higher magnification). Running inference directly on these high-resolution images produces poor results because:

- Spheroids that were ~58 pixels in diameter during training appear as ~145 pixels in high-res images (2.5x larger)
- The model sees magnified texture instead of whole spheroid structures
- Tile boundaries intersect spheroids differently than during training

**Scale factor:** 2.76 / 1.1 = **2.51x**

Users frequently switch between both magnifications, so the solution must handle either resolution seamlessly.

---

## Solution: Input Rescaling

Rescale input images to match training resolution before inference, then rescale output masks back to original resolution.

### Approach

```
Input Image (1.1 µm/px)
        │
        ▼
┌─────────────────────────┐
│  Downsample by 2.51x    │  ◄── scale_factor = training_pixel_size / input_pixel_size
│  (to match 2.76 µm/px)  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Standard Tiled         │
│  Inference (256×256)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Upsample probability   │
│  map to original size   │
└───────────┬─────────────┘
            │
            ▼
Output Mask (original resolution)
```

### Why This Works

- Model sees spheroids at the same apparent size as during training
- All learned features (edges, textures, shapes) apply correctly
- Output masks retain full input resolution for downstream analysis
- No retraining required

---

## Design Specification

### Model Configuration

The training pixel size is stored in the checkpoint's `model_config` metadata:

```python
"model_config": {
    "encoder_name": "resnet18",
    "in_channels": 1,
    "classes": 1,
    "training_pixel_size": 2.76,  # µm/pixel
}
```

**Backward Compatibility:** For older checkpoints without this field, fall back to `TRAINING_PIXEL_SIZE = 2.76` as the default.

**Rationale:** If the model is retrained at a different resolution in the future, the checkpoint remains self-describing and inference automatically adapts.

### Input Parameters

Add a new optional parameter to inference functions and CLI:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pixel_size` | float | None | Pixel size of input image in µm. When provided, images are rescaled to match training resolution. When None, no rescaling is performed (assumes input matches training resolution). |

### Scale Factor Calculation

```
If pixel_size is provided and pixel_size ≠ TRAINING_PIXEL_SIZE:
    scale_factor = TRAINING_PIXEL_SIZE / pixel_size
Else:
    scale_factor = 1.0 (no rescaling)
```

**Examples:**

| Input Pixel Size | Scale Factor | Effect |
|------------------|--------------|--------|
| None | 1.0 | No rescaling (original behavior) |
| 2.76 µm | 1.0 | No rescaling (matches training) |
| 1.1 µm | 2.51 | Downsample to ~40% size before inference |
| 5.52 µm | 0.5 | Upsample to 2x size before inference |

---

## Implementation Details

### Code Architecture

**Rescaling location:** Implement rescaling in the **wrapper functions**, not in model-specific inference functions.

- `predict_single_image()` — Unified wrapper (handles both PyTorch and ONNX)
  - ✓ Apply rescaling here
  - Calls either `predict_full_image()` (PyTorch) or `predict_full_image_onnx()` (ONNX)

- `predict_full_image()` — PyTorch-specific inference
  - ✗ No rescaling logic here
  - Operates on image as-is

- `predict_full_image_onnx()` — ONNX-specific inference
  - ✗ No rescaling logic here
  - Operates on image as-is

This ensures:
1. Rescaling logic is implemented once
2. Both backends benefit automatically
3. Backend-specific code remains simple and focused

**Note:** `predict_image_from_path()` is an older legacy function that directly calls PyTorch inference. It should also support rescaling for backward compatibility.

### Image Rescaling

**Downsampling (scale_factor < 1.0, i.e., input is higher resolution than training):**
- Use `skimage.transform.rescale()` with `anti_aliasing=True`
- This preserves detail and avoids aliasing artifacts
- New dimensions: `(width × scale_factor, height × scale_factor)`
- Maintains consistency with existing scipy/skimage stack

**Upsampling (scale_factor > 1.0, i.e., input is lower resolution than training):**
- Use `skimage.transform.rescale()` with `order=1` (bilinear interpolation)
- Less common case but should be supported
- Anti-aliasing not needed for upsampling

### Probability Map Rescaling

After inference on the scaled image, rescale the probability map back to original dimensions:
- Use `skimage.transform.resize()` with `order=1` (bilinear interpolation)
- **Critical:** Rescale the float probability map, NOT the thresholded binary mask
- Apply threshold after rescaling to preserve boundary accuracy
- Apply post-processing (small object removal, hole filling) after thresholding

### Processing Flow

```
1. Load image
2. Store original dimensions (H_orig, W_orig)
3. Load model and read training_pixel_size from checkpoint (fallback: 2.76)
4. If pixel_size provided and pixel_size ≠ training_pixel_size:
   a. Calculate scale_factor = training_pixel_size / pixel_size
   b. Calculate scaled dimensions (H_scaled, W_scaled)
   c. Rescale image using skimage.transform.rescale()
      - Use anti_aliasing=True if scale_factor < 1.0 (downsampling)
   d. Log: "Rescaling from {H_orig}×{W_orig} to {H_scaled}×{W_scaled}
           (pixel size {pixel_size} µm → {training_pixel_size} µm, scale={scale_factor:.2f}x)"
5. Run standard tiled inference on (possibly rescaled) image → prob_map_scaled
6. If rescaling was applied:
   a. Rescale probability map back to original dimensions using skimage.transform.resize()
      with order=1 (bilinear) → prob_map_original
   b. Calculate effective_min_area = min_object_area / (scale_factor²)
      (scales post-processing threshold to match original resolution)
7. Apply threshold to probability map → binary_mask
8. Apply post-processing with effective_min_area:
   - Remove small objects (< effective_min_area pixels)
   - Fill holes
9. Save outputs at original resolution
```

**Note on min_object_area scaling:** Since output masks are at original resolution, the minimum object area threshold must be adjusted. For example, if training used `min_object_area=100` pixels at 2.76 µm/px, and input is 1.1 µm/px (scale_factor=2.51), then `effective_min_area = 100 / 2.51² ≈ 15.8 pixels` at the original resolution. This ensures objects are filtered based on physical size, not pixel count.

### Edge Cases

| Case | Handling |
|------|----------|
| `pixel_size = None` | Skip rescaling entirely (backward compatible) |
| `pixel_size = training_pixel_size` | Skip rescaling (no-op, scale_factor=1.0) |
| Very large/small scale factors | Proceed without warnings (see Caveats below) |
| Non-square pixels | Not supported; assume square pixels |

**Caveats:** See README for guidance on scale factor limits and when rescaling may degrade quality.

---

## CLI Changes

Update the `predict_full` command to accept the new parameter:

```
predict_full --checkpoint <path> --manifest <path> [OPTIONS]

Options:
  --pixel-size FLOAT    Pixel size of input images in µm. Images will be
                        rescaled to match training resolution (2.76 µm/px).
                        Omit for images already at training resolution.
```

**Example usage:**

```bash
# Standard resolution (no rescaling)
predict_full --checkpoint model.pth --manifest images.csv --output-dir output/

# High resolution images at 1.1 µm/px
predict_full --checkpoint model.pth --manifest images.csv --output-dir output/ \
             --pixel-size 1.1
```

---

## GUI Changes

Add a simple input field for pixel size:

```
┌─ Inference Settings ─────────────────────────┐
│                                              │
│  Pixel Size (µm/pixel): [2.76____]          │
│  Leave at 2.76 for standard resolution      │
│  images. Enter 1.1 for high-mag images.     │
│                                              │
└──────────────────────────────────────────────┘
```

**Keep it simple:** No dropdowns, no real-time preview for first iteration. User enters the value, inference handles rescaling automatically.

---

## Output Considerations

### Probability Maps
- Saved at original (input) resolution
- Rescaling may cause slight smoothing of probability values

### Binary Masks
- Saved at original (input) resolution
- Boundaries are accurate to ~1-2 pixels (interpolation artifact)

### Metrics

When `pixel_size` is provided, output metrics include both pixel and physical measurements:

**CSV Format:**
```csv
image,object_id,area_px,area_um2,equivalent_diameter_px,equivalent_diameter_um,perimeter_px,perimeter_um,circularity
sample_001,1,5230,15862.8,81.5,89.65,290.3,319.3,0.782
sample_001,2,3891,11798.3,70.4,77.44,251.8,277.0,0.771
```

**Column definitions:**
- `area_px`: Object area in pixels² (at original image resolution)
- `area_um2`: Object area in µm² (= area_px × pixel_size²)
- `equivalent_diameter_px`: Diameter of circle with same area, in pixels
- `equivalent_diameter_um`: Same in µm (= diameter_px × pixel_size)
- `perimeter_px`, `perimeter_um`: Similarly scaled
- `circularity`: Dimensionless, same in both units

When `pixel_size = None`, only pixel-based columns are output.

**Physical units use input pixel_size, not training_pixel_size.** This ensures measurements reflect actual physical dimensions of the input images.

### Metadata and Logging

**Console logging** (INFO level):
- When rescaling is applied, log: `"Rescaling from {H}×{W} to {H'}×{W'} (pixel size {input} µm → {training} µm, scale={factor:.2f}x)"`
- When skipped: `"No rescaling (input matches training resolution)"`

**Output metadata** (optional enhancement):
Consider adding a `_metadata.json` file alongside predictions:
```json
{
  "image": "sample_001.tif",
  "input_pixel_size": 1.1,
  "training_pixel_size": 2.76,
  "scale_factor": 2.509,
  "rescaling_applied": true,
  "original_dimensions": [4000, 4000],
  "scaled_dimensions": [1594, 1594],
  "inference_time_seconds": 12.3
}
```

This is **not required** for first iteration but useful for debugging and reproducibility.

---

## Testing Criteria

### Unit Tests

1. **Scale factor calculation**
   - `pixel_size=1.1` → `scale_factor=2.509...`
   - `pixel_size=2.76` → `scale_factor=1.0`
   - `pixel_size=None` → `scale_factor=1.0`

2. **Image rescaling dimensions**
   - Input (1000, 1000) with scale_factor=0.4 → intermediate (400, 400)
   - Output mask shape equals input shape

3. **Interpolation methods**
   - Downsample uses area interpolation
   - Probability map upsample uses linear interpolation

### Integration Tests

1. **Backward compatibility** (Critical)
   - Run inference on standard images without `--pixel-size`
   - Verify results are identical to previous version (bit-for-bit if possible)

2. **End-to-end with real high-res images** (Critical)
   - Process batch of 1.1 µm/px images with `--pixel-size 1.1`
   - Verify output masks are at original resolution
   - Use `review_predictions` to visually confirm masks align with spheroids
   - Compare to results without rescaling to confirm improvement

3. **Synthetic round-trip test** (Optional, nice-to-have)
   - Take training image (2.76 µm/px) → inference → mask A
   - Upsample same image 2.51x (simulate 1.1 µm/px)
   - Run with `--pixel-size 1.1` → mask B (downsampled to original size)
   - Compare masks: Dice should be >0.95
   - This validates rescaling preserves results

**Priority:** Focus on tests 1 and 2 for first trial. Test 3 can be added later if needed.

---

## Performance Notes

**Speed:**
- Rescaling adds minimal overhead (<1 second per image for typical sizes)
- Downsampled inference is actually **faster** (fewer tiles to process)

Example: 4000×4000 image at 1.1 µm/px:
- Without rescaling: 4000×4000 → ~324 tiles (256×256 with 25% overlap)
- With rescaling: ~1594×1594 → ~52 tiles → **~6x faster inference**

**Memory:**
- Peak memory usage is approximately **2.5× the original image size**
- Both original and rescaled images are in memory during rescaling
- Probability map at both resolutions (float32) during upscaling

Example: 8000×8000 16-bit TIFF at 1.1 µm/px:
- Original image: 128 MB
- Scaled image (~3200×3200): 20 MB
- Probability map (original size, float32): 256 MB
- **Peak usage: ~400 MB**

For very large images (>10,000×10,000), monitor memory usage. Consider processing in batches if needed.

**See README** for additional guidance on scale factor limits and quality considerations.

---

## Limitations and Caveats

**Scale Factor Quality Guidelines:**

| Scale Factor | Input Pixel Size | Quality | Recommendation |
|--------------|------------------|---------|----------------|
| 0.8 - 1.2 | 2.3 - 3.5 µm/px | Excellent | Safe to use |
| 0.5 - 0.8 or 1.2 - 2.0 | 1.4 - 2.3 or 3.5 - 5.5 µm/px | Good | Should work well |
| 0.3 - 0.5 or 2.0 - 3.0 | 0.9 - 1.4 or 5.5 - 9.2 µm/px | Fair | Expect some degradation |
| <0.3 or >3.0 | <0.9 or >9.2 µm/px | Poor | Not recommended |

**Why extreme scale factors degrade quality:**
- **Very high magnification (scale < 0.3):** Model sees overly magnified texture, loses context of whole spheroid shape
- **Very low magnification (scale > 3.0):** Model sees blurry, undersampled spheroids, loses fine boundary details

**Recommendation:** If you regularly process images at extreme magnifications, consider retraining the model at that target resolution for optimal results.

**Memory limitations:**
- Images larger than ~15,000×15,000 pixels may exceed available RAM depending on system
- For very large images, consider tiling the image externally before inference

**Interpolation artifacts:**
- Probability map boundaries may have 1-2 pixel smoothing/rounding
- This is acceptable for most use cases but may affect precise boundary-based metrics

**This section provides guidance for README documentation.** No runtime warnings are implemented in code.

---

## Future Considerations

This specification addresses immediate inference needs. Future enhancements may include:

1. **Per-image pixel size in manifest** — Allow CSV to specify pixel_size per row for mixed-resolution datasets:
   ```csv
   basename,image_path,mask_path,pixel_size
   img001,path/img001.tif,,1.1
   img002,path/img002.tif,,2.76
   ```
   CLI `--pixel-size` would become the default for rows without pixel_size specified.

2. **Scale-augmented retraining** — Train model to handle multiple resolutions natively

3. **Automatic pixel size detection** — Read from TIFF metadata if available

4. **Resolution-specific models** — Maintain separate optimized models if quality demands

These are deferred to a separate specification.

---

## Implementation Order

**Phase 1: Core Functionality**
1. Add `training_pixel_size` to `model_config` in `train.py` (extend recent checkpoint refactor)
2. Update `save_checkpoint()` to include `training_pixel_size` in model_config
3. Implement rescaling logic in `predict.py`:
   - Add rescaling to `predict_single_image()` (unified wrapper)
   - Add rescaling to `predict_image_from_path()` (legacy function)
   - Update probability map upscaling
   - Scale `min_object_area` by `1 / scale_factor²`
4. Update metrics output to include both pixel and physical measurements
5. Add `--pixel-size` parameter to CLI `predict_full` command

**Phase 2: Testing**
6. Add unit tests for scale factor calculation and dimension handling
7. Run backward compatibility test (no --pixel-size produces same results)
8. Run end-to-end test on real 1.1 µm/px images with visual validation

**Phase 3: GUI and Documentation**
9. Add pixel size input field to GUI
10. Update README with usage examples, caveats, and quality guidelines
11. (Optional) Add synthetic round-trip validation test

**Dependencies:**
- Phase 1 step 1-2 depend on recent model_config refactor being complete
- Phase 2 tests depend on Phase 1 being complete
- Phase 3 GUI can proceed in parallel with documentation

---

## Summary

| Item | Value |
|------|-------|
| New parameter | `--pixel-size` (float, optional, µm/pixel) |
| Default behavior | No rescaling (backward compatible) |
| Training resolution | Stored in checkpoint `model_config` (fallback: 2.76 µm/px) |
| Scale calculation | `training_pixel_size / input_pixel_size` |
| Downsample method | `skimage.transform.rescale()` with `anti_aliasing=True` |
| Upsample method | `skimage.transform.rescale()` with `order=1` (bilinear) |
| Probability map rescaling | `skimage.transform.resize()` with `order=1` (bilinear) |
| Post-processing | `min_object_area` scaled by `1 / scale_factor²` |
| Output resolution | Always matches input resolution |
| Output metrics | Both pixel and physical units (µm, µm²) when pixel_size provided |
| Rescaling location | Wrapper functions (`predict_single_image()`), not model-specific code |
