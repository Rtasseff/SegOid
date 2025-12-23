# Spheroid Segmentation Pipeline (Fiji/ilastik Labels → PyTorch U-Net → Full-Image Metrics)

**Version:** 2.0  
**Last updated:** 2025-01-15

## Current Status
**Active Phase:** 0 (Project bootstrap)
**Last Completed:** Creation of this software design doc (SDD)
**Blocking Issues:** None
**Next Action:** Read sections 1 to 5 for context and start phase 0

## 1. Problem statement

We need a reproducible, researcher-friendly pipeline that trains a semantic segmentation model to identify spheroids in microscopy images. Initial data consists of well-plate images where each well may contain a single spheroid. Labels are provided as binary masks created in Fiji (ROI-based) and/or ilastik. The model will be trained on patches and deployed via tiled inference to produce full-resolution segmentation masks. A downstream analysis stage will extract individual spheroid objects and compute quantitative morphology metrics.

### 1.1 Strategic context

This pipeline is designed as general-purpose infrastructure for cell microscopy segmentation, not a point solution for well-plate spheroids. The current dataset serves as a simple test case that simultaneously solves a real analysis need. The architecture and workflow are intentionally kept general to support future extension to:

* 2D cell cultures (adherent cells, colonies)
* More complex 3D organoid imaging
* Variable imaging conditions and plate formats

This motivates the CNN-based approach over classical image analysis methods that might be more efficient for the initial dataset but would not generalize.

### 1.2 Scope assumptions (POC)

* Initial images are well-plate format with one spheroid per well (or empty wells).
* Spheroids are well-separated, clearly defined ("nice" cases).
* Ground-truth is binary: spheroid vs background.
* Final deliverable: full-image segmentation + per-object quantification with instance-level evaluation.

### 1.3 Non-goals (for POC)

* Multi-class segmentation (e.g., diffuse plating vs spheroid).
* Robust handling of touching/overlapping objects.
* Per-well localization preprocessing (revisit if needed—see Section 11).
* Automated well-grid detection.

## 2. Users and operational constraints

* Non-technical annotators create masks in Fiji and save to a fixed folder structure.
* The computational workflow must be:
  * Runnable on a **MacBook Air M1 (2020)** for development and smoke tests.
  * Portable to a **Linux workstation** for heavy training and inference:
    * 64 GB DDR5
    * Intel Core Ultra 9 285 vPro (24 cores)
    * NVIDIA GeForce RTX 5060 Ti, 16 GB GDDR7

## 3. Inputs, outputs, and file organization

### 3.1 Canonical folder structure

```
project/
  data/                     # data directory (not committed to git)
    raw/                    # original microscope exports (never edited)
    working/
      images/               # standardized 8-bit copies for labeling + ML
      masks/                # final binary masks (0/255) aligned to images
      rois/                 # optional Fiji ROI Manager exports
    splits/                 # train/val/test manifests (CSV)
  runs/                     # training outputs, checkpoints, logs
  inference/                # full-image predictions
  metrics/                  # object tables and summary plots
  src/                      # Python package (pipeline code)
  notebooks/                # optional exploratory notebooks
  docs/                     # documentation (this SDD, SOP stubs, etc.)
  configs/                  # YAML configuration files
```

### 3.2 Naming convention

* Image: `<basename>.tif`
* Mask: `<basename>_mask.tif`
* ROI: `<basename>_rois.zip`

Constraint: for any image `X.tif`, the corresponding mask must be `X_mask.tif`.

### 3.3 Data manifests

Explicit manifests for reproducibility:

* `data/splits/all.csv` — complete dataset inventory
* `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`

Each row contains:

| Column | Description |
|--------|-------------|
| `basename` | Filename stem |
| `image_path` | Relative path to image |
| `mask_path` | Relative path to mask |
| `mask_coverage` | Fraction of foreground pixels |
| `object_count` | Connected components in mask |
| `empty_confirmed` | Boolean flag for confirmed empty images |
| `date` | Acquisition date (optional) |
| `batch` | Batch identifier (optional) |
| `operator` | Annotator ID (optional) |
| `notes` | Free text (optional) |

Split rule: split **by image**, not by patches, to prevent data leakage.

## 4. Tech stack

* **Language:** Python 3.11+ (pin exact version per environment)
* **DL framework:** PyTorch
* **Key libraries:**
  * `segmentation-models-pytorch` (SMP) — pretrained encoder architectures
  * `albumentations` — augmentation with paired image/mask transforms
  * `tifffile` / `imageio` — TIFF I/O
  * `numpy`, `pandas`
  * `scikit-image` — connected components, morphology, regionprops
  * `scipy` — Hungarian matching for instance evaluation
  * `opencv-python` — optional fast morphology
  * `tqdm` — progress bars
  * `tensorboard` — training logs and visualization

### 4.1 Repository and environment

* Git repository at project root
* Python virtual environment:
  * macOS: `python -m venv .venv`
  * Linux: same, with CUDA-enabled PyTorch
* Dependencies specified in `pyproject.toml` 

## 5. System overview (phases)

| Phase | Name | Goal |
|-------|------|------|
| 0 | Project bootstrap | Reproducible scaffolding |
| 1 | Image curation and dataset splits | Validated, split dataset with QC |
| 1.5 | Sanity check | Pipeline validation before full training |
| 2 | Patch extraction and augmentation | Training data pipeline |
| 3 | Model training | Baseline U-Net on patches |
| 4 | Full-image inference | Tiled prediction at full resolution |
| 5 | Object quantification | Instance extraction, matching, and metrics |

---

## 6. Phase 0 — Project bootstrap

**Goal:** Create reproducible project scaffolding.

**Tasks:**

1. Initialize Git repository
2. Create virtual environment and install dependencies
3. Create `src/` package structure with `__init__.py` files
4. Create baseline CLI entrypoints (placeholder commands)
5. Add `docs/` with this SDD

**Deliverables:**

* Repository with `src/` package, minimal CLI, pinned dependencies
* This SDD in `docs/`

---

## 7. Phase 1 — Image curation and dataset splits

**Goal:** Produce a traceable, validated mapping between images and masks; define train/val/test splits with quality metrics.

### 7.1 Dataset validation

For every image in `data/working/images/`:

1. **Pairing check:** Corresponding mask exists in `data/working/masks/` with correct naming
2. **Dimension check:** Image and mask have identical (H, W)
3. **Format check:** Mask is binary (0/255) or convertible via thresholding
4. **Coverage computation:** Calculate `mask_coverage = foreground_pixels / total_pixels`
5. **Object count:** Run connected components labeling, record count

### 7.2 Empty image handling

Images with zero mask coverage require explicit confirmation:

* If `mask_coverage == 0` and `empty_confirmed == False`: flag for review
* Annotator must confirm whether image is truly empty (no spheroid present) or annotation is missing
* Confirmed empty images are valid negative examples, not errors

### 7.3 QC report

Generate `data/splits/qc_report.csv` containing per-image:

* All manifest columns
* Validation status (pass/fail with reason)
* Warnings (e.g., unusually low/high coverage)

Console summary:

* Total images, passed, failed
* Coverage distribution (min, max, median, quartiles)
* Object count distribution
* Count of confirmed empty vs. flagged for review

### 7.4 Split generation

* Deterministic split using fixed random seed
* Default ratio: 70% train, 15% val, 15% test
* Optional stratification by coverage buckets to ensure balanced foreground representation

**Implementation:**

* Module: `src/data/validate.py`
* CLI command: `validate_dataset --input-dir data/working/ --output-dir data/splits/`
* CLI command: `make_splits --manifest data/splits/all.csv --seed 42 --output-dir data/splits/`

**Deliverables:**

* `data/splits/all.csv`, `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`
* `data/splits/qc_report.csv`
* Console QC summary

---

## 8. Phase 1.5 — Sanity check

**Goal:** Validate the entire pipeline (data loading, model forward pass, loss computation, mask alignment) before committing to full training.

### 8.1 Procedure

1. Select a small subset: ~10% of training data or minimum 5 images
2. Configure minimal training: 5 epochs, reduced patches per image
3. Train and record loss curve
4. Generate prediction overlays on 3–5 validation images

### 8.2 Exit criteria

* Loss decreases over epochs (model is learning something)
* Predictions are spatially coherent (not random noise or uniform)
* Predicted mask regions correspond to actual spheroid locations (no systematic offset)
* No obvious data pipeline bugs (e.g., flipped axes, wrong normalization)

### 8.3 Visual inspection

Save overlay images to `runs/sanity_check/overlays/`:

* Original image with GT mask contour (green)
* Original image with predicted mask contour (red)
* Side-by-side or blended comparison

**Implementation:**

* CLI command: `sanity_check --config configs/sanity_check.yaml`
* Config specifies reduced data fraction, epochs, output directory

**Deliverables:**

* `runs/sanity_check/` with loss log and overlay images
* Go/no-go decision documented before proceeding to Phase 2

---

## 9. Phase 2 — Patch extraction and augmentation

**Goal:** Implement efficient patch-based training without manual cropping.

### 9.1 Patch size selection

Patch size should accommodate a full spheroid even when the center pixel is at the object edge. Formula:

```
patch_size ≈ 2.5 × estimated_spheroid_diameter
```

Measure average spheroid diameter from masks during Phase 1 QC. Round to nearest power of 2 or convenient multiple (e.g., 256, 384, 512) for GPU efficiency.

Rationale: With mask-driven sampling that jitters around foreground pixels, the spheroid may be off-center. A 2.5× margin ensures the full object plus surrounding context fits within the patch.

### 9.2 Patch sampling policy

Implement a PyTorch `Dataset` that loads full image+mask pairs and returns sampled patches.

**Sampling mix per epoch:**

* **70% positive-centered:** Select a random foreground pixel, apply random jitter (up to 25% of patch size), crop patch centered on jittered location
* **30% negative:** Select a random location where local mask coverage is below 5%

This balances learning spheroid features against background discrimination without wasting batches on pure background.

### 9.3 Augmentation strategy

**Initial conservative set:**

* Horizontal flip (p=0.5)
* Vertical flip (p=0.5)
* 90° rotation (p=0.5, randomly 90°/180°/270°)
* Brightness adjustment (±10%)
* Contrast adjustment (±10%)

**Intentionally excluded for now:**

* Aggressive brightness/contrast changes
* Gaussian noise/blur
* Elastic deformation

Rationale: The well-plate images have consistent dark ring artifacts at well edges. Aggressive augmentation might teach the model to over-rely on or ignore these features in ways that hurt generalization to non-well-plate data. Revisit augmentation strength when extending to new image types.

**Note for future:** If the model fails to generalize to 2D cultures or other formats, investigate whether it learned to depend on well-edge features. Consider training with augmentations that mask or vary the ring appearance.

### 9.4 Implementation

* Use `albumentations` with `ReplayCompose` for reproducibility
* Ensure identical transforms applied to image and mask
* Seed control for reproducible sampling during debugging

**Module:** `src/data/dataset.py`

**Classes:**

* `PatchDataset(image_paths, mask_paths, patch_size, positive_ratio, transforms, seed)`

**Config:** `configs/dataset.yaml`

```yaml
patch_size: 256  # adjust based on measured diameter
positive_ratio: 0.7
patches_per_image: 20
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotate_90: 0.5
  brightness_limit: 0.1
  contrast_limit: 0.1
```

**Deliverables:**

* `src/data/dataset.py` with `PatchDataset`
* `configs/dataset.yaml`
* Unit tests verifying patch shapes and coverage properties

---

## 10. Phase 3 — Model training on patches

**Goal:** Train a standard semantic segmentation model using established architectures.

### 10.1 Model selection

Use `segmentation-models-pytorch` (SMP) with:

* **Architecture:** U-Net
* **Encoder:** ResNet18 (lightweight, sufficient for POC) or EfficientNet-B0
* **Pretrained:** ImageNet weights for encoder
* **Input channels:** 1 (grayscale) — requires adapter layer or channel replication

### 10.2 Loss function

Combined loss for sparse foreground:

```
loss = 0.5 × BCE + 0.5 × DiceLoss
```

Both averaged over the batch.

### 10.3 Metrics (validation)

* **Dice coefficient** (primary)
* **IoU / Jaccard index**
* **Precision and recall** at fixed threshold (0.5)

Computed on validation patches each epoch.

### 10.4 Training configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Optimizer | AdamW | |
| Learning rate | 1e-4 | |
| LR scheduler | ReduceLROnPlateau | patience=5, factor=0.5 |
| Batch size | 16 | adjust for GPU memory |
| Epochs | 100 | early stopping patience=15 |
| Mixed precision | Yes (Linux) | AMP with GradScaler |

**Checkpointing:**

* Save best model by validation Dice
* Save checkpoint every 10 epochs
* Save final model

### 10.5 Platform notes

* **Linux GPU:** Full training with mixed precision
* **Mac M1:** CPU or MPS backend for smoke tests; expect 10–20× slower

### 10.6 Implementation

**Module:** `src/training/train.py`

**CLI:** `train --config configs/train.yaml`

**Config:** `configs/train.yaml`

```yaml
model:
  architecture: unet
  encoder: resnet18
  pretrained: true
  in_channels: 1
  classes: 1

training:
  optimizer: adamw
  learning_rate: 1.0e-4
  batch_size: 16
  epochs: 100
  early_stopping_patience: 15
  mixed_precision: true

loss:
  bce_weight: 0.5
  dice_weight: 0.5

data:
  train_manifest: data/splits/train.csv
  val_manifest: data/splits/val.csv
  patch_size: 256
  patches_per_image: 20

output:
  run_dir: runs/
  checkpoint_interval: 10
```

**Deliverables:**

* `runs/<run_id>/` containing:
  * `config.yaml` (snapshot)
  * `checkpoints/` (best, periodic, final)
  * `tensorboard/` logs
  * `val_metrics.csv`

---

## 11. Phase 4 — Full-image segmentation via tiled inference

**Goal:** Apply the patch-trained model to full-resolution images.

### 11.1 Tiled inference method

1. Slide a window across the image with specified overlap
2. Predict probability map for each tile
3. Accumulate predictions in overlap regions (average probabilities)
4. Threshold averaged probabilities to binary mask
5. Apply post-processing

### 11.2 Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Tile size | Same as training patch size | Required for consistent receptive field |
| Overlap | 25% | Sufficient for round objects |
| Threshold | 0.5 | Calibrate on validation set if needed |

### 11.3 Post-processing

1. **Remove small objects:** Delete connected components below minimum area (e.g., 100 px²)
2. **Fill holes:** Binary fill within each object
3. **Optional morphological smoothing:** Small opening/closing to clean edges

### 11.4 Outputs

For each input image `<basename>.tif`:

* `inference/<basename>_pred_mask.tif` — binary mask (0/255)
* `inference/<basename>_pred_prob.tif` — probability map (float32 or 16-bit)

### 11.5 Full-image evaluation

Compare predicted masks to GT masks on test set:

* Pixel-level Dice and IoU
* Summary statistics across test set

### 11.6 Implementation

**Module:** `src/inference/predict.py`

**CLI:** `predict_full --config configs/predict.yaml --checkpoint runs/<run_id>/checkpoints/best.pth`

**Config:** `configs/predict.yaml`

```yaml
inference:
  tile_size: 256
  overlap: 0.25
  threshold: 0.5
  
postprocess:
  min_object_area: 100
  fill_holes: true
  
input:
  manifest: data/splits/test.csv
  
output:
  output_dir: inference/
  save_probabilities: true
```

**Deliverables:**

* `inference/` with predicted masks and probability maps
* `inference/pixel_metrics.csv` — per-image Dice/IoU
* Console summary of test set performance

---

## 12. Phase 5 — Object identification and quantification

**Goal:** Extract individual spheroid objects, match predictions to ground truth, and compute morphology metrics.

### 12.1 Connected components extraction

For each predicted mask:

1. Run connected components labeling
2. Filter by minimum area (remove debris, use same threshold as post-processing)
3. Extract region properties via `skimage.measure.regionprops`

### 12.2 Morphology metrics

| Metric | Description | Units |
|--------|-------------|-------|
| `area` | Object area | pixels (convert to µm² if calibration known) |
| `perimeter` | Boundary length | pixels |
| `equivalent_diameter` | Diameter of circle with same area | pixels |
| `major_axis_length` | Major axis of fitted ellipse | pixels |
| `minor_axis_length` | Minor axis of fitted ellipse | pixels |
| `eccentricity` | Ellipse eccentricity (0=circle, 1=line) | dimensionless |
| `circularity` | 4πA/P² (1=perfect circle) | dimensionless |
| `centroid_x`, `centroid_y` | Object center | pixels |
| `bbox` | Bounding box coordinates | pixels |

### 12.3 Instance-level evaluation

Match predicted objects to GT objects to assess detection quality separately from segmentation quality.

**Matching procedure:**

1. Extract objects from both predicted and GT masks
2. Compute IoU matrix: `iou[i,j] = IoU(pred_object_i, gt_object_j)`
3. Apply Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to find optimal assignment maximizing total IoU
4. Apply threshold: matches with IoU < 0.5 are rejected

**Instance metrics:**

| Metric | Definition |
|--------|------------|
| True Positives (TP) | Predicted objects matched to GT with IoU ≥ 0.5 |
| False Positives (FP) | Predicted objects with no valid match (hallucinated) |
| False Negatives (FN) | GT objects with no valid match (missed) |
| Detection Precision | TP / (TP + FP) |
| Detection Recall | TP / (TP + FN) |
| Detection F1 | Harmonic mean of precision and recall |
| Mean Matched IoU | Average IoU of true positive matches (segmentation quality) |

This decomposition separates "did we find the right objects?" (detection) from "how accurate are the boundaries?" (segmentation).

### 12.4 Outputs

**Per-image object tables:**

* `metrics/per_image/<basename>_objects.csv` — one row per detected object with all morphology metrics

**Aggregated tables:**

* `metrics/all_objects.csv` — concatenated object table with `image_basename` column
* `metrics/instance_eval.csv` — per-image TP/FP/FN counts and detection metrics
* `metrics/summary.csv` — dataset-level summary statistics

**Visualizations:**

* Histograms of area, equivalent diameter, circularity
* Scatter plot of predicted vs GT object counts per image
* Detection precision/recall summary

### 12.5 Physical units (optional)

If pixel size is known (from microscope metadata):

* Store `pixel_size_um` in config
* Compute area_um2, diameter_um, etc.
* Include units in CSV headers

### 12.6 Implementation

**Module:** `src/analysis/quantify.py`

**CLI:** `quantify_objects --config configs/quantify.yaml`

**Config:** `configs/quantify.yaml`

```yaml
input:
  pred_mask_dir: inference/
  gt_manifest: data/splits/test.csv

analysis:
  min_object_area: 100
  iou_match_threshold: 0.5

calibration:
  pixel_size_um: null  # set if known, e.g., 0.65

output:
  output_dir: metrics/
  generate_plots: true
```

**Deliverables:**

* `metrics/per_image/` — per-image object CSVs
* `metrics/all_objects.csv`
* `metrics/instance_eval.csv`
* `metrics/summary.csv`
* `metrics/plots/` — histogram and scatter plot PNGs

---

## 13. Interfaces summary

### 13.1 CLI commands

| Command | Purpose |
|---------|---------|
| `validate_dataset` | Check image/mask pairing, compute QC metrics |
| `make_splits` | Generate train/val/test manifests |
| `sanity_check` | Quick pipeline validation before full training |
| `train` | Train segmentation model |
| `predict_full` | Run tiled inference on full images |
| `quantify_objects` | Extract objects and compute metrics |

### 13.2 Configuration files

All in `configs/`:

* `dataset.yaml` — patch sampling and augmentation
* `sanity_check.yaml` — sanity check parameters
* `train.yaml` — model and training hyperparameters
* `predict.yaml` — inference parameters
* `quantify.yaml` — analysis parameters

---

## 14. Reproducibility and QA

### 14.1 Random seeds

* Fixed seed for dataset splits (default: 42)
* Fixed seed for training initialization
* Seed logged in config snapshots

### 14.2 Version control

* All code in Git
* Config snapshots saved with each run
* Dataset manifests versioned in `data/splits/`

### 14.3 Unit tests

Minimum coverage:

* Mask/image alignment validation
* Patch sampler output shapes
* Patch coverage distribution matches configured ratio
* Tiled inference reconstruction produces correct output shape
* Instance matching with known IoU matrix

### 14.4 Integration tests

* End-to-end pipeline on synthetic data (circles on noisy background)
* Sanity check phase as pipeline validation

---

## 15. Performance expectations

### 15.1 Development (MacBook Air M1)

* Dataset validation and splits: seconds
* Sanity check (5 epochs, 10% data): ~10–15 minutes
* Full training: not recommended (hours to days)
* Inference on single image: ~1–2 minutes

### 15.2 Production (Linux GPU workstation)

* Full training (100 epochs): ~1–2 hours (estimate, depends on dataset size)
* Inference on single image: ~5–15 seconds
* Full test set inference + quantification: minutes

### 15.3 POC success criteria

* Baseline model produces visually correct segmentations on held-out images
* Validation Dice > 0.8 (threshold may be adjusted based on data quality)
* Detection F1 > 0.9 on well-separated spheroid test set
* Pipeline runs end-to-end without manual intervention

---

## 16. Future considerations

### 16.1 Per-well localization

If accuracy on well-plate data is insufficient with the current approach, consider adding a preprocessing step:

* Detect well locations via Hough circles or template matching
* Crop individual wells before segmentation
* May improve accuracy by removing confounding context

This is intentionally deferred to keep the pipeline general-purpose.

### 16.2 Well-edge artifact sensitivity

The current training data has consistent dark rings at well edges. Monitor for:

* Model relying on well edges as segmentation cues
* Poor generalization to non-well-plate images

Mitigation options:

* Augmentations that mask or vary ring appearance
* Training on mixed data (wells + non-wells) when available
* Feature visualization to inspect learned representations

### 16.3 Extension to harder cases

Planned future development:

* **Touching objects:** Watershed post-processing or instance segmentation models
* **Diffuse plating:** Multi-class segmentation (spheroid vs diffuse vs background)
* **2D cultures:** Adapt patch size and augmentation for adherent cell morphology
* **3D organoids:** Extend to volumetric data (3D U-Net or slice-by-slice with context)

### 16.4 Active learning

As dataset grows, consider:

* Uncertainty-based sample selection for annotation
* Hard example mining during training

---

## 17. Open decisions log

| Decision | Status | Default | Notes |
|----------|--------|---------|-------|
| Augmentation library | Resolved | albumentations | |
| Model library | Resolved | segmentation-models-pytorch | |
| Logging framework | Resolved | TensorBoard | |
| Mask format | Resolved | 0/255 | Standardize in Phase 1 |
| Patch size | Pending measurement | 256 | Set to 2.5× measured diameter |
| Pixel calibration | Pending | None | Add if metadata available |

---

## Appendix A: Checklist for phase completion

### Phase 0
- [ ] Git repository initialized
- [ ] Virtual environment created
- [ ] Dependencies installed and pinned
- [ ] Package structure created
- [ ] SDD in docs/

### Phase 1
- [ ] All images have matching masks
- [ ] QC report generated
- [ ] Empty images confirmed or flagged
- [ ] Splits created with seed logged
- [ ] Manifests committed to repo

### Phase 1.5
- [ ] Sanity check training completed
- [ ] Loss decreased over epochs
- [ ] Visual overlays inspected
- [ ] No data pipeline bugs identified
- [ ] Go decision documented

### Phase 2
- [ ] PatchDataset implemented
- [ ] Patch size set based on measured diameter
- [ ] Augmentation config created
- [ ] Unit tests passing

### Phase 3
- [ ] Training completes without errors
- [ ] Validation Dice tracked
- [ ] Best checkpoint saved
- [ ] TensorBoard logs available

### Phase 4
- [ ] Tiled inference runs on test set
- [ ] Predicted masks saved
- [ ] Pixel-level metrics computed

### Phase 5
- [ ] Object extraction working
- [ ] Instance matching implemented
- [ ] Morphology metrics computed
- [ ] Summary statistics and plots generated
- [ ] Final evaluation documented
