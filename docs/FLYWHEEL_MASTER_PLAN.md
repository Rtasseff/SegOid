# SegOid Data Flywheel — Phases 6-10 Master Plan

**Version:** 1.0  
**Created:** 2025-12-27

## Overview

With the POC complete (Phases 0-5), we now build infrastructure to maximize learning from limited data and create a sustainable annotation workflow. The goal is a human-in-the-loop data flywheel that iteratively improves model performance.

## Strategic Goals

1. **Maximize current data:** Use all 6 labeled images effectively via cross-validation
2. **Scale inference:** Apply model to entire unlabeled image library
3. **Efficient review:** Visual interface to quickly identify prediction failures
4. **Accelerate annotation:** Use predictions as starting points for expert correction
5. **Iterate:** Expanded dataset → better model → better predictions → easier annotation

## Phase Summary

| Phase | Name | Input | Output |
|-------|------|-------|--------|
| 6 | Cross-Validation Infrastructure | 6 labeled images | 6 models, performance estimate |
| 7 | Batch Inference Pipeline | Unlabeled image directory | Predicted masks for all images |
| 8 | Visual Review Interface | Image/mask pairs | Flagged images needing review |
| 9 | Annotation Workflow | Flagged images + predictions | Corrected masks, expanded dataset |
| 10 | Iteration & Retraining | Expanded dataset | Improved model |

---

## Phase 6: Cross-Validation Infrastructure

### Goal
Implement leave-one-out cross-validation to get robust performance estimates and train models that have seen all available data.

### Key Deliverables
- `generate_cv_splits` command — create fold manifests from source data
- `run_cv` command — orchestrate training across all folds
- `aggregate_cv_results` — compute mean ± std metrics across folds
- Configurable via YAML (strategy, folds, base training config)

### Output Structure
```
runs/cv_<experiment_id>/
  cv_config.yaml
  folds/
    fold_0/
      train.csv, val.csv
      checkpoints/best_model.pth
      tensorboard/
    fold_1/
      ...
  results/
    fold_metrics.csv
    summary.csv
```

### Success Criteria
- All 6 folds train successfully
- Aggregated val Dice reported with confidence interval
- Each fold's model saved for potential ensemble use

---

## Phase 7: Batch Inference Pipeline

### Goal
Apply trained model(s) to an arbitrary directory of unlabeled images, producing predicted masks at scale.

### Key Deliverables
- `batch_predict` command — process entire directories
- Support for model selection: single best, specific fold, or ensemble
- Progress tracking for large batches
- Output organization matching input structure

### Input/Output
```
Input:                          Output:
unlabeled/                      predictions/
  batch_001/                      batch_001/
    img_001.tif                     img_001_pred_mask.tif
    img_002.tif                     img_001_pred_prob.tif
    ...                             img_002_pred_mask.tif
  batch_002/                        ...
    ...                           batch_002/
                                    ...
                                  batch_manifest.csv
```

### Configuration
```yaml
batch_inference:
  input_dir: /path/to/unlabeled/
  output_dir: /path/to/predictions/
  checkpoint: runs/cv_001/folds/fold_0/checkpoints/best_model.pth
  # or
  ensemble:
    strategy: average  # average, vote, best_confidence
    checkpoints:
      - runs/cv_001/folds/fold_0/checkpoints/best_model.pth
      - runs/cv_001/folds/fold_1/checkpoints/best_model.pth
      # ...
  
  inference:
    tile_size: 256
    overlap: 0.25
    threshold: 0.5
    min_object_area: 100
```

### Success Criteria
- Process 100+ images without manual intervention
- Output masks compatible with review interface
- Manifest tracking all processed images

---

## Phase 8: Visual Review Interface

### Goal
Create a streamlined interface for human reviewers to quickly assess prediction quality and flag failures.

### Key Deliverables
- `review_predictions` command — launch review interface
- Display modes:
  - Slideshow: image → mask → image → mask (configurable timing)
  - Side-by-side: image | mask overlay
  - Toggle: spacebar switches between image and mask
- Flagging mechanism: keyboard shortcut marks current image as "needs review"
- Session persistence: can pause and resume review
- Export flagged list for correction workflow

### Interface Concept
```
┌─────────────────────────────────────────────────────────────────┐
│  Review: batch_001/img_042.tif                    [023/156]     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                                                                 │
│                    [ Image or Mask Display ]                    │
│                                                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  [SPACE] Toggle  [F] Flag  [←/→] Prev/Next  [Q] Quit & Save    │
│  Status: ✓ Good (38)  ⚠ Flagged (4)  ○ Unreviewed (114)        │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration
```yaml
review:
  input_dir: predictions/batch_001/
  display_mode: slideshow  # slideshow, sidebyside, toggle
  slideshow_delay: 3.0     # seconds per frame
  auto_advance: true       # auto-advance after delay
  
output:
  session_file: review_sessions/batch_001_session.yaml
  flagged_list: review_sessions/batch_001_flagged.csv
```

### Output: Flagged List
```csv
image_path,mask_path,flag_reason,reviewer,timestamp
batch_001/img_042.tif,batch_001/img_042_pred_mask.tif,poor_boundary,user,2025-12-27T14:32:00
batch_001/img_089.tif,batch_001/img_089_pred_mask.tif,missed_spheroid,user,2025-12-27T14:35:12
```

### Success Criteria
- Review 100 images in <15 minutes
- Flagged images exported for correction
- Session state preserved if interrupted

---

## Phase 9: Annotation Workflow (Mask Correction)

### Goal
Establish a workflow for correcting predicted masks and integrating them as new training data.

### Key Deliverables
- `export_for_correction` command — prepare flagged images for external editing
- `import_corrections` command — validate and import corrected masks
- Documentation/SOP for correction in ImageJ/Fiji
- Training data registry tracking provenance

### Workflow
```
1. Export flagged images + predicted masks to correction workspace
   └── Creates: correction_workspace/
         img_042.tif
         img_042_pred_mask.tif  (starting point for correction)
         
2. Expert corrects masks in ImageJ/Fiji
   └── Saves: img_042_mask.tif (corrected)
   
3. Import corrected masks back into training data
   └── Validates: dimensions, format, naming
   └── Copies to: data/working/images/, data/working/masks/
   └── Updates: data manifest with new entries
```

### Export Structure
```
correction_workspace/
  batch_001_flagged/
    README.txt                    # Instructions for corrector
    images/
      img_042.tif
      img_089.tif
    predicted_masks/              # Starting points (editable)
      img_042_pred_mask.tif
      img_089_pred_mask.tif
    corrected_masks/              # Corrector saves here
      (empty, to be filled)
    manifest.csv                  # Tracks correction status
```

### Fiji/ImageJ Correction SOP (outline)
1. Open image and predicted mask as overlay
2. Use ROI tools to correct boundaries
3. Fill/delete regions as needed
4. Save as binary mask (0/255) with naming convention
5. Mark as complete in manifest

### Import Validation
- Dimension match (image and mask same H×W)
- Binary format (only 0 and 255 values)
- Naming convention followed
- No duplicate basenames in existing data

### Training Data Registry
```yaml
# data/registry.yaml
images:
  - basename: dECM_1_1
    source: original
    annotator: expert_A
    date: 2025-12-01
    
  - basename: img_042
    source: model_assisted  # was predicted, then corrected
    base_prediction: runs/cv_001/fold_2
    corrector: expert_B
    date: 2025-12-27
```

### Success Criteria
- Seamless round-trip: export → correct → import
- Provenance tracked for all training data
- New data integrates with existing manifest structure

---

## Phase 10: Iteration & Retraining

### Goal
Close the loop—retrain with expanded dataset and measure improvement.

### Process
1. Regenerate manifests including new training data
2. Re-run cross-validation (now with N+k images)
3. Compare metrics to previous iteration
4. Repeat Phases 7-10 as needed

### Tracking
```
iterations/
  iteration_001/
    data_snapshot.yaml    # Which images were in training set
    cv_results.yaml       # Performance metrics
    notes.md              # Observations, decisions
  iteration_002/
    ...
```

### Stopping Criteria (soft guidelines)
- Val Dice plateaus (improvement < 0.01 across iterations)
- Visual review shows <5% flagged images
- Sufficient performance for downstream analysis

---

## Implementation Order

| Order | Phase | Estimated Effort | Dependencies |
|-------|-------|------------------|--------------|
| 1 | Phase 6 (CV Infrastructure) | 1-2 sessions | POC complete |
| 2 | Phase 7 (Batch Inference) | 1 session | Phase 6 |
| 3 | Phase 8 (Review Interface) | 1-2 sessions | Phase 7 |
| 4 | Phase 9 (Annotation Workflow) | 1 session | Phase 8 |
| 5 | Phase 10 (Iteration) | Ongoing | Phase 9 |

---

## Configuration Philosophy

All phases follow the established pattern:
- **YAML configs** for parameters
- **Config snapshots** saved with outputs
- **CLI commands** for each operation
- **Manifests/registries** for data tracking

This ensures every experiment is reproducible and traceable.

---

## Open Questions

1. **Ensemble strategy:** Average probabilities vs. majority vote vs. best-confidence selection?
2. **Review interface technology:** Simple matplotlib/OpenCV window vs. web-based vs. napari?
3. **Fiji integration:** Macro for batch correction or manual one-by-one?
4. **Version control for training data:** Git LFS, DVC, or simple manifest versioning?

These will be resolved as each phase is implemented.
