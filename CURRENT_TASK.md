# Current Task

**Project:** SegOid (Spheroid Segmentation Pipeline)  
**Date:** 2025-12-27  
**Session:** Phase 6 — Cross-Validation Infrastructure

---

## Status

| Item | Value |
|------|-------|
| **Active Phase** | 6 |
| **Last Completed** | Phase 5 — POC complete (Detection F1: TBD from Phase 5 notes) |
| **Blocking Issues** | None |

---

## Context

**POC Results (Phases 0-5):**
- Val Dice: 0.799, Test Dice: 0.794
- Pipeline validated end-to-end
- Current limitation: trained on only 3 images, validated on 2, tested on 1

**New Goal:** Maximize learning from all 6 labeled images via leave-one-out cross-validation. This provides:
- More robust performance estimate (mean ± std across 6 folds)
- 6 trained models (each has seen 5/6 of the data)
- Foundation for ensemble inference in Phase 7

**Master Plan:** See `docs/FLYWHEEL_MASTER_PLAN.md` for Phases 6-10 overview.

---

## Session Goal

Implement cross-validation infrastructure: configurable split generation, training orchestration across folds, and result aggregation. All configuration-driven and traceable.

---

## Tasks

### 1. Implement CV split generation

- [x] Create `src/data/cross_validation.py` module
- [x] Implement `generate_loocv_splits(manifest, output_dir, seed)`:
  - Read source manifest (all 6 images)
  - Generate N fold directories, each with `train.csv` and `val.csv`
  - Leave-one-out: fold_i trains on all except image_i, validates on image_i
  - Save `cv_meta.yaml` with fold-to-image mapping
- [x] Implement `generate_kfold_splits(manifest, output_dir, n_folds, seed)` (for future flexibility)

### 2. Create CV configuration schema

- [x] Create `configs/cv_config.yaml`:
  ```yaml
  cv:
    strategy: leave_one_out  # or k_fold
    source_manifest: data/splits/all.csv
    seed: 42
  
  training:
    # Inherited from train.yaml or specified inline
    epochs: 50
    batch_size: 4
    learning_rate: 1.0e-4
    early_stopping_patience: 10
    patch_size: 256
    patches_per_image: 20
    positive_ratio: 0.7
  
  output:
    cv_dir: runs/cv_001/
  ```
- [x] Implement config loading with defaults and validation

### 3. Implement CV orchestrator

- [x] Implement `run_cross_validation(cv_config_path)`:
  - Load and validate config
  - Create CV output directory structure
  - Save config snapshot
  - Generate fold splits
  - Loop through folds:
    - Generate fold-specific train config
    - Call existing `train_model()` function
    - Collect results (best val Dice, best epoch, etc.)
  - Aggregate results across folds
  - Save summary statistics

### 4. Implement result aggregation

- [x] Compute per-fold metrics:
  - best_val_dice, best_epoch, final_train_dice, training_time
- [x] Compute aggregate statistics:
  - mean, std, min, max for val Dice
  - identify best and worst performing folds
- [x] Save outputs:
  - `results/fold_metrics.csv` — one row per fold
  - `results/summary.yaml` — aggregate statistics

### 5. Implement `run_cv` CLI command

- [x] Add to `src/cli.py`
- [x] Parameters:
  - `--config` (required): path to CV config YAML
  - `--folds` (optional): specific folds to run (e.g., "0,2,5" for subset)
  - `--resume` (optional): resume interrupted CV run (deferred - not critical for Phase 6)
- [x] Progress output showing fold completion

### 6. Run full CV experiment

- [x] Integration test: 2-fold subset completed successfully (verified pipeline works end-to-end)
- [ ] Execute full 6-fold LOO CV on current dataset (in progress - running in background)
- [ ] Document results in Notes section

### 7. Unit tests

- [x] Test LOOCV split generation (correct train/val sizes per fold)
- [x] Test k-fold split generation
- [x] Test config loading and validation
- [x] Test result aggregation computation
- [x] All 17 tests passing

---

## Reference Sections

- **FLYWHEEL_MASTER_PLAN.md:** Phase 6 specification
- **docs/SDD.md Section 10:** Training configuration (reused for each fold)

---

## Files to Create/Modify

| File | Action | Notes |
|------|--------|-------|
| `src/data/cross_validation.py` | Create | Split generation functions |
| `src/training/cross_validation.py` | Create | CV orchestration and aggregation |
| `src/cli.py` | Modify | Add `run_cv` command |
| `configs/cv_config.yaml` | Create | CV configuration template |
| `tests/test_cross_validation.py` | Create | Unit tests |
| `docs/FLYWHEEL_MASTER_PLAN.md` | Create | Master plan for Phases 6-10 |

---

## Technical Details

### Output Directory Structure

```
runs/cv_001/
  cv_config.yaml              # Config snapshot
  cv_meta.yaml                # Fold-to-image mapping
  folds/
    fold_0/
      train.csv               # 5 images
      val.csv                 # 1 image (e.g., dECM_1_1)
      config.yaml             # Fold-specific train config
      checkpoints/
        best_model.pth
        final_model.pth
      tensorboard/
    fold_1/
      ...
    fold_5/
      ...
  results/
    fold_metrics.csv
    summary.yaml
```

### Fold Metrics CSV Schema

```csv
fold,val_image,best_val_dice,best_val_iou,best_epoch,final_train_dice,training_time_min
0,dECM_1_1,0.812,0.684,28,0.845,35.2
1,dECM_1_2,0.778,0.637,32,0.831,38.1
...
```

### Summary YAML Schema

```yaml
experiment_id: cv_001
timestamp: 2025-12-27T10:30:00
n_folds: 6
strategy: leave_one_out

aggregate_metrics:
  val_dice:
    mean: 0.795
    std: 0.024
    min: 0.762
    max: 0.823
    ci_95: [0.771, 0.819]  # 95% confidence interval
  
  val_iou:
    mean: 0.661
    std: 0.031
    ...

per_fold_summary:
  best_fold: 3
  worst_fold: 1
  
total_training_time_min: 215.4
```

### Reusing Existing Training Code

The key insight: `train_model()` from Phase 3 already does everything we need. The CV orchestrator just:
1. Generates different manifest CSVs per fold
2. Calls `train_model()` with fold-specific config
3. Collects the returned results

```python
# Pseudocode for orchestration loop
for fold_idx, (train_csv, val_csv) in enumerate(fold_paths):
    fold_config = create_fold_config(
        base_config=cv_config["training"],
        train_manifest=train_csv,
        val_manifest=val_csv,
        output_dir=cv_dir / "folds" / f"fold_{fold_idx}"
    )
    
    # This is the existing function from Phase 3!
    results = train_model(fold_config)
    
    fold_results.append(results)
```

---

## What NOT to Do This Session

- Do not modify the core `train_model()` function (reuse as-is)
- Do not implement ensemble inference (that's Phase 7)
- Do not implement batch prediction (that's Phase 7)
- Do not implement the review interface (that's Phase 8)
- Do not optimize for parallel training across folds (sequential is fine)

---

## Completion Criteria

This session is complete when:

1. `run_cv --config configs/cv_config.yaml` executes all 6 folds
2. Each fold produces a trained model in `runs/cv_001/folds/fold_N/`
3. `fold_metrics.csv` contains per-fold performance
4. `summary.yaml` contains aggregated statistics with mean ± std
5. CV experiment completes without manual intervention
6. Unit tests pass

---

## Expected Results

**Training time estimate:**
- ~35-40 min per fold on M1
- 6 folds × 40 min = ~4 hours total
- Can run overnight or in background

**Expected performance:**
- Mean val Dice: 0.75-0.85 (should be similar to POC, possibly better with more training data per fold)
- Std val Dice: 0.02-0.05 (indicates consistency across images)
- If one fold has much lower Dice, that image may have unusual characteristics

**What the results tell us:**
- High mean, low std → model generalizes well, ready for production
- High mean, high std → some images are harder, may need more data or investigation
- Low mean → model or data issues to debug

---

## Notes / Decisions Log

_Update during session:_

**Phase 6 Implementation Complete (2025-12-27)**

**Files Created:**
- `src/data/cross_validation.py` - LOOCV and k-fold split generation (164 lines)
- `src/training/cross_validation.py` - CV orchestration and result aggregation (334 lines)
- `configs/cv_config.yaml` - Production CV configuration
- `configs/cv_test_quick.yaml` - Quick test configuration for verification
- `tests/test_cross_validation.py` - 17 unit tests (all passing)

**CLI Command Added:**
- `run_cv --config <path> [--folds 0,1,2]` - registered in pyproject.toml

**Integration Test Results (2-fold subset with minimal config):**
- Pipeline executed successfully end-to-end
- Directory structure created correctly
- Results properly aggregated in fold_metrics.csv and summary.yaml
- Total time: ~0.19 minutes for 2 folds with 2 epochs, 5 patches/image
- Verified: split generation, config loading, training loop, checkpointing, result aggregation

**Design Decisions:**
- Reused existing `train_model()` function - no modifications needed to core training code
- CV orchestrator creates fold-specific configs and data loaders, then calls `train_model()`
- Supports both LOOCV and k-fold strategies via config
- `--folds` parameter allows running subset of folds for debugging
- scikit-learn added as dependency for k-fold splitting
- Resume functionality deferred (not critical for Phase 6)

**Test Coverage:**
- Split generation: 12 tests (LOOCV and k-fold correctness, overlap checking, metadata)
- CV orchestration: 5 tests (config loading, fold config creation, result aggregation)
- All 17 tests passing

**Next Steps:**
- Full 6-fold LOOCV CV execution (estimated ~4 hours, 50 epochs with early stopping)
- Results will provide robust performance estimate with mean ± std across all 6 images 

---

## Next Session Preview

**Phase 7 (Batch Inference):** Apply trained model(s) to directory of unlabeled images. Support single model or ensemble of CV folds. Output predicted masks for entire image library.
