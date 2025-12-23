# Current Task

**Project:** SegOid (Spheroid Segmentation Pipeline)  
**Date:** 2025-01-15  
**Session:** Phase 1 — Image Curation and Dataset Splits

---

## Status

| Item | Value |
|------|-------|
| **Active Phase** | 1 |
| **Last Completed** | Phase 0 — Package structure, CLI placeholders |
| **Blocking Issues** | None |

---

## Session Goal

Implement dataset validation and split generation: verify image/mask pairing, compute QC metrics, handle empty images, and produce train/val/test manifests.

---

## Tasks

### 1. Implement `validate_dataset` command

- [ ] Create `src/data/validate.py` module
- [ ] For each image in `working/images/`:
  - [ ] Check corresponding mask exists (`<basename>_mask.tif`)
  - [ ] Verify dimensions match (H, W)
  - [ ] Verify mask is binary (0/255) or threshold if needed
  - [ ] Compute `mask_coverage` (foreground pixels / total)
  - [ ] Compute `object_count` via connected components
- [ ] Generate `splits/all.csv` manifest with columns:
  - `basename`, `image_path`, `mask_path`, `mask_coverage`, `object_count`, `empty_confirmed`
- [ ] Generate `splits/qc_report.csv` with validation status and warnings
- [ ] Print console summary: total images, passed/failed, coverage distribution
- [ ] Wire up CLI command in `src/cli.py`

### 2. Implement empty image handling

- [ ] Flag images where `mask_coverage == 0` and `empty_confirmed` is not set
- [ ] QC report should clearly list flagged images for manual review
- [ ] Provide mechanism to mark images as confirmed empty (could be manual CSV edit for POC)

### 3. Implement `make_splits` command

- [ ] Create split logic in `src/data/validate.py` or separate module
- [ ] Split by image (not patches)
- [ ] Default ratio: 70% train, 15% val, 15% test
- [ ] Use deterministic random seed (default: 42)
- [ ] Optional: stratify by `mask_coverage` buckets
- [ ] Output: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- [ ] Wire up CLI command

### 4. Compute spheroid diameter for patch size

- [ ] From masks, estimate average spheroid diameter (e.g., mean equivalent diameter from regionprops)
- [ ] Report recommended patch size (2.5× diameter, rounded to nearest 256/512)
- [ ] Add to QC summary output

### 5. Unit tests

- [ ] Test image/mask pairing detection (valid and missing cases)
- [ ] Test dimension mismatch detection
- [ ] Test mask coverage calculation
- [ ] Test split ratios are approximately correct
- [ ] Test deterministic seed produces identical splits

---

## Reference Sections (in docs/SDD.md)

- **Section 3.2:** Naming convention (critical for pairing logic)
- **Section 3.3:** Data manifest schema (required columns)
- **Section 7:** Full Phase 1 specification (validation, QC, splits)
- **Section 9.1:** Patch size formula (2.5× diameter rationale)

---

## Files to Create/Modify

| File | Action | Notes |
|------|--------|-------|
| `src/data/validate.py` | Create | Main validation and split logic |
| `src/cli.py` | Modify | Add `validate_dataset` and `make_splits` commands |
| `tests/test_validate.py` | Create | Unit tests for validation |
| `data/splits/` | Create dir | Output location (add to .gitignore except schema) |

---

## What NOT to Do This Session

- Do not implement `PatchDataset` (that's Phase 2/3)
- Do not implement training loop
- Do not create config YAML files yet (validation uses CLI args for now)
- Do not handle multi-channel images (assume grayscale)
- Do not implement pixel-to-micron conversion (optional future feature)

---

## Completion Criteria

This session is complete when:

1. `validate_dataset --input-dir data/working/ --output-dir data/splits/` runs successfully
2. `splits/all.csv` contains all images with required columns
3. `splits/qc_report.csv` identifies any validation failures or flagged empties
4. `make_splits` produces `train.csv`, `val.csv`, `test.csv` with no overlap
5. Console output shows coverage distribution and recommended patch size
6. Unit tests pass: `pytest tests/test_validate.py`

---

## Notes / Decisions Log

_Update during session:_

- 

---

## Next Session Preview

**Phase 1.5 (Sanity Check):** Quick training run (5 epochs, 10% data) to validate the full pipeline before committing to real training. Requires `PatchDataset` implementation.