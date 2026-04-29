# Contributing Training Data (TODO — Stub)

**Status:** Not yet implemented. For now, **email the maintainer** corrected (image, mask) pairs.

This file is a placeholder. It exists so the work isn't forgotten.

## Background

SegOid users have already been provided a Fiji/ImageJ protocol for producing initial labeled masks when training the original model. **The need this stub addresses is different.**

Once SegOid has been trained and is being used for inference, predictions will sometimes be wrong. The contribution flow we want to enable is:

1. **Identify problem cases** after running SegOid:
   - Visually wrong masks spotted in the review video output of `review_predictions`.
   - Odd values in the metrics CSV (low circularity, area outside expected range, outlier MAD score), then visually confirmed.
2. **Correct the mask** by editing the *predicted* mask (not labeling from scratch). Starting from the predicted `_mask.tif` is faster than redrawing in Fiji.
3. **Send the (image, corrected_mask) pair back** so it can be folded into a future training round.

## Open questions

- Walkthrough doc only, a Fiji macro, or both?
- Standard layout for a corrected-pair contribution (manifest fragment, overlay sanity check, naming convention, optional metadata about *what* was wrong).
- Sharing mechanism. Default plan: existing institutional cloud (Dropbox / OneDrive / Google Drive) — whatever the user already uses with the maintainer. We do not plan to build a sharing tool.

## TODO

- [ ] Decide doc / macro / both for the correction step.
- [ ] Build a `src/contrib/` (or `data_flywheel/`) validator that ingests (image, corrected_mask) pairs, checks dimensions / binarization, generates an overlay PNG for sanity check, and emits a manifest fragment ready to merge into `data/splits/all.csv`. Reuse `src/data/validate.py`.
- [ ] Document the end-to-end protocol: "spot a wrong mask" → "edit it" → "package it" → "submit to maintainer".
