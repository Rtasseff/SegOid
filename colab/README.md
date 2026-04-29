# Colab Notebooks for SegOid

Browser-based analysis tools for SegOid users who don't want to install Python locally. Each notebook is a thin shell over the canonical Python module in `src/` — analytical logic lives in the package; the notebook is the UI.

## Post-Segmentation Analysis

Group, filter, and export a SegOid metrics CSV to a multi-tab Excel ready for inspection or GraphPad Prism import.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rtasseff/SegOid/blob/main/colab/post_segmentation.ipynb)

**Before running this notebook:** run SegOid with the **"Parse filename into fields"** option enabled in the GUI so the metrics CSV has the metadata columns (e.g., `condition`, `parameters`) you want to group by. The notebook does not re-parse filenames — it consumes the columns produced by SegOid.

**Inputs:** the `metrics.csv` (or similar) produced by SegOid.

**Outputs:** a multi-tab `.xlsx` with:
- One tab per group (filtered rows only).
- `all_with_filter_status`: every row, with `passed_filter` and `flag_robust_outlier_mad` columns visible for auditing.
- `group_stats`: per-group descriptive statistics (n, mean, median, IQR, MAD).
- `graphpad_long`: long-format table for GraphPad Prism import.

**Privacy note:** uploading the metrics CSV uses Google's servers, the same as any other cloud platform. The CSV contains numeric metrics and filenames only — no images. Check with your institution before uploading if your data has restrictions on use of cloud platforms.

## CLI alternative

The same analysis runs locally via `post_seg_analyze` if you have Python and SegOid installed:

```bash
post_seg_analyze \
  --metrics path/to/metrics.csv \
  --output post_seg.xlsx \
  --group-by condition,parameters \
  --circularity-min 0.5 \
  --area-min-px 10000 \
  --area-max-px 50000
```

Use `none` (or `null`/`na`/`nan`/empty) to disable a filter.
