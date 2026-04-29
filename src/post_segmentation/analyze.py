"""Core post-segmentation analysis pipeline.

Pipeline: load metrics CSV → validate columns → flag outliers → apply filters →
group → write multi-tab Excel.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.post_segmentation.config import PostSegConfig
from src.post_segmentation.excel_writer import write_multitab_excel
from src.post_segmentation.stats import (
    add_flag_robust_outlier_mad,
    group_descriptive_stats,
)


def _validate_columns(df: pd.DataFrame, group_by: List[str]) -> None:
    """Raise if any grouping column is missing."""
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(
            f"Grouping columns not found in metrics CSV: {missing}. "
            "If these are filename-derived fields, enable "
            "'Parse filename into fields' in the SegOid GUI when running "
            "inference and re-export the metrics CSV. "
            f"Available columns: {list(df.columns)}"
        )


def _apply_filters(df: pd.DataFrame, config: PostSegConfig) -> pd.DataFrame:
    """Add ``passed_filter`` column reflecting all configured filters.

    Filters with value None are bypassed. Returns a copy of ``df`` with the
    new column. Does not drop any rows.
    """
    df = df.copy()
    passed = np.ones(len(df), dtype=bool)

    if config.circularity_min is not None and "circularity" in df.columns:
        passed &= df["circularity"].to_numpy() >= config.circularity_min
    if config.area_min_px is not None and "area_px" in df.columns:
        passed &= df["area_px"].to_numpy() >= config.area_min_px
    if config.area_max_px is not None and "area_px" in df.columns:
        passed &= df["area_px"].to_numpy() <= config.area_max_px
    if config.area_min_um2 is not None and "area_um2" in df.columns:
        passed &= df["area_um2"].to_numpy() >= config.area_min_um2
    if config.area_max_um2 is not None and "area_um2" in df.columns:
        passed &= df["area_um2"].to_numpy() <= config.area_max_um2
    if config.mad_max is not None and "flag_robust_outlier_mad" in df.columns:
        passed &= df["flag_robust_outlier_mad"].to_numpy() <= config.mad_max

    df["passed_filter"] = passed
    return df


def _ordered_columns(df: pd.DataFrame, group_by: List[str]) -> List[str]:
    """Return columns in the canonical post-seg display order."""
    head = ["object_id", "image"]
    head_present = [c for c in head if c in df.columns]
    group_present = [c for c in group_by if c in df.columns and c not in head_present]
    morph = [c for c in ["circularity", "area_um2", "equivalent_diameter_um"] if c in df.columns]
    intensity = sorted(c for c in df.columns if c.startswith("mean_intensity_"))
    seen = set(head_present + group_present + morph + intensity)
    rest = [c for c in df.columns if c not in seen]
    return head_present + group_present + morph + intensity + rest


def _per_group_tabs(
    df: pd.DataFrame, group_by: List[str], col_order: List[str]
) -> Dict[str, pd.DataFrame]:
    """Split filtered (passed_filter=True) rows into per-group DataFrames.

    Returns dict mapping group label → DataFrame in canonical column order.
    """
    out: Dict[str, pd.DataFrame] = {}
    filtered = df[df["passed_filter"]].copy() if "passed_filter" in df.columns else df.copy()
    for keys, sub in filtered.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = "_".join("NA" if pd.isna(k) else str(k) for k in keys)
        if not label:
            label = "all"
        cols = [c for c in col_order if c in sub.columns]
        out[label] = sub[cols].reset_index(drop=True)
    return out


def _graphpad_long(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """Long-format table for GraphPad Prism import.

    One row per object that passed the filter, columns = group identifier +
    numeric metrics. Group identifier is a single string formed by joining
    grouping columns with ``_`` (matches per-tab labels).
    """
    if "passed_filter" in df.columns:
        sub = df[df["passed_filter"]].copy()
    else:
        sub = df.copy()
    if sub.empty:
        return sub

    def _label(row):
        parts = ["NA" if pd.isna(row[c]) else str(row[c]) for c in group_by]
        return "_".join(parts) if parts else "all"

    sub.insert(0, "group", sub.apply(_label, axis=1))

    excluded = {"group", "object_id", "passed_filter", "flag_robust_outlier_mad"}
    numeric_cols = [
        c
        for c in sub.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(sub[c])
    ]
    keep = ["group", "object_id", "image"] + numeric_cols
    keep = [c for c in keep if c in sub.columns]
    return sub[keep].reset_index(drop=True)


def run(config: PostSegConfig) -> Path:
    """Execute the post-segmentation analysis described by ``config``.

    Returns the path of the written Excel file.
    """
    df = pd.read_csv(config.metrics_csv)
    _validate_columns(df, config.group_by)

    df = add_flag_robust_outlier_mad(df, config.group_by, metric_col="area_px")
    df = _apply_filters(df, config)

    col_order = _ordered_columns(df, config.group_by)

    metric_cols = [
        c
        for c in ["circularity", "area_px", "area_um2", "equivalent_diameter_um"]
        if c in df.columns
    ]
    metric_cols += sorted(c for c in df.columns if c.startswith("mean_intensity_"))

    sheets: Dict[str, pd.DataFrame] = {}
    sheets.update(_per_group_tabs(df, config.group_by, col_order))
    sheets["all_with_filter_status"] = df[col_order]
    sheets["group_stats"] = group_descriptive_stats(df, config.group_by, metric_cols)
    sheets["graphpad_long"] = _graphpad_long(df, config.group_by)

    write_multitab_excel(config.output_xlsx, sheets)
    return config.output_xlsx
