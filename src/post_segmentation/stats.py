"""Robust statistics helpers for post-segmentation analysis."""

from typing import Dict, List

import numpy as np
import pandas as pd


def median_absolute_deviation(values: np.ndarray) -> float:
    """Median absolute deviation (MAD) of a 1D array.

    Returns 0.0 for empty or single-value inputs.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def add_flag_robust_outlier_mad(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_col: str = "area_px",
) -> pd.DataFrame:
    """Add a `flag_robust_outlier_mad` column: distance from group median in MAD units.

    Computed per group defined by ``group_cols``. The metric column defaults to
    ``area_px`` (always present in SegOid metrics). When a group has zero MAD,
    the flag is set to 0.0 for all rows in that group (no spread to compare to).

    Returns a copy of ``df`` with the new column appended.
    """
    df = df.copy()
    if metric_col not in df.columns:
        df["flag_robust_outlier_mad"] = np.nan
        return df

    flags = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        rows = df.loc[idx, metric_col].to_numpy(dtype=float)
        median = np.nanmedian(rows)
        mad = median_absolute_deviation(rows)
        if mad == 0.0:
            flags[df.index.get_indexer(idx)] = 0.0
        else:
            flags[df.index.get_indexer(idx)] = np.abs(rows - median) / mad
    df["flag_robust_outlier_mad"] = flags
    return df


def group_descriptive_stats(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Per-group descriptive stats: n, mean, median, IQR, MAD per metric.

    Returns one row per group, with columns
    ``[*group_cols, n, <metric>_mean, <metric>_median, <metric>_iqr, <metric>_mad]``.
    """
    if not group_cols:
        return pd.DataFrame()

    rows: List[Dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        for metric in metric_cols:
            if metric not in sub.columns:
                continue
            values = sub[metric].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_iqr"] = np.nan
                row[f"{metric}_mad"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            q75, q25 = np.percentile(values, [75, 25])
            row[f"{metric}_iqr"] = float(q75 - q25)
            row[f"{metric}_mad"] = median_absolute_deviation(values)
        rows.append(row)

    return pd.DataFrame(rows)
