"""
Object quantification and instance-level evaluation for segmentation masks.

This module provides functions to extract individual objects from binary masks,
compute morphology metrics, match predicted objects to ground truth objects,
and evaluate instance-level detection performance.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage import measure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def extract_objects(mask: np.ndarray, min_area: int = 100) -> Tuple[np.ndarray, int]:
    """
    Extract individual objects from binary mask using connected components.

    Args:
        mask: Binary mask (H, W) with values 0 (background) and 255 (foreground)
        min_area: Minimum object area in pixels (objects smaller than this are discarded)

    Returns:
        labeled_mask: Integer array (H, W) with unique label per object (0 = background)
        object_count: Number of objects found after filtering
    """
    # Ensure binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Run connected components
    labeled = measure.label(binary_mask, connectivity=2)

    # Filter by area
    props = measure.regionprops(labeled)
    filtered_labels = np.zeros_like(labeled)
    new_label = 1

    for prop in props:
        if prop.area >= min_area:
            filtered_labels[labeled == prop.label] = new_label
            new_label += 1

    object_count = new_label - 1

    logger.debug(f"Extracted {object_count} objects (min_area={min_area})")

    return filtered_labels, object_count


def compute_object_properties(
    labeled_mask: np.ndarray,
    pixel_size: Optional[float] = None
) -> pd.DataFrame:
    """
    Compute morphology metrics for each object in labeled mask.

    Args:
        labeled_mask: Integer array (H, W) with unique label per object (0 = background)
        pixel_size: Optional pixel size in micrometers for physical unit conversion

    Returns:
        DataFrame with one row per object containing:
            - object_id: unique identifier (matches label in labeled_mask)
            - area: pixel count (or µm² if pixel_size provided)
            - perimeter: boundary length (px or µm)
            - equivalent_diameter: diameter of equal-area circle (px or µm)
            - major_axis_length: fitted ellipse major axis (px or µm)
            - minor_axis_length: fitted ellipse minor axis (px or µm)
            - eccentricity: ellipse eccentricity (0=circle, 1=line)
            - circularity: 4πA/P² (1=perfect circle)
            - centroid_x, centroid_y: object center (px)
            - bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col: bounding box (px)
    """
    props = measure.regionprops(labeled_mask)

    if len(props) == 0:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'object_id', 'area', 'perimeter', 'equivalent_diameter',
            'major_axis_length', 'minor_axis_length', 'eccentricity',
            'circularity', 'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col'
        ])

    records = []

    for prop in props:
        # Compute circularity: 4πA/P²
        if prop.perimeter > 0:
            circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        else:
            circularity = 0.0

        # Base metrics (in pixels)
        area = prop.area
        perimeter = prop.perimeter
        equiv_diameter = prop.equivalent_diameter_area

        # Use new property names (axis_major_length/axis_minor_length in scikit-image 0.26+)
        # Try new API first, fallback to deprecated for older versions
        try:
            major_axis = prop.axis_major_length
            minor_axis = prop.axis_minor_length
        except AttributeError:
            major_axis = prop.major_axis_length
            minor_axis = prop.minor_axis_length

        # Convert to physical units if pixel size provided
        if pixel_size is not None:
            area = area * (pixel_size ** 2)
            perimeter = perimeter * pixel_size
            equiv_diameter = equiv_diameter * pixel_size
            major_axis = major_axis * pixel_size
            minor_axis = minor_axis * pixel_size

        record = {
            'object_id': prop.label,
            'area': area,
            'perimeter': perimeter,
            'equivalent_diameter': equiv_diameter,
            'major_axis_length': major_axis,
            'minor_axis_length': minor_axis,
            'eccentricity': prop.eccentricity,
            'circularity': circularity,
            'centroid_x': prop.centroid[1],  # Column (X coordinate)
            'centroid_y': prop.centroid[0],  # Row (Y coordinate)
            'bbox_min_row': prop.bbox[0],
            'bbox_min_col': prop.bbox[1],
            'bbox_max_row': prop.bbox[2],
            'bbox_max_col': prop.bbox[3],
        }

        records.append(record)

    df = pd.DataFrame(records)

    logger.debug(f"Computed properties for {len(df)} objects")

    return df


def compute_iou_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between all predicted and ground truth objects.

    Args:
        pred_labels: Predicted labeled mask (H, W)
        gt_labels: Ground truth labeled mask (H, W)

    Returns:
        IoU matrix (N_pred, N_gt) where entry [i, j] is IoU between pred object i and gt object j
    """
    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids > 0]  # Exclude background

    gt_ids = np.unique(gt_labels)
    gt_ids = gt_ids[gt_ids > 0]  # Exclude background

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt))

    iou_matrix = np.zeros((n_pred, n_gt))

    for i, pred_id in enumerate(pred_ids):
        pred_mask = (pred_labels == pred_id)
        for j, gt_id in enumerate(gt_ids):
            gt_mask = (gt_labels == gt_id)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    return iou_matrix


def match_objects(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match predicted objects to ground truth objects using Hungarian algorithm.

    Args:
        pred_labels: Predicted labeled mask (H, W)
        gt_labels: Ground truth labeled mask (H, W)
        iou_threshold: Minimum IoU for a valid match (default: 0.5)

    Returns:
        matches: List of (pred_id, gt_id, iou) tuples for valid matches
        false_positives: List of predicted object IDs with no match
        false_negatives: List of GT object IDs with no match
    """
    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids > 0].tolist()

    gt_ids = np.unique(gt_labels)
    gt_ids = gt_ids[gt_ids > 0].tolist()

    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return [], [], []

    if len(pred_ids) == 0:
        return [], [], gt_ids

    if len(gt_ids) == 0:
        return [], pred_ids, []

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_labels, gt_labels)

    # Hungarian algorithm (maximize IoU = minimize negative IoU)
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter matches by IoU threshold
    matches = []
    matched_pred = set()
    matched_gt = set()

    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            matches.append((pred_ids[i], gt_ids[j], iou))
            matched_pred.add(pred_ids[i])
            matched_gt.add(gt_ids[j])

    # Identify unmatched objects
    false_positives = [pid for pid in pred_ids if pid not in matched_pred]
    false_negatives = [gid for gid in gt_ids if gid not in matched_gt]

    logger.debug(f"Matched {len(matches)} objects, {len(false_positives)} FP, {len(false_negatives)} FN")

    return matches, false_positives, false_negatives


def compute_instance_metrics(
    matches: List[Tuple[int, int, float]],
    false_positives: List[int],
    false_negatives: List[int]
) -> Dict[str, float]:
    """
    Compute instance-level evaluation metrics.

    Args:
        matches: List of (pred_id, gt_id, iou) tuples for valid matches
        false_positives: List of unmatched predicted object IDs
        false_negatives: List of unmatched GT object IDs

    Returns:
        Dictionary with metrics:
            - tp: true positives (count of valid matches)
            - fp: false positives (predicted objects with no match)
            - fn: false negatives (GT objects with no match)
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1: harmonic mean of precision and recall
            - mean_matched_iou: average IoU of true positive matches
    """
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)

    # Compute precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Mean matched IoU
    mean_iou = np.mean([iou for _, _, iou in matches]) if len(matches) > 0 else 0.0

    metrics = {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_matched_iou': mean_iou
    }

    return metrics


def process_image_pair(
    pred_mask_path: Path,
    gt_mask_path: Path,
    min_object_area: int = 100,
    iou_threshold: float = 0.5,
    pixel_size: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Process a single image pair: extract objects, match, and compute metrics.

    Args:
        pred_mask_path: Path to predicted mask TIFF
        gt_mask_path: Path to ground truth mask TIFF
        min_object_area: Minimum object area in pixels
        iou_threshold: Minimum IoU for valid match
        pixel_size: Optional pixel size in µm for physical units

    Returns:
        object_properties: DataFrame with per-object morphology from predictions
        instance_metrics: Dict with TP, FP, FN, precision, recall, F1, mean IoU
        pred_labels: Labeled predicted mask
        gt_labels: Labeled ground truth mask
    """
    import tifffile

    # Load masks with error handling
    try:
        pred_mask = tifffile.imread(pred_mask_path)
        gt_mask = tifffile.imread(gt_mask_path)
    except Exception as e:
        logger.error(f"Failed to load masks: {e}")
        raise ValueError(f"Could not load masks from {pred_mask_path} and {gt_mask_path}") from e

    # Validate shapes match
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted mask {pred_mask.shape} vs "
            f"ground truth mask {gt_mask.shape}"
        )

    # Extract objects
    pred_labels, n_pred = extract_objects(pred_mask, min_area=min_object_area)
    gt_labels, n_gt = extract_objects(gt_mask, min_area=min_object_area)

    # Compute morphology for predicted objects
    object_properties = compute_object_properties(pred_labels, pixel_size=pixel_size)

    # Match objects
    matches, fps, fns = match_objects(pred_labels, gt_labels, iou_threshold=iou_threshold)

    # Compute instance metrics
    instance_metrics = compute_instance_metrics(matches, fps, fns)
    instance_metrics['n_pred'] = n_pred
    instance_metrics['n_gt'] = n_gt

    return object_properties, instance_metrics, pred_labels, gt_labels


def create_summary_plots(
    all_objects_df: pd.DataFrame,
    instance_eval_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create summary visualization plots.

    Args:
        all_objects_df: DataFrame with all object morphology metrics
        instance_eval_df: DataFrame with per-image instance metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out empty results
    valid_objects = all_objects_df[all_objects_df['area'] > 0]

    if len(valid_objects) == 0:
        logger.warning("No valid objects found for plotting")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Histogram of spheroid areas
    ax = axes[0, 0]
    ax.hist(valid_objects['area'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Area (px²)' if 'area' in valid_objects.columns else 'Area')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Spheroid Areas')
    ax.grid(True, alpha=0.3)

    # 2. Histogram of equivalent diameters
    ax = axes[0, 1]
    ax.hist(valid_objects['equivalent_diameter'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Equivalent Diameter (px)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Equivalent Diameters')
    ax.grid(True, alpha=0.3)

    # 3. Histogram of circularity
    ax = axes[1, 0]
    ax.hist(valid_objects['circularity'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Circularity (1 = perfect circle)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Circularity')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Perfect circle')
    ax.legend()

    # 4. Scatter: predicted vs GT object count
    ax = axes[1, 1]
    ax.scatter(instance_eval_df['n_gt'], instance_eval_df['n_pred'], alpha=0.6, s=50)

    # Add diagonal line (perfect agreement)
    max_count = max(instance_eval_df['n_gt'].max(), instance_eval_df['n_pred'].max())
    ax.plot([0, max_count], [0, max_count], 'r--', linewidth=1, label='Perfect agreement')

    ax.set_xlabel('Ground Truth Object Count')
    ax.set_ylabel('Predicted Object Count')
    ax.set_title('Object Count: Predicted vs Ground Truth')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'summary_plots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary plots to {output_path}")
