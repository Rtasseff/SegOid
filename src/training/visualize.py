"""
Visualization utilities for model predictions.

Generates overlay images comparing predictions to ground truth.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create visualization overlay with GT and predicted masks.

    Args:
        image: Grayscale image [H, W] uint8
        gt_mask: Ground truth binary mask [H, W] uint8 (0 or 255)
        pred_mask: Predicted binary mask [H, W] uint8 (0 or 255)
        alpha: Transparency for filled regions (default: 0.5)

    Returns:
        RGB overlay image [H, W, 3] uint8
    """
    H, W = image.shape

    # Convert grayscale to RGB
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create colored masks
    gt_color = np.zeros((H, W, 3), dtype=np.uint8)
    gt_color[:, :, 1] = gt_mask  # Green for ground truth

    pred_color = np.zeros((H, W, 3), dtype=np.uint8)
    pred_color[:, :, 2] = pred_mask  # Red for predictions

    # Blend filled regions
    gt_filled = cv2.addWeighted(overlay, 1 - alpha, gt_color, alpha, 0)
    overlay = np.where(gt_mask[..., None] > 0, gt_filled, overlay)

    pred_filled = cv2.addWeighted(overlay, 1 - alpha, pred_color, alpha, 0)
    overlay = np.where(pred_mask[..., None] > 0, pred_filled, overlay)

    # Draw contours for clarity
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)

    gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw GT contours in green (brighter)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)

    # Draw prediction contours in red (brighter)
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 2)

    return overlay


def create_side_by_side(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """
    Create side-by-side comparison: Original | GT | Pred | Overlay

    Args:
        image: Grayscale image [H, W] uint8
        gt_mask: Ground truth binary mask [H, W] uint8
        pred_mask: Predicted binary mask [H, W] uint8

    Returns:
        Combined image [H, W*4, 3] uint8
    """
    H, W = image.shape

    # Convert all to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gt_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

    # Create overlay
    overlay = create_overlay(image, gt_mask, pred_mask)

    # Stack horizontally
    combined = np.hstack([image_rgb, gt_rgb, pred_rgb, overlay])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    cv2.putText(combined, "Original", (10, 20), font, font_scale, color, thickness)
    cv2.putText(combined, "Ground Truth", (W + 10, 20), font, font_scale, color, thickness)
    cv2.putText(combined, "Prediction", (2 * W + 10, 20), font, font_scale, color, thickness)
    cv2.putText(combined, "Overlay", (3 * W + 10, 20), font, font_scale, color, thickness)

    return combined


def generate_prediction_overlays(
    model: nn.Module,
    data_loader: DataLoader,
    manifest_df,
    output_dir: Path,
    device: torch.device,
    num_samples: Optional[int] = None,
    threshold: float = 0.5,
) -> None:
    """
    Generate overlay visualizations for validation set.

    For each full image in the validation set, runs inference and creates overlays.

    Args:
        model: Trained model
        data_loader: DataLoader for the dataset (should have batch_size=1 for full images)
        manifest_df: DataFrame with image metadata (basenames, paths)
        output_dir: Directory to save overlay images
        device: Device for inference
        num_samples: Max number of samples to visualize (None = all)
        threshold: Threshold for binarizing predictions (default: 0.5)
    """
    logger.info(f"Generating prediction overlays to {output_dir}")

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # We need to load full images, not patches
    # So we'll iterate through the manifest and load full images directly
    import tifffile

    data_root = data_loader.dataset.data_root
    num_processed = 0

    with torch.no_grad():
        for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Generating overlays"):
            if num_samples is not None and num_processed >= num_samples:
                break

            basename = row["basename"]
            image_path = data_root / row["image_path"]
            mask_path = data_root / row["mask_path"]

            # Load full image and mask
            image = tifffile.imread(image_path)
            gt_mask = tifffile.imread(mask_path)

            # Convert RGB to grayscale if needed
            if image.ndim == 3:
                image = image[:, :, 0]

            # Run inference on full image (simple approach for sanity check)
            # For Phase 1.5, we'll just resize to model input size
            # (Proper tiled inference comes in Phase 4)
            H, W = image.shape

            # Resize for inference (simple approach)
            image_resized = cv2.resize(image, (256, 256))
            image_tensor = torch.from_numpy(image_resized).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 256, 256]

            # Run model
            output = model(image_tensor)
            pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # [256, 256]

            # Resize prediction back to original size
            pred_prob_resized = cv2.resize(pred_prob, (W, H))
            pred_mask = ((pred_prob_resized > threshold) * 255).astype(np.uint8)

            # Create visualizations
            overlay = create_overlay(image, gt_mask, pred_mask)
            side_by_side = create_side_by_side(image, gt_mask, pred_mask)

            # Save
            overlay_path = output_dir / f"{basename}_overlay.png"
            side_by_side_path = output_dir / f"{basename}_comparison.png"

            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(side_by_side_path), side_by_side)

            num_processed += 1

    logger.info(f"Generated {num_processed} overlay visualizations")
