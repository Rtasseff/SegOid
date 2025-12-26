"""
Tiled inference for full-resolution image segmentation (Phase 4).

Provides sliding window inference to apply patch-trained models to full images,
with overlap handling, post-processing, and pixel-level evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import remove_small_objects
from tifffile import imread, imwrite

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same architecture as training
    # (hardcoded for now, should match training config)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,  # Loading from checkpoint
        in_channels=1,
        classes=1,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    return model


def predict_full_image(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 256,
    overlap: float = 0.25,
) -> np.ndarray:
    """
    Perform tiled inference on a full-resolution image.

    Uses a sliding window approach with overlap. Predictions in overlapping
    regions are averaged to reduce edge artifacts.

    Args:
        image: Input image [H, W] (grayscale, 8-bit or normalized float)
        model: Trained segmentation model
        device: Device to run inference on
        tile_size: Size of inference tiles (default: 256)
        overlap: Overlap fraction between tiles (default: 0.25)

    Returns:
        Probability map [H, W] with values in [0, 1]
    """
    H, W = image.shape
    stride = int(tile_size * (1 - overlap))

    # Normalize image to [0, 1] if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Initialize accumulator arrays
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Iterate over tiles with sliding window
    num_tiles = 0
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Extract tile with bounds checking
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            tile = image[y:y_end, x:x_end]

            # Pad tile if at edge (to maintain tile_size)
            tile_h, tile_w = tile.shape
            if tile_h < tile_size or tile_w < tile_size:
                # Use reflection padding to avoid edge artifacts
                pad_h = tile_size - tile_h
                pad_w = tile_size - tile_w
                tile = np.pad(
                    tile,
                    ((0, pad_h), (0, pad_w)),
                    mode='reflect'
                )

            # Convert to tensor [1, 1, H, W]
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(tile_tensor)
                prob = torch.sigmoid(output)

            # Convert back to numpy and remove padding
            prob_tile = prob.squeeze().cpu().numpy()
            prob_tile = prob_tile[:tile_h, :tile_w]

            # Accumulate prediction
            prob_map[y:y_end, x:x_end] += prob_tile
            count_map[y:y_end, x:x_end] += 1.0

            num_tiles += 1

    # Average overlapping predictions
    # Avoid division by zero (shouldn't happen, but be safe)
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)

    logger.info(f"Processed {num_tiles} tiles for image of size {H}x{W}")

    return prob_map


def threshold_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert probability map to binary mask.

    Args:
        prob_map: Probability map [H, W] with values in [0, 1]
        threshold: Threshold for binarization (default: 0.5)

    Returns:
        Binary mask [H, W] with values {0, 255}
    """
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    return binary_mask


def postprocess_mask(
    binary_mask: np.ndarray,
    min_object_area: int = 100,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Apply post-processing to clean up binary mask.

    Steps:
    1. Remove small objects below minimum area
    2. Fill holes within objects (optional)

    Args:
        binary_mask: Binary mask [H, W] with values {0, 255}
        min_object_area: Minimum object size in pixels (default: 100)
        fill_holes: Whether to fill holes (default: True)

    Returns:
        Cleaned binary mask [H, W] with values {0, 255}
    """
    # Convert to boolean for processing
    mask_bool = binary_mask > 0

    # Remove small objects
    if min_object_area > 0:
        # Use connected components to identify objects
        labeled, num_labels = ndimage.label(mask_bool)

        # Compute sizes
        sizes = np.bincount(labeled.ravel())

        # Create mask of objects to keep (size > min_area)
        # Note: sizes[0] is background, so we set it to 0 to exclude it
        mask_sizes = sizes > min_object_area
        mask_sizes[0] = 0

        # Apply mask
        mask_bool = mask_sizes[labeled]

        logger.debug(f"Removed {num_labels - mask_sizes.sum()} small objects (< {min_object_area} px)")

    # Fill holes within objects
    if fill_holes:
        mask_bool = ndimage.binary_fill_holes(mask_bool)
        logger.debug("Filled holes in objects")

    # Convert back to uint8 {0, 255}
    cleaned_mask = mask_bool.astype(np.uint8) * 255

    return cleaned_mask


def compute_dice_coefficient(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient between predicted and ground truth masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred_mask: Predicted binary mask [H, W]
        gt_mask: Ground truth binary mask [H, W]
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Dice coefficient (float in [0, 1])
    """
    # Convert to boolean
    pred = pred_mask > 0
    gt = gt_mask > 0

    # Flatten
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()

    # Compute intersection and union
    intersection = np.sum(pred_flat & gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)

    # Compute Dice
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return float(dice)


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred_mask: Predicted binary mask [H, W]
        gt_mask: Ground truth binary mask [H, W]
        smooth: Smoothing constant to avoid division by zero

    Returns:
        IoU (float in [0, 1])
    """
    # Convert to boolean
    pred = pred_mask > 0
    gt = gt_mask > 0

    # Flatten
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()

    # Compute intersection and union
    intersection = np.sum(pred_flat & gt_flat)
    union = np.sum(pred_flat | gt_flat)

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def predict_image_from_path(
    image_path: Path,
    mask_path: Optional[Path],
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: float = 0.5,
    min_object_area: int = 100,
) -> Dict[str, float]:
    """
    Run full inference pipeline on a single image.

    Steps:
    1. Load image
    2. Perform tiled inference
    3. Threshold probability map
    4. Apply post-processing
    5. Save outputs (probability map and binary mask)
    6. Compute metrics if ground truth is provided

    Args:
        image_path: Path to input image
        mask_path: Path to ground truth mask (optional, for evaluation)
        model: Trained segmentation model
        device: Device to run inference on
        output_dir: Directory to save outputs
        tile_size: Tile size for sliding window
        overlap: Overlap fraction between tiles
        threshold: Threshold for binarization
        min_object_area: Minimum object area for post-processing

    Returns:
        Dictionary with metrics (Dice, IoU) if mask_path provided, else empty dict
    """
    logger.info(f"Processing {image_path.name}")

    # Load image
    image = imread(image_path)

    # Convert RGB to grayscale if needed
    if image.ndim == 3:
        # Average across channels
        image = image.mean(axis=2).astype(np.uint8)
        logger.debug(f"Converted RGB to grayscale")

    # Perform tiled inference
    prob_map = predict_full_image(
        image=image,
        model=model,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )

    # Threshold to binary mask
    binary_mask = threshold_mask(prob_map, threshold=threshold)

    # Post-process
    cleaned_mask = postprocess_mask(
        binary_mask,
        min_object_area=min_object_area,
        fill_holes=True,
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = image_path.stem

    # Save probability map as float32 TIFF
    prob_path = output_dir / f"{basename}_pred_prob.tif"
    imwrite(prob_path, prob_map, compression='lzw')
    logger.info(f"Saved probability map to {prob_path}")

    # Save binary mask as uint8 TIFF
    mask_path_out = output_dir / f"{basename}_pred_mask.tif"
    imwrite(mask_path_out, cleaned_mask, compression='lzw')
    logger.info(f"Saved binary mask to {mask_path_out}")

    # Compute metrics if ground truth provided
    metrics = {}
    if mask_path is not None and mask_path.exists():
        gt_mask = imread(mask_path)

        # Convert RGB to grayscale if needed
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.mean(axis=2).astype(np.uint8)

        # Compute metrics
        dice = compute_dice_coefficient(cleaned_mask, gt_mask)
        iou = compute_iou(cleaned_mask, gt_mask)

        metrics = {
            "image": image_path.name,
            "dice": dice,
            "iou": iou,
        }

        logger.info(f"Metrics - Dice: {dice:.4f}, IoU: {iou:.4f}")

    return metrics
