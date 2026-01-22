"""
Tiled inference for full-resolution image segmentation (Phase 4).

Provides sliding window inference to apply patch-trained models to full images,
with overlap handling, post-processing, and pixel-level evaluation.

Supports both PyTorch and ONNX Runtime backends for inference.
"""

from __future__ import annotations  # Defer type annotation evaluation (allows ONNX-only usage)

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

# Use PIL as primary for reading (better Windows compatibility), tifffile for writing
try:
    from PIL import Image
    _USE_PIL = True
except ImportError:
    _USE_PIL = False

from tifffile import imwrite

def imread(path):
    """Read image file with PIL fallback for better Windows compatibility."""
    if _USE_PIL:
        try:
            img = Image.open(path)
            return np.array(img)
        except Exception:
            pass
    # Fallback to tifffile
    from tifffile import imread as tiff_imread
    return tiff_imread(path)

logger = logging.getLogger(__name__)

# Type alias for log callback: (message, level) -> None
LogCallback = Optional[Callable[[str, str], None]]


def _log(msg: str, level: str = "INFO", callback: LogCallback = None):
    """Internal logging helper that also calls optional callback."""
    logger.log(getattr(logging, level), msg)
    if callback:
        callback(msg, level)


# ============================================================================
# ONNX Runtime Inference
# ============================================================================


class ONNXModel:
    """
    Wrapper for ONNX Runtime inference session.

    Provides a consistent interface for running inference with ONNX models,
    matching the PyTorch model interface where possible.
    """

    def __init__(self, onnx_path: Path, providers: Optional[list] = None):
        """
        Load an ONNX model for inference.

        Args:
            onnx_path: Path to ONNX model file
            providers: List of execution providers (default: CPU only)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )

        self.onnx_path = Path(onnx_path)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers,
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"Loaded ONNX model from {onnx_path}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on input array.

        Args:
            x: Input array [batch, channels, height, width]

        Returns:
            Output array [batch, channels, height, width]
        """
        # Ensure float32
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: x},
        )

        return outputs[0]


def load_onnx_model(
    onnx_path: Path,
    log_callback: LogCallback = None,
) -> ONNXModel:
    """
    Load an ONNX model for inference.

    Args:
        onnx_path: Path to ONNX model file
        log_callback: Optional callback for logging

    Returns:
        ONNXModel wrapper instance
    """
    _log(f"Loading ONNX model from {onnx_path}", "INFO", log_callback)

    model = ONNXModel(onnx_path)

    _log("ONNX model loaded successfully", "INFO", log_callback)

    return model


def predict_full_image_onnx(
    image: np.ndarray,
    model: ONNXModel,
    tile_size: int = 256,
    overlap: float = 0.25,
) -> np.ndarray:
    """
    Perform tiled inference using ONNX Runtime.

    Args:
        image: Input image [H, W] (grayscale, 8-bit or normalized float)
        model: ONNXModel instance
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
    else:
        image = image.astype(np.float32)

    # Initialize accumulator arrays
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Iterate over tiles
    num_tiles = 0
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            tile = image[y:y_end, x:x_end]

            # Pad if needed
            tile_h, tile_w = tile.shape
            if tile_h < tile_size or tile_w < tile_size:
                pad_h = tile_size - tile_h
                pad_w = tile_size - tile_w
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')

            # Prepare input: [1, 1, H, W]
            tile_input = tile[np.newaxis, np.newaxis, :, :].astype(np.float32)

            # Run inference
            output = model(tile_input)

            # Apply sigmoid and extract
            prob_tile = 1.0 / (1.0 + np.exp(-output.squeeze()))
            prob_tile = prob_tile[:tile_h, :tile_w]

            # Accumulate
            prob_map[y:y_end, x:x_end] += prob_tile
            count_map[y:y_end, x:x_end] += 1.0

            num_tiles += 1

    # Average overlapping predictions
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)

    logger.debug(f"Processed {num_tiles} tiles for image of size {H}x{W}")

    return prob_map


# ============================================================================
# PyTorch Inference (original implementation)
# ============================================================================

# Lazy imports for PyTorch to allow ONNX-only usage
_torch = None
_smp = None


def _get_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_smp():
    """Lazy import segmentation_models_pytorch."""
    global _smp
    if _smp is None:
        import segmentation_models_pytorch as smp
        _smp = smp
    return _smp


# Default model config for backward compatibility with old checkpoints
DEFAULT_MODEL_CONFIG = {
    "encoder_name": "resnet18",
    "in_channels": 1,
    "classes": 1,
}


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device=None,
    log_callback: LogCallback = None,
):
    """
    Load trained model from checkpoint.

    The model architecture is read from the checkpoint's model_config field.
    For backward compatibility with older checkpoints that don't have this field,
    falls back to default configuration (resnet18 encoder, 1 input channel, 1 class).

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model onto (torch.device or None for auto)
        log_callback: Optional callback for logging

    Returns:
        Loaded model in eval mode
    """
    torch = _get_torch()
    smp = _get_smp()

    if device is None:
        device = torch.device("cpu")

    _log(f"Loading model from {checkpoint_path}", "INFO", log_callback)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Read model config from checkpoint, fall back to defaults for old checkpoints
    model_config = checkpoint.get("model_config", None)
    if model_config is None:
        _log(
            "No model_config in checkpoint, using defaults (resnet18 encoder)",
            "WARNING",
            log_callback,
        )
        model_config = DEFAULT_MODEL_CONFIG
    else:
        _log(
            f"Using model config from checkpoint: encoder={model_config.get('encoder_name')}",
            "INFO",
            log_callback,
        )

    # Create model with architecture from checkpoint
    model = smp.Unet(
        encoder_name=model_config["encoder_name"],
        encoder_weights=None,  # Don't load pretrained weights, we have trained weights
        in_channels=model_config["in_channels"],
        classes=model_config["classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    _log(f"Model loaded successfully (epoch {epoch})", "INFO", log_callback)

    return model


def predict_full_image(
    image: np.ndarray,
    model,
    device=None,
    tile_size: int = 256,
    overlap: float = 0.25,
) -> np.ndarray:
    """
    Perform tiled inference on a full-resolution image using PyTorch.

    Uses a sliding window approach with overlap. Predictions in overlapping
    regions are averaged to reduce edge artifacts.

    Args:
        image: Input image [H, W] (grayscale, 8-bit or normalized float)
        model: Trained segmentation model (PyTorch nn.Module)
        device: Device to run inference on
        tile_size: Size of inference tiles (default: 256)
        overlap: Overlap fraction between tiles (default: 0.25)

    Returns:
        Probability map [H, W] with values in [0, 1]
    """
    torch = _get_torch()

    if device is None:
        device = torch.device("cpu")

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
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)

    logger.debug(f"Processed {num_tiles} tiles for image of size {H}x{W}")

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


# ============================================================================
# Unified Inference Interface (supports both PyTorch and ONNX)
# ============================================================================


def predict_single_image(
    image_path: Path,
    model: Union[ONNXModel, "torch.nn.Module"],
    output_dir: Path,
    device=None,
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: float = 0.5,
    min_object_area: int = 100,
    save_prob_map: bool = False,
    log_callback: LogCallback = None,
) -> Optional[Path]:
    """
    Run inference on a single image using either ONNX or PyTorch model.

    This is the unified inference function that handles both backends,
    includes graceful error handling, and supports logging callbacks.

    Args:
        image_path: Path to input image
        model: ONNXModel or PyTorch nn.Module
        output_dir: Directory to save output mask
        device: Device for PyTorch inference (ignored for ONNX)
        tile_size: Tile size for sliding window
        overlap: Overlap fraction between tiles
        threshold: Probability threshold for binarization
        min_object_area: Minimum object area for post-processing
        save_prob_map: Whether to save probability map (default: False)
        log_callback: Optional callback for logging progress

    Returns:
        Path to saved mask, or None if processing failed
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    basename = image_path.stem
    _log(f"Processing: {image_path.name}", "INFO", log_callback)

    try:
        # Load image
        image = imread(image_path)
    except Exception as e:
        _log(f"Failed to load image {image_path.name}: {e}", "WARNING", log_callback)
        return None

    try:
        # Convert RGB to grayscale if needed
        if image.ndim == 3:
            image = image.mean(axis=2).astype(np.uint8)

        # Determine backend and run inference
        if isinstance(model, ONNXModel):
            prob_map = predict_full_image_onnx(
                image=image,
                model=model,
                tile_size=tile_size,
                overlap=overlap,
            )
        else:
            # PyTorch model
            prob_map = predict_full_image(
                image=image,
                model=model,
                device=device,
                tile_size=tile_size,
                overlap=overlap,
            )

        # Threshold and post-process
        binary_mask = threshold_mask(prob_map, threshold=threshold)
        cleaned_mask = postprocess_mask(
            binary_mask,
            min_object_area=min_object_area,
            fill_holes=True,
        )

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save binary mask
        mask_path = output_dir / f"{basename}_pred_mask.tif"
        imwrite(mask_path, cleaned_mask, compression='lzw')

        # Optionally save probability map
        if save_prob_map:
            prob_path = output_dir / f"{basename}_pred_prob.tif"
            imwrite(prob_path, prob_map, compression='lzw')

        return mask_path

    except Exception as e:
        _log(f"Inference failed for {image_path.name}: {e}", "WARNING", log_callback)
        return None


def run_inference_batch(
    image_paths: list,
    model: Union[ONNXModel, "torch.nn.Module"],
    output_dir: Path,
    device=None,
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: float = 0.5,
    min_object_area: int = 100,
    save_prob_map: bool = False,
    log_callback: LogCallback = None,
) -> Tuple[int, int, list]:
    """
    Run inference on a batch of images.

    Args:
        image_paths: List of image paths to process
        model: ONNXModel or PyTorch nn.Module
        output_dir: Directory to save output masks
        device: Device for PyTorch inference
        tile_size: Tile size for sliding window
        overlap: Overlap fraction between tiles
        threshold: Probability threshold
        min_object_area: Minimum object area
        save_prob_map: Whether to save probability maps
        log_callback: Optional callback for logging

    Returns:
        Tuple of (successful_count, failed_count, list_of_output_paths)
    """
    output_dir = Path(output_dir)
    total = len(image_paths)

    _log(f"Starting batch inference on {total} images", "INFO", log_callback)

    successful = 0
    failed = 0
    output_paths = []

    for idx, image_path in enumerate(image_paths):
        _log(f"[{idx + 1}/{total}] {Path(image_path).name}", "INFO", log_callback)

        result = predict_single_image(
            image_path=image_path,
            model=model,
            output_dir=output_dir,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            min_object_area=min_object_area,
            save_prob_map=save_prob_map,
            log_callback=log_callback,
        )

        if result is not None:
            successful += 1
            output_paths.append(result)
        else:
            failed += 1

    _log(f"Batch complete: {successful} successful, {failed} failed", "INFO", log_callback)

    return successful, failed, output_paths
