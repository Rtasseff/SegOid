#!/usr/bin/env python3
"""
Test script to demonstrate loading and using the production model.

This script shows how to:
1. Load a trained model checkpoint
2. Prepare an image for inference
3. Run inference and get predictions
4. Access training metadata

Usage:
    python scripts/test_model_loading.py --checkpoint runs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
                                          --image data/working/images/Matri_1_1.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import tifffile


def load_production_model(checkpoint_path: Path, device: str = "auto") -> tuple:
    """
    Load production model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ("auto", "cpu", "cuda", "mps")

    Returns:
        Tuple of (model, checkpoint_metadata)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model architecture
    # NOTE: These must match the training configuration
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,  # We'll load trained weights
        in_channels=1,  # Grayscale
        classes=1,  # Binary segmentation
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode (important!)
    model.eval()

    # Move to device
    model = model.to(device)

    # Extract metadata
    metadata = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_val_dice": checkpoint.get("best_val_dice", "unknown"),
        "training_history": checkpoint.get("history", {}),
    }

    print(f"✓ Model loaded successfully!")
    print(f"  Trained for {metadata['epoch']} epochs")
    print(f"  Best validation Dice: {metadata['best_val_dice']:.4f}")

    return model, metadata


def load_and_preprocess_image(image_path: Path) -> tuple:
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (preprocessed_tensor, original_image)
    """
    print(f"\nLoading image: {image_path}")

    # Load image
    image = tifffile.imread(image_path)

    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min()}, {image.max()}]")

    # Convert to grayscale if RGB
    if image.ndim == 3:
        # Assuming RGB, convert to grayscale
        image = np.mean(image, axis=-1).astype(np.float32)
        print(f"  Converted RGB to grayscale")
    else:
        image = image.astype(np.float32)

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Add batch and channel dimensions: [H, W] -> [1, 1, H, W]
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    print(f"  Preprocessed tensor shape: {image_tensor.shape}")

    return image_tensor, image


def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Run inference on preprocessed image.

    Args:
        model: Loaded model
        image_tensor: Preprocessed image tensor [1, 1, H, W]
        device: Device to run inference on

    Returns:
        Prediction probability map [H, W] in range [0, 1]
    """
    print("\nRunning inference...")

    # Move image to device
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)  # Convert logits to probabilities

    # Move back to CPU and convert to numpy
    prediction = prediction.cpu().numpy()

    # Remove batch and channel dimensions: [1, 1, H, W] -> [H, W]
    prediction = prediction[0, 0]

    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Test production model loading and inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")
    parser.add_argument("--output", help="Optional path to save prediction")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)

    # Validate inputs
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 80)
    print("PRODUCTION MODEL INFERENCE TEST")
    print("=" * 80)

    # Load model
    model, metadata = load_production_model(checkpoint_path, device=args.device)

    # Load and preprocess image
    image_tensor, original_image = load_and_preprocess_image(image_path)

    # Get device
    device = next(model.parameters()).device

    # Run inference
    prediction = run_inference(model, image_tensor, device)

    # Binarize prediction
    binary_mask = (prediction > args.threshold).astype(np.uint8) * 255

    print(f"\nBinary mask (threshold={args.threshold}):")
    print(f"  Shape: {binary_mask.shape}")
    print(f"  Foreground pixels: {(binary_mask > 0).sum()}")
    print(f"  Coverage: {(binary_mask > 0).sum() / binary_mask.size:.2%}")

    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save probability map
        prob_path = output_path.parent / f"{output_path.stem}_prob.tif"
        tifffile.imwrite(prob_path, prediction.astype(np.float32), compression="lzw")
        print(f"\n✓ Saved probability map: {prob_path}")

        # Save binary mask
        mask_path = output_path.parent / f"{output_path.stem}_mask.tif"
        tifffile.imwrite(mask_path, binary_mask, compression="lzw")
        print(f"✓ Saved binary mask: {mask_path}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nModel is ready for production use!")
    print(f"Training metadata: Epoch {metadata['epoch']}, Val Dice {metadata['best_val_dice']:.4f}")


if __name__ == "__main__":
    main()
