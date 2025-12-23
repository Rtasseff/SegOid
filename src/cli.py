"""
Command-line interface entrypoints for the SegOid pipeline.

These are placeholder implementations that will be fleshed out in later phases.
"""

import argparse
import sys


def validate_dataset():
    """Phase 1: Validate image/mask pairing and compute QC metrics."""
    parser = argparse.ArgumentParser(
        description="Validate dataset: check image/mask pairing and compute QC metrics"
    )
    parser.add_argument("--input-dir", required=True, help="Path to data/working/ directory")
    parser.add_argument("--output-dir", required=True, help="Path to data/splits/ directory")
    args = parser.parse_args()

    print(f"[Phase 1] Validating dataset in {args.input_dir}")
    print(f"Output will be written to {args.output_dir}")
    print("⚠️  Not yet implemented - Phase 1 placeholder")
    return 0


def make_splits():
    """Phase 1: Generate train/val/test splits from manifest."""
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits from dataset manifest"
    )
    parser.add_argument("--manifest", required=True, help="Path to all.csv manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", required=True, help="Path to data/splits/ directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    args = parser.parse_args()

    print(f"[Phase 1] Creating splits from {args.manifest}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("⚠️  Not yet implemented - Phase 1 placeholder")
    return 0


def sanity_check():
    """Phase 1.5: Quick pipeline validation before full training."""
    parser = argparse.ArgumentParser(
        description="Sanity check: validate pipeline with minimal training"
    )
    parser.add_argument("--config", required=True, help="Path to sanity_check.yaml")
    args = parser.parse_args()

    print(f"[Phase 1.5] Running sanity check with config: {args.config}")
    print("⚠️  Not yet implemented - Phase 1.5 placeholder")
    return 0


def train():
    """Phase 3: Train segmentation model on patches."""
    parser = argparse.ArgumentParser(
        description="Train segmentation model"
    )
    parser.add_argument("--config", required=True, help="Path to train.yaml")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print(f"[Phase 3] Training model with config: {args.config}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    print("⚠️  Not yet implemented - Phase 3 placeholder")
    return 0


def predict_full():
    """Phase 4: Run tiled inference on full-resolution images."""
    parser = argparse.ArgumentParser(
        description="Run tiled inference on full images"
    )
    parser.add_argument("--config", required=True, help="Path to predict.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    print(f"[Phase 4] Running inference with config: {args.config}")
    print(f"Using checkpoint: {args.checkpoint}")
    print("⚠️  Not yet implemented - Phase 4 placeholder")
    return 0


def quantify_objects():
    """Phase 5: Extract objects and compute morphology metrics."""
    parser = argparse.ArgumentParser(
        description="Quantify objects: extract instances and compute metrics"
    )
    parser.add_argument("--config", required=True, help="Path to quantify.yaml")
    args = parser.parse_args()

    print(f"[Phase 5] Quantifying objects with config: {args.config}")
    print("⚠️  Not yet implemented - Phase 5 placeholder")
    return 0
