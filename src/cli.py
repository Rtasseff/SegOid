"""
Command-line interface entrypoints for the SegOid pipeline.

These are placeholder implementations that will be fleshed out in later phases.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data.validate import (
    validate_dataset as run_validation,
    print_validation_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def validate_dataset():
    """Phase 1: Validate image/mask pairing and compute QC metrics."""
    parser = argparse.ArgumentParser(
        description="Validate dataset: check image/mask pairing and compute QC metrics"
    )
    parser.add_argument("--input-dir", required=True, help="Path to data/working/ directory")
    parser.add_argument("--output-dir", required=True, help="Path to data/splits/ directory")
    args = parser.parse_args()

    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        # Run validation
        all_df, qc_df, result = run_validation(input_dir, output_dir)

        # Print summary
        print_validation_summary(all_df, qc_df, result)

        # Return error code if validation failed
        if result.failed > 0:
            return 1

        return 0

    except Exception as e:
        logging.error(f"Validation failed: {e}", exc_info=True)
        return 1


def make_splits():
    """Phase 1: Generate train/val/test splits from manifest."""
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits from dataset manifest"
    )
    parser.add_argument("--manifest", required=True, help="Path to all.csv manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", required=True, help="Path to data/splits/ directory")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify splits by mask coverage buckets",
    )
    args = parser.parse_args()

    try:
        from src.data.validate import make_splits as run_splits

        manifest_path = Path(args.manifest)
        output_dir = Path(args.output_dir)

        # Run split generation
        run_splits(
            manifest_path,
            output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify=args.stratify,
        )

        return 0

    except Exception as e:
        logging.error(f"Split generation failed: {e}", exc_info=True)
        return 1


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
