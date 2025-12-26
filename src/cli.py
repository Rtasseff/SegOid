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
    parser.add_argument(
        "--config",
        default="configs/sanity_check.yaml",
        help="Path to config YAML file (default: configs/sanity_check.yaml)",
    )
    # Optional CLI overrides
    parser.add_argument("--train-manifest", help="Override train manifest path")
    parser.add_argument("--val-manifest", help="Override val manifest path")
    parser.add_argument("--patches-per-image", type=int, help="Override patches per image")
    parser.add_argument("--patch-size", type=int, help="Override patch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--num-workers", type=int, help="Override number of workers")
    args = parser.parse_args()

    try:
        import pandas as pd
        import torch
        import yaml
        from torch.utils.data import DataLoader

        from src.data.dataset import PatchDataset
        from src.training.train import create_model, train_model
        from src.training.visualize import generate_prediction_overlays

        print("\n" + "=" * 80)
        print("PHASE 1.5: SANITY CHECK")
        print("=" * 80)

        # Load config from YAML
        config_path = Path(args.config)
        print(f"\nLoading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Apply CLI overrides
        if args.train_manifest:
            config['data']['train_manifest'] = args.train_manifest
        if args.val_manifest:
            config['data']['val_manifest'] = args.val_manifest
        if args.patches_per_image is not None:
            config['dataset']['patches_per_image'] = args.patches_per_image
        if args.patch_size is not None:
            config['dataset']['patch_size'] = args.patch_size
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            config['training']['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            config['training']['learning_rate'] = args.learning_rate
        if args.output_dir:
            config['output']['run_dir'] = args.output_dir
        if args.num_workers is not None:
            config['training']['num_workers'] = args.num_workers

        # Extract config values
        train_manifest = Path(config['data']['train_manifest'])
        val_manifest = Path(config['data']['val_manifest'])
        data_root = Path(config['data'].get('data_root', 'data'))
        output_dir = Path(config['output']['run_dir'])

        patch_size = config['dataset']['patch_size']
        patches_per_image = config['dataset']['patches_per_image']
        positive_ratio = config['dataset']['positive_ratio']

        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        learning_rate = config['training']['learning_rate']
        num_workers = config['training']['num_workers']

        print(f"\nConfiguration:")
        print(f"  Train manifest: {train_manifest}")
        print(f"  Val manifest: {val_manifest}")
        print(f"  Patches per image: {patches_per_image}")
        print(f"  Patch size: {patch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Output directory: {output_dir}")

        # Detect device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("\n✓ Using Apple M1/M2 GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("\n⚠️  Using CPU (training will be slow)")

        # Create datasets
        print("\nLoading datasets...")
        train_dataset = PatchDataset(
            manifest_csv=train_manifest,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            positive_ratio=positive_ratio,
            augment=config['dataset']['augmentation']['enabled'],
            data_root=data_root,
        )

        val_dataset = PatchDataset(
            manifest_csv=val_manifest,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            positive_ratio=positive_ratio,
            augment=False,  # No augmentation for validation
            data_root=data_root,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        print(f"  Train patches: {len(train_dataset)}")
        print(f"  Val patches: {len(val_dataset)}")

        # Create model
        print("\nCreating model...")
        model = create_model(
            encoder_name=config['model']['encoder'],
            encoder_weights=config['model']['encoder_weights'],
            in_channels=config['model']['in_channels'],
            classes=config['model']['classes'],
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Train model
        print("\nStarting training...\n")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir,
        )

        # Save config snapshot for reproducibility
        output_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = output_dir / "config.yaml"
        with open(config_snapshot_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n✓ Saved config snapshot to {config_snapshot_path}")

        # Generate visualizations
        print("\nGenerating prediction overlays...")
        val_manifest_df = pd.read_csv(val_manifest)
        overlays_dir = output_dir / "overlays"

        num_samples = config['output'].get('num_overlay_samples')
        generate_prediction_overlays(
            model=model,
            data_loader=val_loader,
            manifest_df=val_manifest_df,
            output_dir=overlays_dir,
            device=device,
            num_samples=num_samples,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("SANITY CHECK COMPLETE")
        print("=" * 80)
        print("\nFinal Results:")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Final Train Dice: {history['train_dice'][-1]:.4f}")
        print(f"  Final Val Dice: {history['val_dice'][-1]:.4f}")

        # Check exit criteria
        print("\nExit Criteria Check:")

        spatial_threshold = config['exit_criteria']['spatial_coherence_threshold']

        # 1. Loss should decrease
        loss_decreased = history["train_loss"][-1] < history["train_loss"][0]
        print(f"  ✓ Loss decreased: {loss_decreased} (from {history['train_loss'][0]:.4f} to {history['train_loss'][-1]:.4f})")

        # 2. Dice should increase
        dice_increased = history["val_dice"][-1] > config['exit_criteria']['min_dice_improvement']
        print(f"  ✓ Val Dice > {config['exit_criteria']['min_dice_improvement']}: {dice_increased} ({history['val_dice'][-1]:.4f})")

        # 3. Check for spatial coherence
        spatially_coherent = history["val_dice"][-1] > spatial_threshold
        print(f"  {'✓' if spatially_coherent else '⚠️ '} Predictions spatially coherent (Dice > {spatial_threshold}): {spatially_coherent}")

        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - Checkpoint: {output_dir / 'final_checkpoint.pth'}")
        print(f"  - Overlays: {overlays_dir}")

        print("\n⚠️  IMPORTANT: Review overlay images for:")
        print("  1. Predictions correspond to actual spheroid locations")
        print("  2. No systematic offset between predictions and ground truth")
        print("  3. Predictions are spatially coherent (not random noise)")

        print("\n" + "=" * 80 + "\n")

        return 0

    except Exception as e:
        logging.error(f"Sanity check failed: {e}", exc_info=True)
        return 1


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
