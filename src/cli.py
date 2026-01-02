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
    """Phase 3: Train segmentation model on patches with production features."""
    parser = argparse.ArgumentParser(
        description="Train segmentation model with checkpointing, early stopping, and TensorBoard"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML file (e.g., configs/train.yaml)",
    )
    parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        import torch
        import yaml
        from datetime import datetime
        from torch.utils.data import DataLoader

        from src.data.dataset import PatchDataset
        from src.training.train import load_config, train_model as run_training

        print("\n" + "=" * 80)
        print("PHASE 3: FULL MODEL TRAINING")
        print("=" * 80)

        # Load config
        config_path = Path(args.config)
        print(f"\nLoading configuration from: {config_path}")

        config = load_config(config_path)

        # Extract config values
        train_manifest = Path(config['data']['train_manifest'])
        val_manifest = Path(config['data']['val_manifest'])
        data_root = Path(config['data'].get('data_root', 'data'))

        patch_size = config['dataset']['patch_size']
        patches_per_image = config['dataset']['patches_per_image']
        positive_ratio = config['dataset']['positive_ratio']

        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        learning_rate = config['training']['learning_rate']
        num_workers = config['training']['num_workers']

        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"train_{timestamp}"
        run_dir = Path(config['output']['run_dir']) / run_name

        print(f"\nConfiguration:")
        print(f"  Train manifest: {train_manifest}")
        print(f"  Val manifest: {val_manifest}")
        print(f"  Patches per image: {patches_per_image}")
        print(f"  Patch size: {patch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Run directory: {run_dir}")

        # Early stopping and LR scheduling
        if config.get('early_stopping', {}).get('enabled', False):
            print(f"  Early stopping: enabled (patience={config['early_stopping']['patience']})")
        if config.get('lr_scheduler', {}).get('enabled', False):
            print(f"  LR scheduler: enabled (ReduceLROnPlateau)")
        if config.get('tensorboard', {}).get('enabled', False):
            print(f"  TensorBoard: enabled")

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
            pin_memory=config['training'].get('pin_memory', True) if device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config['training'].get('pin_memory', True) if device.type == "cuda" else False,
        )

        print(f"  Train patches: {len(train_dataset)}")
        print(f"  Val patches: {len(val_dataset)}")
        print(f"  Train batches per epoch: {len(train_loader)}")
        print(f"  Val batches per epoch: {len(val_loader)}")

        # Parse resume checkpoint if provided
        resume_checkpoint = Path(args.resume) if args.resume else None
        if resume_checkpoint:
            print(f"\n✓ Resuming from checkpoint: {resume_checkpoint}")

        # Train model
        print("\nStarting training...\n")
        history = run_training(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_dir=run_dir,
            resume_checkpoint=resume_checkpoint,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print("\nFinal Results:")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Final Train Dice: {history['train_dice'][-1]:.4f}")
        print(f"  Final Val Dice: {history['val_dice'][-1]:.4f}")
        print(f"  Best Val Dice: {max(history['val_dice']):.4f} (epoch {history['val_dice'].index(max(history['val_dice'])) + 1})")

        print(f"\nOutputs saved to: {run_dir}")
        print(f"  - Config: {run_dir / 'config.yaml'}")
        print(f"  - Best model: {run_dir / 'checkpoints' / 'best_model.pth'}")
        print(f"  - Final model: {run_dir / 'checkpoints' / 'final_model.pth'}")
        print(f"  - TensorBoard logs: {run_dir / 'tensorboard'}")

        if config.get('tensorboard', {}).get('enabled', False):
            print(f"\nView training progress with TensorBoard:")
            print(f"  tensorboard --logdir={run_dir / 'tensorboard'}")

        print("\n" + "=" * 80 + "\n")

        return 0

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return 1


def predict_full():
    """Phase 4: Run tiled inference on full-resolution images."""
    parser = argparse.ArgumentParser(
        description="Run tiled inference on full images to generate predicted masks"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to CSV manifest (e.g., data/splits/test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="inference/",
        help="Output directory for predictions (default: inference/)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size for sliding window (default: 256)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap fraction between tiles (default: 0.25)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarization (default: 0.5)",
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=100,
        help="Minimum object area in pixels (default: 100)",
    )
    parser.add_argument(
        "--data-root",
        default="data/",
        help="Root directory for image/mask paths (default: data/)",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        import torch

        from src.inference.predict import (
            load_model_from_checkpoint,
            predict_image_from_path,
        )

        print("\n" + "=" * 80)
        print("PHASE 4: TILED INFERENCE ON FULL IMAGES")
        print("=" * 80)

        # Parse arguments
        checkpoint_path = Path(args.checkpoint)
        manifest_path = Path(args.manifest)
        output_dir = Path(args.output_dir)
        data_root = Path(args.data_root)

        print(f"\nConfiguration:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Tile size: {args.tile_size}")
        print(f"  Overlap: {args.overlap}")
        print(f"  Threshold: {args.threshold}")
        print(f"  Min object area: {args.min_object_area} px")

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Verify manifest exists
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        # Detect device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("\n✓ Using Apple M1/M2 GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("\n⚠️  Using CPU (inference will be slower)")

        # Load model
        print("\nLoading model...")
        model = load_model_from_checkpoint(checkpoint_path, device)

        # Load manifest
        print("\nLoading manifest...")
        manifest_df = pd.read_csv(manifest_path)
        print(f"  Found {len(manifest_df)} images")

        # Process each image
        print("\nRunning inference...")
        all_metrics = []

        for idx, row in manifest_df.iterrows():
            # Resolve paths relative to data_root
            image_path = data_root / row["image_path"]
            mask_path = data_root / row["mask_path"] if "mask_path" in row and pd.notna(row["mask_path"]) else None

            # Run inference on this image
            metrics = predict_image_from_path(
                image_path=image_path,
                mask_path=mask_path,
                model=model,
                device=device,
                output_dir=output_dir,
                tile_size=args.tile_size,
                overlap=args.overlap,
                threshold=args.threshold,
                min_object_area=args.min_object_area,
            )

            if metrics:  # If ground truth was available
                all_metrics.append(metrics)

        # Save metrics to CSV
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = output_dir / "pixel_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)

            print("\n" + "=" * 80)
            print("INFERENCE COMPLETE")
            print("=" * 80)
            print("\nPixel-level Metrics Summary:")
            print(f"  Mean Dice: {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
            print(f"  Mean IoU:  {metrics_df['iou'].mean():.4f} ± {metrics_df['iou'].std():.4f}")
            print(f"\nPer-image metrics:")
            for _, row in metrics_df.iterrows():
                print(f"  {row['image']}: Dice={row['dice']:.4f}, IoU={row['iou']:.4f}")

            print(f"\n✓ Metrics saved to: {metrics_path}")
        else:
            print("\n" + "=" * 80)
            print("INFERENCE COMPLETE")
            print("=" * 80)
            print("\n⚠️  No ground truth masks available - metrics not computed")

        print(f"\n✓ Predictions saved to: {output_dir}")
        print(f"  - Probability maps: *_pred_prob.tif")
        print(f"  - Binary masks: *_pred_mask.tif")

        print("\n" + "=" * 80 + "\n")

        return 0

    except Exception as e:
        logging.error(f"Inference failed: {e}", exc_info=True)
        return 1


def quantify_objects():
    """Phase 5: Extract objects and compute morphology metrics."""
    parser = argparse.ArgumentParser(
        description="Extract individual objects from segmentation masks, match to ground truth, and compute morphology metrics"
    )
    parser.add_argument(
        "--pred-mask-dir",
        required=True,
        help="Directory with predicted masks (e.g., inference/test_predictions/)",
    )
    parser.add_argument(
        "--gt-manifest",
        required=True,
        help="Path to test manifest CSV (e.g., data/splits/test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/",
        help="Output directory for metrics and plots (default: metrics/)",
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=100,
        help="Minimum object area in pixels (default: 100)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for valid object match (default: 0.5)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Pixel size in micrometers for physical unit conversion (optional)",
    )
    parser.add_argument(
        "--data-root",
        default="data/",
        help="Root directory for relative paths in manifest (default: data/)",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        from src.analysis.quantify import (
            process_image_pair,
            create_summary_plots,
        )

        print("\n" + "=" * 80)
        print("PHASE 5: OBJECT QUANTIFICATION AND INSTANCE EVALUATION")
        print("=" * 80)

        # Parse arguments
        pred_mask_dir = Path(args.pred_mask_dir)
        gt_manifest_path = Path(args.gt_manifest)
        output_dir = Path(args.output_dir)
        data_root = Path(args.data_root)

        print(f"\nConfiguration:")
        print(f"  Predicted masks: {pred_mask_dir}")
        print(f"  GT manifest: {gt_manifest_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Min object area: {args.min_object_area} px")
        print(f"  IoU threshold: {args.iou_threshold}")
        if args.pixel_size:
            print(f"  Pixel size: {args.pixel_size} µm")

        # Verify inputs
        if not pred_mask_dir.exists():
            raise FileNotFoundError(f"Predicted mask directory not found: {pred_mask_dir}")
        if not gt_manifest_path.exists():
            raise FileNotFoundError(f"Ground truth manifest not found: {gt_manifest_path}")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        per_image_dir = output_dir / "per_image"
        per_image_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Load ground truth manifest
        print("\nLoading ground truth manifest...")
        gt_manifest = pd.read_csv(gt_manifest_path)
        print(f"  Found {len(gt_manifest)} images")

        # Process each image
        print("\nProcessing images...")
        all_objects = []
        instance_metrics_list = []

        for idx, row in gt_manifest.iterrows():
            basename = Path(row["image_path"]).stem

            # Find predicted mask
            pred_mask_path = pred_mask_dir / f"{basename}_pred_mask.tif"
            if not pred_mask_path.exists():
                logging.warning(f"Predicted mask not found for {basename}, skipping")
                continue

            # Ground truth mask path
            if "mask_path" not in row or pd.isna(row["mask_path"]):
                logging.warning(f"No mask path specified for {basename}, skipping")
                continue
            gt_mask_path = data_root / row["mask_path"]
            if not gt_mask_path.exists():
                logging.warning(f"Ground truth mask not found: {gt_mask_path}, skipping")
                continue

            print(f"  [{idx + 1}/{len(gt_manifest)}] Processing {basename}...")

            # Process this image pair
            obj_props, inst_metrics, _, _ = process_image_pair(
                pred_mask_path=pred_mask_path,
                gt_mask_path=gt_mask_path,
                min_object_area=args.min_object_area,
                iou_threshold=args.iou_threshold,
                pixel_size=args.pixel_size,
            )

            # Save per-image object properties
            if len(obj_props) > 0:
                obj_props["image"] = basename
                per_image_csv = per_image_dir / f"{basename}_objects.csv"
                obj_props.to_csv(per_image_csv, index=False)
                all_objects.append(obj_props)

            # Record instance metrics
            inst_metrics["image"] = basename
            instance_metrics_list.append(inst_metrics)

            # Print per-image summary
            print(f"      Objects: {inst_metrics['n_pred']} predicted, {inst_metrics['n_gt']} ground truth")
            print(f"      TP={inst_metrics['tp']}, FP={inst_metrics['fp']}, FN={inst_metrics['fn']}")
            print(f"      Precision={inst_metrics['precision']:.3f}, Recall={inst_metrics['recall']:.3f}, F1={inst_metrics['f1']:.3f}")

        # Combine all results
        print("\nGenerating summary statistics...")

        # All objects CSV
        if all_objects:
            all_objects_df = pd.concat(all_objects, ignore_index=True)
            all_objects_path = output_dir / "all_objects.csv"
            all_objects_df.to_csv(all_objects_path, index=False)
            print(f"  ✓ Saved {len(all_objects_df)} object records to {all_objects_path}")
        else:
            all_objects_df = pd.DataFrame()
            logging.warning("No objects found in any image")

        # Instance evaluation CSV
        instance_eval_df = pd.DataFrame(instance_metrics_list)
        instance_eval_path = output_dir / "instance_eval.csv"
        instance_eval_df.to_csv(instance_eval_path, index=False)
        print(f"  ✓ Saved per-image instance metrics to {instance_eval_path}")

        # Summary statistics
        summary = {
            "total_images": len(instance_eval_df),
            "total_pred_objects": instance_eval_df["n_pred"].sum(),
            "total_gt_objects": instance_eval_df["n_gt"].sum(),
            "total_tp": instance_eval_df["tp"].sum(),
            "total_fp": instance_eval_df["fp"].sum(),
            "total_fn": instance_eval_df["fn"].sum(),
            "mean_precision": instance_eval_df["precision"].mean(),
            "mean_recall": instance_eval_df["recall"].mean(),
            "mean_f1": instance_eval_df["f1"].mean(),
            "mean_matched_iou": instance_eval_df["mean_matched_iou"].mean(),
        }

        summary_df = pd.DataFrame([summary])
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  ✓ Saved dataset summary to {summary_path}")

        # Create plots
        if len(all_objects_df) > 0:
            print("\nGenerating summary plots...")
            create_summary_plots(all_objects_df, instance_eval_df, plots_dir)
            print(f"  ✓ Saved plots to {plots_dir}")

        # Print final summary
        print("\n" + "=" * 80)
        print("OBJECT QUANTIFICATION COMPLETE")
        print("=" * 80)
        print("\nDataset-Level Summary:")
        print(f"  Images processed: {summary['total_images']}")
        print(f"  Total predicted objects: {summary['total_pred_objects']}")
        print(f"  Total ground truth objects: {summary['total_gt_objects']}")
        print(f"\nInstance-Level Metrics:")
        print(f"  True Positives (TP): {summary['total_tp']}")
        print(f"  False Positives (FP): {summary['total_fp']}")
        print(f"  False Negatives (FN): {summary['total_fn']}")
        print(f"  Mean Precision: {summary['mean_precision']:.4f}")
        print(f"  Mean Recall: {summary['mean_recall']:.4f}")
        print(f"  Mean F1 Score: {summary['mean_f1']:.4f}")
        print(f"  Mean Matched IoU: {summary['mean_matched_iou']:.4f}")

        if len(all_objects_df) > 0:
            print(f"\nMorphology Statistics:")
            print(f"  Area: {all_objects_df['area'].mean():.1f} ± {all_objects_df['area'].std():.1f}")
            print(f"  Equivalent Diameter: {all_objects_df['equivalent_diameter'].mean():.1f} ± {all_objects_df['equivalent_diameter'].std():.1f}")
            print(f"  Circularity: {all_objects_df['circularity'].mean():.3f} ± {all_objects_df['circularity'].std():.3f}")

        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - All objects: {output_dir / 'all_objects.csv'}")
        print(f"  - Instance metrics: {output_dir / 'instance_eval.csv'}")
        print(f"  - Summary: {output_dir / 'summary.csv'}")
        print(f"  - Per-image objects: {per_image_dir}")
        print(f"  - Plots: {plots_dir}")

        # Check success criteria
        print("\n" + "=" * 80)
        if summary['mean_f1'] >= 0.85:
            print("✅ SUCCESS: Detection F1 ≥ 0.85 - POC pipeline complete!")
        else:
            print(f"⚠️  Detection F1 = {summary['mean_f1']:.3f} < 0.85")
            print("   Consider adjusting min_object_area or iou_threshold")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logging.error(f"Object quantification failed: {e}", exc_info=True)
        return 1


def extract_metrics():
    """Extract morphology metrics from predicted masks without ground truth."""
    parser = argparse.ArgumentParser(
        description="Extract morphology metrics from predicted masks (no ground truth required)"
    )
    parser.add_argument(
        "--pred-mask-dir",
        required=True,
        help="Directory with predicted masks (e.g., inference/predictions/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for metrics CSV files",
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=100,
        help="Minimum object area in pixels (default: 100)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Pixel size in micrometers for physical unit conversion (optional)",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        from tifffile import imread
        from src.analysis.quantify import extract_objects, compute_object_properties

        print("\n" + "=" * 80)
        print("EXTRACT MORPHOLOGY METRICS FROM PREDICTIONS")
        print("=" * 80)

        # Parse arguments
        pred_mask_dir = Path(args.pred_mask_dir)
        output_dir = Path(args.output_dir)

        print(f"\nConfiguration:")
        print(f"  Predicted masks: {pred_mask_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Min object area: {args.min_object_area} px")
        if args.pixel_size:
            print(f"  Pixel size: {args.pixel_size} µm")

        # Verify inputs
        if not pred_mask_dir.exists():
            raise FileNotFoundError(f"Predicted mask directory not found: {pred_mask_dir}")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        per_image_dir = output_dir / "per_image"
        per_image_dir.mkdir(parents=True, exist_ok=True)

        # Find all predicted masks
        pred_mask_paths = sorted(pred_mask_dir.glob("*_pred_mask.tif"))
        if len(pred_mask_paths) == 0:
            raise FileNotFoundError(f"No *_pred_mask.tif files found in {pred_mask_dir}")

        print(f"\nFound {len(pred_mask_paths)} predicted masks")

        # Process each mask
        print("\nExtracting metrics...")
        all_objects = []

        for idx, pred_mask_path in enumerate(pred_mask_paths):
            basename = pred_mask_path.stem.replace("_pred_mask", "")
            print(f"  [{idx + 1}/{len(pred_mask_paths)}] Processing {basename}...")

            # Load predicted mask
            pred_mask = imread(pred_mask_path)

            # Extract objects
            labeled_mask, n_objects = extract_objects(pred_mask, min_area=args.min_object_area)

            if n_objects == 0:
                logging.warning(f"No objects found in {basename}")
                continue

            # Compute morphology metrics
            obj_props = compute_object_properties(labeled_mask, pixel_size=args.pixel_size)

            # Add image identifier
            obj_props["image"] = basename

            # Save per-image CSV
            per_image_csv = per_image_dir / f"{basename}_objects.csv"
            obj_props.to_csv(per_image_csv, index=False)

            all_objects.append(obj_props)

        # Combine all objects
        if len(all_objects) > 0:
            all_objects_df = pd.concat(all_objects, ignore_index=True)

            # Save combined CSV
            combined_csv = output_dir / "all_objects.csv"
            all_objects_df.to_csv(combined_csv, index=False)
            print(f"\n  ✓ Saved combined metrics to {combined_csv}")

            # Compute summary statistics
            summary_stats = all_objects_df.drop(columns=['image', 'object_id']).describe()
            summary_csv = output_dir / "summary_statistics.csv"
            summary_stats.to_csv(summary_csv)
            print(f"  ✓ Saved summary statistics to {summary_csv}")

            # Print summary
            print("\n" + "=" * 80)
            print("METRIC EXTRACTION COMPLETE")
            print("=" * 80)
            print(f"\nSummary:")
            print(f"  Images processed: {len(pred_mask_paths)}")
            print(f"  Total objects extracted: {len(all_objects_df)}")
            print(f"\nMorphology Statistics:")
            print(f"  Mean area: {all_objects_df['area'].mean():.2f}")
            print(f"  Mean circularity: {all_objects_df['circularity'].mean():.4f}")
            print(f"  Mean equivalent diameter: {all_objects_df['equivalent_diameter'].mean():.2f}")
            print(f"\nOutput files:")
            print(f"  Combined metrics: {combined_csv}")
            print(f"  Summary stats: {summary_csv}")
            print(f"  Per-image CSVs: {per_image_dir}/")
        else:
            logging.warning("No objects found in any image")

        return 0

    except Exception as e:
        logging.error(f"Metric extraction failed: {e}", exc_info=True)
        return 1


def run_cv():
    """Phase 6: Run cross-validation experiment."""
    parser = argparse.ArgumentParser(
        description="Run cross-validation experiment with configurable strategy"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to CV config YAML file (e.g., configs/cv_config.yaml)",
    )
    parser.add_argument(
        "--folds",
        help="Comma-separated list of specific folds to run (e.g., '0,1,2'). "
        "If not specified, all folds will be run.",
    )
    args = parser.parse_args()

    try:
        from src.training.cross_validation import run_cross_validation

        print("\n" + "=" * 80)
        print("PHASE 6: CROSS-VALIDATION")
        print("=" * 80)

        config_path = Path(args.config)
        print(f"\nLoading configuration from: {config_path}")

        # Parse folds argument if provided
        folds = None
        if args.folds:
            try:
                folds = [int(f.strip()) for f in args.folds.split(",")]
                print(f"Running specific folds: {folds}")
            except ValueError:
                print(f"Error: Invalid folds specification '{args.folds}'")
                print("Expected format: '0,1,2' (comma-separated integers)")
                return 1

        # Run cross-validation
        summary = run_cross_validation(config_path, folds=folds)

        # Return success
        return 0

    except Exception as e:
        logging.error(f"Cross-validation failed: {e}", exc_info=True)
        return 1


def review_predictions():
    """Interactive review tool for segmentation predictions."""
    parser = argparse.ArgumentParser(
        description="Interactive slideshow for reviewing predicted masks with flagging capability"
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing original images (e.g., data/working/images)",
    )
    parser.add_argument(
        "--pred-mask-dir",
        required=True,
        help="Directory containing predicted masks (e.g., inference/full_dataset_review)",
    )
    parser.add_argument(
        "--display-duration",
        type=float,
        default=3.0,
        help="Duration in seconds to display each view (default: 3.0)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Transparency of mask overlay, 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--output-flagged",
        help="Path to save flagged images list (optional)",
    )
    parser.add_argument(
        "--video-export",
        help="Export video instead of interactive review (useful for WSL/headless). Provide output video path (e.g., review.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second when using --video-export (default: 30)",
    )
    args = parser.parse_args()

    try:
        # Video export mode (for WSL/headless environments)
        if args.video_export:
            from src.visualization.review import export_review_video

            n_images = export_review_video(
                image_dir=args.image_dir,
                pred_mask_dir=args.pred_mask_dir,
                output_video=args.video_export,
                frame_duration=args.display_duration,
                overlay_alpha=args.overlay_alpha,
                fps=args.fps,
            )

            return 0

        # Interactive mode
        else:
            from src.visualization.review import run_review

            # Run review
            flagged = run_review(
                image_dir=args.image_dir,
                pred_mask_dir=args.pred_mask_dir,
                display_duration=args.display_duration,
                overlay_alpha=args.overlay_alpha,
                output_flagged=args.output_flagged,
            )

            return 0

    except Exception as e:
        logging.error(f"Review failed: {e}", exc_info=True)
        return 1
