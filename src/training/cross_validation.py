"""Cross-validation orchestration."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.cross_validation import generate_kfold_splits, generate_loocv_splits
from src.data.dataset import PatchDataset
from src.training.train import load_config, save_config, train_model

logger = logging.getLogger(__name__)


def load_cv_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate CV configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["cv", "dataset", "model", "training", "output"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate CV-specific fields
    cv_config = config["cv"]
    if "strategy" not in cv_config:
        raise ValueError("CV config must specify 'strategy'")
    if "source_manifest" not in cv_config:
        raise ValueError("CV config must specify 'source_manifest'")

    strategy = cv_config["strategy"]
    if strategy not in ["leave_one_out", "k_fold"]:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    if strategy == "k_fold" and "n_folds" not in cv_config:
        raise ValueError("k_fold strategy requires 'n_folds' parameter")

    return config


def create_fold_config(
    base_cv_config: Dict[str, Any],
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Create a fold-specific training configuration.

    Merges base CV config with fold-specific paths.
    """
    fold_config = {
        "model": base_cv_config["model"],
        "data": {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "data_root": base_cv_config.get("data", {}).get("data_root", "data"),
        },
        "dataset": base_cv_config["dataset"],
        "training": base_cv_config["training"],
        "loss": base_cv_config.get("loss", {"bce_weight": 0.5, "dice_weight": 0.5}),
        "early_stopping": base_cv_config.get("early_stopping", {"enabled": False}),
        "lr_scheduler": base_cv_config.get("lr_scheduler", {"enabled": False}),
        "checkpointing": base_cv_config.get(
            "checkpointing", {"save_best": True, "save_final": True}
        ),
        "tensorboard": base_cv_config.get("tensorboard", {"enabled": True}),
        "output": {"run_dir": str(output_dir)},
        "validation": base_cv_config.get("validation", {"prediction_threshold": 0.5}),
    }

    return fold_config


def run_cross_validation(
    cv_config_path: Path, folds: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Run full cross-validation experiment.

    Args:
        cv_config_path: Path to CV configuration YAML
        folds: Optional list of specific fold indices to run (for debugging)

    Returns:
        Dictionary with aggregated results
    """
    logger.info(f"Starting cross-validation from config: {cv_config_path}")

    # Load and validate config
    config = load_cv_config(cv_config_path)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = Path(config["output"]["cv_dir"])
    if not cv_dir.is_absolute():
        cv_dir = Path("runs") / f"cv_{timestamp}"
    else:
        # If absolute path is specified, append timestamp to make it unique
        cv_dir = cv_dir.parent / f"{cv_dir.name}_{timestamp}"

    cv_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CV experiment directory: {cv_dir}")

    # Save config snapshot
    save_config(config, cv_dir / "cv_config.yaml")

    # Generate splits
    source_manifest = Path(config["cv"]["source_manifest"])
    strategy = config["cv"].get("strategy", "leave_one_out")
    seed = config["cv"].get("seed", 42)

    folds_dir = cv_dir / "folds"

    print(f"\nGenerating {strategy} splits...")
    if strategy == "leave_one_out":
        fold_paths = generate_loocv_splits(source_manifest, folds_dir, seed)
    elif strategy == "k_fold":
        n_folds = config["cv"].get("n_folds", 5)
        fold_paths = generate_kfold_splits(source_manifest, folds_dir, n_folds, seed)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    print(f"Generated {len(fold_paths)} folds")

    # Filter to specific folds if requested
    if folds is not None:
        print(f"\nRunning subset of folds: {folds}")
        fold_paths_filtered = []
        fold_indices_filtered = []
        for i in folds:
            if i < len(fold_paths):
                fold_paths_filtered.append(fold_paths[i])
                fold_indices_filtered.append(i)
            else:
                logger.warning(f"Fold {i} out of range (max: {len(fold_paths)-1}), skipping")
        fold_paths = fold_paths_filtered
        fold_indices = fold_indices_filtered
    else:
        fold_indices = list(range(len(fold_paths)))

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

    # Run training for each fold
    fold_results = []
    total_start_time = time.time()

    for fold_idx, (train_csv, val_csv) in zip(fold_indices, fold_paths):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{len(fold_indices)} (index {fold_idx})")
        print(f"Training on: {train_csv}")
        print(f"Validating on: {val_csv}")
        print(f"{'='*80}\n")

        fold_output_dir = folds_dir / f"fold_{fold_idx}"

        # Create fold-specific config
        fold_config = create_fold_config(
            base_cv_config=config,
            train_manifest=train_csv,
            val_manifest=val_csv,
            output_dir=fold_output_dir,
        )

        # Save fold config
        save_config(fold_config, fold_output_dir / "config.yaml")

        # Create datasets
        data_root = Path(fold_config["data"].get("data_root", "data"))
        patch_size = fold_config["dataset"]["patch_size"]
        patches_per_image = fold_config["dataset"]["patches_per_image"]
        positive_ratio = fold_config["dataset"]["positive_ratio"]

        print(f"Loading datasets...")
        train_dataset = PatchDataset(
            manifest_csv=train_csv,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            positive_ratio=positive_ratio,
            augment=fold_config["dataset"]["augmentation"]["enabled"],
            data_root=data_root,
        )

        val_dataset = PatchDataset(
            manifest_csv=val_csv,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            positive_ratio=positive_ratio,
            augment=False,  # No augmentation for validation
            data_root=data_root,
        )

        # Create data loaders
        batch_size = fold_config["training"]["batch_size"]
        num_workers = fold_config["training"]["num_workers"]
        pin_memory = fold_config["training"].get("pin_memory", True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory if device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory if device.type == "cuda" else False,
        )

        print(f"  Train patches: {len(train_dataset)}")
        print(f"  Val patches: {len(val_dataset)}")

        # Train fold
        fold_start_time = time.time()
        history = train_model(
            config=fold_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_dir=fold_output_dir,
            resume_checkpoint=None,
        )
        fold_training_time = (time.time() - fold_start_time) / 60.0  # minutes

        # Extract results
        best_val_dice = max(history["val_dice"]) if history["val_dice"] else 0.0
        best_epoch = (
            history["val_dice"].index(best_val_dice) + 1 if history["val_dice"] else 0
        )
        final_train_dice = history["train_dice"][-1] if history["train_dice"] else 0.0

        # Get validation image name(s)
        val_df = pd.read_csv(val_csv)
        val_images = val_df["basename"].tolist()
        val_image_str = val_images[0] if len(val_images) == 1 else f"{len(val_images)} images"

        fold_result = {
            "fold": fold_idx,
            "val_image": val_image_str,
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "final_train_dice": final_train_dice,
            "training_time_min": fold_training_time,
        }
        fold_results.append(fold_result)

        print(f"\nFold {fold_idx} complete:")
        print(f"  Best val Dice: {best_val_dice:.4f} (epoch {best_epoch})")
        print(f"  Training time: {fold_training_time:.1f} min")

    total_training_time = (time.time() - total_start_time) / 60.0  # minutes

    # Aggregate results
    results_dir = cv_dir / "results"
    results_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    logger.info(f"Saved fold metrics to {results_dir / 'fold_metrics.csv'}")

    summary = aggregate_results(fold_results)
    summary["experiment_dir"] = str(cv_dir)
    summary["total_training_time_min"] = total_training_time
    summary["cv_strategy"] = strategy
    summary["n_folds_completed"] = len(fold_results)

    with open(results_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info(f"Saved summary to {results_dir / 'summary.yaml'}")

    # Print summary
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"Folds completed: {len(fold_results)}")
    print(
        f"Val Dice: {summary['val_dice']['mean']:.4f} ± {summary['val_dice']['std']:.4f}"
    )
    print(f"  Min: {summary['val_dice']['min']:.4f}")
    print(f"  Max: {summary['val_dice']['max']:.4f}")
    print(f"Best fold: {summary['best_fold']} (Dice: {summary['best_fold_dice']:.4f})")
    print(f"Worst fold: {summary['worst_fold']} (Dice: {summary['worst_fold_dice']:.4f})")
    print(f"Total training time: {total_training_time:.1f} min")
    print(f"\nResults saved to: {results_dir}")
    print(f"{'='*80}\n")

    return summary


def aggregate_results(fold_results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics across folds."""
    df = pd.DataFrame(fold_results)

    def compute_stats(series):
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    # Find best and worst folds
    best_idx = df["best_val_dice"].idxmax()
    worst_idx = df["best_val_dice"].idxmin()

    summary = {
        "n_folds": len(fold_results),
        "val_dice": compute_stats(df["best_val_dice"]),
        "best_fold": int(df.loc[best_idx, "fold"]),
        "best_fold_dice": float(df.loc[best_idx, "best_val_dice"]),
        "worst_fold": int(df.loc[worst_idx, "fold"]),
        "worst_fold_dice": float(df.loc[worst_idx, "best_val_dice"]),
    }

    return summary
