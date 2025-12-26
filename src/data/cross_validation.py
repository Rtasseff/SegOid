"""Cross-validation split generation utilities."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def generate_loocv_splits(
    manifest_path: Path, output_dir: Path, seed: int = 42
) -> List[Tuple[Path, Path]]:
    """
    Generate leave-one-out cross-validation splits.

    Args:
        manifest_path: Path to source manifest CSV (all images)
        output_dir: Directory to write fold manifests
        seed: Random seed (not used for LOOCV, but kept for API consistency)

    Returns:
        List of (train_csv_path, val_csv_path) tuples, one per fold
    """
    logger.info(f"Generating LOOCV splits from {manifest_path}")
    logger.info(f"Output directory: {output_dir}")

    df = pd.read_csv(manifest_path)
    n_images = len(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Total images: {n_images}")
    logger.info(f"Number of folds (LOOCV): {n_images}")

    fold_paths = []
    fold_metadata = {
        "strategy": "leave_one_out",
        "n_folds": n_images,
        "seed": seed,
        "source_manifest": str(manifest_path),
        "folds": [],
    }

    for fold_idx in range(n_images):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        # Leave one out
        val_df = df.iloc[[fold_idx]].reset_index(drop=True)
        train_df = df.drop(index=fold_idx).reset_index(drop=True)

        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        fold_paths.append((train_path, val_path))
        fold_metadata["folds"].append(
            {
                "fold": fold_idx,
                "val_image": df.iloc[fold_idx]["basename"],
                "n_train": len(train_df),
                "n_val": len(val_df),
            }
        )

        logger.info(
            f"  Fold {fold_idx}: val_image={df.iloc[fold_idx]['basename']}, "
            f"n_train={len(train_df)}, n_val={len(val_df)}"
        )

    # Save metadata
    meta_path = output_dir / "cv_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(fold_metadata, f, default_flow_style=False)

    logger.info(f"Saved CV metadata to {meta_path}")
    logger.info(f"Generated {len(fold_paths)} fold pairs")

    return fold_paths


def generate_kfold_splits(
    manifest_path: Path, output_dir: Path, n_folds: int = 5, seed: int = 42
) -> List[Tuple[Path, Path]]:
    """
    Generate k-fold cross-validation splits.

    Args:
        manifest_path: Path to source manifest CSV
        output_dir: Directory to write fold manifests
        n_folds: Number of folds
        seed: Random seed for shuffling

    Returns:
        List of (train_csv_path, val_csv_path) tuples, one per fold
    """
    logger.info(f"Generating {n_folds}-fold CV splits from {manifest_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {seed}")

    df = pd.read_csv(manifest_path)
    n_images = len(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_folds > n_images:
        raise ValueError(
            f"Number of folds ({n_folds}) cannot exceed number of images ({n_images})"
        )

    logger.info(f"Total images: {n_images}")
    logger.info(f"Number of folds (k-fold): {n_folds}")

    # Use sklearn KFold for consistent splitting
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_paths = []
    fold_metadata = {
        "strategy": "k_fold",
        "n_folds": n_folds,
        "seed": seed,
        "source_manifest": str(manifest_path),
        "folds": [],
    }

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(df)):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        # Split data
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)

        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        fold_paths.append((train_path, val_path))

        # Get validation image basenames
        val_images = val_df["basename"].tolist()
        fold_metadata["folds"].append(
            {
                "fold": fold_idx,
                "val_images": val_images,
                "n_train": len(train_df),
                "n_val": len(val_df),
            }
        )

        logger.info(
            f"  Fold {fold_idx}: val_images={val_images}, "
            f"n_train={len(train_df)}, n_val={len(val_df)}"
        )

    # Save metadata
    meta_path = output_dir / "cv_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(fold_metadata, f, default_flow_style=False)

    logger.info(f"Saved CV metadata to {meta_path}")
    logger.info(f"Generated {len(fold_paths)} fold pairs")

    return fold_paths
