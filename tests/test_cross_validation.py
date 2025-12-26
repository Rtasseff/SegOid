"""Unit tests for cross-validation split generation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.cross_validation import generate_kfold_splits, generate_loocv_splits


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a sample manifest CSV with 6 images."""
    manifest_path = tmp_path / "all.csv"

    data = {
        "basename": [f"img_{i}" for i in range(6)],
        "image_path": [f"working/images/img_{i}.tif" for i in range(6)],
        "mask_path": [f"working/masks/img_{i}_mask.tif" for i in range(6)],
        "mask_coverage": [0.1, 0.2, 0.15, 0.3, 0.25, 0.18],
        "object_count": [5, 8, 6, 10, 9, 7],
    }

    df = pd.DataFrame(data)
    df.to_csv(manifest_path, index=False)

    return manifest_path


def test_generate_loocv_splits_creates_correct_number_of_folds(tmp_path, sample_manifest):
    """Test that LOOCV creates one fold per image."""
    output_dir = tmp_path / "loocv_splits"

    fold_paths = generate_loocv_splits(sample_manifest, output_dir, seed=42)

    # Should have 6 folds (one per image)
    assert len(fold_paths) == 6


def test_generate_loocv_splits_correct_train_val_sizes(tmp_path, sample_manifest):
    """Test that each LOOCV fold has correct train/val sizes."""
    output_dir = tmp_path / "loocv_splits"

    fold_paths = generate_loocv_splits(sample_manifest, output_dir, seed=42)

    for fold_idx, (train_path, val_path) in enumerate(fold_paths):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Each fold should have 5 training images and 1 validation image
        assert len(train_df) == 5, f"Fold {fold_idx}: Expected 5 train images"
        assert len(val_df) == 1, f"Fold {fold_idx}: Expected 1 val image"


def test_generate_loocv_splits_no_overlap(tmp_path, sample_manifest):
    """Test that train and val sets don't overlap for each fold."""
    output_dir = tmp_path / "loocv_splits"

    fold_paths = generate_loocv_splits(sample_manifest, output_dir, seed=42)

    for fold_idx, (train_path, val_path) in enumerate(fold_paths):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        train_basenames = set(train_df["basename"])
        val_basenames = set(val_df["basename"])

        # No overlap
        overlap = train_basenames & val_basenames
        assert len(overlap) == 0, f"Fold {fold_idx}: Found overlap {overlap}"


def test_generate_loocv_splits_all_images_used_as_val(tmp_path, sample_manifest):
    """Test that each image appears as validation exactly once across all folds."""
    output_dir = tmp_path / "loocv_splits"

    fold_paths = generate_loocv_splits(sample_manifest, output_dir, seed=42)

    val_images = []
    for train_path, val_path in fold_paths:
        val_df = pd.read_csv(val_path)
        val_images.extend(val_df["basename"].tolist())

    # Should have 6 unique validation images (one per fold)
    assert len(val_images) == 6
    assert len(set(val_images)) == 6


def test_generate_loocv_splits_creates_metadata(tmp_path, sample_manifest):
    """Test that LOOCV creates cv_meta.yaml with correct structure."""
    output_dir = tmp_path / "loocv_splits"

    generate_loocv_splits(sample_manifest, output_dir, seed=42)

    meta_path = output_dir / "cv_meta.yaml"
    assert meta_path.exists(), "cv_meta.yaml not created"

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    assert meta["strategy"] == "leave_one_out"
    assert meta["n_folds"] == 6
    assert meta["seed"] == 42
    assert len(meta["folds"]) == 6

    # Check each fold metadata
    for fold_idx, fold_info in enumerate(meta["folds"]):
        assert fold_info["fold"] == fold_idx
        assert "val_image" in fold_info
        assert fold_info["n_train"] == 5
        assert fold_info["n_val"] == 1


def test_generate_kfold_splits_creates_correct_number_of_folds(tmp_path, sample_manifest):
    """Test that k-fold creates requested number of folds."""
    output_dir = tmp_path / "kfold_splits"

    fold_paths = generate_kfold_splits(sample_manifest, output_dir, n_folds=3, seed=42)

    # Should have 3 folds
    assert len(fold_paths) == 3


def test_generate_kfold_splits_correct_train_val_sizes(tmp_path, sample_manifest):
    """Test that k-fold splits have approximately correct sizes."""
    output_dir = tmp_path / "kfold_splits"

    fold_paths = generate_kfold_splits(sample_manifest, output_dir, n_folds=3, seed=42)

    for fold_idx, (train_path, val_path) in enumerate(fold_paths):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # For 6 images with 3 folds, each fold should have 2 val images and 4 train images
        assert len(train_df) == 4, f"Fold {fold_idx}: Expected 4 train images"
        assert len(val_df) == 2, f"Fold {fold_idx}: Expected 2 val images"


def test_generate_kfold_splits_no_overlap(tmp_path, sample_manifest):
    """Test that train and val sets don't overlap for each fold."""
    output_dir = tmp_path / "kfold_splits"

    fold_paths = generate_kfold_splits(sample_manifest, output_dir, n_folds=3, seed=42)

    for fold_idx, (train_path, val_path) in enumerate(fold_paths):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        train_basenames = set(train_df["basename"])
        val_basenames = set(val_df["basename"])

        # No overlap
        overlap = train_basenames & val_basenames
        assert len(overlap) == 0, f"Fold {fold_idx}: Found overlap {overlap}"


def test_generate_kfold_splits_all_images_used_as_val(tmp_path, sample_manifest):
    """Test that each image appears as validation exactly once across all folds."""
    output_dir = tmp_path / "kfold_splits"

    fold_paths = generate_kfold_splits(sample_manifest, output_dir, n_folds=3, seed=42)

    val_images = []
    for train_path, val_path in fold_paths:
        val_df = pd.read_csv(val_path)
        val_images.extend(val_df["basename"].tolist())

    # Should have 6 validation images total (each image in exactly one fold)
    assert len(val_images) == 6
    assert len(set(val_images)) == 6


def test_generate_kfold_splits_creates_metadata(tmp_path, sample_manifest):
    """Test that k-fold creates cv_meta.yaml with correct structure."""
    output_dir = tmp_path / "kfold_splits"

    generate_kfold_splits(sample_manifest, output_dir, n_folds=3, seed=42)

    meta_path = output_dir / "cv_meta.yaml"
    assert meta_path.exists(), "cv_meta.yaml not created"

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    assert meta["strategy"] == "k_fold"
    assert meta["n_folds"] == 3
    assert meta["seed"] == 42
    assert len(meta["folds"]) == 3

    # Check each fold metadata
    for fold_idx, fold_info in enumerate(meta["folds"]):
        assert fold_info["fold"] == fold_idx
        assert "val_images" in fold_info
        assert fold_info["n_train"] == 4
        assert fold_info["n_val"] == 2


def test_generate_kfold_splits_reproducible(tmp_path, sample_manifest):
    """Test that k-fold splits are reproducible with same seed."""
    output_dir_1 = tmp_path / "kfold_splits_1"
    output_dir_2 = tmp_path / "kfold_splits_2"

    fold_paths_1 = generate_kfold_splits(
        sample_manifest, output_dir_1, n_folds=3, seed=42
    )
    fold_paths_2 = generate_kfold_splits(
        sample_manifest, output_dir_2, n_folds=3, seed=42
    )

    # Check that same images are in same folds
    for (train_path_1, val_path_1), (train_path_2, val_path_2) in zip(
        fold_paths_1, fold_paths_2
    ):
        train_df_1 = pd.read_csv(train_path_1)
        train_df_2 = pd.read_csv(train_path_2)

        val_df_1 = pd.read_csv(val_path_1)
        val_df_2 = pd.read_csv(val_path_2)

        # Same validation images
        assert set(val_df_1["basename"]) == set(val_df_2["basename"])
        # Same training images
        assert set(train_df_1["basename"]) == set(train_df_2["basename"])


def test_generate_kfold_splits_raises_error_if_too_many_folds(
    tmp_path, sample_manifest
):
    """Test that k-fold raises error if n_folds > n_images."""
    output_dir = tmp_path / "kfold_splits"

    with pytest.raises(ValueError, match="cannot exceed number of images"):
        generate_kfold_splits(sample_manifest, output_dir, n_folds=10, seed=42)
