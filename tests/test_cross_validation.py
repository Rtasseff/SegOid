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


# Tests for CV orchestration


@pytest.fixture
def sample_cv_config(tmp_path, sample_manifest):
    """Create a sample CV config YAML."""
    config_path = tmp_path / "cv_config.yaml"

    config = {
        "cv": {
            "strategy": "leave_one_out",
            "source_manifest": str(sample_manifest),
            "seed": 42,
        },
        "dataset": {
            "patch_size": 256,
            "patches_per_image": 2,
            "positive_ratio": 0.7,
            "negative_threshold": 0.05,
            "max_jitter": 0.25,
            "augmentation": {
                "enabled": False,
            },
        },
        "model": {
            "architecture": "unet",
            "encoder": "resnet18",
            "encoder_weights": "imagenet",
            "in_channels": 1,
            "classes": 1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "num_workers": 0,
            "pin_memory": False,
        },
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
        "early_stopping": {"enabled": False},
        "lr_scheduler": {"enabled": False},
        "checkpointing": {"save_best": True, "save_final": True},
        "tensorboard": {"enabled": False},
        "output": {"cv_dir": "cv_test"},
        "validation": {"prediction_threshold": 0.5},
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def test_load_cv_config_valid(sample_cv_config):
    """Test loading a valid CV config."""
    from src.training.cross_validation import load_cv_config

    config = load_cv_config(sample_cv_config)

    assert "cv" in config
    assert "dataset" in config
    assert "model" in config
    assert config["cv"]["strategy"] == "leave_one_out"


def test_load_cv_config_missing_required_field(tmp_path):
    """Test that loading config with missing field raises error."""
    from src.training.cross_validation import load_cv_config

    config_path = tmp_path / "bad_config.yaml"
    config = {"cv": {"strategy": "leave_one_out"}}  # Missing other required fields

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    with pytest.raises(ValueError, match="Missing required config field"):
        load_cv_config(config_path)


def test_load_cv_config_invalid_strategy(tmp_path, sample_manifest):
    """Test that invalid CV strategy raises error."""
    from src.training.cross_validation import load_cv_config

    config_path = tmp_path / "bad_config.yaml"
    config = {
        "cv": {"strategy": "invalid_strategy", "source_manifest": str(sample_manifest)},
        "dataset": {},
        "model": {},
        "training": {},
        "output": {},
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    with pytest.raises(ValueError, match="Unknown CV strategy"):
        load_cv_config(config_path)


def test_create_fold_config(tmp_path, sample_cv_config):
    """Test creation of fold-specific config."""
    from src.training.cross_validation import create_fold_config, load_cv_config

    cv_config = load_cv_config(sample_cv_config)

    train_manifest = tmp_path / "train.csv"
    val_manifest = tmp_path / "val.csv"
    output_dir = tmp_path / "fold_0"

    # Create dummy manifests
    pd.DataFrame({"basename": ["img_0"]}).to_csv(train_manifest, index=False)
    pd.DataFrame({"basename": ["img_1"]}).to_csv(val_manifest, index=False)

    fold_config = create_fold_config(
        cv_config, train_manifest, val_manifest, output_dir
    )

    # Check structure
    assert "model" in fold_config
    assert "data" in fold_config
    assert "dataset" in fold_config
    assert "training" in fold_config
    assert "output" in fold_config

    # Check fold-specific values
    assert fold_config["data"]["train_manifest"] == str(train_manifest)
    assert fold_config["data"]["val_manifest"] == str(val_manifest)
    assert fold_config["output"]["run_dir"] == str(output_dir)


def test_aggregate_results():
    """Test aggregation of fold results."""
    from src.training.cross_validation import aggregate_results

    fold_results = [
        {
            "fold": 0,
            "val_image": "img_0",
            "best_val_dice": 0.80,
            "best_epoch": 10,
            "final_train_dice": 0.85,
            "training_time_min": 30.0,
        },
        {
            "fold": 1,
            "val_image": "img_1",
            "best_val_dice": 0.75,
            "best_epoch": 8,
            "final_train_dice": 0.82,
            "training_time_min": 28.0,
        },
        {
            "fold": 2,
            "val_image": "img_2",
            "best_val_dice": 0.85,
            "best_epoch": 12,
            "final_train_dice": 0.88,
            "training_time_min": 32.0,
        },
    ]

    summary = aggregate_results(fold_results)

    # Check structure
    assert "n_folds" in summary
    assert "val_dice" in summary
    assert "best_fold" in summary
    assert "worst_fold" in summary

    # Check values
    assert summary["n_folds"] == 3
    assert summary["val_dice"]["mean"] == pytest.approx(0.80, abs=0.01)
    assert summary["val_dice"]["min"] == 0.75
    assert summary["val_dice"]["max"] == 0.85
    assert summary["best_fold"] == 2  # Highest dice
    assert summary["worst_fold"] == 1  # Lowest dice
