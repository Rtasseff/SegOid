"""
Unit tests for src/data/validate.py

Tests cover:
- Image/mask pairing detection (valid and missing cases)
- Dimension mismatch detection
- Mask coverage calculation
- Object count calculation
- Split ratios are approximately correct
- Deterministic seed produces identical splits
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from src.data.validate import (
    validate_image_mask_pair,
    validate_dataset,
    make_splits,
)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create directory structure
        working_dir = tmpdir / "working"
        images_dir = working_dir / "images"
        masks_dir = working_dir / "masks"
        splits_dir = tmpdir / "splits"

        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)

        yield {
            "working": working_dir,
            "images": images_dir,
            "masks": masks_dir,
            "splits": splits_dir,
        }


def create_image_mask_pair(
    images_dir: Path,
    masks_dir: Path,
    basename: str,
    shape: tuple = (100, 100),
    mask_coverage: float = 0.5,
    num_objects: int = 1,
):
    """Helper to create a synthetic image/mask pair."""
    # Create random grayscale image
    image = np.random.randint(0, 256, shape, dtype=np.uint8)

    # Create mask with specified coverage
    mask = np.zeros(shape, dtype=np.uint8)
    if mask_coverage > 0:
        # Create circular objects
        y_center, x_center = shape[0] // 2, shape[1] // 2
        radius = int(np.sqrt(shape[0] * shape[1] * mask_coverage / (num_objects * np.pi)))

        for i in range(num_objects):
            # Offset each object slightly
            offset = i * radius * 2
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            circle = (yy - y_center) ** 2 + (xx - x_center - offset) ** 2 <= radius**2
            mask[circle] = 255

    # Save files
    image_path = images_dir / f"{basename}.tif"
    mask_path = masks_dir / f"{basename}_mask.tif"

    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)

    return image_path, mask_path


class TestValidateImageMaskPair:
    """Tests for validate_image_mask_pair function."""

    def test_valid_pair(self, temp_dataset_dir):
        """Test validation of a valid image/mask pair."""
        image_path, mask_path = create_image_mask_pair(
            temp_dataset_dir["images"],
            temp_dataset_dir["masks"],
            "test_image",
            shape=(100, 100),
            mask_coverage=0.3,
        )

        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        assert is_valid is True
        assert error_msg is None
        assert metrics is not None
        assert "mask_coverage" in metrics
        assert "object_count" in metrics
        assert 0 <= metrics["mask_coverage"] <= 1

    def test_missing_mask(self, temp_dataset_dir):
        """Test detection of missing mask file."""
        # Create only image, no mask
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        image_path = temp_dataset_dir["images"] / "no_mask.tif"
        tifffile.imwrite(image_path, image)

        mask_path = temp_dataset_dir["masks"] / "no_mask_mask.tif"

        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        assert is_valid is False
        assert "not found" in error_msg.lower()
        assert metrics is None

    def test_dimension_mismatch(self, temp_dataset_dir):
        """Test detection of dimension mismatch between image and mask."""
        # Create image and mask with different sizes
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)

        image_path = temp_dataset_dir["images"] / "mismatch.tif"
        mask_path = temp_dataset_dir["masks"] / "mismatch_mask.tif"

        tifffile.imwrite(image_path, image)
        tifffile.imwrite(mask_path, mask)

        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        assert is_valid is False
        assert "mismatch" in error_msg.lower()
        assert metrics is None

    def test_mask_coverage_calculation(self, temp_dataset_dir):
        """Test accurate calculation of mask coverage."""
        shape = (100, 100)
        expected_coverage = 0.25

        image_path, mask_path = create_image_mask_pair(
            temp_dataset_dir["images"],
            temp_dataset_dir["masks"],
            "coverage_test",
            shape=shape,
            mask_coverage=expected_coverage,
        )

        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        assert is_valid is True
        # Allow some tolerance due to rounding in circle creation
        assert abs(metrics["mask_coverage"] - expected_coverage) < 0.1

    def test_object_count(self, temp_dataset_dir):
        """Test object count via connected components."""
        image_path, mask_path = create_image_mask_pair(
            temp_dataset_dir["images"],
            temp_dataset_dir["masks"],
            "objects_test",
            shape=(200, 200),
            mask_coverage=0.3,
            num_objects=1,
        )

        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        assert is_valid is True
        assert metrics["object_count"] >= 1  # Should detect at least 1 object

    def test_empty_mask(self, temp_dataset_dir):
        """Test handling of completely empty mask (all zeros)."""
        image_path, mask_path = create_image_mask_pair(
            temp_dataset_dir["images"],
            temp_dataset_dir["masks"],
            "empty_test",
            shape=(100, 100),
            mask_coverage=0.0,
        )

        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        assert is_valid is True
        assert metrics["mask_coverage"] == 0.0
        assert metrics["object_count"] == 0


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_validate_complete_dataset(self, temp_dataset_dir):
        """Test validation of a complete dataset with multiple images."""
        # Create 5 image/mask pairs
        for i in range(5):
            create_image_mask_pair(
                temp_dataset_dir["images"],
                temp_dataset_dir["masks"],
                f"image_{i}",
                mask_coverage=0.3,
            )

        all_df, qc_df, result = validate_dataset(
            temp_dataset_dir["working"], temp_dataset_dir["splits"]
        )

        assert result.total_images == 5
        assert result.passed == 5
        assert result.failed == 0
        assert len(all_df) == 5
        assert len(qc_df) == 5

        # Check required columns exist
        required_cols = [
            "basename",
            "image_path",
            "mask_path",
            "mask_coverage",
            "object_count",
            "empty_confirmed",
        ]
        for col in required_cols:
            assert col in all_df.columns

    def test_validate_with_missing_masks(self, temp_dataset_dir):
        """Test validation when some masks are missing."""
        # Create 3 complete pairs
        for i in range(3):
            create_image_mask_pair(
                temp_dataset_dir["images"],
                temp_dataset_dir["masks"],
                f"good_{i}",
            )

        # Create 2 images without masks
        for i in range(2):
            image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            tifffile.imwrite(temp_dataset_dir["images"] / f"bad_{i}.tif", image)

        all_df, qc_df, result = validate_dataset(
            temp_dataset_dir["working"], temp_dataset_dir["splits"]
        )

        assert result.total_images == 5
        assert result.passed == 3
        assert result.failed == 2
        assert len(all_df) == 3  # Only valid pairs in all.csv
        assert len(qc_df) == 5  # All images in QC report

    def test_output_files_created(self, temp_dataset_dir):
        """Test that output CSV files are created."""
        # Create a single pair
        create_image_mask_pair(
            temp_dataset_dir["images"],
            temp_dataset_dir["masks"],
            "test",
        )

        validate_dataset(temp_dataset_dir["working"], temp_dataset_dir["splits"])

        # Check files exist
        assert (temp_dataset_dir["splits"] / "all.csv").exists()
        assert (temp_dataset_dir["splits"] / "qc_report.csv").exists()


class TestMakeSplits:
    """Tests for make_splits function."""

    def test_split_ratios(self, temp_dataset_dir):
        """Test that split ratios are approximately correct."""
        # Create manifest with 20 images
        records = []
        for i in range(20):
            records.append(
                {
                    "basename": f"image_{i}",
                    "image_path": f"working/images/image_{i}.tif",
                    "mask_path": f"working/masks/image_{i}_mask.tif",
                    "mask_coverage": 0.3,
                    "object_count": 1,
                    "empty_confirmed": False,
                }
            )

        manifest_df = pd.DataFrame(records)
        manifest_path = temp_dataset_dir["splits"] / "all.csv"
        manifest_df.to_csv(manifest_path, index=False)

        # Make splits with default ratios (0.6, 0.2, 0.2)
        train_df, val_df, test_df = make_splits(
            manifest_path,
            temp_dataset_dir["splits"],
            seed=42,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Check counts (with some tolerance for rounding)
        assert len(train_df) == 12  # 60%
        assert len(val_df) == 4  # 20%
        assert len(test_df) == 4  # 20%
        assert len(train_df) + len(val_df) + len(test_df) == 20

    def test_no_overlap_between_splits(self, temp_dataset_dir):
        """Test that there is no overlap between train/val/test splits."""
        # Create manifest
        records = []
        for i in range(10):
            records.append(
                {
                    "basename": f"image_{i}",
                    "image_path": f"working/images/image_{i}.tif",
                    "mask_path": f"working/masks/image_{i}_mask.tif",
                    "mask_coverage": 0.3,
                    "object_count": 1,
                    "empty_confirmed": False,
                }
            )

        manifest_df = pd.DataFrame(records)
        manifest_path = temp_dataset_dir["splits"] / "all.csv"
        manifest_df.to_csv(manifest_path, index=False)

        train_df, val_df, test_df = make_splits(
            manifest_path, temp_dataset_dir["splits"], seed=42
        )

        # Check no overlap
        train_basenames = set(train_df["basename"])
        val_basenames = set(val_df["basename"])
        test_basenames = set(test_df["basename"])

        assert len(train_basenames & val_basenames) == 0
        assert len(train_basenames & test_basenames) == 0
        assert len(val_basenames & test_basenames) == 0

    def test_deterministic_seed(self, temp_dataset_dir):
        """Test that same seed produces identical splits."""
        # Create manifest
        records = []
        for i in range(15):
            records.append(
                {
                    "basename": f"image_{i}",
                    "image_path": f"working/images/image_{i}.tif",
                    "mask_path": f"working/masks/image_{i}_mask.tif",
                    "mask_coverage": 0.3,
                    "object_count": 1,
                    "empty_confirmed": False,
                }
            )

        manifest_df = pd.DataFrame(records)
        manifest_path = temp_dataset_dir["splits"] / "all.csv"
        manifest_df.to_csv(manifest_path, index=False)

        # Generate splits twice with same seed
        train_df1, val_df1, test_df1 = make_splits(
            manifest_path, temp_dataset_dir["splits"], seed=42
        )

        # Need to reload manifest and re-split
        train_df2, val_df2, test_df2 = make_splits(
            manifest_path, temp_dataset_dir["splits"], seed=42
        )

        # Check identical splits
        assert set(train_df1["basename"]) == set(train_df2["basename"])
        assert set(val_df1["basename"]) == set(val_df2["basename"])
        assert set(test_df1["basename"]) == set(test_df2["basename"])

    def test_different_seeds_produce_different_splits(self, temp_dataset_dir):
        """Test that different seeds produce different splits."""
        # Create manifest
        records = []
        for i in range(15):
            records.append(
                {
                    "basename": f"image_{i}",
                    "image_path": f"working/images/image_{i}.tif",
                    "mask_path": f"working/masks/image_{i}_mask.tif",
                    "mask_coverage": 0.3,
                    "object_count": 1,
                    "empty_confirmed": False,
                }
            )

        manifest_df = pd.DataFrame(records)
        manifest_path = temp_dataset_dir["splits"] / "all.csv"
        manifest_df.to_csv(manifest_path, index=False)

        # Generate splits with different seeds
        train_df1, _, _ = make_splits(manifest_path, temp_dataset_dir["splits"], seed=42)
        train_df2, _, _ = make_splits(manifest_path, temp_dataset_dir["splits"], seed=123)

        # Splits should be different (with high probability)
        assert set(train_df1["basename"]) != set(train_df2["basename"])

    def test_split_files_created(self, temp_dataset_dir):
        """Test that split CSV files are created."""
        # Create minimal manifest
        records = [
            {
                "basename": f"image_{i}",
                "image_path": f"working/images/image_{i}.tif",
                "mask_path": f"working/masks/image_{i}_mask.tif",
                "mask_coverage": 0.3,
                "object_count": 1,
                "empty_confirmed": False,
            }
            for i in range(10)
        ]

        manifest_df = pd.DataFrame(records)
        manifest_path = temp_dataset_dir["splits"] / "all.csv"
        manifest_df.to_csv(manifest_path, index=False)

        make_splits(manifest_path, temp_dataset_dir["splits"], seed=42)

        # Check files exist
        assert (temp_dataset_dir["splits"] / "train.csv").exists()
        assert (temp_dataset_dir["splits"] / "val.csv").exists()
        assert (temp_dataset_dir["splits"] / "test.csv").exists()
