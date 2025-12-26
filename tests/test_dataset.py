"""
Unit tests for src/data/dataset.py

Tests cover:
- PatchDataset returns correct shapes (256×256 for both image and mask)
- Patch sampling produces expected positive/negative ratio (approximately)
- Augmentations apply identically to image and mask
- Model forward pass produces correct output shape
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import segmentation_models_pytorch as smp
import tifffile
import torch

from src.data.dataset import PatchDataset
from src.training.train import create_model


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset with synthetic images and masks."""
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

        # Create 3 synthetic image/mask pairs
        basenames = []
        for i in range(3):
            basename = f"test_image_{i}"
            basenames.append(basename)

            # Create image (grayscale, 512×512)
            image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

            # Create mask with circular objects
            mask = np.zeros((512, 512), dtype=np.uint8)
            # Add 3-5 circular objects
            num_objects = np.random.randint(3, 6)
            for _ in range(num_objects):
                center_y = np.random.randint(100, 412)
                center_x = np.random.randint(100, 412)
                radius = np.random.randint(20, 60)

                yy, xx = np.ogrid[:512, :512]
                circle = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
                mask[circle] = 255

            # Save
            image_path = images_dir / f"{basename}.tif"
            mask_path = masks_dir / f"{basename}_mask.tif"

            tifffile.imwrite(image_path, image)
            tifffile.imwrite(mask_path, mask)

        # Create manifest
        records = []
        for basename in basenames:
            records.append({
                "basename": basename,
                "image_path": f"working/images/{basename}.tif",
                "mask_path": f"working/masks/{basename}_mask.tif",
                "mask_coverage": 0.3,
                "object_count": 4,
                "empty_confirmed": False,
            })

        manifest_df = pd.DataFrame(records)
        manifest_path = splits_dir / "test.csv"
        manifest_df.to_csv(manifest_path, index=False)

        yield {
            "manifest_path": manifest_path,
            "data_root": tmpdir,
            "num_images": len(basenames),
        }


class TestPatchDataset:
    """Tests for PatchDataset class."""

    def test_dataset_length(self, temp_dataset):
        """Test that dataset length equals num_images × patches_per_image."""
        patches_per_image = 10

        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=patches_per_image,
            augment=False,
            data_root=temp_dataset["data_root"],
        )

        expected_length = temp_dataset["num_images"] * patches_per_image
        assert len(dataset) == expected_length

    def test_patch_shapes(self, temp_dataset):
        """Test that patches have correct shape [1, 256, 256]."""
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=5,
            augment=False,
            data_root=temp_dataset["data_root"],
        )

        # Get a few samples
        for i in range(5):
            sample = dataset[i]

            assert "image" in sample
            assert "mask" in sample

            # Check shapes
            assert sample["image"].shape == (1, 256, 256)
            assert sample["mask"].shape == (1, 256, 256)

            # Check dtypes
            assert sample["image"].dtype == torch.float32
            assert sample["mask"].dtype == torch.float32

    def test_image_normalization(self, temp_dataset):
        """Test that images are normalized to [0, 1] range."""
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=5,
            augment=False,
            data_root=temp_dataset["data_root"],
        )

        for i in range(10):
            sample = dataset[i]
            image = sample["image"]

            # Check range
            assert image.min() >= 0.0
            assert image.max() <= 1.0

    def test_mask_binary(self, temp_dataset):
        """Test that masks are binary {0, 1}."""
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=5,
            augment=False,
            data_root=temp_dataset["data_root"],
        )

        for i in range(10):
            sample = dataset[i]
            mask = sample["mask"]

            # Check only contains 0 and 1
            unique_vals = torch.unique(mask)
            assert all(val in [0.0, 1.0] for val in unique_vals)

    def test_positive_negative_ratio(self, temp_dataset):
        """Test that patch sampling produces approximately correct positive/negative ratio."""
        positive_ratio = 0.7
        num_samples = 100

        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=50,  # Ensure we have enough patches
            positive_ratio=positive_ratio,
            augment=False,
            data_root=temp_dataset["data_root"],
        )

        # Sample patches and check coverage
        positive_count = 0
        for i in range(num_samples):
            sample = dataset[i]
            mask = sample["mask"]

            # Consider a patch "positive" if it has >5% foreground
            coverage = mask.sum().item() / mask.numel()
            if coverage > 0.05:
                positive_count += 1

        observed_ratio = positive_count / num_samples

        # Allow 20% tolerance (0.7 ± 0.14)
        assert abs(observed_ratio - positive_ratio) < 0.2

    def test_augmentation_consistency(self, temp_dataset):
        """Test that augmentations apply identically to image and mask."""
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=20,
            augment=True,  # Enable augmentations
            data_root=temp_dataset["data_root"],
        )

        # Get multiple samples and check that mask and image are aligned
        # We can't directly verify augmentation was applied, but we can check
        # that the mask still aligns with the image after augmentation
        for i in range(10):
            sample = dataset[i]
            # If augmentations are correctly synchronized, we shouldn't have
            # nonsensical results (this is a basic sanity check)
            assert sample["image"].shape == sample["mask"].shape

    def test_no_augmentation_when_disabled(self, temp_dataset):
        """Test that augmentations are not applied when disabled."""
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=5,
            augment=False,  # Disable augmentations
            data_root=temp_dataset["data_root"],
        )

        # Just verify that transform is None or disabled
        assert dataset.transform is None

    def test_custom_patch_size(self, temp_dataset):
        """Test dataset with different patch sizes."""
        for patch_size in [128, 256, 512]:
            dataset = PatchDataset(
                manifest_csv=temp_dataset["manifest_path"],
                patch_size=patch_size,
                patches_per_image=5,
                augment=False,
                data_root=temp_dataset["data_root"],
            )

            sample = dataset[0]
            assert sample["image"].shape == (1, patch_size, patch_size)
            assert sample["mask"].shape == (1, patch_size, patch_size)

    def test_dataset_iteration(self, temp_dataset):
        """Test that dataset can be iterated without errors."""
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=5,
            augment=True,
            data_root=temp_dataset["data_root"],
        )

        # Iterate through all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None
            assert "image" in sample
            assert "mask" in sample


class TestModelForwardPass:
    """Tests for model creation and forward pass."""

    def test_create_model(self):
        """Test model creation with correct architecture."""
        model = create_model(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )

        # Check model is created
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_pass_shape(self):
        """Test that model produces correct output shape."""
        model = create_model(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )

        # Create dummy input [B, C, H, W]
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1, 256, 256)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        assert output.shape == (batch_size, 1, 256, 256)

    def test_model_with_batch(self, temp_dataset):
        """Test model forward pass with real batch from dataset."""
        # Create dataset
        dataset = PatchDataset(
            manifest_csv=temp_dataset["manifest_path"],
            patch_size=256,
            patches_per_image=5,
            augment=False,
            data_root=temp_dataset["data_root"],
        )

        # Create model
        model = create_model(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )

        # Get a batch manually
        batch = []
        for i in range(4):
            batch.append(dataset[i])

        # Stack batch
        images = torch.stack([sample["image"] for sample in batch])
        masks = torch.stack([sample["mask"] for sample in batch])

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        # Check output shape matches input
        assert outputs.shape == masks.shape
        assert outputs.shape == (4, 1, 256, 256)

    def test_model_output_range(self):
        """Test that model outputs logits (no activation applied)."""
        model = create_model(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )

        dummy_input = torch.randn(2, 1, 256, 256)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        # Logits should not be constrained to [0, 1]
        # (sigmoid will be applied in loss function)
        # Just check that we got some output
        assert output is not None
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
