"""
Patch-based dataset for spheroid segmentation.

Implements patch sampling with positive/negative balance and data augmentation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """
    PyTorch dataset for patch-based segmentation.

    Loads full images/masks from manifest and extracts patches on-the-fly.
    Supports positive-centered and negative sampling with configurable balance.

    Args:
        manifest_csv: Path to CSV manifest (train.csv, val.csv, or test.csv)
        patch_size: Size of patches to extract (default: 256)
        patches_per_image: Number of patches to sample per image per epoch (default: 20)
        positive_ratio: Fraction of positive-centered patches (default: 0.7)
        negative_threshold: Max mask coverage for negative patches (default: 0.05)
        max_jitter: Max jitter as fraction of patch size for positive patches (default: 0.25)
        augment: Whether to apply augmentations (default: True, ignored for val/test)
        data_root: Root directory for resolving relative paths in manifest
    """

    def __init__(
        self,
        manifest_csv: Path,
        patch_size: int = 256,
        patches_per_image: int = 20,
        positive_ratio: float = 0.7,
        negative_threshold: float = 0.05,
        max_jitter: float = 0.25,
        augment: bool = True,
        data_root: Optional[Path] = None,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_ratio = positive_ratio
        self.negative_threshold = negative_threshold
        self.max_jitter = max_jitter
        self.augment = augment

        # Load manifest
        self.manifest = pd.read_csv(manifest_csv)
        self.data_root = data_root or manifest_csv.parent.parent

        logger.info(f"Loaded manifest from {manifest_csv}: {len(self.manifest)} images")
        logger.info(f"Data root: {self.data_root}")
        logger.info(
            f"Patch config: size={patch_size}, per_image={patches_per_image}, "
            f"pos_ratio={positive_ratio}"
        )

        # Preload all images and masks into memory (small dataset)
        self._load_images()

        # Setup augmentation pipeline
        self.transform = self._build_transforms() if augment else None

    def _load_images(self) -> None:
        """Preload all images and masks into memory."""
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []

        for idx, row in self.manifest.iterrows():
            # Resolve paths
            image_path = self.data_root / row["image_path"]
            mask_path = self.data_root / row["mask_path"]

            # Load image
            image = tifffile.imread(image_path)

            # Convert RGB to grayscale
            if image.ndim == 3:
                # Take first channel (all channels are identical for grayscale TIFFs)
                image = image[:, :, 0]

            # Load mask
            mask = tifffile.imread(mask_path)

            # Verify dimensions match
            assert image.shape[:2] == mask.shape[:2], \
                f"Shape mismatch: {image.shape} vs {mask.shape} for {row['basename']}"

            self.images.append(image)
            self.masks.append(mask)

        logger.info(f"Preloaded {len(self.images)} image/mask pairs")

    def _build_transforms(self) -> A.Compose:
        """Build albumentations augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
        ])

    def __len__(self) -> int:
        """Total number of patches per epoch."""
        return len(self.manifest) * self.patches_per_image

    def _sample_positive_patch(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a patch centered on a random foreground pixel with jitter.

        Returns:
            Tuple of (image_patch, mask_patch) as uint8 arrays
        """
        H, W = mask.shape
        foreground_coords = np.argwhere(mask > 0)

        if len(foreground_coords) == 0:
            # Fallback to random patch if no foreground
            return self._sample_random_patch(image, mask)

        # Pick random foreground pixel
        center_y, center_x = foreground_coords[np.random.randint(len(foreground_coords))]

        # Apply jitter (up to 25% of patch size)
        max_jitter_px = int(self.patch_size * self.max_jitter)
        jitter_y = np.random.randint(-max_jitter_px, max_jitter_px + 1)
        jitter_x = np.random.randint(-max_jitter_px, max_jitter_px + 1)

        center_y += jitter_y
        center_x += jitter_x

        # Calculate patch coordinates
        half = self.patch_size // 2
        y1 = max(0, center_y - half)
        x1 = max(0, center_x - half)
        y2 = min(H, y1 + self.patch_size)
        x2 = min(W, x1 + self.patch_size)

        # Adjust if we hit the edge
        if y2 - y1 < self.patch_size:
            y1 = max(0, y2 - self.patch_size)
        if x2 - x1 < self.patch_size:
            x1 = max(0, x2 - self.patch_size)

        # Extract patch
        image_patch = image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        # Pad if necessary (shouldn't happen for images >= patch_size)
        if image_patch.shape[0] < self.patch_size or image_patch.shape[1] < self.patch_size:
            image_patch = self._pad_patch(image_patch)
            mask_patch = self._pad_patch(mask_patch)

        return image_patch, mask_patch

    def _sample_negative_patch(
        self, image: np.ndarray, mask: np.ndarray, max_attempts: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a patch with <negative_threshold mask coverage.

        Returns:
            Tuple of (image_patch, mask_patch) as uint8 arrays
        """
        H, W = mask.shape

        for _ in range(max_attempts):
            # Random location
            y1 = np.random.randint(0, max(1, H - self.patch_size + 1))
            x1 = np.random.randint(0, max(1, W - self.patch_size + 1))
            y2 = min(H, y1 + self.patch_size)
            x2 = min(W, x1 + self.patch_size)

            # Extract patch
            mask_patch = mask[y1:y2, x1:x2]

            # Check coverage
            coverage = np.sum(mask_patch > 0) / mask_patch.size

            if coverage < self.negative_threshold:
                image_patch = image[y1:y2, x1:x2]

                # Pad if necessary
                if image_patch.shape[0] < self.patch_size or image_patch.shape[1] < self.patch_size:
                    image_patch = self._pad_patch(image_patch)
                    mask_patch = self._pad_patch(mask_patch)

                return image_patch, mask_patch

        # Fallback to random patch if no suitable negative patch found
        return self._sample_random_patch(image, mask)

    def _sample_random_patch(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a completely random patch (fallback)."""
        H, W = mask.shape

        y1 = np.random.randint(0, max(1, H - self.patch_size + 1))
        x1 = np.random.randint(0, max(1, W - self.patch_size + 1))
        y2 = min(H, y1 + self.patch_size)
        x2 = min(W, x1 + self.patch_size)

        image_patch = image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        # Pad if necessary
        if image_patch.shape[0] < self.patch_size or image_patch.shape[1] < self.patch_size:
            image_patch = self._pad_patch(image_patch)
            mask_patch = self._pad_patch(mask_patch)

        return image_patch, mask_patch

    def _pad_patch(self, patch: np.ndarray) -> np.ndarray:
        """Pad patch to patch_size if needed."""
        if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
            return patch

        padded = np.zeros((self.patch_size, self.patch_size), dtype=patch.dtype)
        h, w = patch.shape[:2]
        padded[:h, :w] = patch
        return padded

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patch.

        Returns:
            Dict with keys:
                - "image": Float tensor [1, H, W] normalized to [0, 1]
                - "mask": Float tensor [1, H, W] with values {0, 1}
        """
        # Determine which image this patch comes from
        image_idx = idx // self.patches_per_image

        # Get image and mask
        image = self.images[image_idx]
        mask = self.masks[image_idx]

        # Decide whether to sample positive or negative
        is_positive = np.random.rand() < self.positive_ratio

        if is_positive:
            image_patch, mask_patch = self._sample_positive_patch(image, mask)
        else:
            image_patch, mask_patch = self._sample_negative_patch(image, mask)

        # Apply augmentations
        if self.transform is not None:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed["image"]
            mask_patch = transformed["mask"]

        # Normalize image to [0, 1]
        image_patch = image_patch.astype(np.float32) / 255.0

        # Normalize mask to binary {0, 1}
        mask_patch = (mask_patch > 0).astype(np.float32)

        # Convert to tensors and add channel dimension
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)    # [1, H, W]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }
