"""
Dataset validation and split generation for Phase 1.

This module provides functionality to:
- Validate image/mask pairing and dimensions
- Compute QC metrics (mask coverage, object counts)
- Handle empty images
- Generate train/val/test splits
- Compute spheroid diameter statistics for patch size recommendations
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from skimage import measure

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.total_images = 0
        self.passed = 0
        self.failed = 0
        self.warnings: List[str] = []
        self.errors: List[str] = []


def validate_image_mask_pair(
    image_path: Path, mask_path: Path
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate a single image/mask pair.

    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file

    Returns:
        Tuple of (is_valid, error_message, metrics_dict)
        metrics_dict contains: mask_coverage, object_count, mean_diameter, image_shape
    """
    try:
        # Check if mask exists
        if not mask_path.exists():
            return False, f"Mask not found: {mask_path}", None

        # Load image and mask
        image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)

        # Handle RGB to grayscale conversion if needed
        if image.ndim == 3:
            # Convert RGB to grayscale (take first channel or average)
            image = image[:, :, 0] if image.shape[2] == 3 else image

        # Check dimensions match (H, W)
        if image.shape[:2] != mask.shape[:2]:
            return (
                False,
                f"Dimension mismatch: image {image.shape} vs mask {mask.shape}",
                None,
            )

        # Verify mask is binary or make it binary
        unique_vals = np.unique(mask)
        if not np.array_equal(unique_vals, [0, 255]) and not np.array_equal(
            unique_vals, [0]
        ):
            # Try thresholding
            if len(unique_vals) > 2:
                logger.warning(
                    f"Mask {mask_path} has {len(unique_vals)} unique values, "
                    "thresholding to binary"
                )
                mask = ((mask > 0) * 255).astype(np.uint8)

        # Compute mask coverage
        foreground_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        mask_coverage = foreground_pixels / total_pixels

        # Compute object count via connected components
        binary_mask = mask > 0
        labeled_mask = measure.label(binary_mask, connectivity=2)
        object_count = labeled_mask.max()

        # Compute mean spheroid diameter (equivalent diameter)
        mean_diameter = None
        if object_count > 0:
            props = measure.regionprops(labeled_mask)
            diameters = [prop.equivalent_diameter_area for prop in props]
            mean_diameter = np.mean(diameters) if diameters else None

        metrics = {
            "mask_coverage": mask_coverage,
            "object_count": int(object_count),
            "mean_diameter": float(mean_diameter) if mean_diameter else None,
            "image_shape": image.shape,
        }

        return True, None, metrics

    except Exception as e:
        return False, f"Error processing pair: {str(e)}", None


def validate_dataset(
    input_dir: Path, output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationResult]:
    """
    Validate all images in the dataset.

    Args:
        input_dir: Path to data/working/ directory containing images/ and masks/
        output_dir: Path to data/splits/ directory for output

    Returns:
        Tuple of (all_manifest_df, qc_report_df, validation_result)
    """
    logger.info(f"Validating dataset in {input_dir}")

    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Find all image files
    image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))
    logger.info(f"Found {len(image_files)} image files")

    result = ValidationResult()
    result.total_images = len(image_files)

    all_records = []
    qc_records = []

    for image_path in image_files:
        basename = image_path.stem
        mask_path = masks_dir / f"{basename}_mask{image_path.suffix}"

        # Validate pair
        is_valid, error_msg, metrics = validate_image_mask_pair(image_path, mask_path)

        if is_valid:
            result.passed += 1

            # Create manifest record
            record = {
                "basename": basename,
                "image_path": str(image_path.relative_to(input_dir.parent)),
                "mask_path": str(mask_path.relative_to(input_dir.parent)),
                "mask_coverage": metrics["mask_coverage"],
                "object_count": metrics["object_count"],
                "mean_diameter": metrics["mean_diameter"],
                "empty_confirmed": False,  # Default, user can update manually
                "date": None,
                "batch": None,
                "operator": None,
                "notes": None,
            }
            all_records.append(record)

            # Create QC record
            qc_record = record.copy()
            qc_record["validation_status"] = "PASS"
            qc_record["validation_message"] = ""

            # Check for warnings
            warnings = []
            if metrics["mask_coverage"] == 0 and not record["empty_confirmed"]:
                warnings.append("Empty mask - needs confirmation")
            if metrics["mask_coverage"] > 0.8:
                warnings.append("Unusually high mask coverage")
            if metrics["mask_coverage"] < 0.01 and metrics["mask_coverage"] > 0:
                warnings.append("Very low mask coverage")

            qc_record["warnings"] = "; ".join(warnings)
            qc_records.append(qc_record)

            if warnings:
                result.warnings.extend([f"{basename}: {w}" for w in warnings])

        else:
            result.failed += 1
            result.errors.append(f"{basename}: {error_msg}")

            # Create QC record for failed validation
            qc_record = {
                "basename": basename,
                "image_path": str(image_path.relative_to(input_dir.parent)),
                "mask_path": str(mask_path.relative_to(input_dir.parent)),
                "validation_status": "FAIL",
                "validation_message": error_msg,
                "warnings": "",
            }
            qc_records.append(qc_record)

    # Create DataFrames
    all_df = pd.DataFrame(all_records)
    qc_df = pd.DataFrame(qc_records)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifests
    all_csv_path = output_dir / "all.csv"
    qc_csv_path = output_dir / "qc_report.csv"

    all_df.to_csv(all_csv_path, index=False)
    qc_df.to_csv(qc_csv_path, index=False)

    logger.info(f"Saved manifest to {all_csv_path}")
    logger.info(f"Saved QC report to {qc_csv_path}")

    return all_df, qc_df, result


def print_validation_summary(
    all_df: pd.DataFrame, qc_df: pd.DataFrame, result: ValidationResult
) -> None:
    """Print console summary of validation results."""
    print("\n" + "=" * 80)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 80)

    # Overall stats
    print(f"\nTotal images: {result.total_images}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")

    if result.failed > 0:
        print("\nValidation Errors:")
        for error in result.errors:
            print(f"  ❌ {error}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings[:10]:  # Show first 10
            print(f"  ⚠️  {warning}")
        if len(result.warnings) > 10:
            print(f"  ... and {len(result.warnings) - 10} more")

    if len(all_df) > 0:
        # Coverage statistics
        print("\nMask Coverage Distribution:")
        coverage = all_df["mask_coverage"]
        print(f"  Min:    {coverage.min():.4f}")
        print(f"  Q1:     {coverage.quantile(0.25):.4f}")
        print(f"  Median: {coverage.median():.4f}")
        print(f"  Q3:     {coverage.quantile(0.75):.4f}")
        print(f"  Max:    {coverage.max():.4f}")

        # Object count statistics
        print("\nObject Count Distribution:")
        obj_counts = all_df["object_count"]
        print(f"  Min:    {obj_counts.min()}")
        print(f"  Median: {obj_counts.median():.0f}")
        print(f"  Max:    {obj_counts.max()}")

        # Empty image handling
        empty_unconfirmed = all_df[
            (all_df["mask_coverage"] == 0) & (~all_df["empty_confirmed"])
        ]
        if len(empty_unconfirmed) > 0:
            print(f"\n⚠️  {len(empty_unconfirmed)} empty images need confirmation:")
            for basename in empty_unconfirmed["basename"]:
                print(f"  - {basename}")
            print("\nTo confirm these are truly empty, set empty_confirmed=True in all.csv")

        # Spheroid diameter statistics for patch size recommendation
        diameters = all_df["mean_diameter"].dropna()
        if len(diameters) > 0:
            mean_diam = diameters.mean()
            recommended_patch = mean_diam * 2.5

            # Round to nearest power of 2 or convenient multiple
            if recommended_patch <= 256:
                patch_size = 256
            elif recommended_patch <= 384:
                patch_size = 384
            elif recommended_patch <= 512:
                patch_size = 512
            else:
                patch_size = int(2 ** np.ceil(np.log2(recommended_patch)))

            print("\nSpheroid Diameter Statistics (pixels):")
            print(f"  Mean diameter: {mean_diam:.1f}")
            print(f"  Median diameter: {diameters.median():.1f}")
            print(f"  Std: {diameters.std():.1f}")
            print(f"\n💡 Recommended patch size: {patch_size} pixels")
            print(f"   (Based on 2.5× mean diameter = {recommended_patch:.1f})")

    print("\n" + "=" * 80)


def make_splits(
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate train/val/test splits from the all.csv manifest.

    Args:
        manifest_path: Path to all.csv
        output_dir: Path to data/splits/ directory
        seed: Random seed for reproducibility
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify: Whether to stratify by mask coverage buckets

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Creating splits from {manifest_path}")
    logger.info(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Random seed: {seed}")

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    # Load manifest
    df = pd.read_csv(manifest_path)
    n_total = len(df)

    logger.info(f"Total images in manifest: {n_total}")

    # Set random seed
    np.random.seed(seed)

    if stratify and "mask_coverage" in df.columns:
        # Stratify by coverage buckets
        df["_coverage_bucket"] = pd.cut(
            df["mask_coverage"], bins=[0, 0.01, 0.1, 0.5, 1.0], labels=[0, 1, 2, 3]
        )

        train_dfs, val_dfs, test_dfs = [], [], []

        for bucket in df["_coverage_bucket"].unique():
            bucket_df = df[df["_coverage_bucket"] == bucket].copy()
            n_bucket = len(bucket_df)

            # Shuffle
            bucket_df = bucket_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            # Calculate split indices
            n_train = int(n_bucket * train_ratio)
            n_val = int(n_bucket * val_ratio)

            train_dfs.append(bucket_df[:n_train])
            val_dfs.append(bucket_df[n_train : n_train + n_val])
            test_dfs.append(bucket_df[n_train + n_val :])

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Drop temporary column
        train_df = train_df.drop("_coverage_bucket", axis=1)
        val_df = val_df.drop("_coverage_bucket", axis=1)
        test_df = test_df.drop("_coverage_bucket", axis=1)

    else:
        # Simple random split
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Calculate split indices
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_df = df[:n_train]
        val_df = df[n_train : n_train + n_val]
        test_df = df[n_train + n_val :]

    # Verify no overlap
    train_basenames = set(train_df["basename"])
    val_basenames = set(val_df["basename"])
    test_basenames = set(test_df["basename"])

    assert len(train_basenames & val_basenames) == 0, "Train/Val overlap detected!"
    assert len(train_basenames & test_basenames) == 0, "Train/Test overlap detected!"
    assert len(val_basenames & test_basenames) == 0, "Val/Test overlap detected!"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train split ({len(train_df)} images) to {train_path}")
    logger.info(f"Saved val split ({len(val_df)} images) to {val_path}")
    logger.info(f"Saved test split ({len(test_df)} images) to {test_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET SPLITS SUMMARY")
    print("=" * 80)
    print(f"\nRandom seed: {seed}")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):3d} images ({len(train_df)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_df):3d} images ({len(val_df)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_df):3d} images ({len(test_df)/n_total*100:.1f}%)")
    print(f"  Total: {n_total:3d} images")
    print("\n✓ No overlap between splits verified")
    print("=" * 80 + "\n")

    return train_df, val_df, test_df
