"""
Manifest builder for creating CSV manifests from image folders.

This module provides functionality to scan input folders for TIFF images
and generate manifest CSV files suitable for the inference pipeline.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def scan_folder_for_images(
    input_folder: Path,
    extensions: tuple = (".tif", ".tiff"),
) -> List[Path]:
    """
    Scan a folder for image files with specified extensions.

    Args:
        input_folder: Path to folder containing images
        extensions: Tuple of file extensions to include (case-insensitive)

    Returns:
        List of image file paths, sorted alphabetically
    """
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if not input_folder.is_dir():
        raise ValueError(f"Path is not a directory: {input_folder}")

    # Find all matching files (case-insensitive)
    image_files = []
    for ext in extensions:
        image_files.extend(input_folder.glob(f"*{ext}"))
        image_files.extend(input_folder.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    image_files = sorted(set(image_files))

    logger.info(f"Found {len(image_files)} image files in {input_folder}")

    return image_files


def build_manifest(
    input_folder: Path,
    output_csv: Optional[Path] = None,
    extensions: tuple = (".tif", ".tiff"),
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> pd.DataFrame:
    """
    Build a manifest CSV from images in a folder.

    Creates a manifest with columns: basename, image_path, mask_path
    The mask_path column is left empty (for inference-only workflows).

    Args:
        input_folder: Path to folder containing images
        output_csv: Optional path to save manifest CSV
        extensions: Tuple of file extensions to include
        log_callback: Optional callback for logging (message, level)

    Returns:
        DataFrame with manifest data
    """
    input_folder = Path(input_folder)

    def log(msg: str, level: str = "INFO"):
        logger.log(getattr(logging, level), msg)
        if log_callback:
            log_callback(msg, level)

    log(f"Scanning folder: {input_folder}")

    # Find images
    image_files = scan_folder_for_images(input_folder, extensions)

    if len(image_files) == 0:
        log(f"No image files found with extensions {extensions}", "WARNING")
        return pd.DataFrame(columns=["basename", "image_path", "mask_path"])

    log(f"Found {len(image_files)} images")

    # Build manifest records
    records = []
    for image_path in image_files:
        basename = image_path.stem
        records.append({
            "basename": basename,
            "image_path": str(image_path.absolute()),
            "mask_path": "",  # Empty for inference-only
        })

    df = pd.DataFrame(records)

    # Optionally save to CSV
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log(f"Saved manifest to {output_csv}")

    return df


def build_manifest_with_masks(
    image_folder: Path,
    mask_folder: Path,
    output_csv: Optional[Path] = None,
    mask_suffix: str = "_mask",
    extensions: tuple = (".tif", ".tiff"),
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> pd.DataFrame:
    """
    Build a manifest CSV from paired image and mask folders.

    Args:
        image_folder: Path to folder containing images
        mask_folder: Path to folder containing masks
        output_csv: Optional path to save manifest CSV
        mask_suffix: Suffix added to image basename to find mask (default: "_mask")
        extensions: Tuple of file extensions to include
        log_callback: Optional callback for logging (message, level)

    Returns:
        DataFrame with manifest data (only includes images with matching masks)
    """
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)

    def log(msg: str, level: str = "INFO"):
        logger.log(getattr(logging, level), msg)
        if log_callback:
            log_callback(msg, level)

    log(f"Scanning image folder: {image_folder}")
    log(f"Scanning mask folder: {mask_folder}")

    # Find images
    image_files = scan_folder_for_images(image_folder, extensions)

    if len(image_files) == 0:
        log(f"No image files found", "WARNING")
        return pd.DataFrame(columns=["basename", "image_path", "mask_path"])

    # Build manifest records (only for images with matching masks)
    records = []
    missing_masks = []

    for image_path in image_files:
        basename = image_path.stem
        mask_basename = f"{basename}{mask_suffix}"

        # Look for mask with any valid extension
        mask_path = None
        for ext in extensions:
            candidate = mask_folder / f"{mask_basename}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
            candidate = mask_folder / f"{mask_basename}{ext.upper()}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is not None:
            records.append({
                "basename": basename,
                "image_path": str(image_path.absolute()),
                "mask_path": str(mask_path.absolute()),
            })
        else:
            missing_masks.append(basename)

    if missing_masks:
        log(f"Missing masks for {len(missing_masks)} images", "WARNING")

    log(f"Found {len(records)} image/mask pairs")

    df = pd.DataFrame(records)

    # Optionally save to CSV
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log(f"Saved manifest to {output_csv}")

    return df
