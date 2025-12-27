"""
Interactive review tool for segmentation predictions.

This module provides an interactive slideshow for reviewing predicted masks
overlaid on original images, with the ability to flag images for follow-up review.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


class InteractiveReview:
    """
    Interactive slideshow for reviewing segmentation predictions.

    Displays each image for a configurable duration, then shows the predicted mask
    overlaid on the image (white parts visible, black transparent). Users can click
    on images to flag them for follow-up review.

    Parameters
    ----------
    image_dir : Path
        Directory containing original images
    pred_mask_dir : Path
        Directory containing predicted masks (*_pred_mask.tif)
    display_duration : float, optional
        Duration in seconds to display each view (default: 3.0)
    overlay_alpha : float, optional
        Transparency of mask overlay (0.0-1.0, default: 0.5)

    Attributes
    ----------
    flagged_images : List[str]
        List of image basenames that were flagged for review
    """

    def __init__(
        self,
        image_dir: Path,
        pred_mask_dir: Path,
        display_duration: float = 3.0,
        overlay_alpha: float = 0.5,
    ):
        self.image_dir = Path(image_dir)
        self.pred_mask_dir = Path(pred_mask_dir)
        self.display_duration = display_duration
        self.overlay_alpha = overlay_alpha

        self.flagged_images: List[str] = []
        self.current_image: Optional[str] = None
        self.is_paused = False

        # Find all image pairs
        self.image_pairs = self._find_image_pairs()

        if not self.image_pairs:
            raise ValueError(
                f"No matching image pairs found in {image_dir} and {pred_mask_dir}"
            )

        logger.info(f"Found {len(self.image_pairs)} image pairs for review")

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find all matching image and predicted mask pairs.

        Returns
        -------
        List[Tuple[Path, Path]]
            List of (image_path, pred_mask_path) tuples
        """
        pairs = []

        # Find all images
        for image_path in sorted(self.image_dir.glob("*.tif")):
            basename = image_path.stem

            # Look for corresponding predicted mask
            pred_mask_path = self.pred_mask_dir / f"{basename}_pred_mask.tif"

            if pred_mask_path.exists():
                pairs.append((image_path, pred_mask_path))
            else:
                logger.warning(
                    f"No predicted mask found for {basename}, skipping"
                )

        return pairs

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load an image file.

        Parameters
        ----------
        path : Path
            Path to the image file

        Returns
        -------
        np.ndarray
            Loaded image array
        """
        img = tifffile.imread(path)

        # Convert RGB to grayscale if needed
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.mean(img, axis=-1).astype(img.dtype)

        return img

    def _create_overlay(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Create an RGBA overlay of the mask on the image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale image
        mask : np.ndarray
            Binary mask (0/255)

        Returns
        -------
        np.ndarray
            RGBA image with mask overlay
        """
        # Normalize image to 0-1
        if image.max() > 1.0:
            image_norm = image / 255.0
        else:
            image_norm = image

        # Create RGB image (grayscale replicated)
        rgb = np.stack([image_norm] * 3, axis=-1)

        # Create overlay: white mask areas with transparency
        # Normalize mask to 0-1
        mask_norm = (mask / 255.0) if mask.max() > 1.0 else mask

        # Create green overlay where mask is white
        overlay = rgb.copy()
        overlay[mask_norm > 0.5, 0] = 0.0  # R
        overlay[mask_norm > 0.5, 1] = 1.0  # G (green for mask)
        overlay[mask_norm > 0.5, 2] = 0.0  # B

        # Blend based on mask presence
        alpha_blend = mask_norm * self.overlay_alpha
        result = rgb * (1 - alpha_blend[:, :, np.newaxis]) + \
                 overlay * alpha_blend[:, :, np.newaxis]

        return result

    def _on_click(self, event):
        """Handle mouse click events to flag images."""
        if event.button == 1 and self.current_image:  # Left click
            if self.current_image not in self.flagged_images:
                self.flagged_images.append(self.current_image)
                logger.info(f"Flagged image: {self.current_image}")
                print(f"\n🚩 Flagged: {self.current_image}")
            else:
                self.flagged_images.remove(self.current_image)
                logger.info(f"Unflagged image: {self.current_image}")
                print(f"\n✓ Unflagged: {self.current_image}")

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == ' ':  # Space to pause/unpause
            self.is_paused = not self.is_paused
            if self.is_paused:
                print("\n⏸  Paused (press SPACE to continue, ESC to quit)")
            else:
                print("\n▶  Resumed")
        elif event.key == 'escape':  # ESC to quit
            plt.close('all')

    def run(self) -> List[str]:
        """
        Run the interactive review slideshow.

        Returns
        -------
        List[str]
            List of flagged image basenames
        """
        print("\n" + "=" * 80)
        print("INTERACTIVE SEGMENTATION REVIEW")
        print("=" * 80)
        print(f"\nReviewing {len(self.image_pairs)} images")
        print(f"Display duration: {self.display_duration}s per view")
        print("\nControls:")
        print("  - LEFT CLICK: Flag/unflag image for review")
        print("  - SPACE: Pause/resume slideshow")
        print("  - ESC: Exit review")
        print("=" * 80 + "\n")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.canvas.manager.set_window_title('Segmentation Review')

        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', self._on_click)
        fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Show in interactive mode
        plt.ion()

        try:
            for idx, (image_path, pred_mask_path) in enumerate(self.image_pairs):
                basename = image_path.stem
                self.current_image = basename

                # Load image and mask
                image = self._load_image(image_path)
                pred_mask = self._load_image(pred_mask_path)

                # Display 1: Original image
                ax.clear()
                ax.imshow(image, cmap='gray')
                ax.set_title(
                    f"[{idx + 1}/{len(self.image_pairs)}] {basename}\n"
                    f"{'🚩 FLAGGED' if basename in self.flagged_images else ''}",
                    fontsize=14,
                    color='red' if basename in self.flagged_images else 'black'
                )
                ax.axis('off')
                plt.draw()
                plt.pause(0.01)

                # Wait for display duration (checking for pause)
                start_time = time.time()
                while time.time() - start_time < self.display_duration:
                    if self.is_paused:
                        plt.pause(0.1)
                        start_time = time.time()  # Reset timer when unpaused
                    else:
                        plt.pause(0.1)

                    # Check if window was closed
                    if not plt.fignum_exists(fig.number):
                        logger.info("Review window closed by user")
                        return self.flagged_images

                # Display 2: Image with overlay
                overlay = self._create_overlay(image, pred_mask)
                ax.clear()
                ax.imshow(overlay)
                ax.set_title(
                    f"[{idx + 1}/{len(self.image_pairs)}] {basename} - PREDICTION OVERLAY\n"
                    f"{'🚩 FLAGGED' if basename in self.flagged_images else ''}",
                    fontsize=14,
                    color='red' if basename in self.flagged_images else 'black'
                )
                ax.axis('off')
                plt.draw()
                plt.pause(0.01)

                # Wait for display duration (checking for pause)
                start_time = time.time()
                while time.time() - start_time < self.display_duration:
                    if self.is_paused:
                        plt.pause(0.1)
                        start_time = time.time()  # Reset timer when unpaused
                    else:
                        plt.pause(0.1)

                    # Check if window was closed
                    if not plt.fignum_exists(fig.number):
                        logger.info("Review window closed by user")
                        return self.flagged_images

        finally:
            plt.ioff()
            plt.close('all')

        print("\n" + "=" * 80)
        print("REVIEW COMPLETE")
        print("=" * 80)
        if self.flagged_images:
            print(f"\n🚩 Flagged {len(self.flagged_images)} images for review:")
            for img in self.flagged_images:
                print(f"  - {img}")
        else:
            print("\n✓ No images flagged")
        print("=" * 80 + "\n")

        return self.flagged_images


def run_review(
    image_dir: str,
    pred_mask_dir: str,
    display_duration: float = 3.0,
    overlay_alpha: float = 0.5,
    output_flagged: Optional[str] = None,
) -> List[str]:
    """
    Run interactive review of segmentation predictions.

    This is the main entry point for the review tool. It can be called
    programmatically or from the command line.

    Parameters
    ----------
    image_dir : str
        Directory containing original images
    pred_mask_dir : str
        Directory containing predicted masks (*_pred_mask.tif)
    display_duration : float, optional
        Duration in seconds to display each view (default: 3.0)
    overlay_alpha : float, optional
        Transparency of mask overlay (0.0-1.0, default: 0.5)
    output_flagged : str, optional
        Path to save flagged images list (text file)

    Returns
    -------
    List[str]
        List of flagged image basenames

    Examples
    --------
    >>> # Review predictions from inference run
    >>> flagged = run_review(
    ...     image_dir="data/working/images",
    ...     pred_mask_dir="inference/full_dataset_review",
    ...     display_duration=2.5
    ... )

    >>> # Use in a script
    >>> from src.visualization.review import run_review
    >>> flagged = run_review("data/working/images", "inference/test_predictions")
    >>> if flagged:
    ...     print(f"Need to review: {flagged}")
    """
    reviewer = InteractiveReview(
        image_dir=Path(image_dir),
        pred_mask_dir=Path(pred_mask_dir),
        display_duration=display_duration,
        overlay_alpha=overlay_alpha,
    )

    flagged = reviewer.run()

    # Optionally save flagged list
    if output_flagged and flagged:
        output_path = Path(output_flagged)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for img in flagged:
                f.write(f"{img}\n")
        logger.info(f"Saved flagged images list to {output_path}")
        print(f"\n✓ Saved flagged list to: {output_path}")

    return flagged
