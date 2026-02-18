"""
Visualize GT vs Predicted masks overlaid on image for a 1.1 µm/px image.

Produces a figure with 3 panels:
  1. Image + GT mask (green)
  2. Image + Predicted mask (red)
  3. GT (green) + Pred (red) overlay — overlap appears yellow

Each object is labeled with its effective diameter in µm.

Usage:
    source .venv/bin/activate
    python scripts/visualize_gt_vs_pred.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure
from matplotlib.patheffects import withStroke

from src.analysis.quantify import extract_objects, compute_object_properties

# --- Config ---
IMAGE_NAME = "GFR_1_1"
IMAGE_PATH = Path("data/working_110/images/GFR_1_1.tif")
GT_MASK_PATH = Path("data/working_110/masks/GFR_1_1_mask.tif")
PRED_MASK_PATH = Path("scripts/_size_analysis_preds/GFR_1_1_pred_mask.tif")
PIXEL_SIZE = 1.10
MIN_AREA = 100
OUTPUT_PATH = Path("scripts/gt_vs_pred_overlay.png")


def mask_to_rgba(mask_binary, color, alpha=0.35):
    """Create an RGBA overlay from a binary mask."""
    h, w = mask_binary.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[mask_binary > 0] = [*color, alpha]
    return overlay


def normalize_image(image):
    """Normalize image to 0-1 float for display."""
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    else:
        img = np.zeros_like(img)
    return img


def label_objects(ax, labeled_mask, pixel_size, color, offset=(0, 0)):
    """Label each object with its effective diameter."""
    props = measure.regionprops(labeled_mask)
    for prop in props:
        if prop.area < MIN_AREA:
            continue
        diam_um = prop.equivalent_diameter_area * pixel_size
        cy, cx = prop.centroid
        ax.text(
            cx + offset[0], cy + offset[1],
            f"{diam_um:.0f}",
            fontsize=4, fontweight="bold",
            color=color,
            ha="center", va="center",
            path_effects=[
                withStroke(linewidth=1.5, foreground="black")
            ],
        )


def main():
    print(f"Loading {IMAGE_NAME}...")
    image = tifffile.imread(IMAGE_PATH)
    gt_mask = tifffile.imread(GT_MASK_PATH)
    pred_mask = tifffile.imread(PRED_MASK_PATH)

    print(f"  Image shape: {image.shape}")
    print(f"  GT mask shape: {gt_mask.shape}")
    print(f"  Pred mask shape: {pred_mask.shape}")

    img_norm = normalize_image(image)

    # Extract labeled objects
    gt_labeled, n_gt = extract_objects(gt_mask, min_area=MIN_AREA)
    pred_labeled, n_pred = extract_objects(pred_mask, min_area=MIN_AREA)
    print(f"  GT objects: {n_gt}, Pred objects: {n_pred}")

    gt_binary = gt_labeled > 0
    pred_binary = pred_labeled > 0

    # Compute summary stats for title
    gt_props_df = compute_object_properties(gt_labeled, pixel_size=PIXEL_SIZE)
    pred_props_df = compute_object_properties(pred_labeled, pixel_size=PIXEL_SIZE)

    gt_mean_diam = gt_props_df["equivalent_diameter_um"].mean()
    pred_mean_diam = pred_props_df["equivalent_diameter_um"].mean()
    gt_mean_area = gt_props_df["area_um2"].mean()
    pred_mean_area = pred_props_df["area_um2"].mean()

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(30, 12))
    fig.suptitle(
        f"{IMAGE_NAME}  (pixel size = {PIXEL_SIZE} µm/px)\n"
        f"GT: {n_gt} objects, mean diam = {gt_mean_diam:.1f} µm, mean area = {gt_mean_area:.0f} µm²\n"
        f"Pred: {n_pred} objects, mean diam = {pred_mean_diam:.1f} µm, mean area = {pred_mean_area:.0f} µm²",
        fontsize=14, fontweight="bold",
    )

    green = (0.0, 1.0, 0.0)
    red = (1.0, 0.2, 0.2)

    # Panel 1: Image + GT (green)
    ax = axes[0]
    ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask_to_rgba(gt_binary, green, alpha=0.35))
    label_objects(ax, gt_labeled, PIXEL_SIZE, color="#00ff00")
    ax.set_title(f"Ground Truth (green)\n{n_gt} objects, mean diam = {gt_mean_diam:.1f} µm", fontsize=11)
    ax.axis("off")

    # Panel 2: Image + Pred (red)
    ax = axes[1]
    ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask_to_rgba(pred_binary, red, alpha=0.35))
    label_objects(ax, pred_labeled, PIXEL_SIZE, color="#ff4444")
    ax.set_title(f"Predicted (red)\n{n_pred} objects, mean diam = {pred_mean_diam:.1f} µm", fontsize=11)
    ax.axis("off")

    # Panel 3: GT (green) + Pred (red) overlay
    ax = axes[2]
    ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1, alpha=0.4)
    # Create combined overlay: green for GT-only, red for Pred-only, yellow for overlap
    h, w = gt_binary.shape
    combined = np.zeros((h, w, 4), dtype=np.float32)
    gt_only = gt_binary & ~pred_binary
    pred_only = pred_binary & ~gt_binary
    overlap = gt_binary & pred_binary
    combined[gt_only] = [0.0, 1.0, 0.0, 0.6]     # green
    combined[pred_only] = [1.0, 0.2, 0.2, 0.6]    # red
    combined[overlap] = [1.0, 1.0, 0.0, 0.6]       # yellow
    ax.imshow(combined)

    # Label GT diameters (slight offset up) and Pred diameters (slight offset down)
    label_objects(ax, gt_labeled, PIXEL_SIZE, color="#00ff00", offset=(0, -15))
    label_objects(ax, pred_labeled, PIXEL_SIZE, color="#ff4444", offset=(0, 15))

    gt_patch = mpatches.Patch(color="green", alpha=0.6, label="GT only")
    pred_patch = mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.6, label="Pred only")
    overlap_patch = mpatches.Patch(color="yellow", alpha=0.6, label="Overlap (agreement)")
    ax.legend(handles=[gt_patch, pred_patch, overlap_patch], loc="upper right", fontsize=9)
    ax.set_title("GT (green) vs Pred (red)\nYellow = overlap / agreement", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(OUTPUT_PATH), dpi=200, bbox_inches="tight")
    print(f"\nSaved to {OUTPUT_PATH}")

    # Also save a zoomed crop for detail
    crop_output = Path("scripts/gt_vs_pred_overlay_crop.png")
    # Pick a central crop ~40% of the image
    ch, cw = img_norm.shape
    y0, y1 = int(ch * 0.3), int(ch * 0.7)
    x0, x1 = int(cw * 0.3), int(cw * 0.7)

    fig2, axes2 = plt.subplots(1, 3, figsize=(30, 12))
    fig2.suptitle(f"{IMAGE_NAME} — Zoomed Center Crop  (diameters in µm)", fontsize=14, fontweight="bold")

    for ax2, ax_orig in zip(axes2, axes):
        ax2.set_title(ax_orig.get_title(), fontsize=11)

    # Panel 1 crop
    axes2[0].imshow(img_norm[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    axes2[0].imshow(mask_to_rgba(gt_binary[y0:y1, x0:x1], green, alpha=0.35))
    # Label objects whose centroid falls in the crop
    gt_props_crop = measure.regionprops(gt_labeled)
    for prop in gt_props_crop:
        if prop.area < MIN_AREA:
            continue
        cy, cx = prop.centroid
        if y0 <= cy < y1 and x0 <= cx < x1:
            diam_um = prop.equivalent_diameter_area * PIXEL_SIZE
            axes2[0].text(
                cx - x0, cy - y0, f"{diam_um:.0f}",
                fontsize=6, fontweight="bold", color="#00ff00", ha="center", va="center",
                path_effects=[withStroke(linewidth=2, foreground="black")],
            )
    axes2[0].axis("off")

    # Panel 2 crop
    axes2[1].imshow(img_norm[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    axes2[1].imshow(mask_to_rgba(pred_binary[y0:y1, x0:x1], red, alpha=0.35))
    pred_props_crop = measure.regionprops(pred_labeled)
    for prop in pred_props_crop:
        if prop.area < MIN_AREA:
            continue
        cy, cx = prop.centroid
        if y0 <= cy < y1 and x0 <= cx < x1:
            diam_um = prop.equivalent_diameter_area * PIXEL_SIZE
            axes2[1].text(
                cx - x0, cy - y0, f"{diam_um:.0f}",
                fontsize=6, fontweight="bold", color="#ff4444", ha="center", va="center",
                path_effects=[withStroke(linewidth=2, foreground="black")],
            )
    axes2[1].axis("off")

    # Panel 3 crop
    axes2[2].imshow(img_norm[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, alpha=0.4)
    combined_crop = combined[y0:y1, x0:x1]
    axes2[2].imshow(combined_crop)
    for prop in gt_props_crop:
        if prop.area < MIN_AREA:
            continue
        cy, cx = prop.centroid
        if y0 <= cy < y1 and x0 <= cx < x1:
            diam_um = prop.equivalent_diameter_area * PIXEL_SIZE
            axes2[2].text(
                cx - x0, (cy - y0) - 15, f"{diam_um:.0f}",
                fontsize=6, fontweight="bold", color="#00ff00", ha="center", va="center",
                path_effects=[withStroke(linewidth=2, foreground="black")],
            )
    for prop in pred_props_crop:
        if prop.area < MIN_AREA:
            continue
        cy, cx = prop.centroid
        if y0 <= cy < y1 and x0 <= cx < x1:
            diam_um = prop.equivalent_diameter_area * PIXEL_SIZE
            axes2[2].text(
                cx - x0, (cy - y0) + 15, f"{diam_um:.0f}",
                fontsize=6, fontweight="bold", color="#ff4444", ha="center", va="center",
                path_effects=[withStroke(linewidth=2, foreground="black")],
            )
    axes2[2].legend(handles=[gt_patch, pred_patch, overlap_patch], loc="upper right", fontsize=9)
    axes2[2].axis("off")

    plt.tight_layout()
    fig2.savefig(str(crop_output), dpi=200, bbox_inches="tight")
    print(f"Saved crop to {crop_output}")


if __name__ == "__main__":
    main()
