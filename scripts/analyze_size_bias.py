"""
Analyze whether predicted spheroid sizes differ from ground truth,
and whether that differs across imaging scales (1.1 vs 2.76 µm/px).

Questions:
  1. Are GT spheroids different sizes between the two scales?
  2. Are predicted spheroids bigger/smaller than GT spheroids?
  3. Is any prediction bias consistent across scales?

Usage:
    source .venv/bin/activate
    python scripts/analyze_size_bias.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tifffile
from scipy import stats

from src.analysis.quantify import extract_objects, compute_object_properties
from src.inference.predict import (
    load_model_from_checkpoint,
    predict_image_from_path,
)

DATA_ROOT = Path("data")
CHECKPOINT = Path("runs/train_20260216_173233/checkpoints/best_model.pth")
TRAINING_PIXEL_SIZE = 2.76

# Define the two datasets
DATASETS = {
    "2.76": {
        "pixel_size": 2.76,
        "images": [
            ("dECM_1_1", "working_276/images/dECM_1_1.tif", "working_276/masks/dECM_1_1_mask.tif"),
            ("dECM_1_2", "working_276/images/dECM_1_2.tif", "working_276/masks/dECM_1_2_mask.tif"),
            ("dECM_2_1", "working_276/images/dECM_2_1.tif", "working_276/masks/dECM_2_1_mask.tif"),
            ("dECM_2_2", "working_276/images/dECM_2_2.tif", "working_276/masks/dECM_2_2_mask.tif"),
            ("Matri_1_1", "working_276/images/Matri_1_1.tif", "working_276/masks/Matri_1_1_mask.tif"),
            ("Matri_1_2", "working_276/images/Matri_1_2.tif", "working_276/masks/Matri_1_2_mask.tif"),
        ],
    },
    "1.10": {
        "pixel_size": 1.10,
        "images": [
            ("GFR_1_1", "working_110/images/GFR_1_1.tif", "working_110/masks/GFR_1_1_mask.tif"),
            ("Normal_1_1", "working_110/images/Normal_1_1.tif", "working_110/masks/Normal_1_1_mask.tif"),
            ("Normal_1_2", "working_110/images/Normal_1_2.tif", "working_110/masks/Normal_1_2_mask.tif"),
        ],
    },
}

MIN_AREA = 100  # pixels


def measure_mask(mask_path: Path, pixel_size: float) -> pd.DataFrame:
    """Extract objects from a mask and compute properties."""
    mask = tifffile.imread(mask_path)
    labeled, n = extract_objects(mask, min_area=MIN_AREA)
    if n == 0:
        return pd.DataFrame()
    props = compute_object_properties(labeled, pixel_size=pixel_size)
    return props


def run_inference(image_path: Path, pixel_size: float, model, device) -> Path:
    """Run inference and return path to predicted mask."""
    output_dir = Path("scripts/_size_analysis_preds")
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_image_from_path(
        image_path=image_path,
        mask_path=None,
        model=model,
        device=device,
        output_dir=output_dir,
        pixel_size=pixel_size,
        training_pixel_size=TRAINING_PIXEL_SIZE,
    )

    basename = image_path.stem
    pred_mask_path = output_dir / f"{basename}_pred_mask.tif"
    return pred_mask_path


def main():
    print("=" * 70)
    print("SPHEROID SIZE BIAS ANALYSIS")
    print("=" * 70)

    # Load model once
    print("\nLoading model...")
    import torch
    device = torch.device("cpu")
    model, training_ps = load_model_from_checkpoint(CHECKPOINT, device=device)
    print(f"  Model loaded on {device} (training_pixel_size={training_ps})")

    all_rows = []

    for scale_label, ds in DATASETS.items():
        pixel_size = ds["pixel_size"]
        print(f"\n{'=' * 70}")
        print(f"SCALE: {scale_label} µm/px  ({len(ds['images'])} images)")
        print(f"{'=' * 70}")

        for basename, img_rel, mask_rel in ds["images"]:
            img_path = DATA_ROOT / img_rel
            mask_path = DATA_ROOT / mask_rel

            print(f"\n  --- {basename} ---")

            # Ground truth measurements
            gt_props = measure_mask(mask_path, pixel_size)
            n_gt = len(gt_props)
            print(f"  GT objects: {n_gt}")

            # Run inference
            pred_mask_path = run_inference(img_path, pixel_size, model, device)
            pred_props = measure_mask(pred_mask_path, pixel_size)
            n_pred = len(pred_props)
            print(f"  Pred objects: {n_pred}")

            # Collect per-object rows
            for _, row in gt_props.iterrows():
                all_rows.append({
                    "scale": scale_label,
                    "image": basename,
                    "source": "GT",
                    "area_px": row["area_px"],
                    "area_um2": row["area_um2"],
                    "equiv_diam_px": row["equivalent_diameter_px"],
                    "equiv_diam_um": row["equivalent_diameter_um"],
                })
            for _, row in pred_props.iterrows():
                all_rows.append({
                    "scale": scale_label,
                    "image": basename,
                    "source": "Pred",
                    "area_px": row["area_px"],
                    "area_um2": row["area_um2"],
                    "equiv_diam_px": row["equivalent_diameter_px"],
                    "equiv_diam_um": row["equivalent_diameter_um"],
                })

    df = pd.DataFrame(all_rows)
    df.to_csv("scripts/_size_analysis_results.csv", index=False)

    # ===== ANALYSIS =====
    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # ----- Question 1: GT size differences between scales -----
    print("\n--- Q1: Ground truth spheroid sizes by scale ---")
    gt = df[df["source"] == "GT"]
    for scale in ["2.76", "1.10"]:
        sub = gt[gt["scale"] == scale]
        print(f"\n  Scale {scale} µm/px (n={len(sub)} objects across {sub['image'].nunique()} images):")
        print(f"    Area (µm²):       mean={sub['area_um2'].mean():.1f}  median={sub['area_um2'].median():.1f}  std={sub['area_um2'].std():.1f}")
        print(f"    Eff. diam (µm):   mean={sub['equiv_diam_um'].mean():.1f}  median={sub['equiv_diam_um'].median():.1f}  std={sub['equiv_diam_um'].std():.1f}")
        print(f"    Area (px):        mean={sub['area_px'].mean():.1f}  median={sub['area_px'].median():.1f}  std={sub['area_px'].std():.1f}")
        print(f"    Eff. diam (px):   mean={sub['equiv_diam_px'].mean():.1f}  median={sub['equiv_diam_px'].median():.1f}  std={sub['equiv_diam_px'].std():.1f}")

    gt_276 = gt[gt["scale"] == "2.76"]["area_um2"]
    gt_110 = gt[gt["scale"] == "1.10"]["area_um2"]
    t_stat, p_val = stats.ttest_ind(gt_276, gt_110, equal_var=False)
    print(f"\n  Welch t-test (GT area_um2): t={t_stat:.3f}, p={p_val:.4f}")
    mw_stat, mw_p = stats.mannwhitneyu(gt_276, gt_110, alternative='two-sided')
    print(f"  Mann-Whitney U (GT area_um2): U={mw_stat:.0f}, p={mw_p:.4f}")

    # ----- Question 2: Pred vs GT sizes -----
    print("\n--- Q2: Predicted vs GT spheroid sizes (all objects) ---")
    pred = df[df["source"] == "Pred"]

    for scale in ["2.76", "1.10"]:
        gt_sub = gt[gt["scale"] == scale]
        pred_sub = pred[pred["scale"] == scale]
        print(f"\n  Scale {scale} µm/px:")
        print(f"    GT   n={len(gt_sub):4d}  area_um2: mean={gt_sub['area_um2'].mean():10.1f}  std={gt_sub['area_um2'].std():10.1f}")
        print(f"    Pred n={len(pred_sub):4d}  area_um2: mean={pred_sub['area_um2'].mean():10.1f}  std={pred_sub['area_um2'].std():10.1f}")
        if len(pred_sub) > 0 and len(gt_sub) > 0:
            ratio = pred_sub['area_um2'].mean() / gt_sub['area_um2'].mean()
            print(f"    Ratio (pred/gt mean area): {ratio:.3f}")
            t, p = stats.ttest_ind(pred_sub['area_um2'], gt_sub['area_um2'], equal_var=False)
            print(f"    Welch t-test: t={t:.3f}, p={p:.4f}")

        print(f"    GT   equiv_diam_um: mean={gt_sub['equiv_diam_um'].mean():8.1f}  std={gt_sub['equiv_diam_um'].std():8.1f}")
        print(f"    Pred equiv_diam_um: mean={pred_sub['equiv_diam_um'].mean():8.1f}  std={pred_sub['equiv_diam_um'].std():8.1f}")
        if len(pred_sub) > 0 and len(gt_sub) > 0:
            ratio_d = pred_sub['equiv_diam_um'].mean() / gt_sub['equiv_diam_um'].mean()
            print(f"    Ratio (pred/gt mean diam): {ratio_d:.3f}")

    # ----- Question 3: Per-image comparison -----
    print("\n--- Q3: Per-image pred vs GT summary ---")
    print(f"\n  {'Image':<14} {'Scale':>6} {'GT_n':>5} {'Pred_n':>6} "
          f"{'GT_area_um2':>12} {'Pred_area_um2':>14} {'Ratio':>7} "
          f"{'GT_diam_um':>11} {'Pred_diam_um':>12} {'D_Ratio':>8}")

    for scale in ["2.76", "1.10"]:
        for img in sorted(df[df["scale"] == scale]["image"].unique()):
            g = gt[(gt["scale"] == scale) & (gt["image"] == img)]
            p = pred[(pred["scale"] == scale) & (pred["image"] == img)]
            gt_area = g["area_um2"].mean() if len(g) > 0 else 0
            pred_area = p["area_um2"].mean() if len(p) > 0 else 0
            gt_diam = g["equiv_diam_um"].mean() if len(g) > 0 else 0
            pred_diam = p["equiv_diam_um"].mean() if len(p) > 0 else 0
            area_ratio = pred_area / gt_area if gt_area > 0 else float('nan')
            diam_ratio = pred_diam / gt_diam if gt_diam > 0 else float('nan')
            print(f"  {img:<14} {scale:>6} {len(g):>5} {len(p):>6} "
                  f"{gt_area:>12.1f} {pred_area:>14.1f} {area_ratio:>7.3f} "
                  f"{gt_diam:>11.1f} {pred_diam:>12.1f} {diam_ratio:>8.3f}")

    # ----- Summary -----
    print("\n--- Summary ---")
    for scale in ["2.76", "1.10"]:
        ratios = []
        for img in df[df["scale"] == scale]["image"].unique():
            g = gt[(gt["scale"] == scale) & (gt["image"] == img)]
            p = pred[(pred["scale"] == scale) & (pred["image"] == img)]
            if len(g) > 0 and len(p) > 0:
                ratios.append(p["area_um2"].mean() / g["area_um2"].mean())
        if ratios:
            print(f"  Scale {scale}: mean pred/gt area ratio = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")

    print(f"\nResults saved to scripts/_size_analysis_results.csv")
    print("Done.")


if __name__ == "__main__":
    main()
