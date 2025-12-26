"""
Quick script to analyze morphology of ground truth masks.
"""
import pandas as pd
from pathlib import Path
import tifffile
from src.analysis.quantify import extract_objects, compute_object_properties

# Load GT manifest
manifest = pd.read_csv('data/splits/test.csv')

all_objects = []

for idx, row in manifest.iterrows():
    # Load GT mask
    gt_mask_path = Path('data') / row['mask_path']
    print(f"Processing {gt_mask_path.name}...")
    
    gt_mask = tifffile.imread(gt_mask_path)
    
    # Extract objects
    labeled, n_objects = extract_objects(gt_mask, min_area=100)
    print(f"  Found {n_objects} objects")
    
    # Compute morphology
    if n_objects > 0:
        props = compute_object_properties(labeled)
        props['image'] = gt_mask_path.stem.replace('_mask', '')
        all_objects.append(props)

# Combine all results
if all_objects:
    all_df = pd.concat(all_objects, ignore_index=True)
    
    print("\n" + "="*80)
    print("GROUND TRUTH MORPHOLOGY STATISTICS")
    print("="*80)
    print(f"\nTotal images: {len(manifest)}")
    print(f"Total objects: {len(all_df)}")
    print(f"\nMorphology Metrics:")
    print(f"  Area (px²):              {all_df['area'].mean():.1f} ± {all_df['area'].std():.1f}")
    print(f"    Min/Max:               {all_df['area'].min():.1f} / {all_df['area'].max():.1f}")
    print(f"  Equivalent Diameter (px): {all_df['equivalent_diameter'].mean():.1f} ± {all_df['equivalent_diameter'].std():.1f}")
    print(f"    Min/Max:               {all_df['equivalent_diameter'].min():.1f} / {all_df['equivalent_diameter'].max():.1f}")
    print(f"  Circularity (1=circle):  {all_df['circularity'].mean():.3f} ± {all_df['circularity'].std():.3f}")
    print(f"    Min/Max:               {all_df['circularity'].min():.3f} / {all_df['circularity'].max():.3f}")
    print(f"  Eccentricity (0=circle): {all_df['eccentricity'].mean():.3f} ± {all_df['eccentricity'].std():.3f}")
    print(f"    Min/Max:               {all_df['eccentricity'].min():.3f} / {all_df['eccentricity'].max():.3f}")
    print(f"  Major Axis (px):         {all_df['major_axis_length'].mean():.1f} ± {all_df['major_axis_length'].std():.1f}")
    print(f"  Minor Axis (px):         {all_df['minor_axis_length'].mean():.1f} ± {all_df['minor_axis_length'].std():.1f}")
    
    print("\nPer-Image Object Counts:")
    for img_name in all_df['image'].unique():
        img_objects = all_df[all_df['image'] == img_name]
        print(f"  {img_name}: {len(img_objects)} objects")
    
    # Save to CSV
    all_df.to_csv('gt_morphology.csv', index=False)
    print(f"\n✓ Saved detailed metrics to gt_morphology.csv")
    print("="*80 + "\n")
else:
    print("No objects found!")
