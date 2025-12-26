"""
Unit tests for src/analysis/quantify.py

Tests object extraction, morphology metrics, matching, and instance evaluation.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.analysis.quantify import (
    extract_objects,
    compute_object_properties,
    compute_iou_matrix,
    match_objects,
    compute_instance_metrics,
    process_image_pair,
    create_summary_plots,
)


class TestExtractObjects:
    """Test connected components extraction."""

    def test_empty_mask(self):
        """Empty mask should return zero objects."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        labeled, count = extract_objects(mask, min_area=10)

        assert count == 0
        assert labeled.shape == (100, 100)
        assert np.all(labeled == 0)

    def test_single_object(self):
        """Single object should be labeled correctly."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 255  # Single rectangle

        labeled, count = extract_objects(mask, min_area=10)

        assert count == 1
        assert np.max(labeled) == 1
        assert np.sum(labeled == 1) == 20 * 20  # 400 pixels

    def test_multiple_objects(self):
        """Multiple distinct objects should be labeled separately."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255  # Object 1
        mask[30:40, 30:40] = 255  # Object 2
        mask[60:70, 60:70] = 255  # Object 3

        labeled, count = extract_objects(mask, min_area=10)

        assert count == 3
        assert np.max(labeled) == 3

    def test_min_area_filtering(self):
        """Objects smaller than min_area should be filtered out."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255  # 100 pixels
        mask[30:33, 30:33] = 255  # 9 pixels (too small)

        labeled, count = extract_objects(mask, min_area=50)

        assert count == 1  # Only large object remains

    def test_binary_conversion(self):
        """Mask with any positive value should be treated as foreground."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 128  # Not 255

        labeled, count = extract_objects(mask, min_area=10)

        assert count == 1


class TestComputeObjectProperties:
    """Test morphology metrics computation."""

    def test_empty_labeled_mask(self):
        """Empty mask should return empty DataFrame with correct columns."""
        labeled = np.zeros((100, 100), dtype=int)
        df = compute_object_properties(labeled)

        assert len(df) == 0
        assert 'object_id' in df.columns
        assert 'area' in df.columns
        assert 'circularity' in df.columns

    def test_single_square(self):
        """Square object should have known properties."""
        labeled = np.zeros((100, 100), dtype=int)
        labeled[10:30, 10:30] = 1  # 20x20 square

        df = compute_object_properties(labeled)

        assert len(df) == 1
        assert df.loc[0, 'object_id'] == 1
        assert df.loc[0, 'area'] == 400  # 20 * 20

        # Square is fairly circular
        assert df.loc[0, 'circularity'] > 0.7
        assert df.loc[0, 'circularity'] <= 1.0

    def test_perfect_circle(self):
        """Circular object should have circularity close to 1."""
        from skimage.draw import disk

        labeled = np.zeros((100, 100), dtype=int)
        rr, cc = disk((50, 50), 20, shape=(100, 100))
        labeled[rr, cc] = 1

        df = compute_object_properties(labeled)

        assert len(df) == 1
        # Circle should have circularity close to 1 (pixelated circles aren't perfect)
        assert df.loc[0, 'circularity'] > 0.94
        assert df.loc[0, 'eccentricity'] < 0.2  # Low eccentricity for circle

    def test_multiple_objects(self):
        """Should compute properties for all objects."""
        labeled = np.zeros((100, 100), dtype=int)
        labeled[10:20, 10:20] = 1
        labeled[30:40, 30:40] = 2
        labeled[60:70, 60:70] = 3

        df = compute_object_properties(labeled)

        assert len(df) == 3
        assert list(df['object_id']) == [1, 2, 3]
        assert all(df['area'] == 100)  # All 10x10 squares

    def test_physical_units_conversion(self):
        """Pixel size should convert to physical units."""
        labeled = np.zeros((100, 100), dtype=int)
        labeled[10:30, 10:30] = 1  # 20x20 square = 400 px²

        pixel_size = 0.5  # µm per pixel
        df = compute_object_properties(labeled, pixel_size=pixel_size)

        # Area in µm²: 400 * (0.5)² = 100
        assert df.loc[0, 'area'] == 100.0

    def test_bounding_box(self):
        """Bounding box should match object location."""
        labeled = np.zeros((100, 100), dtype=int)
        labeled[20:40, 30:50] = 1

        df = compute_object_properties(labeled)

        assert df.loc[0, 'bbox_min_row'] == 20
        assert df.loc[0, 'bbox_max_row'] == 40
        assert df.loc[0, 'bbox_min_col'] == 30
        assert df.loc[0, 'bbox_max_col'] == 50


class TestComputeIoUMatrix:
    """Test IoU matrix computation."""

    def test_identical_masks(self):
        """Identical masks should have IoU = 1."""
        mask1 = np.zeros((100, 100), dtype=int)
        mask1[20:40, 30:50] = 1

        mask2 = mask1.copy()

        iou_matrix = compute_iou_matrix(mask1, mask2)

        assert iou_matrix.shape == (1, 1)
        assert iou_matrix[0, 0] == 1.0

    def test_no_overlap(self):
        """Non-overlapping masks should have IoU = 0."""
        mask1 = np.zeros((100, 100), dtype=int)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros((100, 100), dtype=int)
        mask2[50:60, 50:60] = 1

        iou_matrix = compute_iou_matrix(mask1, mask2)

        assert iou_matrix.shape == (1, 1)
        assert iou_matrix[0, 0] == 0.0

    def test_partial_overlap(self):
        """Partially overlapping masks should have 0 < IoU < 1."""
        mask1 = np.zeros((100, 100), dtype=int)
        mask1[20:40, 20:40] = 1  # 20x20 = 400 pixels

        mask2 = np.zeros((100, 100), dtype=int)
        mask2[30:50, 30:50] = 1  # 20x20 = 400 pixels
        # Overlap: 10x10 = 100 pixels
        # Union: 400 + 400 - 100 = 700 pixels
        # IoU = 100 / 700 = 0.142857

        iou_matrix = compute_iou_matrix(mask1, mask2)

        assert iou_matrix.shape == (1, 1)
        assert 0.14 < iou_matrix[0, 0] < 0.15

    def test_multiple_objects(self):
        """Should compute IoU for all pairs."""
        pred = np.zeros((100, 100), dtype=int)
        pred[10:20, 10:20] = 1
        pred[50:60, 50:60] = 2

        gt = np.zeros((100, 100), dtype=int)
        gt[10:20, 10:20] = 1  # Perfect match with pred object 1
        gt[30:40, 30:40] = 2  # No overlap with any pred

        iou_matrix = compute_iou_matrix(pred, gt)

        assert iou_matrix.shape == (2, 2)
        assert iou_matrix[0, 0] == 1.0  # Perfect match
        assert iou_matrix[0, 1] == 0.0  # No overlap
        assert iou_matrix[1, 0] == 0.0  # No overlap
        assert iou_matrix[1, 1] == 0.0  # No overlap

    def test_empty_masks(self):
        """Empty masks should return empty IoU matrix."""
        pred = np.zeros((100, 100), dtype=int)
        gt = np.zeros((100, 100), dtype=int)

        iou_matrix = compute_iou_matrix(pred, gt)

        assert iou_matrix.shape == (0, 0)


class TestMatchObjects:
    """Test Hungarian matching algorithm."""

    def test_perfect_matches(self):
        """Perfect 1:1 matches should be found."""
        pred = np.zeros((100, 100), dtype=int)
        pred[10:20, 10:20] = 1
        pred[30:40, 30:40] = 2

        gt = pred.copy()

        matches, fps, fns = match_objects(pred, gt, iou_threshold=0.5)

        assert len(matches) == 2
        assert len(fps) == 0
        assert len(fns) == 0
        assert all(iou == 1.0 for _, _, iou in matches)

    def test_false_positives(self):
        """Unmatched predictions should be FPs."""
        pred = np.zeros((100, 100), dtype=int)
        pred[10:20, 10:20] = 1
        pred[30:40, 30:40] = 2  # This has no GT match

        gt = np.zeros((100, 100), dtype=int)
        gt[10:20, 10:20] = 1

        matches, fps, fns = match_objects(pred, gt, iou_threshold=0.5)

        assert len(matches) == 1
        assert len(fps) == 1
        assert len(fns) == 0

    def test_false_negatives(self):
        """Unmatched GT should be FNs."""
        pred = np.zeros((100, 100), dtype=int)
        pred[10:20, 10:20] = 1

        gt = np.zeros((100, 100), dtype=int)
        gt[10:20, 10:20] = 1
        gt[30:40, 30:40] = 2  # This has no pred match

        matches, fps, fns = match_objects(pred, gt, iou_threshold=0.5)

        assert len(matches) == 1
        assert len(fps) == 0
        assert len(fns) == 1

    def test_iou_threshold(self):
        """Matches below threshold should be rejected."""
        pred = np.zeros((100, 100), dtype=int)
        pred[20:40, 20:40] = 1  # 400 pixels

        gt = np.zeros((100, 100), dtype=int)
        gt[30:50, 30:50] = 1  # 400 pixels, 100 overlap
        # IoU = 100 / 700 ≈ 0.14

        # Should match with low threshold
        matches, fps, fns = match_objects(pred, gt, iou_threshold=0.1)
        assert len(matches) == 1

        # Should NOT match with high threshold
        matches, fps, fns = match_objects(pred, gt, iou_threshold=0.5)
        assert len(matches) == 0
        assert len(fps) == 1
        assert len(fns) == 1

    def test_empty_inputs(self):
        """Empty masks should return empty results."""
        pred = np.zeros((100, 100), dtype=int)
        gt = np.zeros((100, 100), dtype=int)

        matches, fps, fns = match_objects(pred, gt, iou_threshold=0.5)

        assert len(matches) == 0
        assert len(fps) == 0
        assert len(fns) == 0


class TestComputeInstanceMetrics:
    """Test instance-level metrics computation."""

    def test_perfect_detection(self):
        """All matches, no errors -> precision=recall=F1=1."""
        matches = [(1, 1, 0.8), (2, 2, 0.9)]
        fps = []
        fns = []

        metrics = compute_instance_metrics(matches, fps, fns)

        assert metrics['tp'] == 2
        assert metrics['fp'] == 0
        assert metrics['fn'] == 0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert 0.8 < metrics['mean_matched_iou'] < 0.9

    def test_false_positives_only(self):
        """FPs lower precision but not recall."""
        matches = [(1, 1, 0.8)]
        fps = [2, 3]  # 2 false positives
        fns = []

        metrics = compute_instance_metrics(matches, fps, fns)

        assert metrics['tp'] == 1
        assert metrics['fp'] == 2
        assert metrics['fn'] == 0
        assert metrics['precision'] == 1 / 3  # TP / (TP + FP)
        assert metrics['recall'] == 1.0  # TP / (TP + FN)
        assert metrics['f1'] == 0.5  # 2 * (1/3 * 1) / (1/3 + 1)

    def test_false_negatives_only(self):
        """FNs lower recall but not precision."""
        matches = [(1, 1, 0.8)]
        fps = []
        fns = [2, 3]  # 2 false negatives

        metrics = compute_instance_metrics(matches, fps, fns)

        assert metrics['tp'] == 1
        assert metrics['fp'] == 0
        assert metrics['fn'] == 2
        assert metrics['precision'] == 1.0  # TP / (TP + FP)
        assert metrics['recall'] == 1 / 3  # TP / (TP + FN)
        assert metrics['f1'] == 0.5

    def test_no_detections(self):
        """No matches should give zero metrics."""
        matches = []
        fps = [1, 2]
        fns = [1, 2]

        metrics = compute_instance_metrics(matches, fps, fns)

        assert metrics['tp'] == 0
        assert metrics['fp'] == 2
        assert metrics['fn'] == 2
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['mean_matched_iou'] == 0.0


class TestProcessImagePair:
    """Test end-to-end image pair processing."""

    def test_process_simple_pair(self):
        """Should process a simple image pair correctly."""
        import tifffile

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple masks
            pred_mask = np.zeros((100, 100), dtype=np.uint8)
            pred_mask[20:40, 20:40] = 255

            gt_mask = np.zeros((100, 100), dtype=np.uint8)
            gt_mask[20:40, 20:40] = 255

            pred_path = tmpdir / "pred_mask.tif"
            gt_path = tmpdir / "gt_mask.tif"

            tifffile.imwrite(pred_path, pred_mask)
            tifffile.imwrite(gt_path, gt_mask)

            # Process
            obj_props, inst_metrics, pred_labels, gt_labels = process_image_pair(
                pred_mask_path=pred_path,
                gt_mask_path=gt_path,
                min_object_area=100,
                iou_threshold=0.5,
            )

            # Check results
            assert len(obj_props) == 1  # One object detected
            assert inst_metrics['tp'] == 1
            assert inst_metrics['fp'] == 0
            assert inst_metrics['fn'] == 0
            assert inst_metrics['f1'] == 1.0


class TestCreateSummaryPlots:
    """Test visualization generation."""

    def test_creates_plots(self):
        """Should create summary plots without error."""
        # Create sample data
        all_objects = pd.DataFrame({
            'area': np.random.uniform(100, 500, 50),
            'equivalent_diameter': np.random.uniform(10, 25, 50),
            'circularity': np.random.uniform(0.7, 1.0, 50),
        })

        instance_eval = pd.DataFrame({
            'n_pred': [10, 12, 11],
            'n_gt': [10, 11, 10],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            create_summary_plots(all_objects, instance_eval, output_dir)

            # Check that plot file was created
            plot_path = output_dir / "summary_plots.png"
            assert plot_path.exists()

    def test_handles_empty_data(self):
        """Should handle empty data gracefully."""
        all_objects = pd.DataFrame(columns=['area', 'equivalent_diameter', 'circularity'])
        instance_eval = pd.DataFrame(columns=['n_pred', 'n_gt'])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Should not raise error
            create_summary_plots(all_objects, instance_eval, output_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
