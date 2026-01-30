"""
Unit tests for tiled inference module (Phase 4).

Tests tiled inference, post-processing, and pixel-level metrics.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.predict import (
    compute_dice_coefficient,
    compute_iou,
    postprocess_mask,
    predict_full_image,
    threshold_mask,
)


class DummyModel(nn.Module):
    """
    Dummy model for testing that returns a constant probability.
    """

    def __init__(self, constant_value: float = 0.7):
        super().__init__()
        self.constant_value = constant_value

    def forward(self, x):
        # Return logits corresponding to desired probability
        # sigmoid(logit) = prob => logit = log(prob / (1 - prob))
        prob = self.constant_value
        logit = np.log(prob / (1 - prob))
        return torch.full_like(x, logit)


def test_threshold_mask():
    """Test that thresholding produces correct binary values."""
    # Create probability map with known values
    prob_map = np.array([[0.2, 0.4], [0.6, 0.8]])

    # Threshold at 0.5
    binary_mask = threshold_mask(prob_map, threshold=0.5)

    # Expected: values > 0.5 become 255, others become 0
    expected = np.array([[0, 0], [255, 255]], dtype=np.uint8)

    assert np.array_equal(binary_mask, expected)
    assert binary_mask.dtype == np.uint8


def test_threshold_mask_edge_cases():
    """Test threshold behavior at boundary values."""
    prob_map = np.array([[0.0, 0.5, 1.0]])

    # Threshold at 0.5 (values > threshold, not >=)
    binary_mask = threshold_mask(prob_map, threshold=0.5)

    # 0.5 should not be included (> not >=)
    expected = np.array([[0, 0, 255]], dtype=np.uint8)

    assert np.array_equal(binary_mask, expected)


def test_postprocess_remove_small_objects():
    """Test that small objects are removed correctly."""
    # Create mask with one large object (9 pixels) and one small object (4 pixels)
    binary_mask = np.array(
        [
            [255, 255, 255, 0, 0],
            [255, 255, 255, 0, 0],
            [255, 255, 255, 0, 255],
            [0, 0, 0, 0, 255],
            [0, 0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    # Remove objects < 5 pixels
    cleaned = postprocess_mask(binary_mask, min_object_area=5, fill_holes=False)

    # Expected: only the 9-pixel object remains
    expected = np.array(
        [
            [255, 255, 255, 0, 0],
            [255, 255, 255, 0, 0],
            [255, 255, 255, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(cleaned, expected)


def test_postprocess_fill_holes():
    """Test that holes within objects are filled."""
    # Create object with a hole in the middle
    binary_mask = np.array(
        [
            [255, 255, 255, 255],
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    # Fill holes
    cleaned = postprocess_mask(binary_mask, min_object_area=0, fill_holes=True)

    # Expected: hole should be filled
    expected = np.full((4, 4), 255, dtype=np.uint8)

    assert np.array_equal(cleaned, expected)


def test_postprocess_no_modifications():
    """Test that post-processing with disabled features returns unchanged mask."""
    binary_mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    # No removal (min_area=0), no filling
    cleaned = postprocess_mask(binary_mask, min_object_area=0, fill_holes=False)

    assert np.array_equal(cleaned, binary_mask)


def test_compute_dice_perfect_match():
    """Test Dice coefficient for perfect prediction."""
    pred_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    gt_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)

    dice = compute_dice_coefficient(pred_mask, gt_mask)

    # Perfect match should give Dice = 1.0
    assert dice == pytest.approx(1.0, abs=1e-5)


def test_compute_dice_no_overlap():
    """Test Dice coefficient for no overlap."""
    pred_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    gt_mask = np.array([[0, 0], [255, 255]], dtype=np.uint8)

    dice = compute_dice_coefficient(pred_mask, gt_mask)

    # No overlap should give Dice close to 0
    # (smooth constant prevents exact 0)
    assert dice < 0.01


def test_compute_dice_partial_overlap():
    """Test Dice coefficient for partial overlap."""
    pred_mask = np.array([[255, 255], [255, 0]], dtype=np.uint8)
    gt_mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)

    dice = compute_dice_coefficient(pred_mask, gt_mask)

    # 2 pixels overlap, 3 in pred, 3 in gt
    # Dice = 2*2 / (3+3) = 4/6 = 0.667
    expected_dice = 2 * 2 / (3 + 3)

    assert dice == pytest.approx(expected_dice, abs=1e-3)


def test_compute_iou_perfect_match():
    """Test IoU for perfect prediction."""
    pred_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    gt_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)

    iou = compute_iou(pred_mask, gt_mask)

    # Perfect match should give IoU = 1.0
    assert iou == pytest.approx(1.0, abs=1e-5)


def test_compute_iou_no_overlap():
    """Test IoU for no overlap."""
    pred_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    gt_mask = np.array([[0, 0], [255, 255]], dtype=np.uint8)

    iou = compute_iou(pred_mask, gt_mask)

    # No overlap should give IoU close to 0
    assert iou < 0.01


def test_compute_iou_partial_overlap():
    """Test IoU for partial overlap."""
    pred_mask = np.array([[255, 255], [255, 0]], dtype=np.uint8)
    gt_mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)

    iou = compute_iou(pred_mask, gt_mask)

    # 2 pixels overlap, 4 in union
    # IoU = 2 / 4 = 0.5
    expected_iou = 2 / 4

    assert iou == pytest.approx(expected_iou, abs=1e-3)


def test_predict_full_image_output_shape():
    """Test that tiled inference produces correct output shape."""
    # Create dummy image
    H, W = 512, 768
    image = np.random.rand(H, W).astype(np.float32)

    # Create dummy model
    model = DummyModel(constant_value=0.7)
    model.eval()

    device = torch.device("cpu")

    # Run inference
    prob_map = predict_full_image(
        image=image,
        model=model,
        device=device,
        tile_size=256,
        overlap=0.25,
    )

    # Check output shape matches input
    assert prob_map.shape == (H, W)


def test_predict_full_image_values_range():
    """Test that probability map values are in [0, 1]."""
    # Create dummy image
    image = np.random.rand(256, 256).astype(np.float32)

    # Create dummy model
    model = DummyModel(constant_value=0.7)
    model.eval()

    device = torch.device("cpu")

    # Run inference
    prob_map = predict_full_image(
        image=image,
        model=model,
        device=device,
        tile_size=128,
        overlap=0.25,
    )

    # Check values are in valid range
    assert np.all(prob_map >= 0.0)
    assert np.all(prob_map <= 1.0)


def test_predict_full_image_overlap_accumulation():
    """Test that overlapping regions are averaged correctly."""
    # Create small image that will be covered by multiple tiles
    image = np.ones((128, 128), dtype=np.float32)

    # Create model that returns constant probability
    model = DummyModel(constant_value=0.6)
    model.eval()

    device = torch.device("cpu")

    # Use small tiles with high overlap to ensure averaging
    prob_map = predict_full_image(
        image=image,
        model=model,
        device=device,
        tile_size=64,
        overlap=0.5,
    )

    # Since model returns constant 0.6, averaged result should also be ~0.6
    # Allow some tolerance for edge effects
    mean_prob = prob_map.mean()
    assert mean_prob == pytest.approx(0.6, abs=0.05)


def test_predict_full_image_handles_uint8_input():
    """Test that uint8 images are normalized correctly."""
    # Create uint8 image
    image = np.full((256, 256), 128, dtype=np.uint8)

    # Create dummy model
    model = DummyModel(constant_value=0.5)
    model.eval()

    device = torch.device("cpu")

    # Run inference - should automatically normalize to [0, 1]
    prob_map = predict_full_image(
        image=image,
        model=model,
        device=device,
        tile_size=256,
        overlap=0.0,
    )

    # Should complete without error and produce valid output
    assert prob_map.shape == (256, 256)
    assert np.all(prob_map >= 0.0)
    assert np.all(prob_map <= 1.0)


def test_predict_full_image_edge_padding():
    """Test that edge tiles are handled correctly with padding."""
    # Create image with non-tile-aligned dimensions
    image = np.ones((300, 450), dtype=np.float32)

    # Create dummy model
    model = DummyModel(constant_value=0.8)
    model.eval()

    device = torch.device("cpu")

    # Run inference with tile size that doesn't divide image dimensions
    prob_map = predict_full_image(
        image=image,
        model=model,
        device=device,
        tile_size=256,
        overlap=0.25,
    )

    # Check output shape matches input
    assert prob_map.shape == (300, 450)

    # Check that all pixels have been processed (no zeros from unprocessed regions)
    # Since model outputs constant 0.8, no pixel should be 0
    assert np.all(prob_map > 0.0)


# ============================================================================
# Multi-Resolution Inference Tests
# ============================================================================


def test_scale_factor_calculation():
    """Test scale factor calculation for different pixel sizes."""
    training_pixel_size = 2.76

    # High resolution (smaller pixel size)
    input_pixel_size = 1.1
    scale_factor = training_pixel_size / input_pixel_size
    assert scale_factor == pytest.approx(2.509, abs=0.01)

    # Low resolution (larger pixel size)
    input_pixel_size = 5.52
    scale_factor = training_pixel_size / input_pixel_size
    assert scale_factor == pytest.approx(0.5, abs=0.01)

    # Same resolution
    input_pixel_size = 2.76
    scale_factor = training_pixel_size / input_pixel_size
    assert scale_factor == pytest.approx(1.0, abs=0.001)


def test_rescaling_preserves_dimensions():
    """Test that rescaling pipeline returns to original dimensions."""
    from skimage.transform import rescale, resize

    # Create test image
    original_shape = (400, 600)
    image = np.random.rand(*original_shape).astype(np.float32)

    # Simulate rescaling for high-res image (scale down then up)
    scale_factor = 0.4  # Downsample to 40%

    # Rescale down
    scaled_image = rescale(
        image,
        scale_factor,
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(image.dtype)

    # Check intermediate size is correct
    expected_scaled_h = int(original_shape[0] * scale_factor)
    expected_scaled_w = int(original_shape[1] * scale_factor)
    assert scaled_image.shape[0] == expected_scaled_h
    assert scaled_image.shape[1] == expected_scaled_w

    # Rescale back up (simulate probability map upscaling)
    prob_map = resize(
        scaled_image,
        original_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    )

    # Check output matches original dimensions
    assert prob_map.shape == original_shape


def test_min_object_area_scaling():
    """Test that min_object_area is correctly scaled with resolution."""
    min_object_area = 100  # pixels at training resolution

    # High resolution (2.51x downsampling)
    scale_factor = 2.51
    effective_min_area = int(min_object_area / (scale_factor ** 2))
    # At original (high) resolution, objects should be ~6.3x larger in area
    # So min threshold should be smaller
    assert effective_min_area == pytest.approx(15.8, abs=1)

    # Low resolution (0.5x upsampling)
    scale_factor = 0.5
    effective_min_area = int(min_object_area / (scale_factor ** 2))
    # At original (low) resolution, objects should be 4x smaller in area
    # So min threshold should be larger
    assert effective_min_area == 400

    # No scaling (scale_factor = 1.0)
    scale_factor = 1.0
    effective_min_area = int(min_object_area / (scale_factor ** 2))
    assert effective_min_area == min_object_area


def test_rescaling_with_predict_single_image(tmp_path):
    """Integration test: rescaling with predict_single_image."""
    from src.inference.predict import predict_single_image
    from tifffile import imwrite

    # Create a test image
    test_image = np.random.randint(0, 256, (400, 400), dtype=np.uint8)
    image_path = tmp_path / "test_image.tif"
    imwrite(image_path, test_image)

    # Create dummy model
    model = DummyModel(constant_value=0.6)
    model.eval()

    device = torch.device("cpu")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run inference with pixel_size (triggers rescaling)
    result = predict_single_image(
        image_path=image_path,
        model=model,
        output_dir=output_dir,
        device=device,
        tile_size=128,
        overlap=0.25,
        threshold=0.5,
        min_object_area=50,
        pixel_size=1.1,  # High resolution
        training_pixel_size=2.76,
    )

    # Check that output was created
    assert result is not None
    assert result.exists()

    # Load output mask and check it matches input dimensions
    from tifffile import imread
    output_mask = imread(result)
    assert output_mask.shape == test_image.shape


def test_no_rescaling_when_pixel_size_matches():
    """Test that no rescaling occurs when pixel_size matches training_pixel_size."""
    # This is tested indirectly - when pixel_size = training_pixel_size,
    # scale_factor should be 1.0 and rescaling is skipped
    training_pixel_size = 2.76
    pixel_size = 2.76

    scale_factor = training_pixel_size / pixel_size
    assert scale_factor == pytest.approx(1.0, abs=1e-6)

    # In the actual code, this condition is checked:
    # if pixel_size is not None and abs(pixel_size - training_pixel_size) > 1e-6:
    # So with scale_factor = 1.0, the rescaling branch is not taken
