"""
Unit tests for training infrastructure (Phase 3).

Tests config loading, checkpointing, early stopping, and core training functions.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from src.training.train import (
    CombinedLoss,
    DiceLoss,
    EarlyStopping,
    compute_dice_metric,
    create_model,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
)


class TestConfigLoading:
    """Test YAML config loading and saving."""

    def test_load_config(self):
        """Test loading config from YAML file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "model": {"encoder": "resnet18"},
                "training": {"epochs": 10, "batch_size": 4},
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded_config = load_config(config_path)
            assert loaded_config["model"]["encoder"] == "resnet18"
            assert loaded_config["training"]["epochs"] == 10
            assert loaded_config["training"]["batch_size"] == 4
        finally:
            config_path.unlink()

    def test_save_config(self):
        """Test saving config to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": {"encoder": "resnet34"},
                "training": {"learning_rate": 0.001},
            }
            output_path = Path(tmpdir) / "config.yaml"

            save_config(config, output_path)

            assert output_path.exists()

            # Verify saved content
            with open(output_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["model"]["encoder"] == "resnet34"
            assert loaded["training"]["learning_rate"] == 0.001


class TestLossFunctions:
    """Test loss functions."""

    def test_dice_loss_perfect_match(self):
        """Test Dice loss with perfect prediction."""
        loss_fn = DiceLoss()

        # Perfect prediction (before sigmoid)
        pred = torch.ones(2, 1, 4, 4) * 10.0  # High logits -> sigmoid ~1.0
        target = torch.ones(2, 1, 4, 4)

        loss = loss_fn(pred, target)

        # Loss should be close to 0 for perfect match
        assert loss < 0.01

    def test_dice_loss_no_match(self):
        """Test Dice loss with completely wrong prediction."""
        loss_fn = DiceLoss()

        # Completely wrong prediction
        pred = torch.ones(2, 1, 4, 4) * 10.0  # Predicting all foreground
        target = torch.zeros(2, 1, 4, 4)  # Ground truth is all background

        loss = loss_fn(pred, target)

        # Loss should be close to 1 for no overlap
        assert loss > 0.99

    def test_combined_loss(self):
        """Test combined BCE + Dice loss."""
        loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

        pred = torch.randn(2, 1, 4, 4)
        target = torch.randint(0, 2, (2, 1, 4, 4)).float()

        loss = loss_fn(pred, target)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestMetrics:
    """Test evaluation metrics."""

    def test_compute_dice_perfect(self):
        """Test Dice metric with perfect prediction."""
        # Perfect prediction (logits)
        pred = torch.ones(2, 1, 4, 4) * 10.0
        target = torch.ones(2, 1, 4, 4)

        dice = compute_dice_metric(pred, target, threshold=0.5)

        # Perfect match should give Dice = 1.0
        assert dice > 0.99

    def test_compute_dice_no_overlap(self):
        """Test Dice metric with no overlap."""
        # No overlap
        pred = torch.ones(2, 1, 4, 4) * 10.0  # Predicting all foreground
        target = torch.zeros(2, 1, 4, 4)  # Ground truth is all background

        dice = compute_dice_metric(pred, target, threshold=0.5)

        # No overlap should give low Dice
        assert dice < 0.01

    def test_compute_dice_partial_overlap(self):
        """Test Dice metric with partial overlap."""
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)

        # Set half of prediction to high (will be 1 after sigmoid)
        pred[0, 0, :2, :] = 10.0
        # Set different half of target to 1 (50% overlap of prediction with target)
        target[0, 0, 1:3, :] = 1.0

        dice = compute_dice_metric(pred, target, threshold=0.5)

        # Prediction: rows 0-1, Target: rows 1-2
        # Overlap: row 1 (4 pixels), Union: rows 0-2 (12 pixels)
        # Dice = 2*4/(8+8) = 8/16 = 0.5
        assert 0.45 < dice < 0.55


class TestModelCreation:
    """Test model creation."""

    def test_create_model_default(self):
        """Test creating model with default parameters."""
        model = create_model()

        assert model is not None
        # Check it's a PyTorch module
        assert isinstance(model, torch.nn.Module)

    def test_create_model_custom(self):
        """Test creating model with custom parameters."""
        model = create_model(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )

        assert model is not None
        assert isinstance(model, torch.nn.Module)


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_and_load_checkpoint(self):
        """Test checkpoint save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"

            # Create a simple model
            model = create_model(encoder_name="resnet18")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

            # Save checkpoint
            history = {"train_loss": [0.5, 0.4], "val_dice": [0.6, 0.7]}
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=5,
                best_val_dice=0.75,
                history=history,
            )

            assert checkpoint_path.exists()

            # Load checkpoint into new model
            new_model = create_model(encoder_name="resnet18")
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(new_optimizer)

            epoch, best_val_dice, loaded_history = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, new_scheduler
            )

            # Verify loaded values
            assert epoch == 5
            assert best_val_dice == 0.75
            assert loaded_history["train_loss"] == [0.5, 0.4]
            assert loaded_history["val_dice"] == [0.6, 0.7]

    def test_save_checkpoint_without_scheduler(self):
        """Test checkpoint saving without scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"

            model = create_model(encoder_name="resnet18")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Save without scheduler
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=3,
                best_val_dice=0.8,
                history={},
            )

            assert checkpoint_path.exists()

            # Load should work even without scheduler
            new_model = create_model(encoder_name="resnet18")
            epoch, best_val_dice, history = load_checkpoint(checkpoint_path, new_model)

            assert epoch == 3
            assert best_val_dice == 0.8


class TestEarlyStopping:
    """Test early stopping mechanism."""

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode (e.g., for Dice)."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode="max")

        # Improving scores - should not stop
        assert early_stopping(0.5) is False
        assert early_stopping(0.6) is False
        assert early_stopping(0.7) is False

        # No improvement for 3 epochs - should trigger
        assert early_stopping(0.69) is False  # Within patience
        assert early_stopping(0.69) is False  # Within patience
        assert early_stopping(0.69) is True  # Triggers early stopping

    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode (e.g., for loss)."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.001, mode="min")

        # Improving scores (decreasing) - should not stop
        assert early_stopping(1.0) is False
        assert early_stopping(0.8) is False
        assert early_stopping(0.6) is False

        # No improvement for 2 epochs - should trigger
        assert early_stopping(0.61) is False  # Within patience
        assert early_stopping(0.61) is True  # Triggers early stopping

    def test_early_stopping_min_delta(self):
        """Test early stopping respects min_delta."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1, mode="max")

        # Small improvements below min_delta should not count
        assert early_stopping(0.5) is False  # Best = 0.5
        assert early_stopping(0.55) is False  # Improvement = 0.05 < 0.1, counter = 1
        assert early_stopping(0.56) is True  # Improvement = 0.06 < 0.1, counter = 2, triggers

    def test_early_stopping_reset_on_improvement(self):
        """Test that early stopping counter resets on improvement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.02, mode="max")

        assert early_stopping(0.5) is False  # Best = 0.5
        assert early_stopping(0.51) is False  # Improvement = 0.01 < 0.02, counter = 1
        assert early_stopping(0.515) is True  # Improvement = 0.015 < 0.02, counter = 2, triggers

        # Test reset with significant improvement
        early_stopping2 = EarlyStopping(patience=2, min_delta=0.02, mode="max")
        assert early_stopping2(0.5) is False  # Best = 0.5
        assert early_stopping2(0.51) is False  # Improvement = 0.01 < 0.02, counter = 1
        assert early_stopping2(0.53) is False  # Improvement = 0.03 > 0.02! Counter resets, best = 0.53
        assert early_stopping2(0.54) is False  # Improvement = 0.01 < 0.02, counter = 1
        assert early_stopping2(0.545) is True  # Improvement = 0.005 < 0.02, counter = 2, triggers


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_forward_backward_pass(self):
        """Test that model can perform forward and backward pass."""
        model = create_model(encoder_name="resnet18")
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dummy batch
        images = torch.randn(2, 1, 256, 256)
        masks = torch.randint(0, 2, (2, 1, 256, 256)).float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_model_eval_mode(self):
        """Test that model switches to eval mode correctly."""
        model = create_model(encoder_name="resnet18")

        # Set to eval mode
        model.eval()

        # Forward pass should work without computing gradients
        with torch.no_grad():
            images = torch.randn(2, 1, 256, 256)
            outputs = model(images)

        assert outputs.shape == (2, 1, 256, 256)
