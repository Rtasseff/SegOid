"""
Training infrastructure for Phase 1.5 sanity check.

Provides model creation, loss functions, metrics, and basic training loop.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Computes 1 - Dice coefficient, where Dice = 2*|A∩B| / (|A| + |B|)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted mask logits or probabilities [B, 1, H, W]
            target: Ground truth binary mask [B, 1, H, W]

        Returns:
            Dice loss value (scalar)
        """
        # Apply sigmoid if needed (logits assumed)
        pred = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return loss (1 - Dice)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    Args:
        bce_weight: Weight for BCE loss (default: 0.5)
        dice_weight: Weight for Dice loss (default: 0.5)
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted mask logits [B, 1, H, W]
            target: Ground truth binary mask [B, 1, H, W]

        Returns:
            Combined loss value (scalar)
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_dice_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient metric for evaluation.

    Args:
        pred: Predicted mask logits [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        Dice coefficient (float)
    """
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred_binary = (pred > threshold).float()

        # Flatten
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        target = target.view(target.size(0), -1)

        # Compute Dice
        intersection = (pred_binary * target).sum(dim=1)
        union = pred_binary.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

        return dice.mean().item()


def create_model(
    encoder_name: str = "resnet18",
    encoder_weights: str = "imagenet",
    in_channels: int = 1,
    classes: int = 1,
) -> nn.Module:
    """
    Create U-Net model with specified encoder.

    Args:
        encoder_name: Encoder backbone (default: "resnet18")
        encoder_weights: Pretrained weights (default: "imagenet")
        in_channels: Number of input channels (default: 1 for grayscale)
        classes: Number of output classes (default: 1 for binary segmentation)

    Returns:
        U-Net model
    """
    logger.info(
        f"Creating U-Net: encoder={encoder_name}, weights={encoder_weights}, "
        f"in_channels={in_channels}, classes={classes}"
    )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number (for logging)

    Returns:
        Tuple of (avg_loss, avg_dice)
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        dice = compute_dice_metric(outputs, masks)

        # Accumulate
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate for one epoch.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (avg_loss, avg_dice)
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)

            # Compute metrics
            dice = compute_dice_metric(outputs, masks)

            # Accumulate
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, list]:
    """
    Full training loop.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoints

    Returns:
        Dictionary containing training history
    """
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")

    # Move model to device
    model = model.to(device)

    # Setup loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
    }

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)

        # Log results
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Dice: {train_dice:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Dice: {val_dice:.4f}"
        )

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Val Dice: {val_dice:.4f}"
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

    # Save final checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "final_checkpoint.pth"

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )

    logger.info(f"Saved final checkpoint to {checkpoint_path}")

    return history
