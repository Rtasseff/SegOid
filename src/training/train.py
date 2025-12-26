"""
Training infrastructure for Phase 3: Full model training.

Provides production-grade training with config loading, checkpointing,
early stopping, LR scheduling, and TensorBoard logging.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def load_config(config_path: Path) -> dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: dict, output_path: Path) -> None:
    """
    Save config snapshot to output directory.

    Args:
        config: Config dictionary
        output_path: Path to save config
    """
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config snapshot to {output_path}")


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_dice: float,
    history: dict,
) -> None:
    """
    Save training checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: LR scheduler to save (optional)
        epoch: Current epoch
        best_val_dice: Best validation Dice so far
        history: Training history
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "history": history,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float, dict]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)

    Returns:
        Tuple of (epoch, best_val_dice, history)
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_val_dice = checkpoint.get("best_val_dice", 0.0)
    history = checkpoint.get("history", {})

    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")

    return epoch, best_val_dice, history


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
) -> Tuple[float, float, dict]:
    """
    Validate for one epoch.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (avg_loss, avg_dice, sample_batch) where sample_batch
        contains images, masks, and predictions for visualization
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    sample_batch = None

    pbar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
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

            # Save first batch for visualization
            if batch_idx == 0:
                sample_batch = {
                    "images": images.cpu(),
                    "masks": masks.cpu(),
                    "predictions": torch.sigmoid(outputs).cpu(),
                }

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice, sample_batch


def log_images_to_tensorboard(
    writer: SummaryWriter,
    sample_batch: dict,
    epoch: int,
    num_samples: int = 4,
) -> None:
    """
    Log sample predictions to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        sample_batch: Dict with 'images', 'masks', 'predictions'
        epoch: Current epoch
        num_samples: Number of samples to log
    """
    images = sample_batch["images"][:num_samples]
    masks = sample_batch["masks"][:num_samples]
    predictions = sample_batch["predictions"][:num_samples]

    # Normalize images to [0, 1] for visualization
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)

    # Create visualization grid: [image, mask, prediction] for each sample
    for i in range(min(num_samples, images.size(0))):
        img = images[i]
        mask = masks[i]
        pred = predictions[i]

        # Stack horizontally: image, mask, prediction
        viz = torch.cat([img, mask, pred], dim=2)  # Concatenate along width

        writer.add_image(f"Sample_{i}/Image_Mask_Prediction", viz, epoch)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics to maximize (e.g., Dice), 'min' for minimize
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def train_model(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, list]:
    """
    Full training loop with all production features.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        run_dir: Directory to save outputs
        resume_checkpoint: Optional checkpoint to resume from

    Returns:
        Dictionary containing training history
    """
    # Create run directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # Save config snapshot
    save_config(config, run_dir / "config.yaml")

    # Create model
    model_config = config["model"]
    model = create_model(
        encoder_name=model_config["encoder"],
        encoder_weights=model_config["encoder_weights"],
        in_channels=model_config["in_channels"],
        classes=model_config["classes"],
    )
    model = model.to(device)

    # Setup loss and optimizer
    loss_config = config["loss"]
    criterion = CombinedLoss(
        bce_weight=loss_config["bce_weight"],
        dice_weight=loss_config["dice_weight"],
    )

    training_config = config["training"]
    optimizer = Adam(model.parameters(), lr=training_config["learning_rate"])

    # Setup LR scheduler
    scheduler = None
    if config.get("lr_scheduler", {}).get("enabled", False):
        lr_config = config["lr_scheduler"]
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=lr_config["mode"],
            factor=lr_config["factor"],
            patience=lr_config["patience"],
            min_lr=lr_config["min_lr"],
        )

    # Setup early stopping
    early_stopping = None
    if config.get("early_stopping", {}).get("enabled", False):
        es_config = config["early_stopping"]
        early_stopping = EarlyStopping(
            patience=es_config["patience"],
            min_delta=es_config["min_delta"],
            mode="max",  # Maximize validation Dice
        )

    # Setup TensorBoard
    writer = None
    if config.get("tensorboard", {}).get("enabled", False):
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Training state
    start_epoch = 0
    best_val_dice = 0.0
    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
        "learning_rate": [],
    }

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        start_epoch, best_val_dice, history = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch

    logger.info(f"Starting training for {training_config['epochs']} epochs")
    logger.info(f"Learning rate: {training_config['learning_rate']}")
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")

    # Training loop
    num_epochs = training_config["epochs"]
    for epoch in range(start_epoch, num_epochs):
        epoch_num = epoch + 1  # 1-indexed for display

        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch_num
        )

        # Validate
        val_loss, val_dice, sample_batch = validate_epoch(
            model, val_loader, criterion, device
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log results
        logger.info(
            f"Epoch {epoch_num}/{num_epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Dice: {train_dice:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Dice: {val_dice:.4f} - "
            f"LR: {current_lr:.6f}"
        )

        print(
            f"Epoch {epoch_num}/{num_epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Val Dice: {val_dice:.4f} - "
            f"LR: {current_lr:.6f}"
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["learning_rate"].append(current_lr)

        # TensorBoard logging
        if writer is not None:
            tb_config = config["tensorboard"]
            writer.add_scalar("Loss/train", train_loss, epoch_num)
            writer.add_scalar("Loss/val", val_loss, epoch_num)
            writer.add_scalar("Dice/train", train_dice, epoch_num)
            writer.add_scalar("Dice/val", val_dice, epoch_num)
            writer.add_scalar("Learning_Rate", current_lr, epoch_num)

            # Log images at specified interval
            if epoch_num % tb_config["image_interval"] == 0:
                log_images_to_tensorboard(
                    writer, sample_batch, epoch_num, tb_config["num_image_samples"]
                )

        # LR scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

            # Log LR changes
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                logger.info(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")

        # Save best model
        if config.get("checkpointing", {}).get("save_best", True):
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_model_path = checkpoint_dir / "best_model.pth"
                save_checkpoint(
                    best_model_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_dice,
                    history,
                )
                logger.info(f"New best model! Val Dice: {best_val_dice:.4f}")

        # Save periodic checkpoint
        if config.get("checkpointing", {}).get("save_periodic", False):
            periodic_interval = config["checkpointing"]["periodic_interval"]
            if epoch_num % periodic_interval == 0:
                periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch_num:03d}.pth"
                save_checkpoint(
                    periodic_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_dice,
                    history,
                )

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_dice):
                logger.info(
                    f"Early stopping triggered after {epoch_num} epochs. "
                    f"Best val Dice: {best_val_dice:.4f}"
                )
                print(
                    f"Early stopping triggered. Best val Dice: {best_val_dice:.4f}"
                )
                break

    # Save final model
    if config.get("checkpointing", {}).get("save_final", True):
        final_path = checkpoint_dir / "final_model.pth"
        save_checkpoint(
            final_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_dice,
            history,
        )

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    logger.info(f"Training complete! Best val Dice: {best_val_dice:.4f}")
    logger.info(f"Outputs saved to {run_dir}")

    return history
