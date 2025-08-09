"""
Defines the model evaluation logic on the validation dataset.

This module calculates metrics such as validation loss and Mean Intersection
over Union (mIoU). It's typically called after each training epoch or periodically.
Gradients are disabled during validation for efficiency.
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import CITYSCAPES_ID_TO_NAME_MAP
from utils import fast_hist, log_segmentation_to_wandb, per_class_iou

ConfigModule = Any


@torch.no_grad()  # Disables gradient calculations for all operations within the decorated function. This is crucial for validation/inference as it reduces memory usage and speeds up computation.
def validate_and_log(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Optional[nn.Module],  # Criterion can be None if only mIoU is needed
    device: torch.device,
    epoch: int,  # Current epoch number, 0-indexed
    global_step: int,
    effective_total_epochs: int,  # For accurate tqdm display
    config_module_ref: ConfigModule,  # Pass the config module for NORM_MEAN/STD
) -> Tuple[float, float, np.ndarray]:  # Returns: mIoU, avg_loss, all_class_ious_array
    """
    Evaluates the model on the validation dataset and logs metrics.

    Calculates validation loss (if criterion is provided) and Mean Intersection
    over Union (mIoU). Logs these metrics and sample segmentation masks to
    Weights & Biases.

    Args:
        model: The neural network model to be evaluated.
        val_loader: PyTorch DataLoader for the validation dataset.
        criterion: The loss function. Can be None if loss is not to be computed.
        device: The PyTorch device to perform validation on.
        epoch: The current epoch number (0-indexed from training loop,
               displayed as 1-indexed for logging).
        global_step: The current global training step, used for aligning
                     W&B logs with training progress.
        effective_total_epochs: The total number of epochs for the current training run,
                                used for accurate tqdm progress bar display.
        config_module_ref: A reference to the configuration module (e.g., `cfg`)
                           to access parameters like NORM_MEAN, NORM_STD for
                           image denormalization during W&B logging.

    Returns:
        A tuple containing:
        - mean_iou (float): The mean Intersection over Union over all classes.
        - avg_val_loss (float): The average validation loss. Returns 0.0 if
                                criterion is None.
    """
    # Set the model to evaluation mode (disables dropout, uses fixed batch norm statistics).
    model.eval()

    # Initialize total validation loss
    total_val_loss = 0

    # Initialize confusion matrix for IoU calculation
    hist = np.zeros((config_module_ref.NUM_CLASSES, config_module_ref.NUM_CLASSES))

    first_batch_images_logged = False  #  Flag to ensure sample images are logged only once per validation run if desired.

    # Wraps val_loader for a progress bar.
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{effective_total_epochs} [Validation]",  # Use effective_total_epochs
        unit="batch",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).long()

        # Forward pass. `model.eval()` mode ensures deeplabv2.py returns a single output tensor.
        outputs = model(images)

        if criterion:  # Can be None if only evaluating mIoU
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Gets the predicted class ID for each pixel by taking the argmax along the channel dimension of the output logits. Get predicted class IDs (shape: B, H, W)
        preds = torch.argmax(outputs, dim=1)

        # Convert tensors to numpy arrays for histogram calculation
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Creates a mask to exclude ignored pixels from mIoU calculation.
        valid_pixels_mask = labels_np != config_module_ref.IGNORE_INDEX

        # Updates the confusion matrix hist using only valid pixels.
        hist += fast_hist(
            labels_np[valid_pixels_mask].flatten(),
            preds_np[valid_pixels_mask].flatten(),
            config_module_ref.NUM_CLASSES,
        )

        # W&B Image Logging
        # Periodically logs a batch of images with their ground truth and predicted masks to W&B.
        if (
            not first_batch_images_logged
            and wandb.run
            and (epoch + 1) % config_module_ref.WANDB_LOG_IMAGES_FREQ_EPOCH == 0
            and (epoch + 1) > 0
        ):  # Ensure not epoch 0 if 0-indexed
            log_segmentation_to_wandb(
                images,
                labels,
                preds,
                epoch + 1,
                config_module_ref,
                global_step=global_step,
            )
            first_batch_images_logged = True

    # Calculate average validation loss
    avg_val_loss = (
        total_val_loss / len(val_loader) if criterion and len(val_loader) > 0 else 0.0
    )

    # Calculate Mean IoU
    all_class_ious = per_class_iou(hist)  # Array of IoUs for each class
    mean_iou_all_classes = float(np.nanmean(all_class_ious))

    # Console print after each validation epoch (general summary)
    print(
        f"\nValidation Epoch {epoch + 1}: Avg Loss: {avg_val_loss:.4f}, Overall Mean IoU: {mean_iou_all_classes:.4f}"
    )

    # W&B Logging (Epoch-level Validation)
    # Logs validation loss, mIoU, and per-class IoUs to W&B, using global_step to align with training.
    if wandb.run:
        log_payload = {
            "val/epoch_loss": avg_val_loss,
            "val/mIoU": mean_iou_all_classes,
            "epoch": epoch + 1,  # Log 1-indexed epoch
        }
        for i, iou_val in enumerate(all_class_ious):
            class_name = CITYSCAPES_ID_TO_NAME_MAP.get(i, f"class_{i}")
            log_payload[f"val_iou_per_class/iou_{class_name}"] = float(iou_val)
        wandb.log(log_payload, step=global_step)  # Log against global training step

    return mean_iou_all_classes, avg_val_loss, all_class_ious
