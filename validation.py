# This file defines the logic to evaluate the model's performance on the validation dataset.
# It calculates metrics like validation loss and Mean Intersection over Union (mIoU).
import numpy as np
import torch
import wandb
from tqdm import tqdm

import config as cfg
from utils import fast_hist, log_segmentation_to_wandb, per_class_iou


@torch.no_grad()  # Disables gradient calculations for all operations within the decorated function. This is crucial for validation/inference as it reduces memory usage and speeds up computation.
def validate_and_log(
    model, val_loader, criterion, device, epoch, global_step
):  # Added global_step for W&B
    """
    Validates the model and logs metrics to W&B.
    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run validation on.
        epoch: The current epoch number (0-indexed for logging).
        global_step: Current global training step for W&B logging.
    Returns:
        tuple: (mean_iou, average_validation_loss)
    """
    # Set the model to evaluation mode (disables dropout, uses fixed batch norm statistics).
    model.eval()

    # Initialize total validation loss
    total_val_loss = 0

    # Initialize confusion matrix for IoU calculation
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))

    first_batch_images_logged = False  #  Flag to ensure sample images are logged only once per validation run if desired.

    # Wraps val_loader for a progress bar.
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{cfg.TRAIN_EPOCHS} [Validation]",
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

        # Gets the predicted class ID for each pixel by taking the argmax along the channel dimension of the output logits.
        preds = torch.argmax(outputs, dim=1)

        # Convert tensors to numpy arrays for histogram calculation
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Creates a mask to exclude ignored pixels from mIoU calculation.
        valid_pixels_mask = labels_np != cfg.IGNORE_INDEX

        # Updates the confusion matrix hist using only valid pixels.
        hist += fast_hist(
            labels_np[valid_pixels_mask].flatten(),
            preds_np[valid_pixels_mask].flatten(),
            cfg.NUM_CLASSES,
        )

        # W&B Image Logging
        # Periodically logs a batch of images with their ground truth and predicted masks to W&B.
        if (
            not first_batch_images_logged
            and wandb.run
            and (epoch + 1) % cfg.WANDB_LOG_IMAGES_FREQ_EPOCH == 0
            and (epoch + 1) > 0
        ):  # Ensure not epoch 0 if 0-indexed
            log_segmentation_to_wandb(
                images, labels, preds, epoch + 1, cfg
            )  # Pass cfg for NORM_MEAN/STD
            first_batch_images_logged = True

    # Calculate average validation loss
    avg_val_loss = (
        total_val_loss / len(val_loader) if criterion and len(val_loader) > 0 else 0.0
    )

    # Calculate Mean IoU
    ious = per_class_iou(hist)  # This now handles nan_to_num
    # Calculate mean IoU across all classes
    mean_iou = np.mean(ious)

    # W&B Logging (Epoch-level Validation)
    # Logs validation loss, mIoU, and per-class IoUs to W&B, using global_step to align with training.
    if wandb.run:
        wandb_log_dict = {
            "val/epoch_loss": avg_val_loss,
            "val/mIoU": mean_iou,
            "epoch": epoch + 1,  # Log 1-indexed epoch
        }
        for i, iou_val in enumerate(ious):
            wandb_log_dict[f"val/iou_class_{i}"] = iou_val
        wandb.log(wandb_log_dict, step=global_step)  # Log against global_step

    print(
        f"\nValidation Epoch {epoch + 1}: Avg Loss: {avg_val_loss:.4f}, Mean IoU: {mean_iou:.4f}"
    )

    return mean_iou, avg_val_loss
