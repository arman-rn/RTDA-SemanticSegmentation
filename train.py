"""
Defines the training logic for a single epoch of model training.

This module handles the forward pass, loss calculation, backward pass
(gradient computation), and optimizer step (weight updates) for each batch
in the training dataset. It supports mixed-precision training via
`torch.cuda.amp.GradScaler` if provided.
"""
# This file defines the logic for training the model for a single epoch.
# It handles the forward pass, loss calculation, backward pass (gradient computation), and optimizer step (weight updates).

from typing import Any, Optional, Tuple

import torch
from torch import GradScaler, nn, optim  # For type hints
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg  # Assuming config.py is accessible
import wandb  # For logging to Weights & Biases
from utils import poly_lr_scheduler

# Type alias for the config module for clarity
ConfigModule = Any  # Could be replaced with a Protocol if config structure is strict


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,  # e.g., nn.CrossEntropyLoss
    device: torch.device,
    epoch: int,  # Current epoch number, 0-indexed
    global_step_offset: int,
    max_iter: int,
    initial_base_lr: float,
    effective_total_epochs: int,  # For accurate tqdm display
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, int]:
    """
    Trains the model for one epoch.

    Iterates over the training data, performs forward and backward passes,
    updates model weights, and logs training metrics. Supports optional
    mixed-precision training using a GradScaler.

    Args:
        model: The neural network model to be trained.
        train_loader: PyTorch DataLoader for the training dataset.
        optimizer: The PyTorch optimizer (e.g., SGD, Adam).
        criterion: The loss function (e.g., CrossEntropyLoss).
        device: The PyTorch device to perform training on (e.g., 'cuda', 'cpu').
        epoch: The current epoch number (0-indexed for internal logic,
               often displayed as 1-indexed).
        global_step_offset: The total number of batches processed in previous epochs.
                            Used to maintain a continuous global step count.
        max_iter: The total number of training iterations (batches) over all epochs,
                  used by the learning rate scheduler.
        initial_base_lr: The initial learning rate that was set for the
                         optimizer's primary parameter group, used by the LR scheduler.
        effective_total_epochs: The total number of epochs this training run will perform
                                (can be from config or overridden by CLI). Used for display.
        scaler: An optional `torch.GradScaler` instance for mixed-precision
                training. If None, full precision (FP32) training is performed.

    Returns:
        A tuple containing:
        - avg_epoch_loss (float): The average training loss over the epoch.
        - current_global_step (int): The updated global step count after the epoch.
    """

    model.train()  # Sets the model to training mode (enables dropout, updates batch norm statistics if they weren't frozen).
    running_loss = 0.0  # Initialize running loss for the epoch

    # Progress bar for iterating over batches
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{effective_total_epochs} [Training]",  # Use effective_total_epochs
        unit="batch",
        leave=False,  # Keep the bar from staying after completion if nested
    )
    current_global_step = global_step_offset

    # Batch loop
    # Loop through the batches in the training data
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)  # Moves data to the target device.
        labels = labels.to(device).long()  # Moves data to the target device.

        # Update LR using poly scheduler based on current_global_step
        current_lr = poly_lr_scheduler(
            optimizer,
            initial_base_lr,
            current_global_step,
            max_iter,
            cfg.LR_SCHEDULER_POWER,
        )

        # Clears old gradients. `set_to_none=True` can be slightly more memory efficient.
        optimizer.zero_grad(set_to_none=True)

        if scaler:  # Mixed-Precision Handling
            # Enables automatic mixed precision for the forward pass.
            # Operations are performed in FP16 where possible.
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=True
            ):
                # Forward pass. deeplabv2.py returns (final_output, None, None) during training.
                outputs_tuple = model(images)

                # Extract the primary output.
                main_output = (
                    outputs_tuple[0]
                    if isinstance(outputs_tuple, tuple)
                    else outputs_tuple
                )

                # Compute the loss.
                loss = criterion(main_output, labels)

            # Scale the loss (for FP16 stability) and computes gradients.
            scaler.scale(loss).backward()
            # Unscale gradients and calls optimizer.step().
            scaler.step(optimizer)
            # Updates the scaler for the next iteration.
            scaler.update()
        else:  # Full-Precision Handling
            outputs_tuple = model(images)
            main_output = (
                outputs_tuple[0] if isinstance(outputs_tuple, tuple) else outputs_tuple
            )
            loss = criterion(main_output, labels)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

        # Accumulate the batch loss.
        running_loss += loss.item()

        # Update the tqdm progress bar with current loss and LR.
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        # Log batch loss and LR to W&B periodically, using current_global_step as the x-axis.
        if wandb.run and (
            current_global_step % cfg.PRINT_FREQ_BATCH == 0
            or batch_idx == len(train_loader) - 1
        ):
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": current_lr,
                },
                step=current_global_step,
            )

        current_global_step += 1

    avg_epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
    return avg_epoch_loss, current_global_step
