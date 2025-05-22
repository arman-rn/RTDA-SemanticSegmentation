# This file defines the logic for training the model for a single epoch.
# It handles the forward pass, loss calculation, backward pass (gradient computation), and optimizer step (weight updates).

import torch
from tqdm import tqdm

import config as cfg
import wandb
from utils import poly_lr_scheduler


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    global_step_offset,
    max_iter,
    initial_base_lr,
    scaler=None,
):
    # """
    # Trains the model for one epoch.
    # Args:
    #     model: The neural network model.
    #     train_loader: DataLoader for training data.
    #     optimizer: The optimizer.
    #     criterion: The loss function.
    #     device: The device to train on (e.g., 'cuda' or 'cpu').
    #     epoch: The current epoch number (0-indexed).
    #     global_step_offset: The total number of batches processed in previous epochs, used to correctly update the learning rate scheduler and for W&B logging.
    #     max_iter: Total number of iterations (batches) over all epochs, for poly_lr_scheduler.
    #     scaler:  A torch.cuda.amp.GradScaler object for mixed-precision training (if enabled and on GPU). Mixed precision can speed up training and reduce memory usage.
    # Returns:
    #     tuple: (average_training_loss_for_the_epoch, current_global_step)
    # """
    model.train()  # Sets the model to training mode (enables dropout, updates batch norm statistics if they weren't frozen).
    running_loss = 0.0  # Initialize running loss for the epoch

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{cfg.TRAIN_EPOCHS} [Training]",
        unit="batch",
        leave=False,
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
            loss.backward()
            optimizer.step()

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
