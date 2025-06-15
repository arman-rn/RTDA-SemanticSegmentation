"""
Defines the training logic for a single epoch of model training.

Includes vanilla segmentation training and adversarial domain adaptation training.

This module handles the forward pass, loss calculation, backward pass
(gradient computation), and optimizer step (weight updates) for each batch
in the training dataset. It supports mixed-precision training via
`torch.cuda.amp.GradScaler` if provided.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch import GradScaler, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from data_loader import InfiniteDataLoader
from utils import poly_lr_scheduler

# Type alias for the config module for clarity
ConfigModule = Any


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


# --- Adversarial Training Function ---
def train_one_epoch_adversarial(
    # --- Generator (Segmentation Model) Components ---
    model_G: nn.Module,
    optimizer_G: optim.Optimizer,
    criterion_seg: nn.Module,  # Segmentation loss (e.g., CrossEntropy)
    train_loader_source: DataLoader,  # DataLoader for labeled source domain data
    initial_base_lr_G: float,
    # --- Discriminator Components ---
    model_D: nn.Module,
    optimizer_D: optim.Optimizer,
    criterion_adv: nn.Module,  # Adversarial loss (e.g., BCEWithLogitsLoss)
    train_loader_target: InfiniteDataLoader,  # DataLoader for unlabeled target domain data
    initial_base_lr_D: float,
    # --- Common Training Loop Parameters ---
    device: torch.device,
    epoch: int,  # Current epoch number, 0-indexed
    global_step_offset: int,
    max_iter: int,  # Max iterations for the entire training, for LR scheduling (for G)
    effective_total_epochs: int,
    config_module_ref: ConfigModule,  # Reference to the config module (cfg)
    scaler: Optional[GradScaler] = None,
) -> Tuple[Dict[str, float], int]:  # Returns dict of average losses and new global_step
    """
    Trains the Generator (model_G) and Discriminator (model_D) for one epoch
    using an adversarial domain adaptation strategy.
    """
    model_G.train()
    model_D.train()

    running_loss_seg_G = 0.0  # Segmentation loss for Generator
    running_loss_adv_G = 0.0  # Adversarial loss for Generator (to fool D)
    running_loss_D_total = 0.0  # Total loss for Discriminator

    # Labels for adversarial training (Discriminator's perspective)
    # D wants to output 1 for source (real), 0 for target (fake)
    # As per Paper [7] (Tsai et al.) interpretation for BCEWithLogitsLoss:
    # L_d = - sum(z*log(D(P)) + (1-z)*log(1-D(P))), where z=1 for source, z=0 for target.
    # If D(P) is sigmoid(logit), then for BCEWithLogitsLoss, target for source is 1, target for target is 0.
    real_label = 1.0  # Label for source domain samples (real)
    fake_label = 0.0  # Label for target domain samples (fake)

    progress_bar = tqdm(
        enumerate(train_loader_source),
        total=len(train_loader_source),
        desc=f"Epoch {epoch + 1}/{effective_total_epochs} [Adv. Training]",
        unit="batch",
        leave=False,
    )
    current_global_step = global_step_offset
    num_batches_source = len(train_loader_source)

    for batch_idx, (images_s, labels_s) in progress_bar:
        images_s = images_s.to(device)  # Source images (e.g., GTA5)
        labels_s = labels_s.to(device).long()  # Source labels

        # Fetch target domain images (unlabeled)
        images_t, _ = next(train_loader_target)  # Labels from target_loader are ignored
        images_t = images_t.to(device)  # Target images (e.g., Cityscapes)

        # Ensure batch sizes are consistent if models/losses require it (usually they do)
        if images_s.size(0) != images_t.size(0):
            print(
                f"Warning: Batch size mismatch. Source: {images_s.size(0)}, Target: {images_t.size(0)}. Skipping."
            )
            continue

        # --- Update Learning Rates for BOTH G and D ---
        lr_power = config_module_ref.LR_SCHEDULER_POWER
        current_lr_G = poly_lr_scheduler(
            optimizer_G, initial_base_lr_G, current_global_step, max_iter, lr_power
        )
        current_lr_D = poly_lr_scheduler(
            optimizer_D, initial_base_lr_D, current_global_step, max_iter, lr_power
        )

        # --- 1. Train Discriminator (model_D) ---
        # Maximize log(D(G_s(x_s))) + log(1 - D(G_t(x_t)))
        # D wants to assign "real_label" to source and "fake_label" to target
        optimizer_D.zero_grad(set_to_none=True)

        # On source data (real)
        with torch.no_grad():  # Don't track gradients for G during D's source pass
            pred_s_logits_G_detached = (
                model_G(images_s)[0]
                if isinstance(model_G(images_s), tuple)
                else model_G(images_s)
            )

        # Discriminator takes probability maps as input [cite: 76, 86]
        input_d_source = F.softmax(pred_s_logits_G_detached, dim=1).detach()

        if scaler:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=True
            ):
                d_out_source = model_D(input_d_source)
                loss_d_real = criterion_adv(
                    d_out_source,
                    torch.full_like(d_out_source, real_label, device=device),
                )
        else:
            d_out_source = model_D(input_d_source)
            loss_d_real = criterion_adv(
                d_out_source, torch.full_like(d_out_source, real_label, device=device)
            )

        # On target data (fake)
        with torch.no_grad():  # Don't track gradients for G during D's target pass
            pred_t_logits_G_detached = (
                model_G(images_t)[0]
                if isinstance(model_G(images_t), tuple)
                else model_G(images_t)
            )
        input_d_target = F.softmax(pred_t_logits_G_detached, dim=1).detach()

        if scaler:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=True
            ):
                d_out_target = model_D(input_d_target)
                loss_d_fake = criterion_adv(
                    d_out_target,
                    torch.full_like(d_out_target, fake_label, device=device),
                )
        else:
            d_out_target = model_D(input_d_target)
            loss_d_fake = criterion_adv(
                d_out_target, torch.full_like(d_out_target, fake_label, device=device)
            )

        loss_D = (loss_d_real + loss_d_fake) * 0.5
        if scaler:
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            # scaler.update() will be called after G's step
        else:
            loss_D.backward()
            optimizer_D.step()
        running_loss_D_total += loss_D.item()

        # --- 2. Train Generator (model_G) ---
        # Maximize log(D(G_t(x_t))) (to fool D) + Minimize SegLoss(G_s(x_s), y_s)
        optimizer_G.zero_grad(set_to_none=True)

        # a) Segmentation loss on source data
        if scaler:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=True
            ):
                pred_s_logits_G = (
                    model_G(images_s)[0]
                    if isinstance(model_G(images_s), tuple)
                    else model_G(images_s)
                )
                loss_seg = criterion_seg(pred_s_logits_G, labels_s)
        else:
            pred_s_logits_G = (
                model_G(images_s)[0]
                if isinstance(model_G(images_s), tuple)
                else model_G(images_s)
            )
            loss_seg = criterion_seg(pred_s_logits_G, labels_s)
        running_loss_seg_G += loss_seg.item()

        # b) Adversarial loss on target data (G wants D to predict target as "real_label")
        # [cite: 89] (describes G's adv loss as maximizing D(P_t) being considered source)
        pred_t_logits_G_for_adv = (
            model_G(images_t)[0]
            if isinstance(model_G(images_t), tuple)
            else model_G(images_t)
        )
        input_d_target_for_g = F.softmax(pred_t_logits_G_for_adv, dim=1)

        if scaler:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=True
            ):
                d_out_target_for_g = model_D(input_d_target_for_g)
                loss_adv = criterion_adv(
                    d_out_target_for_g,
                    torch.full_like(d_out_target_for_g, real_label, device=device),
                )
        else:
            d_out_target_for_g = model_D(input_d_target_for_g)
            loss_adv = criterion_adv(
                d_out_target_for_g,
                torch.full_like(d_out_target_for_g, real_label, device=device),
            )
        running_loss_adv_G += loss_adv.item()

        # Total Generator loss
        lambda_adv = config_module_ref.ADVERSARIAL_LAMBDA_ADV_GENERATOR
        loss_G_total = loss_seg + lambda_adv * loss_adv
        if scaler:
            scaler.scale(loss_G_total).backward()
            scaler.step(optimizer_G)
        else:
            loss_G_total.backward()
            optimizer_G.step()

        # Single scaler update after both G and D steps
        if scaler:
            scaler.update()

        # --- Logging & Progress Bar ---
        postfix_dict = {
            "L_seg": f"{loss_seg.item():.3f}",
            "L_adv_G": f"{loss_adv.item():.3f}",
            "L_D": f"{loss_D.item():.3f}",
            "lr_G": f"{current_lr_G:.2e}",
            "lr_D": f"{current_lr_D:.2e}",
        }
        progress_bar.set_postfix(**postfix_dict)

        if wandb.run and (
            current_global_step % config_module_ref.PRINT_FREQ_BATCH == 0
            or batch_idx == num_batches_source - 1
        ):
            log_payload = {
                "train_adv/batch_loss_seg_G": loss_seg.item(),
                "train_adv/batch_loss_adv_G": loss_adv.item(),
                "train_adv/batch_loss_D": loss_D.item(),
                "train_adv/learning_rate_G": current_lr_G,
                "train_adv/learning_rate_D": current_lr_D,
            }
            wandb.log(log_payload, step=current_global_step)

        current_global_step += 1

    # Calculate average losses for the epoch
    avg_losses_epoch: Dict[str, float] = {
        "seg_loss_G": running_loss_seg_G / num_batches_source
        if num_batches_source > 0
        else 0.0,
        "adv_loss_G": running_loss_adv_G / num_batches_source
        if num_batches_source > 0
        else 0.0,
        "loss_D_total": running_loss_D_total / num_batches_source
        if num_batches_source > 0
        else 0.0,
    }

    return avg_losses_epoch, current_global_step
