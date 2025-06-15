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
    model_D_main: nn.Module,
    optimizer_D_main: optim.Optimizer,
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
    model_D_aux: Optional[
        nn.Module
    ],  # Optional auxiliary discriminator for multi-level training
    optimizer_D_aux: Optional[
        optim.Optimizer
    ],  # Optional optimizer for auxiliary discriminator
    scaler: Optional[GradScaler] = None,
) -> Tuple[Dict[str, float], int]:  # Returns dict of average losses and new global_step
    """
    Trains the Generator (model_G) and Discriminator (model_D) for one epoch
    using an adversarial domain adaptation strategy.
    """
    is_multilevel = model_D_aux is not None and optimizer_D_aux is not None

    model_G.train()
    model_D_main.train()
    if is_multilevel and model_D_aux is not None:
        model_D_aux.train()

    # --- Loss Accumulators for the epoch ---
    running_losses = {
        "seg_main": 0.0,  # Main segmentation loss for Generator
        "seg_aux": 0.0,  # Auxiliary segmentation loss for Generator (if applicable)
        "adv_main": 0.0,  # Main adversarial loss for Generator
        "adv_aux": 0.0,  # Auxiliary adversarial loss for Generator (if applicable)
        "d_main": 0.0,  # Main loss for Discriminator
        "d_aux": 0.0,  # Auxiliary loss for Discriminator (if applicable)
    }

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
        desc=f"Epoch {epoch + 1}/{effective_total_epochs} [{'Multi' if is_multilevel else 'Single'} Adv. Training]",
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
            optimizer_D_main, initial_base_lr_D, current_global_step, max_iter, lr_power
        )

        if is_multilevel and optimizer_D_aux is not None:
            poly_lr_scheduler(
                optimizer_D_aux,
                initial_base_lr_D,
                current_global_step,
                max_iter,
                lr_power,
            )

        # ========================== 1. Train Discriminators ==========================
        optimizer_D_main.zero_grad(set_to_none=True)
        if is_multilevel and optimizer_D_aux is not None:
            optimizer_D_aux.zero_grad(set_to_none=True)

        with torch.no_grad():
            out_s_main, _, out_s_aux = model_G(images_s)
            out_t_main, _, out_t_aux = model_G(images_t)

        d_main_loss = train_discriminator(
            model_D_main,
            criterion_adv,
            out_s_main,
            out_t_main,
            scaler,
            real_label,
            fake_label,
            device,
        )
        running_losses["d_main"] += d_main_loss.item()

        d_aux_loss = torch.tensor(0.0, device=device)
        if is_multilevel:
            d_aux_loss = train_discriminator(
                model_D_aux,
                criterion_adv,
                out_s_aux,
                out_t_aux,
                scaler,
                real_label,
                fake_label,
                device,
            )
            running_losses["d_aux"] += d_aux_loss.item()

        d_total_loss = d_main_loss + d_aux_loss
        if scaler:
            scaler.scale(d_total_loss).backward()
            scaler.step(optimizer_D_main)
            if is_multilevel and optimizer_D_aux is not None:
                scaler.step(optimizer_D_aux)
        else:
            d_total_loss.backward()
            optimizer_D_main.step()
            if is_multilevel and optimizer_D_aux is not None:
                optimizer_D_aux.step()

        # ========================== 2. Train Generator ==========================
        optimizer_G.zero_grad(set_to_none=True)

        out_s_main_G, out_s_aux1_G, out_s_aux2_G = model_G(images_s)
        out_t_main_G, _, out_t_aux_G = model_G(images_t)

        loss_G_total = torch.tensor(0.0, device=device)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(scaler is not None)
        ):
            loss_seg_main = criterion_seg(out_s_main_G, labels_s)
            loss_seg_total = loss_seg_main

            if is_multilevel:
                loss_seg_aux1 = criterion_seg(out_s_aux1_G, labels_s)
                loss_seg_aux2 = criterion_seg(out_s_aux2_G, labels_s)
                loss_seg_total += config_module_ref.ADVERSARIAL_LAMBDA_SEG_AUX * (
                    loss_seg_aux1 + loss_seg_aux2
                )

            loss_adv_main = adversarial_loss_for_generator(
                model_D_main, criterion_adv, out_t_main_G, real_label, device
            )

            loss_adv_aux = torch.tensor(0.0, device=device)
            if is_multilevel:
                loss_adv_aux = adversarial_loss_for_generator(
                    model_D_aux, criterion_adv, out_t_aux_G, real_label, device
                )

            loss_G_total = (
                loss_seg_total
                + config_module_ref.ADVERSARIAL_LAMBDA_ADV_GENERATOR * loss_adv_main
                + (
                    config_module_ref.ADVERSARIAL_LAMBDA_ADV_AUX * loss_adv_aux
                    if is_multilevel
                    else 0
                )
            )

        if scaler:
            scaler.scale(loss_G_total).backward()
            scaler.step(optimizer_G)
        else:
            loss_G_total.backward()
            optimizer_G.step()

        # Single scaler update after both G and D steps
        if scaler:
            scaler.update()

        # --- Accumulate Generator losses for epoch average ---
        running_losses["seg_main"] += loss_seg_main.item()
        if is_multilevel:
            running_losses["seg_aux"] += loss_seg_aux1.item() + loss_seg_aux2.item()
        running_losses["adv_main"] += loss_adv_main.item()
        if is_multilevel:
            running_losses["adv_aux"] += loss_adv_aux.item()

        # --- Logging & Progress Bar ---
        postfix_dict = {
            "L_seg": f"{loss_seg_main.item():.2f}",
            "L_adv": f"{loss_adv_main.item():.2f}",
            "L_D": f"{d_main_loss.item():.2f}",
        }
        if is_multilevel:
            postfix_dict["L_adv_aux"] = f"{loss_adv_aux.item():.2f}"
            postfix_dict["L_D_aux"] = f"{d_aux_loss.item():.2f}"
        progress_bar.set_postfix(**postfix_dict)

        if wandb.run and (
            current_global_step % config_module_ref.PRINT_FREQ_BATCH == 0
        ):
            log_data = {
                "train_adv/batch_loss_seg_main": loss_seg_main.item(),
                "train_adv/batch_loss_adv_main": loss_adv_main.item(),
                "train_adv/batch_loss_D_main": d_main_loss.item(),
                "train_adv/lr_G": current_lr_G,
                "train_adv/lr_D": current_lr_D,
            }
            if is_multilevel:
                log_data["train_adv/batch_loss_seg_aux"] = (
                    loss_seg_aux1.item() + loss_seg_aux2.item()
                )
                log_data["train_adv/batch_loss_adv_aux"] = loss_adv_aux.item()
                log_data["train_adv/batch_loss_D_aux"] = d_aux_loss.item()
            wandb.log(log_data, step=current_global_step)

        current_global_step += 1

    avg_losses_epoch = {
        key: val / num_batches_source for key, val in running_losses.items() if val > 0
    }
    return avg_losses_epoch, current_global_step


# --- Helper Functions for Clarity ---
def train_discriminator(
    model_D, criterion, out_s, out_t, scaler, real_label, fake_label, device
):
    """Helper to compute loss for one discriminator, with autocast if scaler is used."""
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(scaler is not None)
    ):
        input_s = F.softmax(out_s, dim=1).detach()
        input_t = F.softmax(out_t, dim=1).detach()

        d_out_s = model_D(input_s)
        loss_real = criterion(
            d_out_s, torch.full_like(d_out_s, real_label, device=device)
        )

        d_out_t = model_D(input_t)
        loss_fake = criterion(
            d_out_t, torch.full_like(d_out_t, fake_label, device=device)
        )

    return (loss_real + loss_fake) * 0.5


def adversarial_loss_for_generator(model_D, criterion, out_t_G, real_label, device):
    """Helper to compute adversarial loss for the generator."""
    input_d_t_G = F.softmax(out_t_G, dim=1)
    d_out_t_G = model_D(input_d_t_G)
    return criterion(d_out_t_G, torch.full_like(d_out_t_G, real_label, device=device))
