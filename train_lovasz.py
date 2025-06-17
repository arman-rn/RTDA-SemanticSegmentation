"""
Defines the training logic for a single epoch of model training.

Includes vanilla segmentation training, adversarial domain adaptation training,
and a new version for training with a combined Cross-Entropy and Lovasz-Softmax loss.

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


def train_one_epoch_lovasz(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_ce: nn.Module,
    criterion_lovasz: nn.Module,
    lovasz_weight: float,
    device: torch.device,
    epoch: int,
    global_step_offset: int,
    max_iter: int,
    initial_base_lr: float,
    effective_total_epochs: int,
    scaler: Optional[GradScaler] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Trains the model for one epoch using a combined Cross-Entropy and Lovasz-Softmax loss.
    """

    model.train()  # Sets the model to training mode (enables dropout, updates batch norm statistics if they weren't frozen).
    running_loss_total = 0.0
    running_loss_ce = 0.0
    running_loss_lovasz = 0.0

    # Progress bar for iterating over batches
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{effective_total_epochs} [Training w/ Lovasz]",  # Use effective_total_epochs
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
                # a) Standard Cross-Entropy loss
                loss_ce = criterion_ce(main_output, labels)

                # b) Lovasz-Softmax loss (requires probabilities from softmax)
                probas = F.softmax(main_output, dim=1)
                loss_lovasz = criterion_lovasz(probas, labels)

                # c) Combine the losses using the weight from the config
                loss_total = loss_ce + lovasz_weight * loss_lovasz

            # Scale the loss (for FP16 stability) and computes gradients.
            scaler.scale(loss_total).backward()
            # Unscale gradients and calls optimizer.step().
            scaler.step(optimizer)
            # Updates the scaler for the next iteration.
            scaler.update()
        else:  # Full-Precision Handling
            outputs_tuple = model(images)
            main_output = (
                outputs_tuple[0] if isinstance(outputs_tuple, tuple) else outputs_tuple
            )
            # Compute the loss.
            # a) Standard Cross-Entropy loss
            loss_ce = criterion_ce(main_output, labels)

            # b) Lovasz-Softmax loss (requires probabilities from softmax)
            probas = F.softmax(main_output, dim=1)
            loss_lovasz = criterion_lovasz(probas, labels)

            # c) Combine the losses using the weight from the config
            loss_total = loss_ce + lovasz_weight * loss_lovasz

            loss_total.backward()  # Compute gradients
            optimizer.step()  # Update weights

        # --- Accumulate each loss component ---
        running_loss_total += loss_total.item()
        running_loss_ce += loss_ce.item()
        running_loss_lovasz += loss_lovasz.item()

        # Update the tqdm progress bar with current loss and LR.
        progress_bar.set_postfix(
            loss=f"{loss_total.item():.4f}",
            ce=f"{loss_ce.item():.4f}",
            lovasz=f"{loss_lovasz.item():.4f}",
            lr=f"{current_lr:.2e}",
        )

        # Log batch loss and LR to W&B periodically, using current_global_step as the x-axis.
        if wandb.run and (
            current_global_step % cfg.PRINT_FREQ_BATCH == 0
            or batch_idx == len(train_loader) - 1
        ):
            wandb.log(
                {
                    "train_batch/loss_total": loss_total.item(),
                    "train_batch/loss_ce": loss_ce.item(),
                    "train_batch/loss_lovasz": loss_lovasz.item(),
                    "train/learning_rate": current_lr,
                },
                step=current_global_step,
            )

        current_global_step += 1

    # --- Calculate average for each loss component ---
    num_batches = len(train_loader)
    avg_losses_epoch = {
        "total": running_loss_total / num_batches if num_batches > 0 else 0,
        "ce": running_loss_ce / num_batches if num_batches > 0 else 0,
        "lovasz": running_loss_lovasz / num_batches if num_batches > 0 else 0,
    }
    return avg_losses_epoch, current_global_step


def train_one_epoch_adversarial_lovasz(
    # --- Generator (Segmentation Model) Components ---
    model_G: nn.Module,
    optimizer_G: optim.Optimizer,
    criterion_seg_ce: nn.Module,
    criterion_seg_lovasz: nn.Module,
    lovasz_weight: float,
    train_loader_source: DataLoader,
    initial_base_lr_G: float,
    # --- Discriminator Components ---
    model_D: nn.Module,
    optimizer_D: optim.Optimizer,
    criterion_adv: nn.Module,
    train_loader_target: InfiniteDataLoader,
    initial_base_lr_D: float,
    # --- Common Training Loop Parameters ---
    device: torch.device,
    epoch: int,
    global_step_offset: int,
    max_iter: int,
    effective_total_epochs: int,
    config_module_ref: ConfigModule,
    scaler: Optional[GradScaler] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Trains the Generator and Discriminator for one epoch using an adversarial
    strategy, where the Generator's segmentation loss is a combination of
    Cross-Entropy and Lovasz-Softmax.
    """
    model_G.train()
    model_D.train()

    running_loss_seg_ce_G = 0.0
    running_loss_seg_lovasz_G = 0.0
    running_loss_adv_G = 0.0
    running_loss_D_total = 0.0

    real_label = 1.0
    fake_label = 0.0

    progress_bar = tqdm(
        enumerate(train_loader_source),
        total=len(train_loader_source),
        desc=f"Epoch {epoch + 1}/{effective_total_epochs} [Adv. Training w/ Lovasz]",
        unit="batch",
        leave=False,
    )
    current_global_step = global_step_offset
    num_batches_source = len(train_loader_source)

    for batch_idx, (images_s, labels_s) in progress_bar:
        images_s = images_s.to(device)
        labels_s = labels_s.to(device).long()
        images_t, _ = next(train_loader_target)
        images_t = images_t.to(device)

        if images_s.size(0) != images_t.size(0):
            continue

        lr_power = config_module_ref.LR_SCHEDULER_POWER
        current_lr_G = poly_lr_scheduler(
            optimizer_G, initial_base_lr_G, current_global_step, max_iter, lr_power
        )
        current_lr_D = poly_lr_scheduler(
            optimizer_D, initial_base_lr_D, current_global_step, max_iter, lr_power
        )

        # --- EFFICIENT FORWARD PASS for GENERATOR ---
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler is not None
        ):
            pred_s_logits = model_G(images_s)[0]
            pred_t_logits = model_G(images_t)[0]

        # --- 1. Train Discriminator (model_D) ---
        optimizer_D.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler is not None
        ):
            d_out_source = model_D(F.softmax(pred_s_logits, dim=1).detach())
            loss_d_real = criterion_adv(
                d_out_source, torch.full_like(d_out_source, real_label, device=device)
            )
            d_out_target = model_D(F.softmax(pred_t_logits, dim=1).detach())
            loss_d_fake = criterion_adv(
                d_out_target, torch.full_like(d_out_target, fake_label, device=device)
            )
            loss_D = (loss_d_real + loss_d_fake) * 0.5
        if scaler:
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
        else:
            loss_D.backward()
            optimizer_D.step()
        running_loss_D_total += loss_D.item()

        # --- 2. Train Generator (model_G) ---
        optimizer_G.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler is not None
        ):
            # a) Adversarial loss
            d_out_target_for_g = model_D(F.softmax(pred_t_logits, dim=1))
            loss_adv = criterion_adv(
                d_out_target_for_g,
                torch.full_like(d_out_target_for_g, real_label, device=device),
            )
            # b) Combined Segmentation loss
            loss_seg_ce = criterion_seg_ce(pred_s_logits, labels_s)
            probas_s = F.softmax(pred_s_logits, dim=1)
            loss_seg_lovasz = criterion_seg_lovasz(probas_s, labels_s)
            # c) Final combined loss
            lambda_adv = config_module_ref.ADVERSARIAL_LAMBDA_ADV_GENERATOR
            loss_G_total = (
                loss_seg_ce
                + (lovasz_weight * loss_seg_lovasz)
                + (lambda_adv * loss_adv)
            )

        if scaler:
            scaler.scale(loss_G_total).backward()
            scaler.step(optimizer_G)
        else:
            loss_G_total.backward()
            optimizer_G.step()

        # Update running losses
        running_loss_seg_ce_G += loss_seg_ce.item()
        running_loss_seg_lovasz_G += loss_seg_lovasz.item()
        running_loss_adv_G += loss_adv.item()

        if scaler:
            scaler.update()

        postfix_dict = {
            "L_total_G": f"{loss_G_total.item():.3f}",
            "L_ce_G": f"{loss_seg_ce.item():.3f}",
            "L_lov_G": f"{loss_seg_lovasz.item():.3f}",
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
                "train_adv_lov/batch_loss_G_total": loss_G_total.item(),
                "train_adv_lov/batch_loss_seg_ce_G": loss_seg_ce.item(),
                "train_adv_lov/batch_loss_seg_lov_G": loss_seg_lovasz.item(),
                "train_adv_lov/batch_loss_adv_G": loss_adv.item(),
                "train_adv_lov/batch_loss_D": loss_D.item(),
                "train_adv_lov/learning_rate_G": current_lr_G,
                "train_adv_lov/learning_rate_D": current_lr_D,
            }
            wandb.log(log_payload, step=current_global_step)

        current_global_step += 1

    avg_losses_epoch: Dict[str, float] = {
        "seg_loss_ce_G": running_loss_seg_ce_G / num_batches_source
        if num_batches_source > 0
        else 0.0,
        "seg_loss_lovasz_G": running_loss_seg_lovasz_G / num_batches_source
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
