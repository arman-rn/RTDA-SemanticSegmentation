"""
Main script for orchestrating the training and validation pipeline
for DeepLabV2 semantic segmentation.

It handles:
- Command-line argument parsing for overriding configurations.
- Initialization of datasets, data loaders, and the model.
- Setup of the optimizer, loss function, and mixed-precision scaler.
- The main training loop, calling epoch-level training and validation functions.
- Saving the best performing model based on validation mIoU.
- Final calculation and reporting of performance metrics (FLOPs, Latency, etc.).
- Integration with Weights & Biases for experiment tracking.
"""

import argparse
import importlib  # For reloading config in notebooks
import os
from typing import Any, Optional

import torch
from torch import GradScaler, nn, optim  # For type hints

import config as cfg  # Your project's configuration module
import wandb  # For Weights & Biases integration
from data_loader import get_loaders
from model_loader import get_model
from train import train_one_epoch
from utils import (
    calculate_performance_metrics,
    init_wandb,
    load_checkpoint,
    save_checkpoint,
)
from validation import validate_and_log

# Type alias for the config module for clarity
ConfigModule = Any  # Could be replaced with a Protocol if config structure is strict


def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="DeepLabV2 Semantic Segmentation Training Script"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,  # Default to None, will use config
        choices=["sgd", "adam"],
        help="Optimizer type to use (sgd or adam). Overrides config if set.",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate from config."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,  # Default is None, will check config next
        help="Path to a checkpoint file to resume training from. Overrides config RESUME_CHECKPOINT_PATH.",
    )

    args = parser.parse_args()

    # --- Determine Effective Configuration ---
    # Reload config in case it was changed (especially useful in notebooks)
    importlib.reload(cfg)

    effective_optimizer_type = args.optimizer if args.optimizer else cfg.OPTIMIZER_TYPE

    effective_epochs = args.epochs if args.epochs is not None else cfg.TRAIN_EPOCHS

    effective_optimizer_config_log = {"optimizer_type": effective_optimizer_type}
    current_weight_decay = cfg.WEIGHT_DECAY

    if effective_optimizer_type.lower() == "sgd":
        current_base_lr = args.lr if args.lr is not None else cfg.SGD_LEARNING_RATE
        effective_optimizer_config_log.update(
            {
                "learning_rate": current_base_lr,
                "momentum": cfg.SGD_MOMENTUM,
                "weight_decay": current_weight_decay,
            }
        )
    elif effective_optimizer_type.lower() == "adam":
        current_base_lr = args.lr if args.lr is not None else cfg.ADAM_LEARNING_RATE
        effective_optimizer_config_log.update(
            {
                "learning_rate": current_base_lr,
                "beta1": getattr(cfg, "ADAM_BETA1", 0.9),
                "beta2": getattr(cfg, "ADAM_BETA2", 0.999),
                "weight_decay": current_weight_decay,
            }
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {effective_optimizer_type}")

    print("--- Effective Configuration ---")
    print(f"Optimizer: {effective_optimizer_type.upper()}")
    print(f"Base Learning Rate (for all params): {current_base_lr}")
    print(f"Weight Decay: {current_weight_decay}")
    print(f"Training for {effective_epochs} epochs.")
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch Size (from config): {cfg.BATCH_SIZE}")
    print("-----------------------------")

    # --- Initial Checks & Setup ---
    if not os.path.exists(cfg.DATASET_PATH):
        print(
            f"CRITICAL ERROR: 'DATASET_PATH' in 'config.py' is not set or invalid: '{cfg.DATASET_PATH}'"
        )
        return
    if not os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        print(
            f"CRITICAL ERROR: 'PRETRAINED_MODEL_PATH' in 'config.py' does not exist: '{cfg.PRETRAINED_MODEL_PATH}'"
        )
        return

    # Create the directory to save models if it doesn't exist.
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)  # Ensure checkpoint directory exists

    # Initialize the Weights & Biases run.
    init_wandb(cfg, effective_optimizer_config_log)

    # --- DataLoaders ---
    try:
        train_loader, val_loader = get_loaders(cfg)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize DataLoaders: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        return
    if not train_loader or not len(train_loader.dataset):  # Check if dataset is empty
        print(
            "CRITICAL ERROR: Training dataset is empty or DataLoader failed. Check dataset path and 'datasets/cityscapes.py'."
        )
        if wandb.run:
            wandb.finish(exit_code=1)
        return
    print(
        f"Train loader: {len(train_loader)} batches, {len(train_loader.dataset)} images."
    )
    print(f"Val loader: {len(val_loader)} batches, {len(val_loader.dataset)} images.")

    # --- Model ---
    # Create the DeepLabV2 model instance
    model = get_model(cfg.NUM_CLASSES, cfg.PRETRAINED_MODEL_PATH, cfg.DEVICE)

    # --- Optimizer Initialization ---
    if effective_optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=current_base_lr,
            momentum=cfg.SGD_MOMENTUM,
            weight_decay=current_weight_decay,
        )
    elif effective_optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=current_base_lr,
            betas=(getattr(cfg, "ADAM_BETA1", 0.9), getattr(cfg, "ADAM_BETA2", 0.999)),
            weight_decay=current_weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {effective_optimizer_type}")

    # Define the Cross Entropy loss function, ignoring the specified index.
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)

    # Initializes the gradient scaler for mixed-precision training.
    scaler: Optional[GradScaler] = None
    if cfg.DEVICE.type == "cuda":
        scaler = torch.GradScaler()
        print("Using mixed-precision training with GradScaler on CUDA device.")
    else:
        print("Using full-precision (FP32) training (CPU or no scaler).")

    # --- Initialize tracking variables & Load Checkpoint if specified ---
    start_epoch = 0
    global_step = 0
    best_miou = 0.0

    checkpoint_to_resume_from = (
        args.resume_checkpoint if args.resume_checkpoint else cfg.RESUME_CHECKPOINT_PATH
    )

    if checkpoint_to_resume_from and os.path.exists(checkpoint_to_resume_from):
        print(f"Attempting to resume from checkpoint: {checkpoint_to_resume_from}")
        checkpoint_data = load_checkpoint(
            checkpoint_to_resume_from, model, optimizer, scaler, cfg.DEVICE
        )
        if checkpoint_data:  # If checkpoint was successfully loaded
            start_epoch = (
                checkpoint_data.get("epoch", -1) + 1
            )  # Resume from the next epoch
            global_step = checkpoint_data.get("global_step", 0)
            best_miou = checkpoint_data.get("best_miou", 0.0)
            print(
                f"Resumed training from Epoch {start_epoch}. Global Step: {global_step}. Best mIoU so far: {best_miou:.4f}"
            )
        else:
            print(
                "Failed to load checkpoint data, or checkpoint was empty. Starting from scratch."
            )
    else:
        if checkpoint_to_resume_from:  # Path was given but file not found
            print(
                f"Warning: Specified resume checkpoint '{checkpoint_to_resume_from}' not found. Starting training from scratch."
            )
        else:
            print("No resume checkpoint specified. Starting training from scratch.")

    if wandb.run:
        wandb.watch(model, log="all", log_freq=cfg.PRINT_FREQ_BATCH * 5, log_graph=True)

    # --- Training & Validation Loop ---
    max_iter = effective_epochs * len(train_loader) if len(train_loader) > 0 else 0

    for epoch in range(start_epoch, effective_epochs):  # epoch is 0-indexed
        print(f"\n--- Epoch {epoch + 1}/{effective_epochs} ---")

        avg_train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.DEVICE,
            epoch=epoch,
            global_step_offset=global_step,
            max_iter=max_iter,
            initial_base_lr=current_base_lr,
            effective_total_epochs=effective_epochs,
            scaler=scaler,
        )
        if wandb.run:
            wandb.log(
                {"train/epoch_loss": avg_train_loss, "epoch": epoch + 1},
                step=global_step,
            )

        # --- Validation ---
        current_epoch_miou = 0.0  # mIoU specifically for this epoch's validation
        if (epoch + 1) % cfg.VALIDATE_FREQ_EPOCH == 0 or (
            epoch + 1
        ) == effective_epochs:
            current_epoch_miou, avg_val_loss = validate_and_log(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=cfg.DEVICE,
                epoch=epoch,
                global_step=global_step,
                effective_total_epochs=effective_epochs,
                config_module_ref=cfg,
            )

        # --- Save Checkpoint ---
        is_best = current_epoch_miou > best_miou
        if is_best:
            best_miou = current_epoch_miou  # Update overall best mIoU
            print(f"New best mIoU: {best_miou:.4f} at epoch {epoch + 1}")
            if wandb.run:
                wandb.summary["best_val_mIoU"] = best_miou  # Update W&B summary

        checkpoint_state = {
            "epoch": epoch,  # Save 0-indexed epoch
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,  # Persist the overall best mIoU found so far
        }
        if scaler:
            checkpoint_state["scaler_state_dict"] = scaler.state_dict()

        latest_checkpoint_path = os.path.join(
            cfg.CHECKPOINT_DIR, cfg.LATEST_CHECKPOINT_FILENAME
        )
        save_checkpoint(checkpoint_state, latest_checkpoint_path)

        if is_best:
            best_checkpoint_path = os.path.join(
                cfg.CHECKPOINT_DIR, cfg.BEST_CHECKPOINT_FILENAME
            )
            save_checkpoint(checkpoint_state, best_checkpoint_path)

        if (
            cfg.SAVE_CHECKPOINT_FREQ_EPOCH > 0
            and (epoch + 1) % cfg.SAVE_CHECKPOINT_FREQ_EPOCH == 0
            and (epoch + 1) < effective_epochs
        ):
            periodic_checkpoint_path = os.path.join(
                cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(checkpoint_state, periodic_checkpoint_path)

    print("\n--- Training Finished ---")
    if wandb.run and "avg_train_loss" in locals():
        wandb.summary["final_train_epoch_loss_last_epoch"] = avg_train_loss

    # --- Final Performance Metrics ---
    final_eval_model_path = os.path.join(
        cfg.CHECKPOINT_DIR, cfg.BEST_CHECKPOINT_FILENAME
    )
    if os.path.exists(final_eval_model_path):
        print(
            f"Loading best model from checkpoint {final_eval_model_path} for final metrics calculation..."
        )
        # Only model weights are strictly needed for eval, but load_checkpoint is fine
        load_checkpoint(final_eval_model_path, model, device=cfg.DEVICE)
    else:
        print(
            f"Best checkpoint not found at {final_eval_model_path}. Using last model state for metrics."
        )

    print("\nCalculating final performance metrics (FLOPs, Latency)...")
    perf_metrics = calculate_performance_metrics(
        model,
        cfg.DEVICE,
        cfg.IMG_HEIGHT,
        cfg.IMG_WIDTH,
        cfg.LATENCY_ITERATIONS,
        cfg.WARMUP_ITERATIONS,
    )

    print("\n--- Table 1: Classic Cityscapes (DeepLabV2 - Step 2a) ---")
    print("| Metric                      | Value                               |")
    print("|-----------------------------|-------------------------------------|")
    print(
        f"| Best mIoU (%)               | {best_miou * 100:.2f}                              |"
    )
    print(
        f"| Latency (ms)                | {perf_metrics.get('mean_latency_ms', -1.0):.2f} +/- {perf_metrics.get('std_latency_ms', -1.0):.2f} |"
    )
    print(
        f"| FLOPs (G)                   | {perf_metrics.get('flops_g', -1.0):.2f}                                |"
    )
    print(
        f"| Params (M)                  | {perf_metrics.get('params_m', -1.0):.2f}                                |"
    )

    if wandb.run:
        wandb.summary["table1_best_val_mIoU_percent"] = best_miou * 100
        wandb.summary["table1_latency_ms_mean"] = perf_metrics.get(
            "mean_latency_ms", -1.0
        )
        wandb.summary["table1_flops_g"] = perf_metrics.get("flops_g", -1.0)
        wandb.summary["table1_params_m"] = perf_metrics.get("params_m", -1.0)

        print("\nFull FLOPs Table (from fvcore calculation):")
        print(
            perf_metrics.get(
                "flop_table", "FLOPs table not calculated or error during calculation."
            )
        )

        wandb.finish()
    print("Run completed.")


if __name__ == "__main__":
    main()
