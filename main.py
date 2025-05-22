# main script that orchestrates the entire training and validation pipeline.
# It initializes everything, runs the training loop (calling functions from train.py and validation.py), saves the best model, and performs final metric calculations.
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import config as cfg
from data_loader import get_loaders
from model_loader import get_model
from train import train_one_epoch
from utils import calculate_performance_metrics, init_wandb
from validation import validate_and_log


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="DeepLabV2 Semantic Segmentation Training"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,  # Default to None, will use config
        choices=["sgd", "adam"],
        help="Optimizer type to use (sgd or adam). Overrides config if set.",
    )
    # Add other arguments you might want to control from CLI, e.g., learning_rate, batch_size
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate from config."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config.",
    )

    args = parser.parse_args()

    # --- Determine Effective Configuration ---
    # Start with config file values, then override with CLI args if provided
    effective_optimizer_type = args.optimizer if args.optimizer else cfg.OPTIMIZER_TYPE

    effective_epochs = args.epochs if args.epochs is not None else cfg.TRAIN_EPOCHS

    # Determine learning rate and other optimizer params based on choice
    effective_optimizer_config = {"optimizer_type": effective_optimizer_type}
    if effective_optimizer_type.lower() == "sgd":
        current_base_lr = args.lr if args.lr is not None else cfg.SGD_LEARNING_RATE
        current_weight_decay = cfg.WEIGHT_DECAY  # Using common weight decay
        effective_optimizer_config.update(
            {
                "learning_rate": current_base_lr,
                "momentum": cfg.SGD_MOMENTUM,
                "weight_decay": current_weight_decay,
            }
        )
    elif effective_optimizer_type.lower() == "adam":
        current_base_lr = args.lr if args.lr is not None else cfg.ADAM_LEARNING_RATE
        current_weight_decay = (
            cfg.WEIGHT_DECAY
        )  # Using common weight decay, or define ADAM_WEIGHT_DECAY
        effective_optimizer_config.update(
            {
                "learning_rate": current_base_lr,
                "beta1": cfg.ADAM_BETA1,
                "beta2": cfg.ADAM_BETA2,
                "weight_decay": current_weight_decay,
            }
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {effective_optimizer_type}")

    print(f"Using Optimizer: {effective_optimizer_type.upper()}")
    print(f"Effective Base Learning Rate: {current_base_lr}")
    print(f"Effective Weight Decay: {current_weight_decay}")
    print(f"Training for {effective_epochs} epochs.")

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
    os.makedirs(cfg.SAVE_MODEL_DIR, exist_ok=True)

    # Initialize the Weights & Biases run.
    init_wandb(cfg, effective_optimizer_config)

    print(f"Using device: {cfg.DEVICE}")

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

    # Tells W&B to "watch" the model, logging gradients and parameter distributions (can be resource-intensive).
    if wandb.run:
        wandb.watch(model, log="all", log_freq=cfg.PRINT_FREQ_BATCH * 5, log_graph=True)

    # --- Optimizer, Loss, Scaler ---

    # Create the SGD optimizer.
    # `model.optim_parameters()` provides the parameter groups for differential learning rates (backbone vs. head).

    # --- Optimizer Initialization ---
    # model.optim_parameters() will use current_base_lr to set different LRs for head/backbone
    param_groups = model.optim_parameters(current_base_lr)

    if effective_optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            param_groups,  # This contains params and their specific initial LRs
            # lr=current_base_lr, # Not needed here, already set in param_groups
            momentum=cfg.SGD_MOMENTUM,
            weight_decay=current_weight_decay,
        )
    elif effective_optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            param_groups,  # This contains params and their specific initial LRs
            # lr=current_base_lr, # Not needed here, already set in param_groups
            betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
            weight_decay=current_weight_decay,
        )
    else:  # Should have been caught earlier, but good practice
        raise ValueError(f"Unsupported optimizer: {effective_optimizer_type}")

    # Store initial LRs in each param group for the scheduler (if not already there from optim_parameters)
    # The revised poly_lr_scheduler now handles this internally on its first call.

    # Define the Cross Entropy loss function, ignoring the specified index.
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)

    # Initializes the gradient scaler for mixed-precision training.
    scaler = None
    if cfg.DEVICE.type == "cuda":
        scaler = torch.GradScaler()
        print("Using mixed-precision training with GradScaler on CUDA device.")
    else:
        print("Using full-precision (FP32) training (CPU or no scaler).")

    # --- Training & Validation Loop ---

    # Initialize tracking variables
    best_miou = 0.0
    global_step = 0
    max_iter = cfg.TRAIN_EPOCHS * len(train_loader)

    for epoch in range(cfg.TRAIN_EPOCHS):  # epoch is 0-indexed
        print(f"\n--- Epoch {epoch + 1}/{cfg.TRAIN_EPOCHS} ---")

        avg_train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg.DEVICE,
            epoch,
            global_step,
            max_iter,
            scaler,
        )
        print(f"Epoch {epoch + 1} Training: Average Loss: {avg_train_loss:.4f}")

        if wandb.run:  # Log epoch-level training loss
            wandb.log(
                {"train/epoch_loss": avg_train_loss, "epoch": epoch + 1},
                step=global_step,
            )

        if (epoch + 1) % cfg.VALIDATE_FREQ_EPOCH == 0 or (
            epoch + 1
        ) == cfg.TRAIN_EPOCHS:
            current_miou, avg_val_loss = validate_and_log(
                model, val_loader, criterion, cfg.DEVICE, epoch, global_step
            )
            # validate_and_log handles its own W&B logging for validation metrics

            # Save Best Model
            # If the current validation mIoU is better than the previous best_miou,
            # the model's state dictionary is saved, and W&B is updated.
            if current_miou > best_miou:
                best_miou = current_miou
                save_path = os.path.join(cfg.SAVE_MODEL_DIR, cfg.BEST_MODEL_NAME)
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with mIoU: {best_miou:.4f} to {save_path}")
                if wandb.run:
                    wandb.summary["best_val_mIoU"] = best_miou
                    wandb.save(save_path)  # Upload best model to W&B

    print("\n--- Training Finished ---")
    if wandb.run and "avg_train_loss" in locals():  # Check if training loop ran
        wandb.summary["final_train_epoch_loss_last_epoch"] = avg_train_loss

    # --- Final Performance Metrics ---

    # Load the state dictionary of the best_model saved during training for final metrics calculation
    best_model_path = os.path.join(cfg.SAVE_MODEL_DIR, cfg.BEST_MODEL_NAME)
    if os.path.exists(best_model_path):
        print(
            f"Loading best model from {best_model_path} for final metrics calculation..."
        )
        # Ensure model is on the correct device before loading state_dict if it was moved
        model.to(cfg.DEVICE)
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))
    else:
        print(
            f"Best model not found at {best_model_path}. Using last model state for metrics."
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

    print("\n--- Table 1: Classic Cityscapes (DeepLabV2 - Step 2a) ---")  # [cite: 15]
    print("| Metric                      | Value                               |")
    print("|-----------------------------|-------------------------------------|")
    print(
        f"| Best mIoU (%)               | {best_miou * 100:.2f}                              |"
    )  # [cite: 14]
    print(
        f"| Latency (ms)                | {perf_metrics.get('mean_latency_ms', -1):.2f} +/- {perf_metrics.get('std_latency_ms', -1):.2f} |"
    )  # [cite: 14]
    print(
        f"| FLOPs (G)                   | {perf_metrics.get('flops_g', -1):.2f}                                |"
    )  # [cite: 14]
    print(
        f"| Params (M)                  | {perf_metrics.get('params_m', -1):.2f}                                |"
    )  # [cite: 14]

    if wandb.run:
        wandb.summary["table1_best_val_mIoU_percent"] = best_miou * 100
        wandb.summary["table1_latency_ms_mean"] = perf_metrics.get(
            "mean_latency_ms", -1
        )
        wandb.summary["table1_flops_g"] = perf_metrics.get("flops_g", -1)
        wandb.summary["table1_params_m"] = perf_metrics.get("params_m", -1)

        print("\nFull FLOPs Table:")
        print(perf_metrics.get("flop_table", "Not calculated"))

        # Properly end the W&B run
        wandb.finish()
    print("Run completed.")


if __name__ == "__main__":
    main()
