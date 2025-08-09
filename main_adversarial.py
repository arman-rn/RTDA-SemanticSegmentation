"""
Main script for orchestrating ADVERSARIAL domain adaptation training
for semantic segmentation (e.g., GTA5 -> Cityscapes).

Supports both single-level and multi-level adversarial training.
"""

import argparse
import importlib
import os
from typing import Any, Optional

import numpy as np
import torch
import wandb
from torch import GradScaler, nn, optim

import config as cfg
from data_loader import CITYSCAPES_ID_TO_NAME_MAP, get_loaders
from losses.lovasz_loss import LovaszSoftmax
from model_loader import get_discriminator, get_model
from train import train_one_epoch_adversarial
from train_lovasz import train_one_epoch_adversarial_lovasz
from utils import (
    calculate_performance_metrics,
    init_wandb,
    load_adversarial_checkpoint,
    log_best_model_predictions,
    save_checkpoint,
    set_seeds,
)
from validation import validate_and_log

ConfigModule = Any


def main_adversarial():
    set_seeds(cfg.SEED_VALUE)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Adversarial Domain Adaptation for Semantic Segmentation"
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default=None,
        choices=["bisenet", "deeplabv2"],
        help="Generator model name (e.g., bisenet). Overrides config.MODEL_NAME.",
    )
    parser.add_argument(
        "--generator_optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam"],
        help="Optimizer for Generator. Overrides config.OPTIMIZER_TYPE.",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=None,
        help="Learning rate for Generator. Overrides config ADAM_LEARNING_RATE or SGD_LEARNING_RATE.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides config.TRAIN_EPOCHS.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from. Overrides config.RESUME_CHECKPOINT_PATH.",
    )
    parser.add_argument(
        "--gta5_path", type=str, default=None, help="Overrides config.GTA5_DATASET_PATH"
    )
    parser.add_argument(
        "--cityscapes_path",
        type=str,
        default=None,
        help="Overrides config.CITYSCAPES_DATASET_PATH",
    )

    args = parser.parse_args()

    # --- Reload and Apply CLI Overrides to Config ---
    importlib.reload(cfg)  # Reload to ensure fresh config state if in interactive env

    if args.generator_model is not None:
        cfg.MODEL_NAME = args.generator_model  # This is for the Generator
    if args.generator_optimizer is not None:
        cfg.OPTIMIZER_TYPE = args.generator_optimizer
    if args.epochs is not None:
        cfg.TRAIN_EPOCHS = args.epochs
    if args.resume_checkpoint is not None:
        cfg.RESUME_CHECKPOINT_PATH = args.resume_checkpoint
    if args.gta5_path is not None:
        cfg.GTA5_DATASET_PATH = args.gta5_path
    if args.cityscapes_path is not None:
        cfg.CITYSCAPES_DATASET_PATH = args.cityscapes_path

    # Generator LR override
    if args.generator_lr is not None:
        if cfg.OPTIMIZER_TYPE.lower() == "sgd":
            cfg.SGD_LEARNING_RATE = args.generator_lr
        elif cfg.OPTIMIZER_TYPE.lower() == "adam":
            cfg.ADAM_LEARNING_RATE = args.generator_lr

    # --- Setup Checkpoint Directory (specific for adversarial runs) ---
    cfg.CHECKPOINT_DIR = (
        f"{cfg.ROOT_DIR}/checkpoints/{cfg.MODEL_NAME}_adversarial_GTA2City"
    )
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # --- Log Effective Configuration ---
    print("--- ADVERSARIAL TRAINING - Effective Configuration ---")
    print("Training Mode: SINGLE-LEVEL Adversarial")
    print(f"  Lambda Adversarial: {cfg.ADVERSARIAL_LAMBDA_ADV_GENERATOR}")

    # Determine Generator LR based on its optimizer type
    current_g_base_lr = 0.0
    if cfg.OPTIMIZER_TYPE.lower() == "sgd":
        current_g_base_lr = cfg.SGD_LEARNING_RATE
    elif cfg.OPTIMIZER_TYPE.lower() == "adam":
        current_g_base_lr = cfg.ADAM_LEARNING_RATE
    print(
        f"Generator Optimizer: {cfg.OPTIMIZER_TYPE.upper()} with LR: {current_g_base_lr}"
    )

    print(f"  Source Dataset (Labeled): {cfg.ADVERSARIAL_SOURCE_DATASET_NAME.upper()}")
    print(
        f"  Target Dataset (Unlabeled): {cfg.ADVERSARIAL_TARGET_DATASET_NAME.upper()} (Split: {cfg.ADVERSARIAL_TARGET_DATASET_SPLIT})"
    )
    print(f"  Validation Dataset: {cfg.VAL_DATASET.upper()}")
    print(f"  Lambda Adversarial (Generator): {cfg.ADVERSARIAL_LAMBDA_ADV_GENERATOR}")
    print(
        f"Discriminator Optimizer: {cfg.ADVERSARIAL_DISCRIMINATOR_OPTIMIZER_TYPE.upper()} with LR: {cfg.ADVERSARIAL_DISCRIMINATOR_LEARNING_RATE}"
    )
    print(f"Training for {cfg.TRAIN_EPOCHS} epochs.")
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Checkpoint Directory: {cfg.CHECKPOINT_DIR}")
    print(
        f"Resume from checkpoint: {cfg.RESUME_CHECKPOINT_PATH if cfg.RESUME_CHECKPOINT_PATH else 'No'}"
    )
    print("----------------------------------------------------")

    # --- W&B Initialization ---
    generator_opt_log_config = {"type": cfg.OPTIMIZER_TYPE, "lr": current_g_base_lr}
    if cfg.OPTIMIZER_TYPE.lower() == "sgd":
        generator_opt_log_config["momentum"] = cfg.SGD_MOMENTUM
    elif cfg.OPTIMIZER_TYPE.lower() == "adam":
        generator_opt_log_config["beta1"] = getattr(cfg, "ADAM_BETA1", 0.9)
        generator_opt_log_config["beta2"] = getattr(cfg, "ADAM_BETA2", 0.999)
    generator_opt_log_config["weight_decay"] = cfg.WEIGHT_DECAY
    init_wandb(
        cfg,
        effective_optimizer_config=generator_opt_log_config,
        is_adversarial_training=True,
    )

    # --- DataLoaders ---
    try:
        source_loader, val_loader, target_loader_infinite = get_loaders(
            config_obj=cfg,
            train_dataset_name=cfg.ADVERSARIAL_SOURCE_DATASET_NAME,
            val_dataset_name=cfg.VAL_DATASET,
            load_target_loader=True,
            target_dataset_name=cfg.ADVERSARIAL_TARGET_DATASET_NAME,
            target_dataset_split=cfg.ADVERSARIAL_TARGET_DATASET_SPLIT,
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize DataLoaders: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        return
    if not source_loader or not target_loader_infinite:
        print(
            "CRITICAL ERROR: Source or Target DataLoader failed to initialize for adversarial training."
        )
        if wandb.run:
            wandb.finish(exit_code=1)
        return
    print(
        f"Source loader: {len(source_loader.dataset)} images, Target (unlabeled) loader: {len(target_loader_infinite.data_loader.dataset)} images (infinite wrapper)."
    )

    # --- Models ---
    model_G = get_model(config_obj=cfg)  # Generator (BiSeNet)
    model_D = get_discriminator(config_obj=cfg)  # Discriminators

    if model_D is None:
        print("CRITICAL ERROR: Main Discriminator model could not be initialized.")
        if wandb.run:
            wandb.finish(exit_code=1)
        return

    # --- Optimizers ---
    # Generator Optimizer
    if cfg.OPTIMIZER_TYPE.lower() == "sgd":
        optimizer_G = optim.SGD(
            model_G.parameters(),
            lr=current_g_base_lr,
            momentum=cfg.SGD_MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER_TYPE.lower() == "adam":
        optimizer_G = optim.Adam(
            model_G.parameters(),
            lr=current_g_base_lr,
            betas=(getattr(cfg, "ADAM_BETA1", 0.9), getattr(cfg, "ADAM_BETA2", 0.999)),
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unsupported generator optimizer type: {cfg.OPTIMIZER_TYPE}")

    # Discriminator Optimizer(s)
    optimizer_D = optim.Adam(
        model_D.parameters(),
        lr=cfg.ADVERSARIAL_DISCRIMINATOR_LEARNING_RATE,
        betas=(
            cfg.ADVERSARIAL_DISCRIMINATOR_ADAM_BETA1,
            cfg.ADVERSARIAL_DISCRIMINATOR_ADAM_BETA2,
        ),
    )

    # --- Loss Functions ---
    criterion_seg_ce = nn.CrossEntropyLoss(
        ignore_index=cfg.IGNORE_INDEX
    )  # For Generator's segmentation task (Cross-Entropy)
    criterion_seg_lovasz = None
    if cfg.USE_LOVASZ_LOSS:
        criterion_seg_lovasz = LovaszSoftmax(
            ignore=cfg.IGNORE_INDEX
        )  # For Generator's segmentation task (Lovasz-Softmax)
    criterion_adv = (
        nn.BCEWithLogitsLoss()
    )  # For adversarial training (D loss, and G's adv loss)

    # --- Mixed Precision Scaler ---
    scaler: Optional[GradScaler] = None
    if cfg.DEVICE.type == "cuda":
        scaler = torch.GradScaler()
        print("Using mixed-precision training with GradScaler on CUDA device.")

    # --- Load Checkpoint (handles both G and D) ---
    start_epoch = 0
    global_step = 0
    best_miou = 0.0  # Based on Generator's performance on val set

    if cfg.RESUME_CHECKPOINT_PATH and os.path.exists(cfg.RESUME_CHECKPOINT_PATH):
        print(f"Attempting to resume from checkpoint: {cfg.RESUME_CHECKPOINT_PATH}")
        checkpoint_data = load_adversarial_checkpoint(
            filepath=cfg.RESUME_CHECKPOINT_PATH,
            model_G=model_G,
            model_D=model_D,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            scaler=scaler,
            device=cfg.DEVICE,
        )

        if checkpoint_data:
            start_epoch = checkpoint_data.get("epoch", -1) + 1
            global_step = checkpoint_data.get("global_step", 0)
            best_miou = checkpoint_data.get("best_miou", 0.0)
            print(
                f"Resumed from Epoch {start_epoch}. Global Step: {global_step}. Best mIoU (Generator): {best_miou:.4f}"
            )

    else:
        if cfg.RESUME_CHECKPOINT_PATH:
            print(
                f"Warning: Resume checkpoint '{cfg.RESUME_CHECKPOINT_PATH}' not found. Starting from scratch."
            )
        else:
            print("No resume checkpoint. Starting adversarial training from scratch.")

    # --- W&B Watch Models ---
    if wandb.run:
        wandb.watch(
            model_G,
            log="all",
            log_freq=cfg.PRINT_FREQ_BATCH * 5,
            log_graph=True,
            criterion=criterion_seg_ce,
        )
        wandb.watch(model_D, log="all", log_freq=cfg.PRINT_FREQ_BATCH * 10)

    # --- Adversarial Training & Validation Loop ---
    max_iter = cfg.TRAIN_EPOCHS * len(source_loader)

    for epoch in range(start_epoch, cfg.TRAIN_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{cfg.TRAIN_EPOCHS} [Single-Level] ---")

        if cfg.USE_LOVASZ_LOSS and criterion_seg_lovasz is not None:
            avg_losses_dict, global_step = train_one_epoch_adversarial_lovasz(
                model_G=model_G,
                optimizer_G=optimizer_G,
                criterion_seg_ce=criterion_seg_ce,
                criterion_seg_lovasz=criterion_seg_lovasz,
                lovasz_weight=cfg.LOVASZ_LOSS_WEIGHT,
                train_loader_source=source_loader,
                initial_base_lr_G=current_g_base_lr,
                model_D=model_D,
                optimizer_D=optimizer_D,
                criterion_adv=criterion_adv,
                train_loader_target=target_loader_infinite,
                initial_base_lr_D=cfg.ADVERSARIAL_DISCRIMINATOR_LEARNING_RATE,
                device=cfg.DEVICE,
                epoch=epoch,
                global_step_offset=global_step,
                max_iter=max_iter,
                effective_total_epochs=cfg.TRAIN_EPOCHS,
                config_module_ref=cfg,
                scaler=scaler,
            )
        else:
            avg_losses_dict, global_step = train_one_epoch_adversarial(
                model_G=model_G,
                optimizer_G=optimizer_G,
                criterion_seg=criterion_seg_ce,
                train_loader_source=source_loader,
                initial_base_lr_G=current_g_base_lr,
                model_D=model_D,
                optimizer_D=optimizer_D,
                criterion_adv=criterion_adv,
                train_loader_target=target_loader_infinite,
                initial_base_lr_D=cfg.ADVERSARIAL_DISCRIMINATOR_LEARNING_RATE,
                device=cfg.DEVICE,
                epoch=epoch,
                global_step_offset=global_step,
                max_iter=max_iter,
                effective_total_epochs=cfg.TRAIN_EPOCHS,
                config_module_ref=cfg,
                scaler=scaler,
            )

        if wandb.run:  # Log epoch-level average losses
            log_payload_epoch = {"epoch": epoch + 1}
            if "seg_loss_G" in avg_losses_dict:
                log_payload_epoch["train_adv_epoch/avg_loss_seg_G"] = avg_losses_dict[
                    "seg_loss_G"
                ]
            if "adv_loss_G" in avg_losses_dict:
                log_payload_epoch["train_adv_epoch/avg_loss_adv_G"] = avg_losses_dict[
                    "adv_loss_G"
                ]
            if "loss_D_total" in avg_losses_dict:
                log_payload_epoch["train_adv_epoch/avg_loss_D"] = avg_losses_dict[
                    "loss_D_total"
                ]
            if "adv_loss_G" in avg_losses_dict:
                log_payload_epoch["train_adv_epoch/avg_loss_adv_G"] = avg_losses_dict[
                    "adv_loss_G"
                ]
            if "seg_loss_lovasz_G" in avg_losses_dict:
                log_payload_epoch["train_adv_epoch/avg_loss_seg_lov_G"] = (
                    avg_losses_dict["seg_loss_lovasz_G"]
                )
            if "seg_loss_ce_G" in avg_losses_dict:
                log_payload_epoch["train_adv_epoch/avg_loss_seg_ce_G"] = (
                    avg_losses_dict["seg_loss_ce_G"]
                )

            wandb.log(log_payload_epoch, step=global_step)

        # --- Validation (on Generator model_G) ---
        current_epoch_miou = 0.0
        current_per_class_ious = np.zeros(cfg.NUM_CLASSES)
        if (epoch + 1) % cfg.VALIDATE_FREQ_EPOCH == 0 or (
            epoch + 1
        ) == cfg.TRAIN_EPOCHS:
            print(f"--- Validating Generator model_G at end of Epoch {epoch + 1} ---")
            current_epoch_miou, avg_val_loss, current_per_class_ious = validate_and_log(
                model=model_G,  # Validate generator
                val_loader=val_loader,
                criterion=criterion_seg_ce,  # Use segmentation loss for validation
                device=cfg.DEVICE,
                epoch=epoch,
                global_step=global_step,
                effective_total_epochs=cfg.TRAIN_EPOCHS,
                config_module_ref=cfg,
            )

        # --- Save Checkpoint (includes G and D states) ---
        is_best = current_epoch_miou > best_miou
        if is_best:
            best_miou = current_epoch_miou
            print(f"New best mIoU (Generator): {best_miou:.4f} at epoch {epoch + 1}")
            if wandb.run:
                wandb.summary["best_val_mIoU_generator"] = best_miou

        checkpoint_state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_G_state_dict": model_G.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "model_D_state_dict": model_D.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
            "best_miou": best_miou,
        }

        if scaler:
            checkpoint_state["scaler_state_dict"] = scaler.state_dict()

        if is_best:  # Save best model checkpoint
            checkpoint_state["best_model_per_class_ious"] = current_per_class_ious
            best_checkpoint_path = os.path.join(
                cfg.CHECKPOINT_DIR, cfg.BEST_CHECKPOINT_FILENAME
            )
            save_checkpoint(checkpoint_state, best_checkpoint_path)
            if "best_model_per_class_ious" in checkpoint_state:
                del checkpoint_state["best_model_per_class_ious"]

        if (
            cfg.SAVE_CHECKPOINT_FREQ_EPOCH > 0
            and (epoch + 1) % cfg.SAVE_CHECKPOINT_FREQ_EPOCH == 0
            and (epoch + 1) < cfg.TRAIN_EPOCHS
        ):  # Avoid saving periodic on last epoch if best is already saved
            periodic_checkpoint_path = os.path.join(
                cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_FILENAME
            )
            save_checkpoint(checkpoint_state, periodic_checkpoint_path)

    print("\n--- Adversarial Training Finished ---")

    # --- Final Metrics and Per-Class IoU for Best Generator Model ---
    final_per_class_ious_to_report = np.zeros(cfg.NUM_CLASSES)
    best_miou_for_summary_report = 0.0
    final_eval_model_path = os.path.join(
        cfg.CHECKPOINT_DIR, cfg.BEST_CHECKPOINT_FILENAME
    )

    if os.path.exists(final_eval_model_path):
        print(
            f"Loading best generator model from {final_eval_model_path} for final evaluation..."
        )
        # Load only generator for final eval on val set. model_G is updated in-place.
        checkpoint_summary = load_adversarial_checkpoint(
            filepath=final_eval_model_path,
            model_G=model_G,  # The generator model to load into
            model_D=model_D,  # Pass discriminator models, though they won't be used
            device=cfg.DEVICE,
        )
        if checkpoint_summary:
            best_miou_for_summary_report = checkpoint_summary.get("best_miou", 0.0)
            loaded_per_class_ious = checkpoint_summary.get("best_model_per_class_ious")
            if loaded_per_class_ious is not None:
                final_per_class_ious_to_report = loaded_per_class_ious
                print("Loaded per-class IoUs from the best generator checkpoint.")
    else:
        print(
            f"CRITICAL WARNING: Best checkpoint file '{cfg.BEST_CHECKPOINT_FILENAME}' not found at '{final_eval_model_path}'."
        )
        best_miou_for_summary_report = best_miou

    print(
        "\nCalculating performance metrics (FLOPs, Latency on final generator model state)..."
    )
    perf_metrics = calculate_performance_metrics(
        model_G,
        cfg.DEVICE,
        cfg.CITYSCAPES_IMG_HEIGHT,
        cfg.CITYSCAPES_IMG_WIDTH,
        cfg.LATENCY_ITERATIONS,
        cfg.WARMUP_ITERATIONS,
    )

    run_name_prefix = f"{cfg.MODEL_NAME.lower()}_adv_{cfg.ADVERSARIAL_SOURCE_DATASET_NAME.lower()}_to_{cfg.ADVERSARIAL_TARGET_DATASET_NAME.lower()}"
    print(f"\n--- Final Results for Adversarial Run: {run_name_prefix} ---")
    print(
        f"| Best Overall mIoU (Generator) on {cfg.VAL_DATASET.upper()} (%) | {best_miou_for_summary_report * 100:.2f} |"
    )
    print(
        f"| Latency (ms) @ {cfg.CITYSCAPES_IMG_WIDTH}x{cfg.CITYSCAPES_IMG_HEIGHT} | {perf_metrics.get('mean_latency_ms', -1.0):.2f} +/- {perf_metrics.get('std_latency_ms', -1.0):.2f} |"
    )
    print(
        f"| FLOPs (G) @ {cfg.CITYSCAPES_IMG_WIDTH}x{cfg.CITYSCAPES_IMG_HEIGHT}    | {perf_metrics.get('flops_g', -1.0):.2f} |"
    )
    print(
        f"| Parameters (M) - Generator   | {perf_metrics.get('params_m', -1.0):.2f} |"
    )

    print("\nPer-Class IoUs from Best Generator Model Checkpoint:")
    print("| Class Name           | IoU     |")
    print("|----------------------|---------|")
    if (
        final_per_class_ious_to_report is not None
        and len(final_per_class_ious_to_report) == cfg.NUM_CLASSES
    ):
        for i, iou_val in enumerate(final_per_class_ious_to_report):
            class_name = CITYSCAPES_ID_TO_NAME_MAP.get(i, f"Class_{i}")
            print(f"| {class_name:<20} | {iou_val:.4f} |")

    if wandb.run:
        wandb.summary[f"{run_name_prefix}_best_val_mIoU_generator_percent"] = (
            best_miou_for_summary_report * 100
        )
        wandb.summary[f"{run_name_prefix}_latency_ms_mean"] = perf_metrics.get(
            "mean_latency_ms", -1.0
        )
        wandb.summary[f"{run_name_prefix}_flops_g"] = perf_metrics.get("flops_g", -1.0)
        wandb.summary[f"{run_name_prefix}_params_m"] = perf_metrics.get(
            "params_m", -1.0
        )

        if (
            final_per_class_ious_to_report is not None
            and len(final_per_class_ious_to_report) == cfg.NUM_CLASSES
        ):
            for i, iou_val in enumerate(final_per_class_ious_to_report):
                class_name = CITYSCAPES_ID_TO_NAME_MAP.get(i, f"class_{i}")
                wandb.summary[f"{run_name_prefix}_final_iou_G_{class_name}"] = float(
                    iou_val
                )

        # ---  Log final predictions from the best generator model ---
        log_best_model_predictions(
            model=model_G,  # Use the generator model
            val_loader=val_loader,
            device=cfg.DEVICE,
            config_module_ref=cfg,
            num_images=6,
        )
        wandb.finish()

    print("Adversarial run completed.")


if __name__ == "__main__":
    main_adversarial()
