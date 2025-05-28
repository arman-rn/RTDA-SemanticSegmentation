"""
Collection of utility functions for the semantic segmentation project.

Includes learning rate schedulers, metrics calculation (mIoU, FLOPs, latency),
Weights & Biases integration helpers, and image visualization utilities.
"""
# This file is a collection of various helper functions used across different parts of the project.
# This includes learning rate scheduling, metrics calculation (mIoU, FLOPs, latency), and Weights & Biases integration helpers.

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import GradScaler, nn, optim

from data_loader import CITYSCAPES_ID_TO_NAME_MAP

# Type alias for the config module for clarity
ConfigModule = Any  # Could be replaced with a Protocol if config structure is strict


# --- Learning Rate Scheduler ---
def poly_lr_scheduler(
    optimizer: optim.Optimizer,
    initial_learning_rate: float,
    current_iter: int,
    max_iter: int,
    power: float,
) -> float:
    """
    Applies polynomial decay to the learning rate.

    This scheduler assumes the optimizer has a single parameter group,
    and it updates the learning rate of this group based on the provided
    initial learning rate and the polynomial decay formula.

    Args:
        optimizer: The PyTorch optimizer instance.
        initial_learning_rate: The starting learning rate from which to decay.
        current_iter: The current training iteration (batch number).
        max_iter: The total number of training iterations.
        power: The exponent for the polynomial decay formula (e.g., 0.9).

    Returns:
        The newly calculated (decayed) learning rate applied to the optimizer.
    """

    lr_scale_factor = (1 - current_iter / max_iter) ** power
    new_lr = initial_learning_rate * lr_scale_factor

    # Update the learning rate in the optimizer's single parameter group
    optimizer.param_groups[0]["lr"] = new_lr

    return new_lr


# --- mIoU Calculation Helpers ---
def fast_hist(
    label_true: np.ndarray, label_pred: np.ndarray, n_class: int
) -> np.ndarray:
    """
    Computes the confusion matrix for evaluating semantic segmentation.

    Args:
        label_true: A flattened NumPy array of ground truth labels (integers).
        label_pred: A flattened NumPy array of predicted class IDs (integers).
        n_class: The total number of classes.

    Returns:
        An N x N NumPy array representing the confusion matrix, where N is n_class.
        The element at (i, j) is the number of pixels with true label i
        and predicted label j.
    """
    mask = (
        (label_true >= 0)
        & (label_true < n_class)
        & (label_pred >= 0)
        & (label_pred < n_class)
    )
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class**2,
    ).reshape(n_class, n_class)

    return hist


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """
    Calculates Intersection over Union (IoU) for each class from a confusion matrix.

    Args:
        hist: An N x N NumPy array representing the confusion matrix,
              where N is the number of classes.

    Returns:
        A 1D NumPy array of length N, where each element is the IoU for the
        corresponding class. IoU = TP / (TP + FP + FN).
    """
    epsilon = 1e-5  # Small value to avoid division by zero
    # TP = diag(hist)
    # FP = sum(hist, axis=0) - diag(hist)
    # FN = sum(hist, axis=1) - diag(hist)
    # IoU = TP / (TP + FP + FN) = diag(hist) / (sum(axis=1) + sum(axis=0) - diag(hist))
    ious = np.diag(hist) / (
        hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + epsilon
    )
    # Replace potential NaN values (e.g., if a class has no true positives
    # and no false positives/negatives, resulting in 0/0) with 0.0.
    ious = np.nan_to_num(ious, nan=0.0)

    return ious


# --- W&B Initialization ---
def init_wandb(
    cfg_module: ConfigModule, effective_optimizer_config: Dict[str, Any]
) -> None:
    """
    Initializes a Weights & Biases (wandb) run for experiment tracking.

    Args:
        cfg_module: The imported configuration module (e.g., `config.py`).
        effective_optimizer_config: A dictionary containing the actual optimizer
            settings being used for the run (e.g., type, learning rate, momentum).
    """
    try:
        base_config_dict: Dict[str, Any] = {
            "script_epochs": cfg_module.TRAIN_EPOCHS,  # Renamed to avoid conflict with wandb internal 'epochs'
            "batch_size": cfg_module.BATCH_SIZE,
            "train_dataset": cfg_module.TRAIN_DATASET,
            "validation_dataset": cfg_module.VAL_DATASET,
            "num_classes": cfg_module.NUM_CLASSES,
            "lr_scheduler_power": cfg_module.LR_SCHEDULER_POWER,
            "device": str(cfg_module.DEVICE),
            "norm_mean": cfg_module.NORM_MEAN,
            "norm_std": cfg_module.NORM_STD,
        }

        if hasattr(cfg_module, "DEEPLABV2_PRETRAINED_BACKBONE_PATH"):
            base_config_dict["deeplabv2_pretrained_path"] = (
                cfg_module.DEEPLABV2_PRETRAINED_BACKBONE_PATH
            )
        if hasattr(cfg_module, "BISENET_CONTEXT_PATH"):
            base_config_dict["bisenet_context_path"] = cfg_module.BISENET_CONTEXT_PATH

        base_config_dict["train_image_height"] = (
            cfg_module.GTA5_IMG_HEIGHT
            if cfg_module.TRAIN_DATASET == "gta5"
            else cfg_module.CITYSCAPES_IMG_HEIGHT
        )
        base_config_dict["train_image_width"] = (
            cfg_module.GTA5_IMG_WIDTH
            if cfg_module.TRAIN_DATASET == "gta5"
            else cfg_module.CITYSCAPES_IMG_WIDTH
        )

        base_config_dict["validation_image_height"] = (
            cfg_module.GTA5_IMG_HEIGHT
            if cfg_module.VAL_DATASET == "gta5"
            else cfg_module.CITYSCAPES_IMG_HEIGHT
        )
        base_config_dict["validation_image_width"] = (
            cfg_module.GTA5_IMG_WIDTH
            if cfg_module.VAL_DATASET == "gta5"
            else cfg_module.CITYSCAPES_IMG_WIDTH
        )

        # Merge base config with effective optimizer config
        full_config: Dict[str, Any] = {**base_config_dict, **effective_optimizer_config}

        wandb.init(
            project=cfg_module.WANDB_PROJECT_NAME,
            entity=cfg_module.WANDB_ENTITY,
            config=full_config,
        )
        print("Weights & Biases initialized successfully.")
    except Exception as e:
        print(f"Error initializing W&B: {e}. W&B logging will be disabled.")


# --- W&B Image Logging ---
def log_segmentation_to_wandb(
    images: torch.Tensor,
    true_masks: torch.Tensor,  # These are the 2D label ID tensors (B, H, W)
    pred_masks: torch.Tensor,  # These are also 2D label ID tensors (B, H, W)
    epoch: int,  # Current epoch number (1-indexed for display)
    current_config: ConfigModule,  # The config module (e.g., cfg)
    max_images: int = 4,
) -> None:
    """
    Logs sample input images with their ground truth and predicted segmentation
    masks to Weights & Biases for visual inspection.

    Args:
        images: A batch of input image tensors, expected shape (B, C, H, W),
                typically normalized.
        true_masks: A batch of ground truth label tensors, shape (B, H, W) or (B, 1, H, W).
        pred_masks: A batch of predicted label tensors, shape (B, H, W) or (B, 1, H, W).
        epoch: The current epoch number (1-indexed, for labeling in W&B).
        current_config: The configuration module (e.g., imported `config.py`)
                        containing `NORM_MEAN` and `NORM_STD` for denormalization.
        max_images: The maximum number of image-mask sets to log from the batch.
    """
    if not wandb.run:  # Check if wandb run is active
        return

    num_to_log: int = min(max_images, images.shape[0])
    log_dict: Dict[str, wandb.Image] = {}

    # Ensure NORM_MEAN and NORM_STD are available in current_config
    norm_mean = getattr(current_config, "NORM_MEAN", (0.0, 0.0, 0.0))
    norm_std = getattr(current_config, "NORM_STD", (1.0, 1.0, 1.0))

    mean_tensor = torch.tensor(norm_mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(norm_std, device=images.device).view(1, 3, 1, 1)

    for i in range(num_to_log):
        # Denormalize image for visualization
        img_to_log: torch.Tensor = (images[i].clone() * std_tensor) + mean_tensor
        img_to_log = img_to_log.clamp(0, 1)

        # Prepare 2D numpy arrays for mask_data from the label ID tensors
        true_mask_np_2d: np.ndarray = true_masks[i].cpu().numpy()
        pred_mask_np_2d: np.ndarray = pred_masks[i].cpu().numpy()

        # Ensure masks are 2D (H, W) by squeezing if they are (1, H, W)
        if true_mask_np_2d.ndim == 3 and true_mask_np_2d.shape[0] == 1:
            true_mask_np_2d = true_mask_np_2d.squeeze(0)
        if pred_mask_np_2d.ndim == 3 and pred_mask_np_2d.shape[0] == 1:
            pred_mask_np_2d = pred_mask_np_2d.squeeze(0)

        # Final check for 2D
        if true_mask_np_2d.ndim != 2 or pred_mask_np_2d.ndim != 2:
            print(
                f"Warning: Skipping W&B image log for sample {i} due to unexpected mask dimensions. "
                f"True mask shape: {true_masks[i].shape}, Pred mask shape: {pred_masks[i].shape}"
            )
            continue

        wandb_image = wandb.Image(
            img_to_log,  # The denormalized input image
            masks={
                "ground_truth": {
                    "mask_data": true_mask_np_2d,  # Pass the 2D numpy array of class IDs
                    "class_labels": CITYSCAPES_ID_TO_NAME_MAP,  # Use ID -> String Name map
                },
                "prediction": {
                    "mask_data": pred_mask_np_2d,  # Pass the 2D numpy array of class IDs
                    "class_labels": CITYSCAPES_ID_TO_NAME_MAP,  # Use ID -> String Name map
                },
            },
        )
        log_dict[f"Validation_Epoch_{epoch}_Sample_{i + 1}"] = wandb_image

    if log_dict:
        wandb.log(log_dict)


# --- Performance Metrics (FLOPs, Latency) ---
@torch.no_grad()
def calculate_performance_metrics(
    model: torch.nn.Module,
    device: torch.device,
    img_height: int,
    img_width: int,
    latency_iters: int,
    warmup_iters: int,
) -> Dict[str, Any]:
    """
    Calculates FLOPs, number of parameters, latency, and FPS for a given model.

    Args:
        model: The PyTorch model (nn.Module) to evaluate.
        device: The PyTorch device to run the model on for latency tests.
        img_height: The height of the dummy input image for FLOPs/latency.
        img_width: The width of the dummy input image for FLOPs/latency.
        latency_iters: The number of iterations for averaging latency.
        warmup_iters: The number of warmup iterations before timing latency.

    Returns:
        A dictionary containing performance metrics:
        - 'flops_g' (float): GigaFLOPs.
        - 'params_m' (float): Total parameters in Millions.
        - 'flop_table' (str): Formatted string table of FLOPs by module.
        - 'mean_latency_ms' (float): Mean inference latency in milliseconds.
        - 'std_latency_ms' (float): Standard deviation of latency.
        - 'mean_fps' (float): Mean Frames Per Second.
        - 'std_fps' (float): Standard deviation of FPS.
        Values might be -1 or error strings if calculation fails.
    """
    model.eval()  # Set model to evaluation mode
    results: Dict[str, Any] = {}
    # Create a dummy input tensor with batch size 1, 3 color channels
    dummy_image: torch.Tensor = torch.randn(1, 3, img_height, img_width).to(device)

    print("Calculating FLOPs and Parameters...")
    try:
        # fvcore for FLOPs and parameter count
        flops_analyzer = FlopCountAnalysis(model, dummy_image)
        results["flops_g"] = flops_analyzer.total() / 1e9
        # Calculate total parameters (both trainable and non-trainable)
        results["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
        results["flop_table"] = flop_count_table(flops_analyzer, max_depth=3)
    except Exception as e:
        print(f"Error calculating FLOPs/Params: {e}")
        results["flops_g"] = -1.0
        results["params_m"] = -1.0
        results["flop_table"] = "Error during FLOPs/Params calculation."

    print(
        f"Calculating Latency (warmup: {warmup_iters} iters, measurement: {latency_iters} iters)..."
    )
    latencies_ms: List[float] = []
    try:
        # Warm-up iterations
        for _ in range(warmup_iters):
            _ = model(dummy_image)

        # Latency measurement iterations
        for i in range(latency_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(
                    device=device
                )  # Ensure previous CUDA ops are done
            start_time = time.perf_counter()  # More precise for short durations
            _ = model(dummy_image)
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)  # Ensure model inference is done
            end_time = time.perf_counter()
            latencies_ms.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds
            if (i + 1) % (latency_iters // 10 or 1) == 0:
                print(f" Latency iteration {i + 1}/{latency_iters} completed.")
    except Exception as e:
        print(f"Error during latency calculation: {e}")

    if latencies_ms:
        results["mean_latency_ms"] = np.mean(latencies_ms)
        results["std_latency_ms"] = np.std(latencies_ms)
        fps_list: List[float] = [1000.0 / l for l in latencies_ms if l > 0]
        results["mean_fps"] = np.mean(fps_list) if fps_list else 0.0
        results["std_fps"] = np.std(fps_list) if fps_list else 0.0
    else:  # Fill with defaults if latency calculation failed or produced no results
        results["mean_latency_ms"] = -1.0
        results["std_latency_ms"] = -1.0
        results["mean_fps"] = -1.0
        results["std_fps"] = -1.0

    return results


# --- Checkpoint Helper Functions ---
def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """
    Saves the training state to a checkpoint file.

    Args:
        state: A dictionary containing the state to save (e.g., epoch,
               model_state_dict, optimizer_state_dict, scaler_state_dict, best_miou).
        filepath: The path where the checkpoint file will be saved.
    """
    print(f"Saving checkpoint to {filepath}...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved successfully to {filepath}")

    # Optional: Save to W&B as an artifact if a run is active
    if wandb.run:
        try:
            # Use policy="live" if you want W&B to keep this file updated (e.g., for 'latest_checkpoint.pth')
            # For best model or periodic, you might not need "live" or can use default.
            wandb.save(filepath, base_path=os.path.dirname(filepath), policy="live")
            print(
                f"Checkpoint '{os.path.basename(filepath)}' also saved to W&B artifacts."
            )
        except Exception as e:
            print(f"Warning: Could not save checkpoint to W&B artifacts: {e}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads training state from a checkpoint file.

    Args:
        filepath: Path to the checkpoint file.
        model: The model instance to load the state_dict into.
        optimizer: The optimizer instance to load the state_dict into (optional).
        scaler: The GradScaler instance to load the state_dict into (optional).
        device: The torch.device to map the loaded tensors to. If None, loads to CPU first.

    Returns:
        A dictionary containing the loaded state (epoch, global_step, best_miou, etc.),
        or an empty dictionary if the checkpoint file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at '{filepath}'. Cannot resume.")
        return {}  # Return empty dict to indicate no checkpoint was loaded

    print(f"Loading checkpoint from '{filepath}'...")
    map_location = (
        device if device else torch.device("cpu")
    )  # Load to specified device or CPU

    try:
        checkpoint: Dict[str, Any] = torch.load(filepath, map_location=map_location)
    except Exception as e:
        print(
            f"Error loading checkpoint file '{filepath}': {e}. Returning empty state."
        )
        return {}

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(
            f"Warning: 'model_state_dict' not found in checkpoint '{filepath}'. Model weights not loaded."
        )

    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # After loading optimizer state, if tensors were loaded to CPU via map_location,
            # and the model/optimizer are meant to be on GPU, move optimizer states to GPU.
            if device and device.type == "cuda":
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            print("Optimizer state loaded.")
        except Exception as e:
            print(
                f"Warning: Could not load optimizer state: {e}. Optimizer state may be reset."
            )

    if scaler and "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("GradScaler state loaded.")
        except Exception as e:
            print(
                f"Warning: Could not load GradScaler state: {e}. Scaler state may be reset."
            )

    print("Checkpoint loaded successfully.")
    # Return relevant information for resuming training
    return {
        "epoch": checkpoint.get(
            "epoch", -1
        ),  # Return -1 if not found, so start_epoch becomes 0
        "global_step": checkpoint.get("global_step", 0),
        "best_miou": checkpoint.get("best_miou", 0.0),
        "best_model_per_class_ious": checkpoint.get("best_model_per_class_ious"),
        # Include any other saved metadata you need
    }
