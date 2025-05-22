# This file is a collection of various helper functions used across different parts of the project.
# This includes learning rate scheduling, metrics calculation (mIoU, FLOPs, latency), and Weights & Biases integration helpers.

import time

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

import wandb
from data_loader import (  # For W&B image logging
    CITYSCAPES_COLOR_MAP_TRAIN_IDS,
    tensor_to_rgb,
)


# --- Learning Rate Scheduler ---
def poly_lr_scheduler(optimizer, initial_learning_rate, current_iter, max_iter, power):
    """
    Polynomial decay of learning rate.
    Applies decay to the provided 'initial_learning_rate' and updates
    the single parameter group in the optimizer.

    Args:
        optimizer: The PyTorch optimizer.
        initial_learning_rate (float): The starting learning rate from which to decay.
        current_iter (int): Current training iteration.
        max_iter (int): Total number of training iterations.
        power (float): The exponent for the polynomial decay.
    Returns:
        float: The newly calculated (decayed) learning rate.
    """

    lr_scale_factor = (1 - current_iter / max_iter) ** power
    new_lr = initial_learning_rate * lr_scale_factor

    # Update the learning rate in the optimizer's single parameter group
    optimizer.param_groups[0]["lr"] = new_lr

    return new_lr


# --- mIoU Calculation Helpers ---
def fast_hist(label_true, label_pred, n_class):
    """
    Function for calculating the confusion matrix (hist)

    Args:
        label_true (_type_): flattened arrays of ground truth
        label_pred (_type_): predicted class IDs
        n_class (_type_): number of classes

    Returns:
        _type_: N x N confusion matrix where N is n_class.
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


def per_class_iou(hist):
    """
    Takes the confusion matrix hist and calculates IoU for each class

    Args:
        hist (_type_): confusion matrix

    Returns:
        _type_: _description_
    """
    epsilon = 1e-5  # Small value to avoid division by zero
    ious = np.diag(hist) / (
        hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + epsilon
    )
    ious = np.nan_to_num(ious, nan=0.0)  # Replace NaN with 0.0

    return ious


# --- W&B Initialization ---
def init_wandb(cfg, effective_optimizer_config):
    """Initializes a new Weights & Biases run.

    Args:
        cfg (_type_): configuration object containing the parameters for the run.
    """
    try:
        base_config_dict = {
            "epochs": cfg.TRAIN_EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "image_height": cfg.IMG_HEIGHT,
            "image_width": cfg.IMG_WIDTH,
            "num_classes": cfg.NUM_CLASSES,
            "lr_scheduler_power": cfg.LR_SCHEDULER_POWER,
            "device": str(cfg.DEVICE),
            "dataset_path": cfg.DATASET_PATH,
            "pretrained_model_path": cfg.PRETRAINED_MODEL_PATH,
            "norm_mean": cfg.NORM_MEAN,
            "norm_std": cfg.NORM_STD,
        }
        # Merge base config with effective optimizer config
        full_config = {**base_config_dict, **effective_optimizer_config}

        wandb.init(
            project=cfg.WANDB_PROJECT_NAME,
            entity=cfg.WANDB_ENTITY,
            config=full_config,
        )
        print("Weights & Biases initialized successfully.")
    except Exception as e:
        print(f"Error initializing W&B: {e}. W&B logging will be disabled.")
        # Optionally, create a dummy wandb object so calls to wandb.log don't crash
        # class DummyWandB:
        #     def __init__(self): self.run = None
        #     def log(self, *args, **kwargs): pass
        #     def summary(self, *args, **kwargs): pass # if you use summary
        #     def finish(self, *args, **kwargs): pass
        # wandb = DummyWandB() # This is a bit hacky, better to check wandb.run


# --- W&B Image Logging ---
def log_segmentation_to_wandb(
    images, true_masks, pred_masks, epoch, current_config, max_images=4
):
    """Logs sample input images, their ground truth segmentation masks, and the model's predicted masks to W&B.

    Args:
        images (_type_): Batch of input images.
        true_masks (_type_): Corresponding label tensors.
        pred_masks (_type_): Corresponding label tensors.
        epoch (_type_): _description_
        current_config (_type_): _description_
        max_images (int, optional): _description_. Defaults to 4.
    """
    if not wandb.run:
        return

    num_to_log = min(max_images, images.shape[0])
    log_dict = {}

    mean = torch.tensor(current_config.NORM_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(current_config.NORM_STD, device=images.device).view(1, 3, 1, 1)

    for i in range(num_to_log):
        img = (
            (images[i].clone() * std) + mean
        )  # Denormalize a clone, It denormalizes the input images for better visualization.
        img = img.clamp(0, 1)

        # Use CITYSCAPES_COLOR_MAP_TRAIN_IDS from data_loader.py for class labels if needed
        wandb_image = wandb.Image(
            img,
            masks={
                "ground_truth": {
                    "mask_data": tensor_to_rgb(true_masks[i]),
                    "class_labels": CITYSCAPES_COLOR_MAP_TRAIN_IDS,
                },
                "prediction": {
                    "mask_data": tensor_to_rgb(pred_masks[i]),
                    "class_labels": CITYSCAPES_COLOR_MAP_TRAIN_IDS,
                },
            },
        )
        log_dict[f"Val_Epoch_{epoch}_Sample_{i}"] = wandb_image

    if log_dict:
        wandb.log(
            log_dict
        )  # step will be managed by global step in main training loop or by epoch


# --- Performance Metrics (FLOPs, Latency) ---
@torch.no_grad()
def calculate_performance_metrics(
    model, device, img_height, img_width, latency_iters, warmup_iters
):
    """Calculates FLOPs, number of parameters, latency, and FPS, as required by the project.

    Args:
        model (_type_): _description_
        device (_type_): _description_
        img_height (_type_): _description_
        img_width (_type_): _description_
        latency_iters (_type_): _description_
        warmup_iters (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    results = {}
    dummy_image = torch.randn(1, 3, img_height, img_width).to(
        device
    )  # Creates a sample input tensor.

    print("Calculating FLOPs and Parameters...")
    try:
        flops_analyzer = FlopCountAnalysis(
            model, dummy_image
        )  # (fvcore usage from project description) Uses fvcore to analyze the model's operations with the dummy input.

        results["flops_g"] = flops_analyzer.total() / 1e9  # Total FLOPs in GigaFLOPs

        results["params_m"] = (
            sum(p.numel() for p in model.parameters()) / 1e6
        )  # Calculates total parameters in Millions.

        results["flop_table"] = flop_count_table(
            flops_analyzer, max_depth=3
        )  # Generates a formatted table of FLOPs per module.
    except Exception as e:
        print(f"Error calculating FLOPs/Params: {e}")
        results["flops_g"], results["params_m"], results["flop_table"] = (
            -1,
            -1,
            "Error calculating FLOPs/Params.",
        )

    print(
        f"Calculating Latency (warmup: {warmup_iters}, iterations: {latency_iters})..."
    )
    latencies_ms = []
    # Warm-up GPU (Latency calculation pseudo-code)
    # Runs the model `warmup_iters` times to stabilize GPU clocks and cache memory.
    for _ in range(warmup_iters):
        _ = model(dummy_image)

    # Latency measurement (Latency calculation pseudo-code)
    # Runs the model `latency_iters` times, timing each forward pass (torch.cuda.synchronize is used for accurate GPU timing).
    for i in range(latency_iters):
        torch.cuda.synchronize(device=device)
        start_time = time.time()
        _ = model(dummy_image)
        torch.cuda.synchronize(device=device)
        latencies_ms.append((time.time() - start_time) * 1000)  # milliseconds
        if (i + 1) % (latency_iters // 10 or 1) == 0:
            print(f" Latency iter {i + 1}/{latency_iters}")

    # Calculates mean/std latency (in ms) and mean/std FPS.
    if latencies_ms:
        results["mean_latency_ms"] = np.mean(
            latencies_ms
        )  # (Latency calculation pseudo-code)
        results["std_latency_ms"] = np.std(
            latencies_ms
        )  # (Latency calculation pseudo-code)
        fps_list = [
            1000.0 / l for l in latencies_ms if l > 0
        ]  # (Latency calculation pseudo-code)
        results["mean_fps"] = (
            np.mean(fps_list) if fps_list else 0
        )  # (Latency calculation pseudo-code)
        results["std_fps"] = (
            np.std(fps_list) if fps_list else 0
        )  # (Latency calculation pseudo-code)
    else:
        results["mean_latency_ms"], results["std_latency_ms"] = -1, -1
        results["mean_fps"], results["std_fps"] = -1, -1

    return results
