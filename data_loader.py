"""
Handles dataset loading and provides PyTorch DataLoader instances.

This file uses dataset classes (e.g., CityScapes) and configurations
from config.py to create PyTorch DataLoaders. DataLoaders are responsible
for efficiently loading data in batches, shuffling, and enabling
multi-process data loading for training and validation.
It also includes utility functions for visualizing segmentation masks.
"""

# This file uses datasets class (cityscapes/gta5) and the configurations from config.py to create PyTorch DataLoader objects.
# DataLoaders are responsible for efficiently loading data in batches, shuffling, and enabling multi-process data loading.
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import the CityScapes class from your existing file
from datasets.cityscapes import CityScapes

ConfigModule = Any


# --- DataLoader Getter Function ---
def get_loaders(config_obj: ConfigModule) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns training and validation PyTorch DataLoaders.

    It uses the CityScapes dataset class (imported from datasets.cityscapes)
    and applies transformations specified in the configuration object.

    Args:
        config_obj: The configuration module or object (e.g., imported `config.py`)
            containing paths, batch size, number of workers, and transformations.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
        (train_loader, val_loader)

    Raises:
        ValueError: If the training or validation dataset is found to be empty
            after initialization, indicating a likely issue with `DATASET_PATH`
            or the dataset class implementation.
    """
    print(f"Loading training data from: {config_obj.DATASET_PATH}")
    print("Using CityScapes class from: datasets.cityscapes.py")
    train_dataset = CityScapes(
        cityscapes_path=config_obj.DATASET_PATH,
        split="train",
        transforms=config_obj.TRAIN_TRANSFORMS,
    )
    if not len(train_dataset):
        print(
            f"CRITICAL: Training dataset is empty. Check DATASET_PATH in config.py ('{config_obj.DATASET_PATH}') and the implementation of 'datasets/cityscapes.py'."
        )
    else:
        print(f"Found {len(train_dataset)} training images.")

    print(f"Loading validation data from: {config_obj.DATASET_PATH}")
    val_dataset = CityScapes(
        cityscapes_path=config_obj.DATASET_PATH,
        split="val",
        transforms=config_obj.VAL_TRANSFORMS,
    )
    if not len(val_dataset):
        print(
            f"CRITICAL: Validation dataset is empty. Check DATASET_PATH in config.py ('{config_obj.DATASET_PATH}') and the implementation of 'datasets/cityscapes.py'."
        )
    else:
        print(f"Found {len(val_dataset)} validation images.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_obj.BATCH_SIZE,
        shuffle=True,  # Shuffle training data
        num_workers=config_obj.DATALOADER_NUM_WORKERS,  # Number of parallel processes for data loading.
        pin_memory=True,  # For faster data transfer to GPU. If True, copies tensors into CUDA pinned memory before returning them, which can speed up GPU transfers.
        drop_last=True,  # Drops the last batch if it's smaller than BATCH_SIZE (good for training). If the dataset size is not divisible by the batch size, the last batch will be smaller. drop_last=True ignores this smaller batch.
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Usually 1 for validation to get per-image metrics
        shuffle=False,  # No need to shuffle validation data
        num_workers=config_obj.DATALOADER_NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


# --- For Visualization with W&B ---
# A dictionary mapping Cityscapes class IDs (the 19 training IDs) to specific RGB colors for visualization.
CITYSCAPES_COLOR_MAP_TRAIN_IDS: Dict[int, Tuple[int, int, int]] = {
    0: (128, 64, 128),  # road
    1: (244, 35, 232),  # sidewalk
    2: (70, 70, 70),  # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),  # traffic light
    7: (220, 220, 0),  # traffic sign
    8: (107, 142, 35),  # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),  # person
    12: (255, 0, 0),  # rider
    13: (0, 0, 142),  # car
    14: (0, 0, 70),  # truck
    15: (0, 60, 100),  # bus
    16: (0, 80, 100),  # train
    17: (0, 0, 230),  # motorcycle
    18: (119, 11, 32),  # bicycle
    255: (0, 0, 0),  # ignore/void label often mapped to black
}


def tensor_to_rgb(
    label_tensor: torch.Tensor,
    color_map: Dict[int, Tuple[int, int, int]] = CITYSCAPES_COLOR_MAP_TRAIN_IDS,
) -> np.ndarray:
    """
    Converts a 2D integer label tensor to a 3D RGB color-coded NumPy image.

    This function takes a tensor where each pixel value represents a class ID
    and maps these IDs to specified RGB colors for visualization purposes,
    often used for logging segmentation masks to tools like Weights & Biases.

    Args:
        label_tensor: A PyTorch tensor of shape (H, W) containing integer class IDs.
            It's expected to be on the CPU before conversion to NumPy.
        color_map: A dictionary mapping class IDs (int) to RGB color tuples
            (Tuple[int, int, int]). Defaults to `CITYSCAPES_COLOR_MAP_TRAIN_IDS`.

    Returns:
        A NumPy array of shape (H, W, 3) with dtype uint8, representing the
        RGB color-coded segmentation mask.
    """
    if label_tensor.is_cuda:
        label_tensor = label_tensor.cpu()
    label_np: np.ndarray = label_tensor.numpy()

    # Ensure label_np is 2D if it came from a batch of 1 e.g. (1, H, W)
    if label_np.ndim == 3 and label_np.shape[0] == 1:
        label_np = label_np.squeeze(0)

    if label_np.ndim != 2:
        raise ValueError(
            f"Expected label_tensor to be 2D (H, W) or (1, H, W), but got shape {label_tensor.shape}"
        )

    output_rgb = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        output_rgb[label_np == label_id] = color
    return output_rgb
