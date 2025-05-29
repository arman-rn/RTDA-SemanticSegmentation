# data_loader.py
"""
Handles dataset loading and provides PyTorch DataLoader instances.

This file uses dataset classes (e.g., CityScapes) and configurations
from config.py to create PyTorch DataLoaders. DataLoaders are responsible
for efficiently loading data in batches, shuffling, and enabling
multi-process data loading for training and validation.
It also includes utility functions for visualizing segmentation masks and class mappings.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5

# Type alias for the config module for clarity
ConfigModule = Any


# --- DataLoader Getter Function ---
def get_loaders(
    config_obj: ConfigModule,
    train_dataset_name: str = "gta5",
    val_dataset_name: str = "cityscapes",
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns training and validation PyTorch DataLoaders.

    It uses the dataset classes (imported from datasets.cityscapes)
    and applies transformations specified in the configuration object.

    Args:
        config_obj: The configuration module or object (e.g., imported `config.py`)
            containing paths, batch size, number of workers, and transformations.
        train_dataset_name (str): Name of the dataset for training ("cityscapes" or "gta5").
        val_dataset_name (str): Name of the dataset for validation ("cityscapes" or "gta5").

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
        (train_loader, val_loader)

    Raises:
        ValueError: If the training or validation dataset is found to be empty
            after initialization, indicating a likely issue with `DATASET_PATH`
            or the dataset class implementation.
    """

    # --- Training Dataset ---
    if train_dataset_name.lower() == "cityscapes":
        print(
            f"Loading Cityscapes training data from: {config_obj.CITYSCAPES_DATASET_PATH}"
        )
        print(
            f"Using Cityscapes training transforms (resize to {config_obj.CITYSCAPES_IMG_WIDTH}x{config_obj.CITYSCAPES_IMG_HEIGHT})."
        )
        train_dataset = CityScapes(
            cityscapes_path=config_obj.CITYSCAPES_DATASET_PATH,
            split="train",
            transforms=config_obj.CITYSCAPES_TRAIN_TRANSFORMS,
        )
    elif train_dataset_name.lower() == "gta5":
        print(f"Loading GTA5 training data from: {config_obj.GTA5_DATASET_PATH}")
        print(
            f"Using GTA5 training transforms (resize to {config_obj.GTA5_IMG_WIDTH}x{config_obj.GTA5_IMG_HEIGHT})."
        )
        # Determine which label subdirectory to use and if conversion is on-the-fly
        convert_on_the_fly = config_obj.GTA5_CONVERT_LABELS_ON_THE_FLY
        if convert_on_the_fly:
            labels_subdir = config_obj.GTA5_ORIGINAL_LABELS_SUBDIR
            print(
                f"GTA5 labels will be converted on-the-fly from subdir: '{labels_subdir}'"
            )
        else:
            labels_subdir = config_obj.GTA5_PRECONVERTED_LABELS_SUBDIR
            print(f"GTA5 will use pre-converted labels from subdir: '{labels_subdir}'")

        train_dataset = GTA5(
            gta5_path=config_obj.GTA5_DATASET_PATH,
            labels_subdir=labels_subdir,
            convert_on_the_fly=convert_on_the_fly,
            transforms=config_obj.GTA5_TRAIN_TRANSFORMS,
        )
    else:
        raise ValueError(f"Unsupported training dataset: {train_dataset_name}")

    if not len(train_dataset):
        raise ValueError(f"CRITICAL: Training dataset '{train_dataset_name}' is empty.")

    print(f"Found {len(train_dataset)} training images for {train_dataset_name}.")

    # --- Validation Dataset ---
    if val_dataset_name.lower() == "cityscapes":
        print(
            f"Loading Cityscapes validation data from: {config_obj.CITYSCAPES_DATASET_PATH}"
        )
        print(
            f"Using Cityscapes validation transforms (resize to {config_obj.CITYSCAPES_IMG_WIDTH}x{config_obj.CITYSCAPES_IMG_HEIGHT})."
        )
        val_dataset = CityScapes(
            cityscapes_path=config_obj.CITYSCAPES_DATASET_PATH,
            split="val",
            transforms=config_obj.CITYSCAPES_VAL_TRANSFORMS,
        )
    elif val_dataset_name.lower() == "gta5":
        # This case is not needed but supported
        print(f"Loading GTA5 validation data from: {config_obj.GTA5_DATASET_PATH}")
        print(
            f"Using GTA5 validation transforms (resize to {config_obj.GTA5_IMG_WIDTH}x{config_obj.GTA5_IMG_HEIGHT})."
        )
        convert_on_the_fly_val = config_obj.GTA5_CONVERT_LABELS_ON_THE_FLY
        if convert_on_the_fly_val:
            labels_subdir_val = config_obj.GTA5_ORIGINAL_LABELS_SUBDIR
        else:
            labels_subdir_val = config_obj.GTA5_PRECONVERTED_LABELS_SUBDIR
        print(
            f"GTA5 validation labels from subdir '{labels_subdir_val}', convert_on_the_fly={convert_on_the_fly_val}"
        )

        val_dataset = GTA5(
            gta5_path=config_obj.GTA5_DATASET_PATH,
            labels_subdir=labels_subdir_val,
            convert_on_the_fly=convert_on_the_fly_val,
            transforms=config_obj.GTA5_TRAIN_TRANSFORMS,  # Or a dedicated GTA5_VAL_TRANSFORMS if you create one
        )
    else:
        raise ValueError(f"Unsupported validation dataset: {val_dataset_name}")

    if not len(val_dataset):
        raise ValueError(f"CRITICAL: Validation dataset '{val_dataset_name}' is empty.")

    print(f"Found {len(val_dataset)} validation images for {val_dataset_name}.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_obj.BATCH_SIZE,
        shuffle=True,
        num_workers=config_obj.DATALOADER_NUM_WORKERS,
        pin_memory=True,  # Helps speed up CPU to GPU memory transfers
        drop_last=True,  # Drop the last incomplete batch during training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Typically 1 for validation for per-image metrics
        shuffle=False,
        num_workers=config_obj.DATALOADER_NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


# --- Color and Name Mappings for Visualization ---

# Color map for Cityscapes 19 evaluation classes + ignore label
# Each class ID is mapped to an RGB tuple.
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
    255: (0, 0, 0),  # ignore/void label (often mapped to black)
}

# Map of Class IDs to String Names for W&B class_labels argument
CITYSCAPES_ID_TO_NAME_MAP: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "ignore/void",  # Include the ignore label if it might appear in masks
}


def tensor_to_rgb(
    label_tensor: torch.Tensor,
    color_map: Dict[int, Tuple[int, int, int]] = CITYSCAPES_COLOR_MAP_TRAIN_IDS,
) -> np.ndarray:
    """
    Converts a 2D integer label tensor to a 3D RGB color-coded NumPy image.

    This function takes a tensor where each pixel value represents a class ID
    and maps these IDs to specified RGB colors for visualization purposes.

    Args:
        label_tensor: A PyTorch tensor of shape (H, W) or (1, H, W) containing
            integer class IDs. Expected to be on CPU or moved to CPU.
        color_map: A dictionary mapping class IDs (int) to RGB color tuples
            (Tuple[int, int, int]). Defaults to `CITYSCAPES_COLOR_MAP_TRAIN_IDS`.

    Returns:
        A NumPy array of shape (H, W, 3) with dtype uint8, representing the
        RGB color-coded segmentation mask.

    Raises:
        ValueError: If the input label_tensor, after potential squeezing, is not 2D.
    """
    if label_tensor.is_cuda:
        label_tensor = label_tensor.cpu()
    label_np: np.ndarray = label_tensor.numpy()

    # Ensure label_np is 2D if it came from a batch of 1 e.g. (1, H, W)
    if label_np.ndim == 3 and label_np.shape[0] == 1:
        label_np = label_np.squeeze(0)

    if label_np.ndim != 2:
        raise ValueError(
            f"Expected label_tensor to be 2D (H, W) or (1, H, W) after squeeze, "
            f"but got shape {label_tensor.shape} (processed as {label_np.shape})"
        )

    output_rgb = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        output_rgb[label_np == label_id] = color
    return output_rgb
