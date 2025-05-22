# This file uses datasets class (cityscapes/gta5) and the configurations from config.py to create PyTorch DataLoader objects.
# DataLoaders are responsible for efficiently loading data in batches, shuffling, and enabling multi-process data loading.
import numpy as np
from torch.utils.data import DataLoader

# Import the CityScapes class from your existing file
from datasets.cityscapes import CityScapes  # This will import your class


# --- DataLoader Getter Function ---
def get_loaders(cfg):
    """
    Creates and returns train and validation DataLoaders using the CityScapes
    dataset class from datasets.cityscapes.
    Args:
        cfg: Configuration object (e.g., from config.py)
    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"Loading training data from: {cfg.DATASET_PATH}")
    print("Using CityScapes class from: datasets.cityscapes.py")
    train_dataset = CityScapes(
        cityscapes_path=cfg.DATASET_PATH, split="train", transforms=cfg.TRAIN_TRANSFORMS
    )
    if not len(train_dataset):
        print(
            f"CRITICAL: Training dataset is empty. Check DATASET_PATH in config.py ('{cfg.DATASET_PATH}') and the implementation of 'datasets/cityscapes.py'."
        )
    else:
        print(f"Found {len(train_dataset)} training images.")

    print(f"Loading validation data from: {cfg.DATASET_PATH}")
    val_dataset = CityScapes(
        cityscapes_path=cfg.DATASET_PATH, split="val", transforms=cfg.VAL_TRANSFORMS
    )
    if not len(val_dataset):
        print(
            f"CRITICAL: Validation dataset is empty. Check DATASET_PATH in config.py ('{cfg.DATASET_PATH}') and the implementation of 'datasets/cityscapes.py'."
        )
    else:
        print(f"Found {len(val_dataset)} validation images.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,  # Shuffle training data
        num_workers=4,  # Number of parallel processes for data loading.
        pin_memory=True,  # For faster data transfer to GPU. If True, copies tensors into CUDA pinned memory before returning them, which can speed up GPU transfers.
        drop_last=True,  # Drops the last batch if it's smaller than BATCH_SIZE (good for training). If the dataset size is not divisible by the batch size, the last batch will be smaller. drop_last=True ignores this smaller batch.
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Usually 1 for validation to get per-image metrics
        shuffle=False,  # No need to shuffle validation data
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


# --- For Visualization with W&B ---
# A dictionary mapping Cityscapes class IDs (the 19 training IDs) to specific RGB colors for visualization.
CITYSCAPES_COLOR_MAP_TRAIN_IDS = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    255: (0, 0, 0),  # ignore/void
}


def tensor_to_rgb(label_tensor, color_map=CITYSCAPES_COLOR_MAP_TRAIN_IDS):
    """Converts a label tensor to an RGB image using the provided color map.
    A utility function that takes a 2D label tensor (H, W) containing class IDs and converts it into a 3D RGB image (H, W, 3) using the color_map.
    This is used to create visually interpretable segmentation masks for logging to Weights & Biases.

    Args:
        label_tensor (_type_): _description_
        color_map (_type_, optional): _description_. Defaults to CITYSCAPES_COLOR_MAP_TRAIN_IDS.

    Returns:
        _type_: _description_
    """
    label_np = label_tensor.cpu().numpy()
    output_rgb = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        output_rgb[label_np == label_id] = color
    return output_rgb
