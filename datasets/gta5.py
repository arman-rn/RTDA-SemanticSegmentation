from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class _GTA5LabelDef:
    """Helper dataclass to store GTA5 label properties."""

    name: str  # Added name for clarity, though not strictly used in mapping here
    ID: int  # This ID will be used as the trainId
    color: Tuple[int, int, int]


class GTA5LabelInfo:
    """
    Manages GTA5 label definitions and provides a color-to-ID map.
    The IDs are consistent with Cityscapes `trainId` for common classes.
    """

    road = _GTA5LabelDef(name="road", ID=0, color=(128, 64, 128))
    sidewalk = _GTA5LabelDef(name="sidewalk", ID=1, color=(244, 35, 232))
    building = _GTA5LabelDef(name="building", ID=2, color=(70, 70, 70))
    wall = _GTA5LabelDef(name="wall", ID=3, color=(102, 102, 156))
    fence = _GTA5LabelDef(name="fence", ID=4, color=(190, 153, 153))
    pole = _GTA5LabelDef(name="pole", ID=5, color=(153, 153, 153))
    light = _GTA5LabelDef(name="traffic light", ID=6, color=(250, 170, 30))
    sign = _GTA5LabelDef(name="traffic sign", ID=7, color=(220, 220, 0))
    vegetation = _GTA5LabelDef(name="vegetation", ID=8, color=(107, 142, 35))
    terrain = _GTA5LabelDef(name="terrain", ID=9, color=(152, 251, 152))
    sky = _GTA5LabelDef(name="sky", ID=10, color=(70, 130, 180))
    person = _GTA5LabelDef(name="person", ID=11, color=(220, 20, 60))
    rider = _GTA5LabelDef(name="rider", ID=12, color=(255, 0, 0))
    car = _GTA5LabelDef(name="car", ID=13, color=(0, 0, 142))
    truck = _GTA5LabelDef(name="truck", ID=14, color=(0, 0, 70))
    bus = _GTA5LabelDef(name="bus", ID=15, color=(0, 60, 100))
    train = _GTA5LabelDef(name="train", ID=16, color=(0, 80, 100))
    motorcycle = _GTA5LabelDef(name="motorcycle", ID=17, color=(0, 0, 230))
    bicycle = _GTA5LabelDef(name="bicycle", ID=18, color=(119, 11, 32))
    # Note: Class ID 19-254 are typically not used, 255 is ignore_index

    # List of all defined labels
    definitions: List[_GTA5LabelDef] = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motorcycle,
        bicycle,
    ]

    # Create a mapping from color tuples to a class ID for efficient conversion
    color_to_id_map = {label_def.color: label_def.ID for label_def in definitions}

    # The ignore index to use for pixels that don't map to any defined color
    # This should match config.IGNORE_INDEX
    ignore_id = 255


class GTA5(Dataset):
    """
    GTA5 Dataset for Semantic Segmentation.
    This class loads images and their corresponding semantic segmentation labels from the GTA5 dataset.
    It converts the color-coded labels to class IDs compatible with Cityscapes trainIds.
    """

    def __init__(
        self,
        gta5_path: str,
        transforms: Optional[Compose] = None,
    ):
        """
        Initializes the GTA5 dataset class.

        Args:
            gta5_path (str): The root directory path where the GTA5 dataset is stored.
                             Expected structure: gta5_path/images/*.png and gta5_path/labels/*.png
            transforms (Optional[Compose], optional): A function/transform that takes in an
                                                      image and label and returns a transformed version.
                                                      Defaults to None.
        Raises:
            FileNotFoundError: If the image or label root directory doesn't exist or no pairs are found.
        """

        self.gta5_path = Path(gta5_path)
        self.transforms = transforms
        self.image_label_pairs = self._get_image_label_pairs()

        if not self.image_label_pairs:
            raise FileNotFoundError(
                f"No image-label pairs found in {gta5_path}. "
                "Please check the path and ensure 'images' and 'labels' subdirectories exist and are populated."
            )

    def _get_image_label_pairs(self) -> list[Tuple[Path, Path]]:
        image_label_pairs = []
        image_root = self.gta5_path / "images"
        label_root = self.gta5_path / "labels"

        if not image_root.is_dir():
            raise FileNotFoundError(
                f"Image directory not found or is not a directory: {image_root}"
            )
        if not label_root.is_dir():
            raise FileNotFoundError(
                f"Label directory not found or is not a directory: {label_root}"
            )

        image_paths = sorted(list(image_root.rglob("*.png")))

        if not image_paths:
            print(f"Warning: No images found in {image_root}. Check for .png files.")

        for image_path in image_paths:
            label_path = label_root / image_path.name
            if label_path.exists():
                image_label_pairs.append((image_path, label_path))
            else:
                print(
                    f"Warning: Label not found for image {image_path}. Expected at {label_path}"
                )

        print(f"Found {len(image_label_pairs)} image-label pairs in {self.gta5_path}")
        return image_label_pairs

    def _convert_rgb_to_id(self, label_rgb_np: np.ndarray) -> np.ndarray:
        """Converts an RGB label image (H, W, 3) to a 2D class ID map (H, W)."""
        if label_rgb_np.ndim != 3 or label_rgb_np.shape[2] != 3:
            raise ValueError(
                f"Input label_rgb_np must be an RGB image with shape (H, W, 3), got {label_rgb_np.shape}"
            )

        # Initialize the ID map with the ignore_id
        id_map_np = np.full(
            (label_rgb_np.shape[0], label_rgb_np.shape[1]),
            GTA5LabelInfo.ignore_id,
            dtype=np.uint8,
        )

        for color_tuple, class_id in GTA5LabelInfo.color_to_id_map.items():
            # Create a mask for all pixels matching the current color
            mask = np.all(
                label_rgb_np == np.array(color_tuple, dtype=np.uint8).reshape(1, 1, 3),
                axis=2,
            )
            id_map_np[mask] = class_id

        return id_map_np

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.image_label_pairs[index]

        try:
            img_pil = Image.open(img_path).convert("RGB")
            label_pil = Image.open(label_path).convert(
                "RGB"
            )  # Ensure label is also loaded as RGB
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not open image or label at index {index}. Path: {img_path} or {label_path}. Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading image/label at index {index} ({img_path}, {label_path}): {e}"
            )

        img_np = np.array(img_pil)
        label_rgb_np = np.array(label_pil)

        # Convert RGB label to single-channel ID map
        label_id_np = self._convert_rgb_to_id(
            label_rgb_np
        )  # This is now (H, W) with class IDs

        if self.transforms:
            try:
                # Albumentations expects 'mask' to be the 2D label map
                transformed = self.transforms(image=img_np, mask=label_id_np)
                image_tensor, label_tensor = transformed["image"], transformed["mask"]
            except Exception as e:
                raise RuntimeError(
                    f"Error applying transforms to image/label ({img_path}, {label_path}): {e}"
                )
        else:
            # Manual conversion if no albumentations transforms are provided
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            label_tensor = torch.from_numpy(label_id_np).long()

        # Ensure label tensor is 2D (H, W)
        if label_tensor.ndim == 3 and label_tensor.shape[0] == 1:  # (1, H, W)
            label_tensor = label_tensor.squeeze(0)
        elif label_tensor.ndim != 2:
            raise ValueError(
                f"Label tensor for {label_path} has unexpected shape {label_tensor.shape} after transforms/conversion. Expected (H,W) or (1,H,W)."
            )

        return image_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.image_label_pairs)
