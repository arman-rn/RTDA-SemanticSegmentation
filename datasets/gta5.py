from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset

from .label_definitions import GTA5LabelInfo


class GTA5(Dataset):
    """
    GTA5 Dataset for Semantic Segmentation.
    This class loads images and their corresponding semantic segmentation labels from the GTA5 dataset.
    It converts the color-coded labels to class IDs compatible with Cityscapes trainIds.
    """

    _COLOR_TO_ID_LUT = None

    @staticmethod
    def _initialize_lut():
        if GTA5._COLOR_TO_ID_LUT is None:
            print("Initializing GTA5 Color-to-ID LUT for on-the-fly conversion...")
            # Use the imported GTA5LabelInfo
            lut = np.full((256, 256, 256), GTA5LabelInfo.ignore_id, dtype=np.uint8)
            for color_tuple, class_id in GTA5LabelInfo.color_to_id_map.items():
                lut[color_tuple[0], color_tuple[1], color_tuple[2]] = class_id
            GTA5._COLOR_TO_ID_LUT = lut
            print("GTA5 LUT initialized.")
        return GTA5._COLOR_TO_ID_LUT

    def __init__(
        self,
        gta5_path: str,
        labels_subdir: str,
        convert_on_the_fly: bool,
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
        self.labels_subdir = labels_subdir
        self.convert_on_the_fly = convert_on_the_fly
        self.transforms = transforms

        self.image_label_pairs = self._get_image_label_pairs()

        if self.convert_on_the_fly:
            self.lut = GTA5._initialize_lut()
        else:
            self.lut = None

        if not self.image_label_pairs:
            raise FileNotFoundError(
                f"No image-label pairs found in {gta5_path}. "
                "Please check the path and ensure 'images' and 'labels' subdirectories exist and are populated."
            )

    def _get_image_label_pairs(self) -> list[Tuple[Path, Path]]:
        image_label_pairs = []
        image_root = self.gta5_path / "images"
        label_root = self.gta5_path / self.labels_subdir

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

    def _convert_rgb_to_id_with_lut(self, label_rgb_np: np.ndarray) -> np.ndarray:
        if self.lut is None:
            raise RuntimeError("LUT not initialized for on-the-fly conversion.")
        if label_rgb_np.ndim != 3 or label_rgb_np.shape[2] != 3:
            raise ValueError(
                f"Input label_rgb_np must be (H, W, 3), got {label_rgb_np.shape}"
            )
        return self.lut[
            label_rgb_np[..., 0], label_rgb_np[..., 1], label_rgb_np[..., 2]
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.image_label_pairs[index]
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading PIL image {img_path}: {e}")

        if self.convert_on_the_fly:
            try:
                label_pil_rgb = Image.open(label_path).convert("RGB")
                label_rgb_np = np.array(label_pil_rgb)
                label_id_np = self._convert_rgb_to_id_with_lut(label_rgb_np)
            except Exception as e:
                raise RuntimeError(
                    f"Error converting RGB label {label_path} on-the-fly: {e}"
                )
        else:
            try:
                label_pil_gray = Image.open(label_path)
                if label_pil_gray.mode != "L" and label_pil_gray.mode != "P":
                    print(
                        f"Warning: Pre-converted label {label_path} mode {label_pil_gray.mode}. Converting to 'L'."
                    )
                    label_pil_gray = label_pil_gray.convert("L")
                label_id_np = np.array(label_pil_gray, dtype=np.uint8)
                if label_id_np.ndim == 3:
                    if label_id_np.shape[2] == 1:
                        label_id_np = label_id_np.squeeze(-1)
                    else:
                        raise ValueError(
                            f"Pre-converted label {label_path} not single channel. Shape: {label_id_np.shape}"
                        )
            except Exception as e:
                raise RuntimeError(
                    f"Error loading pre-converted label {label_path}: {e}"
                )
        img_np = np.array(img_pil)
        if self.transforms:
            try:
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
