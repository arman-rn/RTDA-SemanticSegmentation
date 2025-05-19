from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset


class CityScapes(Dataset):
    """
    CityScapes Dataset for Semantic Segmentation.
    This class is designed to load the CityScapes dataset, which contains images and their corresponding semantic segmentation labels.
    This class provides methods to load the images and labels, apply transformations, and return them in a format suitable for training deep learning models.
    """

    def __init__(
        self, cityscapes_path: str, split: str, transforms: Optional[Compose] = None
    ):
        """
        Initializes the CityScapes dataset class.

        This constructor sets up the dataset for use, either for training or validation, based on the provided path and subset selection. It optionally applies a transformation to the data.

        Args:
            cityscapes_path (str): The root directory path where the CityScapes dataset is stored.
            split (str): A string that specifies whether to load the 'train' or 'val' subset of the dataset.
            transforms (Optional[Compose], optional): A function/transform that takes in an image and label and returns a transformed version. Defaults to None.

        Raises:
            ValueError: If split is not 'train' or 'val'.
        """

        if split not in ["train", "val"]:
            raise ValueError("split must be 'train' or 'val'")

        self.cityscapes_path = Path(cityscapes_path)
        self.transforms = transforms
        self.image_label_pairs = self._get_image_label_pairs(split)

    def _get_image_label_pairs(self, split: str) -> list[Tuple[str, str]]:
        image_label_pairs = []

        # rglob & sort for deterministic order
        image_root = self.cityscapes_path / "images" / split
        image_paths = sorted(image_root.rglob("*.png"))

        for image_path in image_paths:
            # Get the corresponding label path
            label_path = Path(
                str(image_path)
                .replace("images", "gtFine")
                .replace("_leftImg8bit", "_gtFine_labelTrainIds")
            )
            image_label_pairs.append((image_path, label_path))

        return image_label_pairs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.image_label_pairs[index]
        img = Image.open(
            img_path
        ).convert(
            "RGB"
        )  # .convert('RGB') ensures the image has 3 color channels, even if it was grayscale or RGBA.

        label = Image.open(
            label_path
        )  # This is a grayscale image where each pixel represents a class index (e.g., 0 = road, 1 = sidewalk, etc.)

        img, label = (
            np.array(img),
            np.array(label),
        )  # Converts both the image and label from PIL to NumPy arrays. img: shape (H, W, 3) with pixel values 0–255 , label: shape (H, W) with integer class IDs

        if self.transforms:
            transformed = self.transforms(image=img, mask=label)
            image_tensor, label_tensor = (
                transformed["image"],
                transformed["mask"],
            )
            # If a transform is provided (an Albumentations Compose object), apply it. Albumentations applies the same spatial transforms (e.g., resize, crop, flip) to both image and mask in sync.
            # Returns a dict:
            # {
            #   'image': transformed_image,
            #   'mask': transformed_label
            # }

            # ToTensorV2() transforms the image and label to PyTorch tensors. The image is converted to a float tensor with values in the range [0, 1], while the label is converted to a long tensor with integer values.
            # The output tensors are:
            # image: torch.FloatTensor (C, H, W)
            # label: torch.LongTensor (H, W)
        else:
            # If no transform is provided, we need to convert the image and label to tensors manually.

            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255
            # torch.from_numpy(img): Converts the NumPy array (shape (H, W, C), dtype uint8, values 0–255) into a PyTorch tensor (still (H, W, C), still uint8).
            # permute(2, 0, 1): Reorders the dimensions from (H, W, C) → (C, H, W). This is the format expected by PyTorch models (channel-first).
            # float(): Converts from uint8 (0–255) to float32.
            # / 255: Normalizes pixel values from [0, 255] to [0.0, 1.0].

            label_tensor = torch.from_numpy(label).long()
            # torch.from_numpy(label): Converts the label mask from a NumPy array (shape (H, W), dtype usually uint8 or int32) to a PyTorch tensor.

            # .long(): Converts the tensor to int64 (PyTorch's default long integer type).
            # This is important because: PyTorch loss functions like nn.CrossEntropyLoss require target tensors to be of type torch.LongTensor.

        return image_tensor, label_tensor

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        This method returns the total number of items in the dataset, which corresponds to the number of image-label pairs loaded.

        Returns:
            int: The total number of items in the dataset.
        """

        return len(self.image_label_pairs)


# Notes:
# gtFine is the folder with the labels ("gtFine_labelTrainIds" files are the ones with labels for each pixel and the "gtFine_color" files are the ones with colors for each label for visualization),
# the images with "gtFine_labelTrainIds" name are used for training because they are the ones with the labels (each pixel has a label),
# we need to get the corresponding image for each label, which is in the "/images" folder with the same name but "leftImg8bit" instead of "gtFine_labelTrainIds".
