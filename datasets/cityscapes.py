import os
from typing import Optional, Tuple

import numpy as np
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class CityScapes(Dataset):
    """
    CityScapes Dataset for Semantic Segmentation.
    This class is designed to load the CityScapes dataset, which contains images and their corresponding semantic segmentation labels.
    This class provides methods to load the images and labels, apply transformations, and return them in a format suitable for training deep learning models.
    """

    def __init__(
        self, cityscapes_path: str, train_val: str, transforms: Optional[Compose] = None
    ):
        """
        Initializes the CityScapes dataset class.

        This constructor sets up the dataset for use, either for training or validation, based on the provided path and subset selection. It optionally applies a transformation to the data.

        Args:
            cityscapes_path (str): The root directory path where the CityScapes dataset is stored.
            train_val (str): A string that specifies whether to load the 'train' or 'val' subset of the dataset.
            transforms (Optional[Compose], optional): A function/transform that takes in an image and label and returns a transformed version. Defaults to None.

        Raises:
            ValueError: If train_val is not 'train' or 'val'.
        """

        if train_val not in ["train", "val"]:
            raise ValueError("train_val must be 'train' or 'val'")

        self.cityscapes_path = cityscapes_path
        self.transforms = transforms
        self.image_label_pairs = self._get_image_label_pairs(train_val)

    def _get_image_label_pairs(self, train_val: str) -> list[Tuple[str, str]]:
        image_label_pairs = []
        path = os.path.join(self.cityscapes_path, "Cityscapes", "gtFine", train_val)

        # Iterate through the directory and get all the image paths
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".png"):
                    if "_gtFine_labelTrainIds" not in file:
                        continue
                    # Get the full path to the image
                    label_path = os.path.join(root, file)
                    # Get the corresponding label path
                    img_path = label_path.replace(
                        "_gtFine_labelTrainIds", "_leftImg8bit"
                    ).replace("gtFine", "images")
                    # Append the paths to the image_label_pairs list
                    image_label_pairs.append((img_path, label_path))
                else:
                    print(f"File {file} is not a .png file. Skipping.")
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
            img, label = (
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

            if not any(isinstance(t, ToTensorV2) for t in self.transforms.transforms):
                # If ToTensorV2() is not in the transform, we need to convert the image and label to tensors manually.
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
                label = torch.from_numpy(label).long()
        else:
            # If no transform is provided, we need to convert the image and label to tensors manually.

            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
            # torch.from_numpy(img): Converts the NumPy array (shape (H, W, C), dtype uint8, values 0–255) into a PyTorch tensor (still (H, W, C), still uint8).
            # permute(2, 0, 1): Reorders the dimensions from (H, W, C) → (C, H, W). This is the format expected by PyTorch models (channel-first).
            # float(): Converts from uint8 (0–255) to float32.
            # / 255: Normalizes pixel values from [0, 255] to [0.0, 1.0].

            label = torch.from_numpy(label).long()
            # torch.from_numpy(label): Converts the label mask from a NumPy array (shape (H, W), dtype usually uint8 or int32) to a PyTorch tensor.

            # .long(): Converts the tensor to int64 (PyTorch's default long integer type).
            # This is important because: PyTorch loss functions like nn.CrossEntropyLoss require target tensors to be of type torch.LongTensor.

        return img, label

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

if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt

    dataset = CityScapes("./data", train_val="train")

    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]

    # Plot the image and label
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image.permute(1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="gray", vmin=0, vmax=20)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    plt.show()
    plt.show()
