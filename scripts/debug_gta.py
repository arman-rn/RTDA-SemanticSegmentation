# import random

import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

from datasets import GTA5


def get_transforms():
    return A.Compose(
        [
            A.Resize(height=720, width=1280),
            # A.ISONoise(intensity=(0.1, 0.3), color_shift=(0.01, 0.05), p=1),
            ToTensorV2(),
        ]
    )


def main():
    dataset = GTA5(
        gta5_path="data/GTA5",
        labels_subdir="labels_trainids",
        convert_on_the_fly=False,
        transforms=get_transforms(),
    )
    # idx = random.randrange(len(dataset))
    idx = 0
    img, label = dataset[idx]

    # img is a FloatTensor C×H×W, label is LongTensor H×W
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mask_np = label.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    axes[0].imshow(img_np)  # Convert from [C, H, W] to [H, W, C]
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=19)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    plt.show()


if __name__ == "__main__":
    main()
