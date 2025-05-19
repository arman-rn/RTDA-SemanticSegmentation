import random

import matplotlib.pyplot as plt
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

from datasets import CityScapes


def get_transforms():
    return Compose(
        [
            Resize(512, 1024),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def main():
    dataset = CityScapes(
        cityscapes_path="data/Cityscapes", split="train", transforms=get_transforms()
    )
    idx = random.randrange(len(dataset))
    img, label = dataset[idx]

    # img is a FloatTensor C×H×W, label is LongTensor H×W
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mask_np = label.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np)  # Convert from [C, H, W] to [H, W, C]
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=19)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    plt.show()


if __name__ == "__main__":
    main()
