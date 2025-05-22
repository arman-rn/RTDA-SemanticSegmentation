"""
Handles the instantiation and loading of the DeepLabV2 model.

This utility file is responsible for creating an instance of the DeepLabV2
model using the factory function `get_deeplab_v2` from the
`models.deeplabv2.deeplabv2` module, loading pretrained weights if specified,
and moving the model to the appropriate computing device.
"""

import torch
from torch import nn

from models.deeplabv2.deeplabv2 import get_deeplab_v2


def get_model(
    num_classes: int, pretrained_model_path: str, device: torch.device
) -> nn.Module:
    """
    Loads and initializes the DeepLabV2 model with a ResNet-101 backbone.

    This function instantiates the DeepLabV2 model, attempts to load
    ImageNet-pretrained weights for its backbone, and moves the model to the
    specified PyTorch device.

    Args:
        num_classes: The number of semantic classes for the output layer.
        pretrained_model_path: The file path to the pretrained weights
            (e.g., ResNet-101 weights trained on ImageNet) for the backbone.
        device: The PyTorch device (e.g., torch.device('cuda') or
            torch.device('cpu')) to which the model should be moved.

    Returns:
        An initialized DeepLabV2 model (subclass of torch.nn.Module)
        ready for training or inference on the specified device.
    """

    print(f"Loading DeepLabV2 model with {num_classes} classes.")
    print(f"Using pretrained weights from: {pretrained_model_path}")

    # Creates the DeepLabV2 model instance (which is ResNetMulti)
    # and attempts to load the pretrained backbone weights.
    model = get_deeplab_v2(
        num_classes=num_classes,
        pretrain=True,  # Instructs get_deeplab_v2 to load ImageNet weights
        pretrain_model_path=pretrained_model_path,
    )

    model.to(device)
    print(f"Model moved to device: {device}")
    return model
