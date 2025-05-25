"""
Handles the instantiation and loading of segmentation models (DeepLabV2, BiSeNet).
"""

import torch
from torch import nn

import config as cfg
from models.bisenet.build_bisenet import BiSeNet
from models.deeplabv2.deeplabv2 import get_deeplab_v2


def get_model(
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Loads and initializes the specified segmentation model.

    This function instantiates the model chosen in `config.MODEL_NAME`.
    - For DeepLabV2, it uses `config.DEEPLABV2_PRETRAINED_BACKBONE_PATH`.
    - For BiSeNet, it uses `config.BISENET_CONTEXT_PATH`. The BiSeNet's
      `build_contextpath.py` (expected in `models/bisenet/`) handles
      loading pretrained weights for its backbone (e.g., ResNet18 from torchvision).

    Args:
        num_classes: The number of semantic classes for the output layer.
        device: The PyTorch device (e.g., torch.device('cuda') or
            torch.device('cpu')) to which the model should be moved.

    Returns:
        An initialized segmentation model (subclass of torch.nn.Module)
        ready for training or inference on the specified device.

    Raises:
        ValueError: If an unsupported `MODEL_NAME` is specified in the config.
    """

    if cfg.MODEL_NAME == "deeplabv2":
        print(f"Loading DeepLabV2 model with {num_classes} classes.")
        print(
            f"Using DeepLabV2 pretrained backbone from: {cfg.DEEPLABV2_PRETRAINED_BACKBONE_PATH}"
        )

        # Creates the DeepLabV2 model instance (which is ResNetMulti)
        # and attempts to load the pretrained backbone weights.
        model = get_deeplab_v2(
            num_classes=num_classes,
            pretrain=True,  # Instructs get_deeplab_v2 to load ImageNet weights
            pretrain_model_path=cfg.DEEPLABV2_PRETRAINED_BACKBONE_PATH,
        )
    elif cfg.MODEL_NAME == "bisenet":
        print(f"Loading BiSeNet model with {num_classes} classes.")
        print(f"Using BiSeNet context path: {cfg.BISENET_CONTEXT_PATH}")
        # BiSeNet's `build_contextpath` handles loading pretrained ResNet18/101 from torchvision
        model = BiSeNet(
            num_classes=num_classes,
            context_path=cfg.BISENET_CONTEXT_PATH,  # e.g., 'resnet18'
        )
        # BiSeNet's own init_weight() is called within its constructor for non-backbone parts.
    else:
        raise ValueError(
            f"Unsupported model_type: '{cfg.MODEL_NAME}'. Choose 'deeplabv2' or 'bisenet'."
        )

    model.to(device)
    print(f"Model '{cfg.MODEL_NAME}' moved to device: {device}")
    return model


if __name__ == "__main__":
    # This block is for quick testing of the model loader.
    print(f"Current MODEL_NAME in config: {cfg.MODEL_NAME}")
    try:
        loaded_model = get_model(num_classes=cfg.NUM_CLASSES, device=cfg.DEVICE)
        print(f"Successfully loaded model: {cfg.MODEL_NAME} using get_model.")
    except Exception as e:
        print(f"Error during model_loader.py __main__ test: {e}")
