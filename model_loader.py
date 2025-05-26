"""
Handles the instantiation and loading of segmentation models (DeepLabV2, BiSeNet).
"""

from typing import Any

from torch import nn

from models.bisenet.build_bisenet import BiSeNet
from models.deeplabv2.deeplabv2 import get_deeplab_v2

ConfigModule = Any


def get_model(config_obj: ConfigModule) -> nn.Module:
    """
    Loads and initializes the specified segmentation model.

    This function instantiates the model chosen in `config.MODEL_NAME`.
    - For DeepLabV2, it uses `config.DEEPLABV2_PRETRAINED_BACKBONE_PATH`.
    - For BiSeNet, it uses `config.BISENET_CONTEXT_PATH`. The BiSeNet's
      `build_contextpath.py` (expected in `models/bisenet/`) handles
      loading pretrained weights for its backbone (e.g., ResNet18 from torchvision).

    Args:
        config_obj: The configuration object (e.g., the imported cfg module)
                    containing all settings like MODEL_NAME, NUM_CLASSES, DEVICE, paths, etc.

    Returns:
        An initialized segmentation model (subclass of torch.nn.Module)
        ready for training or inference on the specified device.

    Raises:
        ValueError: If an unsupported `MODEL_NAME` is specified in the config.
    """

    num_classes = config_obj.NUM_CLASSES
    device = config_obj.DEVICE

    if config_obj.MODEL_NAME == "deeplabv2":
        print(f"Loading DeepLabV2 model with {num_classes} classes.")
        print(
            f"Using DeepLabV2 pretrained backbone from: {config_obj.DEEPLABV2_PRETRAINED_BACKBONE_PATH}"
        )

        # Creates the DeepLabV2 model instance (which is ResNetMulti)
        # and attempts to load the pretrained backbone weights.
        model = get_deeplab_v2(
            num_classes=num_classes,
            pretrain=True,  # Instructs get_deeplab_v2 to load ImageNet weights
            pretrain_model_path=config_obj.DEEPLABV2_PRETRAINED_BACKBONE_PATH,
        )
    elif config_obj.MODEL_NAME == "bisenet":
        print(f"Loading BiSeNet model with {num_classes} classes.")
        print(f"Using BiSeNet context path: {config_obj.BISENET_CONTEXT_PATH}")
        # BiSeNet's `build_contextpath` handles loading pretrained ResNet18/101 from torchvision
        model = BiSeNet(
            num_classes=num_classes,
            context_path=config_obj.BISENET_CONTEXT_PATH,  # e.g., 'resnet18'
        )
        # BiSeNet's own init_weight() is called within its constructor for non-backbone parts.
    else:
        raise ValueError(
            f"Unsupported MODEL_NAME: '{config_obj.MODEL_NAME}'. Choose 'deeplabv2' or 'bisenet'."
        )

    model.to(device)
    print(f"Model '{config_obj.MODEL_NAME}' moved to device: {device}")
    return model
