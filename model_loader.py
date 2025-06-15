"""
Handles the instantiation and loading of segmentation models (DeepLabV2, BiSeNet)
and discriminator models for adversarial training.
This version can select between standard BiSeNet and multi-level BiSeNet.
"""

from typing import Any, Optional

from torch import nn

# Import segmentation models (Generator)
from models.bisenet.build_bisenet import BiSeNet
from models.bisenet.build_bisenet_multilevel import BiSeNetMultiLevel
from models.deeplabv2.deeplabv2 import get_deeplab_v2

# Import Discriminator model
from models.discriminator.discriminator import FCDiscriminator

ConfigModule = Any


def get_model(config_obj: ConfigModule) -> nn.Module:
    """
    Loads and initializes the specified segmentation model.

    This function instantiates the model chosen in `config.MODEL_NAME`.
    - For DeepLabV2, it uses `config.DEEPLABV2_PRETRAINED_BACKBONE_PATH`.
    - For BiSeNet, it uses `config.BISENET_CONTEXT_PATH`. The BiSeNet's
      `build_contextpath.py` (expected in `models/bisenet/`) handles
      loading pretrained weights for its backbone (e.g., ResNet18 from torchvision).
      If config_obj.ADVERSARIAL_MULTI_LEVEL is True and the model is bisenet,
      it loads the BiSeNetMultiLevel model. Otherwise, it loads the standard model.

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
    model_name = config_obj.MODEL_NAME

    if model_name == "deeplabv2":
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
    elif model_name == "bisenet":
        is_multilevel = getattr(config_obj, "ADVERSARIAL_MULTI_LEVEL", False)

        if is_multilevel:
            print(
                f"Loading BiSeNetMultiLevel (Generator) model with {num_classes} classes."
            )
            model = BiSeNetMultiLevel(
                num_classes=num_classes,
                context_path=config_obj.BISENET_CONTEXT_PATH,
            )
        else:
            print(
                f"Loading standard BiSeNet (Generator) model with {num_classes} classes."
            )
            model = BiSeNet(
                num_classes=num_classes,
                context_path=config_obj.BISENET_CONTEXT_PATH,
            )
    else:
        raise ValueError(
            f"Unsupported MODEL_NAME: '{model_name}'. Choose 'deeplabv2' or 'bisenet'."
        )

    model.to(device)
    print(f"Model '{model_name}' moved to device: {device}")
    return model


def get_discriminator(config_obj: ConfigModule) -> Optional[nn.Module]:
    """
    Loads and initializes the discriminator model for adversarial training.

    Args:
        config_obj: The configuration object (e.g., the imported cfg module)
                    containing settings like NUM_CLASSES and DEVICE.

    Returns:
        An initialized FCDiscriminator model (subclass of torch.nn.Module)
        on the specified device if FCDiscriminator is available, otherwise None.
    """
    print("Initializing Discriminator (FCDiscriminator)...")
    num_classes = (
        config_obj.NUM_CLASSES
    )  # Discriminator input channels depend on generator's output classes
    device = config_obj.DEVICE

    # Instantiate the FCDiscriminator
    discriminator = FCDiscriminator(num_classes=num_classes)
    discriminator.to(device)

    print(
        f"Discriminator (FCDiscriminator) model initialized with {num_classes} input channels."
    )
    print(f"Discriminator moved to device: {device}")

    return discriminator
