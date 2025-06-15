"""
Handles the instantiation and loading of segmentation models (DeepLabV2, BiSeNet).
"""

from typing import Any, Optional, Tuple

from torch import nn

from models.bisenet.build_bisenet import BiSeNet
from models.bisenet.build_bisenet_multilevel import BiSeNetMultiLevel
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.discriminator.discriminator import FCDiscriminator, FCDiscriminator_aux

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
    model_name = config_obj.MODEL_NAME
    is_multi_level = config_obj.ADVERSARIAL_MULTI_LEVEL

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
        if is_multi_level:
            print(
                "Loading BiSeNetMultiLevel model for multi-level adversarial training."
            )
            model = BiSeNetMultiLevel(
                num_classes=num_classes,
                context_path=config_obj.BISENET_CONTEXT_PATH,
            )
        else:
            print(f"Loading standard BiSeNet model with {num_classes} classes.")
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


def get_discriminators(
    config_obj: ConfigModule,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """
    Loads and initializes the discriminator(s).

    - For single-level, returns (D_main, None).
    - For multi-level, returns (D_main, D_aux).
    """
    num_classes = config_obj.NUM_CLASSES
    device = config_obj.DEVICE
    is_multi_level = getattr(config_obj, "ADVERSARIAL_MULTI_LEVEL", False)

    print("Initializing Main Discriminator (FCDiscriminator)...")
    d_main = FCDiscriminator(num_classes=num_classes).to(device)
    print(f"Main Discriminator moved to device: {device}")

    d_aux = None
    if is_multi_level:
        print("Initializing Auxiliary Discriminator (FCDiscriminator_aux)...")
        d_aux = FCDiscriminator_aux(num_classes=num_classes).to(device)
        print(f"Auxiliary Discriminator moved to device: {device}")

    return d_main, d_aux
