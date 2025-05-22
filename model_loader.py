# This small utility file is responsible for instantiating the DeepLabV2 model
# using the factory function get_deeplab_v2 from models.deeplabv2.deeplabv2 module and moving it to the correct device.

# Assuming models.deeplabv2.deeplabv2 will be in python path
# or that the script is run from a directory where 'models' is a subdirectory.
from models.deeplabv2.deeplabv2 import get_deeplab_v2


def get_model(num_classes, pretrained_model_path, device):
    """
    Loads the DeepLabV2 model.
    Args:
        num_classes (int): Number of output classes.
        pretrained_model_path (str): Path to the ImageNet pretrained weights.
        device (torch.device): Device to move the model to.
    Returns:
        torch.nn.Module: The loaded model.
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
    return model
