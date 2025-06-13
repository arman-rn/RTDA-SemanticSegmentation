# This file is the central hub for all settings, hyperparameters, and paths used throughout this project.

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

SEED_VALUE = 42

# --- W&B Project Details ---
WANDB_PROJECT_NAME = "RTDA-SemSeg"
WANDB_ENTITY = "RTDA-SemSeg"  # Your W&B username or team name, if None, it uses default

# --- Project Paths ---
ROOT_DIR = "."

# Local
CITYSCAPES_DATASET_PATH = f"{ROOT_DIR}/data/Cityscapes"
GTA5_DATASET_PATH = f"{ROOT_DIR}/data/GTA5"

# Example for Colab after gdown to /content/datasets/:
# CITYSCAPES_DATASET_PATH = '/content/datasets/cityscapes'
# GTA5_DATASET_PATH = '/content/datasets/gta5'

# --- GTA5 Label Configuration ---
GTA5_CONVERT_LABELS_ON_THE_FLY = False  # Set to False to use pre-converted labels
GTA5_ORIGINAL_LABELS_SUBDIR = "labels"  # Subdir for original RGB GTA5 labels
GTA5_PRECONVERTED_LABELS_SUBDIR = (
    "labels_trainids"  # Subdir for pre-converted ID labels
)

# --- Model Selection ---
MODEL_NAME = "bisenet"  # Options: "deeplabv2", "bisenet"

# --- Dataset Selection ---
TRAIN_DATASET = "gta5"  # Options: "gta5", "cityscapes"
# Validation dataset is always Cityscapes for this project.
VAL_DATASET = "cityscapes"  # Options: "gta5", "cityscapes"

# --- DeepLabV2 Specific ---
# Path to the ResNet-101 weights pretrained on ImageNet, used to initialize the backbone of DeepLabV2.
DEEPLABV2_PRETRAINED_BACKBONE_PATH = f"{ROOT_DIR}/models/deeplabv2/DeepLab_resnet_pretrained_imagenet.pth"  # Path to the ResNet-101 weights pretrained on ImageNet, used to initialize the backbone of your DeepLabV2 model. The project specifies using a backbone pre-trained on ImageNet.
# Example for Colab if model is downloaded by gdown to project: f'{ROOT_DIR}/models/deeplabv2/DeepLab_resnet_pretrained_imagenet.pth'

# --- Checkpoint Settings ---
# Directory to save all checkpoints (latest, best, periodic)
CHECKPOINT_DIR = f"{ROOT_DIR}/checkpoints/{MODEL_NAME}"
# File for the model with the best mIoU found so far (continuously updated)
BEST_CHECKPOINT_FILENAME = "best_miou_checkpoint.pth"
# File for the periodic checkpoint, this single file will be OVERWRITTEN periodically
CHECKPOINT_FILENAME = "checkpoint.pth"

# Path to a specific checkpoint to resume from. Set via CLI --resume_checkpoint argument,
# or manually editing to a valid path if you always want to try resuming from it.
# Example: None, or '/content/RTDA_SemanticSegmentation_Project/checkpoints/best_miou_checkpoint.pth'
RESUME_CHECKPOINT_PATH = None
# How often to save a periodic checkpoint (e.g., every N epochs). Set to 0 or less to disable.
SAVE_CHECKPOINT_FREQ_EPOCH = 5  # Saves/overwrites checkpoint.pth every epoch

# --- Model & Dataset Parameters ---
NUM_CLASSES = 19  # The number of semantic classes the model needs to predict. Cityscapes has 19 evaluation classes.
IGNORE_INDEX = 255  # A special label value that the loss function should ignore. This is typically used for "void" or "unlabeled" regions in the ground truth masks.

# Cityscapes Dimensions
CITYSCAPES_IMG_HEIGHT = 512
CITYSCAPES_IMG_WIDTH = 1024

# GTA5 Dimensions
GTA5_IMG_HEIGHT = 720
GTA5_IMG_WIDTH = 1280

# --- DataLoader Settings ---
# Number of worker processes for data loading.
# For Colab T4, 2 is suggested. For A100, can try 2 or 4.
DATALOADER_NUM_WORKERS = 16  # Default value, can be adjusted based on environment

# --- Training Hyperparameters ---
TRAIN_EPOCHS = 50  # The total number of times the training loop will iterate over the entire training dataset (50 epochs for Step 2a).
# Try 8 if using A100
BATCH_SIZE = 8  # The number of images processed in one forward/backward pass during training. Adjust based on GPU memory.
LR_SCHEDULER_POWER = 0.9  # Parameter for the polynomial learning rate decay scheduler.

# --- Optimizer Settings ---
# Default optimizer type
OPTIMIZER_TYPE = "adam"  # Options: 'sgd', 'adam'

# Common settings
WEIGHT_DECAY = 1e-4  # A general weight decay, can be overridden per optimizer

# SGD specific parameters
SGD_LEARNING_RATE = 2.5e-4  # As per previous setup for SGD
SGD_MOMENTUM = 0.9

# Adam specific parameters
ADAM_LEARNING_RATE = 1e-4  # Typical starting LR for Adam
# ADAM_BETA1 = 0.9  # Beta1 parameter for Adam optimizer
# ADAM_BETA2 = 0.999  # Beta2 parameter for Adam optimizer

# --- BiSeNet Specific Settings ---
BISENET_CONTEXT_PATH = "resnet18"
# Note: Pretrained weights for BiSeNet's ResNet18 backbone will be loaded from torchvision by default
# by the build_contextpath.py script (which should be in models/bisenet/).

# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging & Saving ---
PRINT_FREQ_BATCH = 100  # Print training status every N batches
VALIDATE_FREQ_EPOCH = 1  # Validate every N epochs (set to 1 for validation each epoch)
WANDB_LOG_IMAGES_FREQ_EPOCH = 10  # Log sample images to W&B every N epochs

# --- Metrics Calculation (for final summary) ---
# Parameters for calculating latency and FPS, as suggested in the project description's pseudo-code.
LATENCY_ITERATIONS = 100
WARMUP_ITERATIONS = 10

# --- Augmentations ---
# Defines image preprocessing and augmentation pipelines using the albumentations library.
#   A.Compose([...]): Chains multiple transformations.
#   A.Resize: Resizes images to the specified IMG_HEIGHT and IMG_WIDTH.
#   A.Normalize: Normalizes pixel values using NORM_MEAN and NORM_STD.
#   ToTensorV2(): Converts the image (and mask) from a NumPy array to a PyTorch tensor and permutes image dimensions from (H, W, C) to (C, H, W).

# --- Normalization Parameters (ImageNet) ---
# Standard mean and standard deviation values for datasets pretrained on ImageNet. Used to normalize input images.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# --- Base Transform Components (Applied in all GTA5 training pipelines) ---
gta5_base_resize = A.Resize(height=GTA5_IMG_HEIGHT, width=GTA5_IMG_WIDTH)
common_normalize = A.Normalize(mean=NORM_MEAN, std=NORM_STD)
common_to_tensor = ToTensorV2()

# --- Define Four Individual Augmentation Transforms ---
# All are applied with p=0.5 as per project requirements.

# 1. Horizontal Flip
aug_transform_hflip = A.HorizontalFlip(p=0.5)

# 2. Color Jitter
aug_transform_colorjitter = A.ColorJitter(
    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5
)

# 3. ISONoise
aug_transform_isonoise = A.ISONoise(
    intensity=(0.1, 0.3), color_shift=(0.01, 0.05), p=0.5
)

# 4. Coarse Dropout
aug_transform_coarsedropout = A.CoarseDropout(
    num_holes_range=(1, 8),  # Number of holes will be randomly chosen between 1 and 8.
    hole_height_range=(
        20,
        60,
    ),  # Height of holes (in pixels) will be between 20 and 60.
    hole_width_range=(20, 60),  # Width of holes (in pixels) will be between 20 and 60.
    fill=0,  # Value to fill the dropped regions.
    p=0.5,  # Probability of applying the transform.
)

# --- Define Pipelines for Individual Augmentations ---

# Pipeline for HorizontalFlip only
GTA5_TRAIN_TRANSFORMS_HFLIP_ONLY = A.Compose(
    [
        gta5_base_resize,
        aug_transform_hflip,
        common_normalize,
        common_to_tensor,
    ]
)

# Pipeline for ColorJitter only
GTA5_TRAIN_TRANSFORMS_COLORJITTER_ONLY = A.Compose(
    [
        gta5_base_resize,
        aug_transform_colorjitter,
        common_normalize,
        common_to_tensor,
    ]
)

# Pipeline for ISONoise only
GTA5_TRAIN_TRANSFORMS_ISONOISE_ONLY = A.Compose(
    [
        gta5_base_resize,
        aug_transform_isonoise,
        common_normalize,
        common_to_tensor,
    ]
)

# Pipeline for CoarseDropout only
GTA5_TRAIN_TRANSFORMS_COARSEDROPOUT_ONLY = A.Compose(
    [
        gta5_base_resize,
        aug_transform_coarsedropout,
        common_normalize,
        common_to_tensor,
    ]
)

# --- Pipeline: Combining ALL FOUR suggested augmentations ---
# The order of augmentations can matter. A common sequence is geometric -> color/intensity -> noise -> structural.
GTA5_TRAIN_TRANSFORMS_ALL_FOUR_COMBINED = A.Compose(
    [
        gta5_base_resize,
        # aug_transform_hflip,  # Geometric
        aug_transform_colorjitter,  # Color/Intensity
        aug_transform_isonoise,  # Noise
        aug_transform_coarsedropout,  # Structural
        common_normalize,
        common_to_tensor,
    ]
)

# --- Original GTA5 Training Transforms (Step 3a - No NEW Augmentations) ---
GTA5_TRAIN_TRANSFORMS_NO_NEW_AUG = A.Compose(
    [
        gta5_base_resize,
        common_normalize,
        common_to_tensor,
    ]
)


# --- Cityscapes Transforms ---
CITYSCAPES_TRAIN_TRANSFORMS = A.Compose(
    [
        A.Resize(height=CITYSCAPES_IMG_HEIGHT, width=CITYSCAPES_IMG_WIDTH),
        common_normalize,
        common_to_tensor,
    ]
)

CITYSCAPES_VAL_TRANSFORMS = A.Compose(
    [
        A.Resize(height=CITYSCAPES_IMG_HEIGHT, width=CITYSCAPES_IMG_WIDTH),
        common_normalize,
        common_to_tensor,
    ]
)


# --- ACTIVATE THE DESIRED GTA5 TRAINING PIPELINE FOR THE CURRENT EXPERIMENT ---
# Uncomment ONE of the following lines to select the transforms for the current run.
# This `GTA5_TRAIN_TRANSFORMS` variable is what `data_loader.py` will use.

# For Step 3a (baseline):
# GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_NO_NEW_AUG

# --- For Step 3b ---
# 1. Experiment with HorizontalFlip only:
# GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_HFLIP_ONLY

# 2. Experiment with ColorJitter only:
# GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_COLORJITTER_ONLY

# 3. Experiment with ISONoise only:
# GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_ISONOISE_ONLY

# 4. Experiment with CoarseDropout only:
# GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_COARSEDROPOUT_ONLY

# 5. Experiment with all four combined:
GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_ALL_FOUR_COMBINED

# --- Adversarial Domain Adaptation Settings ---

# Source dataset for adversarial training (labeled)
ADVERSARIAL_SOURCE_DATASET_NAME = "gta5"
# Target dataset for adversarial training (unlabeled)
ADVERSARIAL_TARGET_DATASET_NAME = "cityscapes"
# Split of the target dataset to use (typically 'train' for unlabeled images)
ADVERSARIAL_TARGET_DATASET_SPLIT = "train"

# Weight for the generator's adversarial loss component.
# Paper [7] (Tsai et al. "Learning to Adapt Structured Output Space...") suggests lambda_adv = 0.001
# for their single-level output space adaptation (see Table 3 and Section 6.1 Parameter Analysis).
ADVERSARIAL_LAMBDA_ADV_GENERATOR = 0.002
# ADVERSARIAL_LAMBDA_ADV_GENERATOR = 0.0002 # More Cautious Adaptation, This is useful if the discriminator's feedback is noisy or hurting the segmentation performance.

# --- Discriminator Optimizer Settings ---
# Paper [7] uses Adam for the discriminator.
ADVERSARIAL_DISCRIMINATOR_OPTIMIZER_TYPE = "adam"
# ADVERSARIAL_DISCRIMINATOR_LEARNING_RATE = 1e-4  # As per Paper [7] for discriminator.
ADVERSARIAL_DISCRIMINATOR_LEARNING_RATE = 2.5e-5

# Adam specific parameters for Discriminator Optimizer
# Paper [7] sets momentum for Adam as 0.9 and 0.99.
# These likely correspond to beta1 and beta2.
ADVERSARIAL_DISCRIMINATOR_ADAM_BETA1 = 0.9
ADVERSARIAL_DISCRIMINATOR_ADAM_BETA2 = 0.99
ADVERSARIAL_DISCRIMINATOR_WEIGHT_DECAY = (
    0  # Common for GAN discriminators, not specified for D in Paper [7].
)
