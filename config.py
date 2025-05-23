# This file is the central hub for all settings, hyperparameters, and paths used throughout this project.

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

# --- W&B Project Details ---
WANDB_PROJECT_NAME = "RTDA-SemSeg"
WANDB_ENTITY = "RTDA-SemSeg"  # Your W&B username or team name, if None, it uses default

# --- Project Paths ---
ROOT_DIR = "."
# Local
DATASET_PATH = f"{ROOT_DIR}/data/Cityscapes"
# Example for Colab after gdown to /content/datasets/: '/content/datasets/cityscapes'

# Local
PRETRAINED_MODEL_PATH = f"{ROOT_DIR}/models/deeplabv2/DeepLab_resnet_pretrained_imagenet.pth"  # Path to the ResNet-101 weights pretrained on ImageNet, used to initialize the backbone of your DeepLabV2 model. The project specifies using a backbone pre-trained on ImageNet.
# Example for Colab if model is downloaded by gdown to project: f'{ROOT_DIR}/models/deeplabv2/DeepLab_resnet_pretrained_imagenet.pth'

# --- Checkpoint Settings ---
# Directory to save all checkpoints (latest, best, periodic)
CHECKPOINT_DIR = f"{ROOT_DIR}/checkpoints"
LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.pth"
BEST_CHECKPOINT_FILENAME = (
    "best_miou_checkpoint.pth"  # For the model with the best mIoU
)
# Path to a specific checkpoint to resume from. Set via CLI --resume_checkpoint argument,
# or manually edit thiing to a valid path if you always want to try resuming from it.
# Example: None, or '/content/RTDA_SemanticSegmentation_Project/checkpoints/latest_checkpoint.pth'
RESUME_CHECKPOINT_PATH = None
# How often to save a periodic checkpoint (e.g., every N epochs). Set to 0 or less to disable.
SAVE_CHECKPOINT_FREQ_EPOCH = 1  # Saves a checkpoint like 'checkpoint_epoch_X.pth'

# --- Model & Dataset Parameters ---
NUM_CLASSES = 19  # The number of semantic classes the model needs to predict. Cityscapes has 19 evaluation classes.
IMG_HEIGHT = 512  # The target resolution for training and testing images (1024x512 for Cityscapes as per Step 2a)
IMG_WIDTH = 1024  # The target resolution for training and testing images (1024x512 for Cityscapes as per Step 2a)
IGNORE_INDEX = 255  # A special label value (often 255 for Cityscapes) that the loss function should ignore. This is typically used for "void" or "unlabeled" regions in the ground truth masks.

# --- DataLoader Settings ---
# Number of worker processes for data loading.
# For Colab T4, 2 is suggested. For A100, can try 2 or 4.
DATALOADER_NUM_WORKERS = 2  # Default value, can be adjusted based on environment

# --- Training Hyperparameters ---
TRAIN_EPOCHS = 50  # The total number of times the training loop will iterate over the entire training dataset (50 epochs for Step 2a).
# Try 8 if using A100
BATCH_SIZE = 4  # The number of images processed in one forward/backward pass during training. Adjust based on GPU memory.
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

# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Normalization Parameters (ImageNet) ---
# Standard mean and standard deviation values for datasets pretrained on ImageNet. Used to normalize input images.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# --- Augmentations ---
# For Step 2a, only resizing and normalization are strictly needed.
# Defines image preprocessing and augmentation pipelines using the albumentations library.
#   A.Compose([...]): Chains multiple transformations.
#   A.Resize: Resizes images to the specified IMG_HEIGHT and IMG_WIDTH.
#   A.Normalize: Normalizes pixel values using NORM_MEAN and NORM_STD.
#   ToTensorV2(): Converts the image (and mask) from a NumPy array to a PyTorch tensor and permutes image dimensions from (H, W, C) to (C, H, W).

TRAIN_TRANSFORMS = A.Compose(
    [
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ]
)

VAL_TRANSFORMS = A.Compose(
    [
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ]
)

# --- Logging & Saving ---
PRINT_FREQ_BATCH = 100  # Print training status every N batches
VALIDATE_FREQ_EPOCH = 1  # Validate every N epochs (set to 1 for validation each epoch)
WANDB_LOG_IMAGES_FREQ_EPOCH = 5  # Log sample images to W&B every N epochs

# --- Metrics Calculation (for final summary) ---
# Parameters for calculating latency and FPS, as suggested in the project description's pseudo-code.
LATENCY_ITERATIONS = 100
WARMUP_ITERATIONS = 10
