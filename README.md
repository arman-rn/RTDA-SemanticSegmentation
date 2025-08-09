# Real-time Domain Adaptation for Semantic Segmentation

This repository contains the PyTorch implementation for the project "Real-time Domain Adaptation for Semantic Segmentation using Adversarial Training and Lovász-Softmax Loss". The primary goal is to adapt a real-time semantic segmentation model (BiSeNet) from a labeled synthetic source domain (GTA5) to an unlabeled real-world target domain (Cityscapes) using Unsupervised Domain Adaptation (UDA).

## Key Features

- **Models Implemented**:
  - **BiSeNet**: A real-time, two-path segmentation network used as the main generator model.
  - **DeepLabV2**: A classic high-performance network used for establishing an upper-bound performance benchmark.
  - **FC-Discriminator**: A fully convolutional discriminator used in the adversarial training framework to distinguish between source and target domain feature maps.
- **Domain Adaptation Framework**:
  - An adversarial training pipeline inspired by Tsai et al., where the generator (BiSeNet) is trained to produce segmentations that can fool the discriminator.
- **Extension with Lovász-Softmax Loss**:
  - The project includes an extension that incorporates the Lovász-Softmax loss into the segmentation objective. This hybrid loss function, combining Cross-Entropy and Lovász-Softmax, directly optimizes the Intersection-over-Union (IoU) metric, improving performance on underrepresented classes.
- **Systematic Evaluation**:
  - Analysis of the domain shift between synthetic and real data.
  - A comprehensive study on data augmentations ('ColorJitter', 'ISONoise', 'Coarse Dropout') to mitigate the domain gap.
  - Quantitative and qualitative results, including mIoU, per-class IoU, FLOPs, and latency metrics.
- **Experiment Tracking**:
  - Full integration with Weights & Biases (W&B) for logging metrics, configurations, and visualizing segmentation results.

## Setup and Installation

**1. Clone the Repository**

```bash
git clone [https://github.com/arman-rn/MLDL-SemSeg.git](https://github.com/arman-rn/MLDL-SemSeg.git)
cd MLDL-SemSeg
```

**2. Install Dependencies**

This project requires PyTorch and other common libraries.

```bash
pip install torch torchvision numpy pillow tqdm albumentations wandb fvcore
```

## Dataset Preparation

**1. Download Datasets**

You need to download the GTA5 and Cityscapes datasets manually.

Cityscapes: `https://www.cityscapes-dataset.com/`
GTA5: `https://download.visinf.tu-darmstadt.de/data/from_games/`

**2. Organize Directories**
The code expects the datasets to be organized in a `/data` directory at the root of the project. The paths are configured in `config.py`.

```text
MLDL-SemSeg/
├── data/
│   ├── Cityscapes/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── gtFine/
│   │       ├── train/
│   │       └── val/
│   └── GTA5/
│       ├── images/
│       └── labels/  <-- Original RGB labels
└── ... (project files)
```

**3. Pre-process GTA5 Labels (Important!)**
The GTA5 dataset provides RGB color-coded labels. To accelerate training, a script is provided to convert these into single-channel class ID maps that are compatible with the Cityscapes `trainId` format.

Run the following command from the root project directory:

```bash
python preprocess_gta5_labels.py --gta5_root ./data/GTA5
```

This will create a new directory: `data/GTA5/labels_trainids`. The data loader will use these pre-converted labels by default, as configured in config.py (`GTA5_CONVERT_LABELS_ON_THE_FLY = False`).

## How to Run Experiments

The project contains two main scripts: `main.py` for standard (source-only) training and `main_adversarial.py` for domain adaptation.

### Central Configuration

All experiment settings are centralized in `config.py`. Before running an experiment, you can modify this file to:

- Select the data augmentation pipeline (`GTA5_TRAIN_TRANSFORMS`).
- Toggle the Lovász-Softmax loss extension (`USE_LOVASZ_LOSS`).
- Adjust hyperparameters like learning rates, batch size, etc.

### Experiment 1: Source-Only Baseline (Quantifying Domain Shift)

This trains the BiSeNet model on the GTA5 dataset and evaluates it on Cityscapes to measure the performance drop from the domain shift. This stage is run using `main.py`.

**1. Modify `config.py`:**

To run the baseline without new augmentations, set `GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_NO_NEW_AUG`.
To run the baseline with your best combination of augmentations, set `GTA5_TRAIN_TRANSFORMS = GTA5_TRAIN_TRANSFORMS_ALL_FOUR_COMBINED`.

**2. Run Training:**

```bash
python main.py --model_name bisenet --optimizer adam --epochs 50
```

### Experiment 2: Adversarial Domain Adaptation

This trains the full UDA framework. All adversarial experiments are run using main_adversarial.py.

**1. Standard Adversarial Training (CE Loss):**

- In `config.py`, ensure `USE_LOVASZ_LOSS = False`.
- Ensure the best data augmentation pipeline (`GTA5_TRAIN_TRANSFORMS_ALL_FOUR_COMBINED`) is active in `config.py`.
- Run the script:
  ```bash
  python main_adversarial.py --generator_model bisenet --epochs 50
  ```

**2. Adversarial Training with Lovász-Softmax Extension (Final Model):**

- This is the final proposed model in the report, which gave the best results.
- In `config.py`, set `USE_LOVASZ_LOSS = True`.
- Run the script:
  ```bash
  python main_adversarial.py --generator_model bisenet --epochs 50
  ```

### Using Command-Line Overrides

Many settings in `config.py` can be overridden via command-line arguments for convenience, such as `--lr`, `--epochs`, `--model_name`, etc.

## Viewing Results

- **Checkpoints:** Models are saved periodically and whenever a new best validation mIoU is achieved. They can be found in the `checkpoints/` directory, organized by model and training type.
- **Console Output:** Final metrics (mIoU, latency, FLOPs) and a per-class IoU table are printed to the console at the end of each training run.
- **Weights & Biases:** For detailed tracking, log into your W&B account. All metrics, losses, sample predictions, and environment configurations are logged there, providing a comprehensive overview of each experiment.
