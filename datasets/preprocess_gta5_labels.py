import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Centralized label definitions
from datasets.label_definitions import GTA5LabelInfo

# Initialize the LUT
COLOR_TO_ID_LUT = np.full((256, 256, 256), GTA5LabelInfo.ignore_id, dtype=np.uint8)
for color_tuple, class_id in GTA5LabelInfo.color_to_id_map.items():
    COLOR_TO_ID_LUT[color_tuple[0], color_tuple[1], color_tuple[2]] = class_id
print("Color-to-ID LUT for GTA5 initialized for pre-processing.")


def convert_rgb_to_id_with_lut(label_rgb_np: np.ndarray) -> np.ndarray:
    if label_rgb_np.ndim != 3 or label_rgb_np.shape[2] != 3:
        raise ValueError(
            f"Input label_rgb_np must be (H, W, 3), got {label_rgb_np.shape}"
        )
    return COLOR_TO_ID_LUT[
        label_rgb_np[..., 0], label_rgb_np[..., 1], label_rgb_np[..., 2]
    ]


def preprocess_labels(
    gta5_root_path: Path, original_labels_subdir: str, converted_labels_subdir: str
):
    original_label_dir = gta5_root_path / original_labels_subdir
    converted_label_dir = gta5_root_path / converted_labels_subdir
    if not original_label_dir.is_dir():
        print(f"Error: Original label directory not found: {original_label_dir}")
        return
    converted_label_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving pre-converted ID labels to: {converted_label_dir}")
    label_files = sorted(list(original_label_dir.rglob("*.png")))
    if not label_files:
        print(f"No .png label files found in {original_label_dir}")
        return
    for label_path in tqdm(label_files, desc="Converting labels"):
        try:
            label_pil_rgb = Image.open(label_path).convert("RGB")
            label_rgb_np = np.array(label_pil_rgb)
            label_id_np = convert_rgb_to_id_with_lut(label_rgb_np)
            output_pil = Image.fromarray(label_id_np, mode="L")
            output_path = converted_label_dir / label_path.name
            output_pil.save(output_path)
        except Exception as e:
            print(f"Error processing {label_path}: {e}")
    print(f"\nPre-conversion complete. {len(label_files)} labels processed.")
    print(f"Converted labels saved in: {converted_label_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-convert GTA5 RGB labels to Class ID maps."
    )
    parser.add_argument(
        "--gta5_root", type=str, required=True, help="Root path of the GTA5 dataset."
    )
    parser.add_argument(
        "--original_subdir",
        type=str,
        default="labels",
        help="Subdir of original RGB labels (default: 'labels').",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="labels_trainids",
        help="Subdir for converted ID maps (default: 'labels_trainids').",
    )
    args = parser.parse_args()
    root_path = Path(args.gta5_root)
    preprocess_labels(root_path, args.original_subdir, args.output_subdir)
