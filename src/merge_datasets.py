#!/usr/bin/env python3
"""
Script to merge datasets and create train/val/test split.
"""

import os
import random
import shutil
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
ORIGINAL_DATASET = Path("dataset")
EXTERNAL_DATASET = Path("dataset_external")
OUTPUT_DATASET = Path("dataset_merged")

# Create output directories
for split in ["train", "valid", "test"]:
    (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_images_and_labels(dataset_path, split_folder):
    """Collect all images and labels from a dataset split folder."""
    images_dir = dataset_path / split_folder / "images"
    labels_dir = dataset_path / split_folder / "labels"

    if not images_dir.exists():
        return []

    image_files = (
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )

    result = []
    for img in image_files:
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            result.append((img, label))

    return result


# Collect all data from original dataset
print("Collecting from original dataset...")
original_train = collect_images_and_labels(ORIGINAL_DATASET, "train")
original_valid = collect_images_and_labels(ORIGINAL_DATASET, "valid")
print(f"  Original train: {len(original_train)}")
print(f"  Original valid: {len(original_valid)}")

# Collect all data from external dataset
print("Collecting from external dataset...")
external_train = collect_images_and_labels(EXTERNAL_DATASET, "train")
external_valid = collect_images_and_labels(EXTERNAL_DATASET, "valid")
external_test = collect_images_and_labels(EXTERNAL_DATASET, "test")
print(f"  External train: {len(external_train)}")
print(f"  External valid: {len(external_valid)}")
print(f"  External test: {len(external_test)}")

# Combine all data
all_data = (
    original_train + original_valid + external_train + external_valid + external_test
)
print(f"\nTotal combined: {len(all_data)} images")

# Shuffle
random.shuffle(all_data)

# Split: 80% train, 10% valid, 10% test
total = len(all_data)
train_size = int(0.8 * total)
valid_size = int(0.1 * total)

train_data = all_data[:train_size]
valid_data = all_data[train_size : train_size + valid_size]
test_data = all_data[train_size + valid_size :]

print(f"\nSplit:")
print(f"  Train: {len(train_data)}")
print(f"  Valid: {len(valid_data)}")
print(f"  Test: {len(test_data)}")


# Copy files to output directories
def copy_files(data, split):
    count = 0
    for img_path, label_path in data:
        # Copy image with original name
        new_img_path = OUTPUT_DATASET / "images" / split / img_path.name
        shutil.copy2(img_path, new_img_path)

        # Copy label with original name
        new_label_path = OUTPUT_DATASET / "labels" / split / label_path.name
        shutil.copy2(label_path, new_label_path)

        count += 1

    return count


print("\nCopying files...")
train_count = copy_files(train_data, "train")
valid_count = copy_files(valid_data, "valid")
test_count = copy_files(test_data, "test")

print(f"  Train: {train_count} images")
print(f"  Valid: {valid_count} images")
print(f"  Test: {test_count} images")
print(f"\nTotal: {train_count + valid_count + test_count} images")

# Verify counts
print("\nVerifying...")
train_imgs = len(list((OUTPUT_DATASET / "images" / "train").glob("*")))
valid_imgs = len(list((OUTPUT_DATASET / "images" / "valid").glob("*")))
test_imgs = len(list((OUTPUT_DATASET / "images" / "test").glob("*")))
print(f"  Train images: {train_imgs}")
print(f"  Valid images: {valid_imgs}")
print(f"  Test images: {test_imgs}")
print(f"  Total: {train_imgs + valid_imgs + test_imgs}")

# Save split info
split_info = {
    "train": train_count,
    "valid": valid_count,
    "test": test_count,
    "total": train_count + valid_count + test_count,
}

import json

with open(OUTPUT_DATASET / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)

print("\nDone!")
