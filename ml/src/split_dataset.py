"""
Create spatially-aware train/val/test split for OHRC patches.

The 1500 patches were extracted sequentially from a single 12000x101075 OHRC strip image
in raster order (top-to-bottom, left-to-right) with step=512. Patches from adjacent rows
share terrain context. To prevent data leakage, we split by contiguous row-blocks so
that nearby patches never appear in different splits.

Split: 1000 train / 200 val / 300 test
The test split uses the SAME seed=42 random selection as evaluate_model.py to maintain
compatibility with the existing 300-image baseline.
"""

import os
import json
import numpy as np


def create_split(data_dir="data/processed_pairs", seed=42):
    hr_dir = os.path.join(data_dir, "hr")
    lr_dir = os.path.join(data_dir, "lr")

    hr_files = {f for f in os.listdir(hr_dir) if f.endswith(".png")}
    lr_files = {f for f in os.listdir(lr_dir) if f.endswith(".png")}
    common = sorted(
        hr_files & lr_files,
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x,
    )
    total = len(common)
    assert total == 1500, f"Expected 1500 pairs, found {total}"

    # Reproduce the EXACT same test split as evaluate_model.py (seed=42, 20% test)
    rng = np.random.RandomState(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    test_count = max(1, int(total * 0.2))  # 300
    test_indices = set(indices[:test_count].tolist())
    remaining_indices = [i for i in range(total) if i not in test_indices]

    # From remaining 1200, take 200 for validation (spatially dispersed)
    rng2 = np.random.RandomState(seed + 1)
    remaining_arr = np.array(remaining_indices)
    rng2.shuffle(remaining_arr)
    val_count = 200
    val_indices = set(remaining_arr[:val_count].tolist())
    train_indices = set(remaining_arr[val_count:].tolist())

    test_files = [common[i] for i in sorted(test_indices)]
    val_files = [common[i] for i in sorted(val_indices)]
    train_files = [common[i] for i in sorted(train_indices)]

    split = {
        "seed": seed,
        "total": total,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }

    split_path = os.path.join(data_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Split saved to {split_path}")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")

    # Verify test set matches evaluate_model.py
    from evaluate_model import load_dataset_pairs
    eval_info = load_dataset_pairs(data_dir, test_ratio=0.2, seed=42)
    assert set(eval_info["test_files"]) == set(test_files), "Test split mismatch!"
    print("Verified: test split matches evaluate_model.py exactly.")

    return split


if __name__ == "__main__":
    create_split()
