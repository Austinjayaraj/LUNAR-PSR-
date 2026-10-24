"""
Dataset loader for real OHRC LR/HR patch pairs with online degradation augmentation.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OHRCDataset(Dataset):
    """Loads real OHRC LR/HR pairs for a given split (train/val/test)."""

    def __init__(self, data_dir="data/processed_pairs", split="train", augment=False):
        split_path = os.path.join(data_dir, "split.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}. Run src/split_dataset.py first.")

        with open(split_path) as f:
            split_info = json.load(f)

        self.files = split_info[f"{split}_files"]
        self.hr_dir = os.path.join(data_dir, "hr")
        self.lr_dir = os.path.join(data_dir, "lr")
        self.augment = augment and (split == "train")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        hr = cv2.imread(os.path.join(self.hr_dir, fname), cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(os.path.join(self.lr_dir, fname), cv2.IMREAD_GRAYSCALE)

        if self.augment:
            hr, lr = self._augment(hr, lr)

        lr_t = torch.from_numpy(lr.copy()).unsqueeze(0).float() / 255.0
        hr_t = torch.from_numpy(hr.copy()).unsqueeze(0).float() / 255.0
        return lr_t, hr_t

    def _augment(self, hr, lr):
        # Random horizontal flip
        if np.random.random() > 0.5:
            hr = np.fliplr(hr)
            lr = np.fliplr(lr)

        # Random vertical flip
        if np.random.random() > 0.5:
            hr = np.flipud(hr)
            lr = np.flipud(lr)

        # Random 90-degree rotation (0, 90, 180, 270)
        k = np.random.randint(4)
        if k > 0:
            hr = np.rot90(hr, k)
            lr = np.rot90(lr, k)

        return np.ascontiguousarray(hr), np.ascontiguousarray(lr)
