"""3D sub-volume dataset built from the label table of build_labels_3d.py."""
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["small_lesion", "large_lesion"]
NUM_CLASSES = 2


def read_label_csv(csv_path, split=None):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            r["label"] = int(r["label"])
            r["lesion_voxels"] = int(r["lesion_voxels"])
            r["slice_list"] = r["slice_paths"].split(";") if r["slice_paths"] else []
            rows.append(r)
    return rows


def _fit_depth(slices, depth):
    """Pad (repeat last) or evenly subsample a slice list to exactly `depth`."""
    n = len(slices)
    if n == depth:
        return slices
    if n < depth:
        return slices + [slices[-1]] * (depth - n)
    idx = np.linspace(0, n - 1, depth).round().astype(int)
    return [slices[i] for i in idx]


class AISDVolumeDataset(Dataset):
    """Stack of axial CT slices -> binary label.

    Returns a tensor of shape ``(1, depth, image_size, image_size)``.
    """

    def __init__(self, rows, depth=16, image_size=160, train=False, mean=0.5, std=0.5):
        self.rows = rows
        self.depth = depth
        self.image_size = image_size
        self.train = train
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rows)

    def _load_slice(self, path):
        img = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        r = self.rows[idx]
        slices = _fit_depth(r["slice_list"], self.depth)
        vol = np.stack([self._load_slice(p) for p in slices], axis=0)  # (D, H, W)
        vol = (vol - self.mean) / self.std

        if self.train:
            if np.random.rand() < 0.5:               # flip left-right (W)
                vol = vol[:, :, ::-1]
            if np.random.rand() < 0.5:               # flip anterior-posterior (H)
                vol = vol[:, ::-1, :]
            vol = np.ascontiguousarray(vol)

        x = torch.from_numpy(vol).unsqueeze(0)        # (1, D, H, W)
        y = torch.tensor(r["label"], dtype=torch.long)
        return x, y


def class_weights(rows, num_classes=NUM_CLASSES):
    counts = np.zeros(num_classes, dtype=np.float64)
    for r in rows:
        counts[r["label"]] += 1
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
