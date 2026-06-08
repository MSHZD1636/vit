"""Dataset that reads the slice-level label CSV produced by build_labels.py."""
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["no_stroke", "small_lesion", "large_lesion"]
NUM_CLASSES = 3


def read_label_csv(csv_path, split=None):
    """Return the list of row dicts, optionally filtered by split."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            r["label"] = int(r["label"])
            r["lesion_area"] = int(r["lesion_area"])
            rows.append(r)
    return rows


class AISDSliceDataset(Dataset):
    """Grayscale CT slice -> 3-class label.

    Parameters
    ----------
    rows : list of dict
        Rows from ``read_label_csv`` (each has ``image_path`` and ``label``).
    image_size : int
        Square size the slices are resized to.
    train : bool
        If True, light augmentation (flip / small rotation) is applied.
    mean, std : float
        Normalisation statistics applied after scaling to [0, 1].
    """

    def __init__(self, rows, image_size=256, train=False, mean=0.5, std=0.5):
        self.rows = rows
        self.image_size = image_size
        self.train = train
        self.mean = mean
        self.std = std
        # torchvision is optional; only imported when augmentation is requested.
        self._tf = None
        if train:
            from torchvision import transforms
            self._tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path):
        img = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        if self._tf is not None:
            img = self._tf(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = self._load_image(r["image_path"])
        y = torch.tensor(r["label"], dtype=torch.long)
        return x, y


def class_weights(rows, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights for a class-balanced cross-entropy."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for r in rows:
        counts[r["label"]] += 1
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
