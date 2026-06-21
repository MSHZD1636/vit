"""3D dataset + sliding-window inference for AIS lesion segmentation.

Reads the per-patient volumes produced by ``reproduce/aisd_dicom_to_nii.py``:

    <data_root>/<pid>/CT.nii.gz
    <data_root>/<pid>/GT_hard.nii.gz

The patient-level split comes from ``data/splits_final.pkl``. Training samples
random foreground-biased patches; validation/test return whole volumes (run
through ``sliding_window_inference``).
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except Exception:  # allow import without SimpleITK (e.g. for unit tests)
    sitk = None


def zscore_window(image, lo=0.0, hi=60.0):
    """Brain window [lo, hi] HU then z-score over the windowed ROI (AISD style)."""
    roi = np.where((image >= lo) & (image <= hi))
    if roi[0].size == 0:
        mu, sigma = image.mean(), image.std()
    else:
        mu, sigma = image[roi].mean(), image[roi].std()
    img = np.clip(image, lo, hi).astype(np.float32)
    return (img - mu) / (sigma + 1e-6)


def load_split(splits_pkl, split):
    with open(splits_pkl, "rb") as f:
        s = pickle.load(f)[0]
    return [str(p) for p in s[split]]


def read_volume(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z, Y, X)
    spacing = img.GetSpacing()[::-1]                        # -> (z, y, x)
    return arr, spacing


class AISDSegDataset(Dataset):
    def __init__(self, data_root, splits_pkl, split="train",
                 patch_size=(16, 160, 160), fg_prob=0.5):
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size
        self.fg_prob = fg_prob
        self.pids = [p for p in load_split(splits_pkl, split)
                     if os.path.isdir(os.path.join(data_root, p))]
        if not self.pids:
            raise RuntimeError(f"No cases for split={split} under {data_root}")

    def __len__(self):
        return len(self.pids)

    def _random_patch(self, img, lab):
        pd, ph, pw = self.patch_size
        D, H, W = img.shape
        img, lab, (D, H, W) = self._pad_to_min(img, lab)
        if np.random.rand() < self.fg_prob and lab.sum() > 0:
            zs, ys, xs = np.where(lab > 0)
            i = np.random.randint(len(zs))
            cz, cy, cx = zs[i], ys[i], xs[i]
            z0 = np.clip(cz - pd // 2, 0, D - pd)
            y0 = np.clip(cy - ph // 2, 0, H - ph)
            x0 = np.clip(cx - pw // 2, 0, W - pw)
        else:
            z0 = np.random.randint(0, D - pd + 1)
            y0 = np.random.randint(0, H - ph + 1)
            x0 = np.random.randint(0, W - pw + 1)
        sl = (slice(z0, z0 + pd), slice(y0, y0 + ph), slice(x0, x0 + pw))
        return img[sl], lab[sl]

    def _pad_to_min(self, img, lab):
        pd, ph, pw = self.patch_size
        D, H, W = img.shape
        pad = [(0, max(0, pd - D)), (0, max(0, ph - H)), (0, max(0, pw - W))]
        if any(a or b for a, b in pad):
            img = np.pad(img, pad, mode="constant", constant_values=img.min())
            lab = np.pad(lab, pad, mode="constant")
        return img, lab, img.shape

    def __getitem__(self, idx):
        pid = self.pids[idx]
        img, spacing = read_volume(os.path.join(self.data_root, pid, "CT.nii.gz"))
        lab, _ = read_volume(os.path.join(self.data_root, pid, "GT_hard.nii.gz"))
        img = zscore_window(img)
        lab = (lab > 0).astype(np.float32)

        if self.split == "train":
            img, lab = self._random_patch(img, lab)
            x = torch.from_numpy(img).unsqueeze(0)
            y = torch.from_numpy(lab).unsqueeze(0)
            return x, y

        # val/test: whole volume (batch size must be 1)
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(lab).unsqueeze(0),
                pid, np.asarray(spacing, dtype=np.float32))


def _gaussian_map(patch_size, sigma_scale=0.125, device="cpu"):
    coords = [torch.arange(s, dtype=torch.float32) for s in patch_size]
    g = torch.ones(patch_size)
    for d, c in enumerate(coords):
        center = (len(c) - 1) / 2.0
        sigma = patch_size[d] * sigma_scale
        line = torch.exp(-((c - center) ** 2) / (2 * sigma ** 2))
        shape = [1, 1, 1]
        shape[d] = len(c)
        g = g * line.view(shape)
    return (g / g.max()).to(device)


@torch.no_grad()
def sliding_window_inference(model, volume, patch_size=(16, 160, 160),
                             overlap=0.5, device="cpu"):
    """Tile a (1,1,D,H,W) volume into patches, predict, Gaussian-blend.

    Returns a (1,1,D,H,W) probability map.
    """
    model.eval()
    _, _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    # pad up to at least one patch
    pad = [max(0, pd - D), max(0, ph - H), max(0, pw - W)]
    vol = torch.nn.functional.pad(volume, [0, pad[2], 0, pad[1], 0, pad[0]])
    _, _, D, H, W = vol.shape

    step = [max(1, int(p * (1 - overlap))) for p in patch_size]

    def starts(total, p, s):
        if total <= p:
            return [0]
        xs = list(range(0, total - p + 1, s))
        if xs[-1] != total - p:
            xs.append(total - p)
        return xs

    zs = starts(D, pd, step[0]); ys = starts(H, ph, step[1]); xs = starts(W, pw, step[2])
    gauss = _gaussian_map(patch_size, device=device)[None, None]
    acc = torch.zeros((1, 1, D, H, W), device=device)
    wmap = torch.zeros((1, 1, D, H, W), device=device)
    vol = vol.to(device)
    for z in zs:
        for y in ys:
            for x in xs:
                patch = vol[:, :, z:z + pd, y:y + ph, x:x + pw]
                out = model(patch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                acc[:, :, z:z + pd, y:y + ph, x:x + pw] += out * gauss
                wmap[:, :, z:z + pd, y:y + ph, x:x + pw] += gauss
    prob = acc / wmap.clamp_min(1e-6)
    return prob[:, :, :D - pad[0], :H - pad[1], :W - pad[2]]
