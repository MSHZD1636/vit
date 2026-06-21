"""Visualization helpers: training curves, metric bar chart, prediction overlays."""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(history, out_path):
    """history: list of dicts with keys epoch, train_loss, val_dice, val_iou."""
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, [h["train_loss"] for h in history], "C0-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(ep, [h.get("val_dice", np.nan) for h in history], "C1-", label="val Dice")
    ax2.plot(ep, [h.get("val_iou", np.nan) for h in history], "C2-", label="val IoU")
    ax2.set_ylabel("metric", color="C1"); ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="C1")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    plt.title("Training curves")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_metric_bar(metrics, out_path, exclude=("HD95",)):
    """Bar chart of [0,1] metrics (HD95 excluded, different unit)."""
    items = [(k, v) for k, v in metrics.items() if k not in exclude]
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, vals, color="C0")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.set_title("Test metrics")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _window(ct, wl=40, ww=80):
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((ct - lo) / (hi - lo), 0, 1)


def save_overlays(ct, gt, pred, out_dir, pid="case", max_slices=12, wl=40, ww=80):
    """Save a montage of slices: red = prediction, green contour = GT.

    ct, gt, pred: numpy (Z, H, W). gt/pred are binary. Only slices with GT or
    prediction are shown.
    """
    from scipy.ndimage import binary_erosion
    os.makedirs(out_dir, exist_ok=True)
    idx = [z for z in range(ct.shape[0]) if gt[z].sum() > 0 or pred[z].sum() > 0]
    if not idx:
        idx = [ct.shape[0] // 2]
    idx = idx[:: max(1, len(idx) // max_slices)][:max_slices]

    cols = min(4, len(idx)); rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, z in enumerate(idx):
        ax = axes[k // cols][k % cols]
        g = _window(ct[z], wl, ww)
        rgb = np.stack([g, g, g], axis=-1)
        m = pred[z].astype(bool)
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array([1, 0, 0])     # red prediction
        gt_b = gt[z].astype(bool)
        edge = gt_b & ~binary_erosion(gt_b)
        rgb[edge] = [0, 1, 0]                                    # green GT contour
        ax.imshow(np.clip(rgb, 0, 1)); ax.set_title(f"z={z}", fontsize=9)
    fig.suptitle(f"{pid}  (red=pred, green=GT)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{pid}_overlay.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path
