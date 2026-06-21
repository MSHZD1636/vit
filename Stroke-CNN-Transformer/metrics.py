"""Segmentation evaluation metrics for binary lesion segmentation.

All count-based metrics are derived from a binary confusion (TP, FP, FN, TN)
with two classes: background (0) and lesion/foreground (1).

Provided:
    Dice, IoU (foreground), mIoU (mean over bg/fg), Pixel Accuracy (PA),
    mean Pixel Accuracy (mPA), Precision, Recall, F1, and HD95
    (95th-percentile Hausdorff distance, computed from binary masks).
"""
import numpy as np

EPS = 1e-7


def confusion_counts(pred, gt):
    """pred, gt: binary arrays (any shape). Returns TP, FP, FN, TN."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    """All count-based metrics from a binary confusion."""
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    iou_fg = tp / (tp + fp + fn + EPS)
    iou_bg = tn / (tn + fp + fn + EPS)
    miou = 0.5 * (iou_fg + iou_bg)
    pa = (tp + tn) / (tp + tn + fp + fn + EPS)
    recall_fg = tp / (tp + fn + EPS)         # sensitivity
    recall_bg = tn / (tn + fp + EPS)         # specificity
    mpa = 0.5 * (recall_fg + recall_bg)      # mean per-class accuracy
    precision = tp / (tp + fp + EPS)
    f1 = (2 * precision * recall_fg) / (precision + recall_fg + EPS)
    return {
        "Dice": dice,
        "IoU": iou_fg,
        "mIoU": miou,
        "PA": pa,
        "mPA": mpa,
        "Precision": precision,
        "Recall": recall_fg,
        "F1": f1,
    }


def _surface_distances(pred, gt, spacing):
    """Symmetric surface distances (mm) between two binary masks."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf = gt ^ binary_erosion(gt)
    dt_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)
    return np.concatenate([dt_gt[pred_surf], dt_pred[gt_surf]])


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """95th-percentile Hausdorff distance (mm). NaN if undefined."""
    sd = _surface_distances(pred, gt, spacing)
    if sd is None:
        # both empty -> perfect; exactly one empty -> undefined
        if pred.sum() == 0 and gt.sum() == 0:
            return 0.0
        return float("nan")
    return float(np.percentile(sd, 95))


def all_metrics(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """Full metric dict for one case (pred, gt binary, same shape)."""
    m = metrics_from_counts(*confusion_counts(pred, gt))
    m["HD95"] = hd95(pred, gt, spacing)
    return m


def aggregate(per_case):
    """Mean over a list of per-case metric dicts (NaN-aware for HD95)."""
    keys = per_case[0].keys()
    out = {}
    for k in keys:
        vals = np.array([c[k] for c in per_case], dtype=np.float64)
        out[k] = float(np.nanmean(vals)) if np.isnan(vals).any() else float(vals.mean())
    return out


METRIC_ORDER = ["Dice", "HD95", "PA", "mPA", "IoU", "mIoU", "Precision", "Recall", "F1"]


def format_table(agg):
    lines = [f"{'metric':<12}{'value':>10}"]
    for k in METRIC_ORDER:
        if k in agg:
            lines.append(f"{k:<12}{agg[k]:>10.4f}")
    return "\n".join(lines)
