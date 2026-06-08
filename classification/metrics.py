"""Classification metrics without a hard scikit-learn dependency."""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm):
    """Return per-class precision/recall/f1, accuracy and macro-f1."""
    num_classes = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64),
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
    }


def format_report(cm, class_names):
    m = metrics_from_cm(cm)
    lines = []
    lines.append(f"{'class':<14}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for i, name in enumerate(class_names):
        lines.append(f"{name:<14}{m['precision'][i]:>8.3f}{m['recall'][i]:>8.3f}"
                     f"{m['f1'][i]:>8.3f}{m['support'][i]:>9d}")
    lines.append("")
    lines.append(f"accuracy = {m['accuracy']:.4f}   macro-F1 = {m['macro_f1']:.4f}")
    lines.append("")
    lines.append("confusion matrix (rows = true, cols = pred):")
    header = "          " + "".join(f"{n[:10]:>12}" for n in class_names)
    lines.append(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{int(v):>12d}" for v in cm[i])
        lines.append(f"{name[:10]:<10}{row}")
    return "\n".join(lines)
