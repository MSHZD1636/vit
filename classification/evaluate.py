"""Evaluate a trained classifier on a split (default: test) and print the report.

Example
-------
    python -m classification.evaluate \
        --labels-csv classification/labels.csv \
        --checkpoint classification/runs/resnet18/best.pth \
        --split test
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from classification.dataset import (CLASS_NAMES, NUM_CLASSES, AISDSliceDataset,
                                    read_label_csv)
from classification.metrics import confusion_matrix, format_report
from classification.model import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-csv", default="classification/labels.csv")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--save-report", default=None, help="Optional path to write the report.")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_label_csv(args.labels_csv, split=args.split)
    if not rows:
        raise SystemExit(f"No '{args.split}' rows in {args.labels_csv}.")
    ds = AISDSliceDataset(rows, image_size=args.image_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(in_channels=1, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(y.numpy())
    cm = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred), NUM_CLASSES)

    report = f"split = {args.split}   checkpoint = {args.checkpoint}\n\n"
    report += format_report(cm, CLASS_NAMES)
    print(report)
    if args.save_report:
        with open(args.save_report, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved report to {args.save_report}")


if __name__ == "__main__":
    main()
