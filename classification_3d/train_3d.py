"""Train the 3D lesion-size classifier (small_lesion / large_lesion).

Backbone: CNN-Transformer hybrid encoder (Coformer3D) reused from Cl-SegNet.

Example
-------
    python -m classification_3d.train_3d \
        --labels-csv classification_3d/labels_3d.csv \
        --out-dir    classification_3d/runs/coformer \
        --depth 16 --image-size 160 --batch-size 4 --epochs 80 --amp --gpu 0
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classification.metrics import confusion_matrix, format_report, metrics_from_cm
from classification_3d.model_3d import build_model
from classification_3d.volume_dataset import (CLASS_NAMES, NUM_CLASSES,
                                              AISDVolumeDataset, class_weights,
                                              read_label_csv)


def make_loaders(args):
    train_rows = read_label_csv(args.labels_csv, split="train")
    val_rows = read_label_csv(args.labels_csv, split="val")
    if not train_rows:
        raise SystemExit(f"No 'train' rows in {args.labels_csv}. Run build_labels_3d.py first.")
    train_ds = AISDVolumeDataset(train_rows, depth=args.depth, image_size=args.image_size, train=True)
    val_ds = AISDVolumeDataset(val_rows, depth=args.depth, image_size=args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    return train_rows, train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(y.numpy())
    if not y_true:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    return confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred), num_classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-csv", default="classification_3d/labels_3d.csv")
    ap.add_argument("--out-dir", default="classification_3d/runs/coformer")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--depth", type=int, default=16, help="Slices per sub-volume (multiple of 4).")
    ap.add_argument("--image-size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--single-scale", action="store_true",
                    help="Use only the deepest encoder feature instead of the multi-scale pyramid.")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rows, train_loader, val_loader = make_loaders(args)
    model = build_model(in_chans=1, num_classes=NUM_CLASSES,
                        use_multiscale=not args.single_scale).to(device)

    weight = None if args.no_class_weights else class_weights(train_rows).to(device)
    print(f"Class weights: {None if weight is None else weight.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / max(1, len(train_loader.dataset))

        cm = evaluate(model, val_loader, device)
        m = metrics_from_cm(cm)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"]})
        print(f"[{epoch:03d}/{args.epochs}] loss={train_loss:.4f} "
              f"val_acc={m['accuracy']:.4f} val_macroF1={m['macro_f1']:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                   os.path.join(args.out_dir, "last.pth"))
        if m["macro_f1"] >= best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       os.path.join(args.out_dir, "best.pth"))
            with open(os.path.join(args.out_dir, "val_report.txt"), "w") as f:
                f.write(f"epoch {epoch}\n" + format_report(cm, CLASS_NAMES) + "\n")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val macro-F1 = {best_macro_f1:.4f}")
    print(f"Checkpoints + report saved to {args.out_dir}")


if __name__ == "__main__":
    main()
