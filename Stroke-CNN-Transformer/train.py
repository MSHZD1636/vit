"""Train Stroke-CNN-Transformer for AIS lesion segmentation, with validation.

Run from the repo root:
    python Stroke-CNN-Transformer/train.py \
        --data-root /path/to/AISD_nii \
        --splits-pkl data/splits_final.pkl \
        --out-dir Stroke-CNN-Transformer/runs/exp1 \
        --epochs 200 --batch-size 2 --amp --gpu 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports
from model import build_model            # noqa: E402
from dataset import AISDSegDataset, sliding_window_inference  # noqa: E402
from losses import DeepSupervisionLoss   # noqa: E402
from metrics import confusion_counts, metrics_from_counts  # noqa: E402
from visualize import plot_history       # noqa: E402


@torch.no_grad()
def validate(model, val_set, patch_size, device):
    dices, ious = [], []
    for i in range(len(val_set)):
        img, lab, _pid, _sp = val_set[i]
        prob = sliding_window_inference(model, img.unsqueeze(0), patch_size, device=device)
        pred = (prob[0, 0].cpu().numpy() > 0.5)
        gt = lab[0].numpy() > 0.5
        m = metrics_from_counts(*confusion_counts(pred, gt))
        dices.append(m["Dice"]); ious.append(m["IoU"])
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-pkl", default="data/splits_final.pkl")
    ap.add_argument("--out-dir", default="Stroke-CNN-Transformer/runs/exp1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[16, 160, 160])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deep-supervision", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--val-every", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = tuple(args.patch_size)
    print(f"Device: {device}  patch={patch}")

    train_set = AISDSegDataset(args.data_root, args.splits_pkl, "train", patch_size=patch)
    val_set = AISDSegDataset(args.data_root, args.splits_pkl, "val", patch_size=patch)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    model = build_model(in_channels=1, num_classes=1,
                        deep_supervision=args.deep_supervision).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f} M")

    criterion = DeepSupervisionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / max(1, len(train_loader.dataset))

        rec = {"epoch": epoch, "train_loss": train_loss}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            vd, vi = validate(model, val_set, patch, device)
            rec["val_dice"] = vd; rec["val_iou"] = vi
            print(f"[{epoch:03d}/{args.epochs}] loss={train_loss:.4f} val_Dice={vd:.4f} val_IoU={vi:.4f}")
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       os.path.join(args.out_dir, "last.pth"))
            if vd >= best_dice:
                best_dice = vd
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                           os.path.join(args.out_dir, "best.pth"))
        else:
            print(f"[{epoch:03d}/{args.epochs}] loss={train_loss:.4f}")
        history.append(rec)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    plot_history([h for h in history if "val_dice" in h],
                 os.path.join(args.out_dir, "training_curves.png"))
    print(f"\nBest val Dice = {best_dice:.4f}. Artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
