"""Evaluate a trained Stroke-CNN-Transformer on a split (default: test).

Computes Dice, HD95, PA, mPA, IoU, mIoU, Precision, Recall, F1 per case, then
reports the mean and saves visualizations (metric bar chart + qualitative
overlays).

Run from the repo root:
    python Stroke-CNN-Transformer/evaluate.py \
        --data-root /path/to/AISD_nii \
        --splits-pkl data/splits_final.pkl \
        --checkpoint Stroke-CNN-Transformer/runs/exp1/best.pth \
        --split test --out-dir Stroke-CNN-Transformer/runs/exp1/eval_test
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model            # noqa: E402
from dataset import AISDSegDataset, sliding_window_inference  # noqa: E402
from metrics import METRIC_ORDER, aggregate, all_metrics, format_table  # noqa: E402
from visualize import plot_metric_bar, save_overlays  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-pkl", default="data/splits_final.pkl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default="Stroke-CNN-Transformer/runs/eval")
    ap.add_argument("--patch-size", type=int, nargs=3, default=[16, 160, 160])
    ap.add_argument("--num-overlays", type=int, default=6,
                    help="How many cases to save qualitative overlays for.")
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = tuple(args.patch_size)

    ds = AISDSegDataset(args.data_root, args.splits_pkl, args.split, patch_size=patch)
    model = build_model(in_channels=1, num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    per_case = []
    rows = []
    for i in range(len(ds)):
        img, lab, pid, spacing = ds[i]
        prob = sliding_window_inference(model, img.unsqueeze(0), patch, device=device)
        pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        gt = (lab[0].numpy() > 0.5).astype(np.uint8)
        m = all_metrics(pred, gt, spacing=tuple(float(s) for s in spacing))
        per_case.append(m)
        rows.append({"pid": pid, **{k: m[k] for k in METRIC_ORDER}})
        print(f"{pid}: Dice={m['Dice']:.4f} IoU={m['IoU']:.4f} HD95={m['HD95']:.2f}")

        if i < args.num_overlays:
            save_overlays(img[0].numpy(), gt, pred, args.out_dir, pid=str(pid))

    agg = aggregate(per_case)

    # per-case CSV
    with open(os.path.join(args.out_dir, "per_case_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid"] + METRIC_ORDER)
        w.writeheader(); w.writerows(rows)

    # summary report
    report = f"split={args.split}  n={len(per_case)}  checkpoint={args.checkpoint}\n\n"
    report += format_table(agg)
    print("\n" + report)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")

    # metric bar chart
    plot_metric_bar(agg, os.path.join(args.out_dir, "metrics_bar.png"))
    print(f"\nSaved metrics, bar chart and overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
