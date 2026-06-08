"""Build a slice-level classification label table from the AISD masks.

For every image slice we look up its mask, count the number of lesion pixels and
assign one of three classes:

    0 - no stroke   (lesion area == 0)
    1 - small lesion (0 < lesion area <= threshold)
    2 - large lesion (lesion area >  threshold)

The patient-level train/val/test assignment is taken from ``splits_final.pkl``
so that all slices of one patient stay in the same split (no patient leakage).

The AISD masks are *multi-valued* (1=remote/old infarct, 2=clear acute infarct,
3=blurred acute infarct, 4=invisible acute infarct, 5=infarct). By default every
non-zero pixel is treated as a lesion; use ``--lesion-values`` to restrict this
(e.g. ``--lesion-values 2 3 4 5`` to ignore old/remote infarcts).

Output: a CSV with columns ``patient_id,split,image_path,mask_path,lesion_area,label``.

Example
-------
    python -m classification.build_labels \
        --images-dir /path/to/aisd/image \
        --masks-dir  /path/to/aisd/mask \
        --splits-pkl data/splits_final.pkl \
        --out-csv    classification/labels.csv \
        --auto-threshold
"""
import argparse
import csv
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_images(root):
    """Return a list of (relative_path, absolute_path) for every image under root."""
    out = []
    for home, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                ap = os.path.join(home, fn)
                rp = os.path.relpath(ap, root)
                out.append((rp, ap))
    out.sort()
    return out


def index_masks(root):
    """Index masks by relative path and by basename for flexible matching."""
    by_rel, by_base = {}, {}
    for rp, ap in find_images(root):
        by_rel[rp] = ap
        by_base.setdefault(os.path.basename(rp), ap)
    return by_rel, by_base


def extract_patient_id(rel_path, id_regex):
    m = re.search(id_regex, rel_path)
    return m.group(1) if m else None


def lesion_area(mask_path, lesion_values):
    """Number of lesion pixels in a mask image."""
    arr = np.array(Image.open(mask_path).convert("L"))
    if lesion_values is None:
        return int(np.count_nonzero(arr))
    mask = np.isin(arr, np.asarray(list(lesion_values)))
    return int(np.count_nonzero(mask))


def build_split_lookup(splits_pkl):
    """patient_id -> 'train'|'val'|'test' from the nnU-Net splits pickle."""
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    fold = splits[0]
    lut = {}
    for split_name in ("train", "val", "test"):
        for pid in fold.get(split_name, []):
            lut[str(pid)] = split_name
    return lut


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, help="Root folder of PNG image slices.")
    ap.add_argument("--masks-dir", required=True, help="Root folder of PNG mask slices.")
    ap.add_argument("--splits-pkl", default="data/splits_final.pkl",
                    help="nnU-Net splits_final.pkl (patient-level train/val/test).")
    ap.add_argument("--out-csv", default="classification/labels.csv")
    ap.add_argument("--id-regex", default=r"(\d{7})",
                    help="Regex whose group(1) extracts the patient id from a slice path.")
    ap.add_argument("--lesion-values", type=int, nargs="*", default=None,
                    help="Mask values counted as lesion. Default: any non-zero value.")
    ap.add_argument("--small-max-area", type=int, default=500,
                    help="Slices with 0 < area <= this are 'small', above are 'large'.")
    ap.add_argument("--auto-threshold", action="store_true",
                    help="Ignore --small-max-area; split small/large at the MEDIAN "
                         "lesion area of all lesion-bearing slices in the TRAIN set.")
    ap.add_argument("--missing-mask-as-negative", action="store_true", default=True,
                    help="Treat an image with no matching mask file as 'no stroke'.")
    ap.add_argument("--keep-unknown-split", action="store_true",
                    help="Keep slices whose patient id is not in the splits pickle "
                         "(assigned split='unknown'). Default: drop them.")
    args = ap.parse_args()

    split_lut = build_split_lookup(args.splits_pkl)
    images = find_images(args.images_dir)
    mask_by_rel, mask_by_base = index_masks(args.masks_dir)
    print(f"Found {len(images)} image slices and {len(mask_by_rel)} mask slices.")

    rows = []
    missing_mask = 0
    unknown_split = 0
    for rel, img_path in images:
        pid = extract_patient_id(rel, args.id_regex)
        if pid is None:
            continue
        split = split_lut.get(pid, "unknown")
        if split == "unknown" and not args.keep_unknown_split:
            unknown_split += 1
            continue

        mask_path = mask_by_rel.get(rel) or mask_by_base.get(os.path.basename(rel))
        if mask_path is None:
            if not args.missing_mask_as_negative:
                continue
            missing_mask += 1
            area = 0
        else:
            area = lesion_area(mask_path, args.lesion_values)

        rows.append({
            "patient_id": pid,
            "split": split,
            "image_path": img_path,
            "mask_path": mask_path or "",
            "lesion_area": area,
        })

    # Decide the small/large boundary.
    if args.auto_threshold:
        train_areas = [r["lesion_area"] for r in rows
                       if r["split"] == "train" and r["lesion_area"] > 0]
        threshold = int(np.median(train_areas)) if train_areas else args.small_max_area
        print(f"Auto threshold (median train lesion area) = {threshold} pixels.")
    else:
        threshold = args.small_max_area

    for r in rows:
        a = r["lesion_area"]
        r["label"] = 0 if a == 0 else (1 if a <= threshold else 2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fields = ["patient_id", "split", "image_path", "mask_path", "lesion_area", "label"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Report.
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
    if missing_mask:
        print(f"  ({missing_mask} images had no matching mask -> labelled 'no stroke')")
    if unknown_split:
        print(f"  ({unknown_split} images dropped: patient id not in splits pickle)")
    names = {0: "no_stroke", 1: "small_lesion", 2: "large_lesion"}
    per_split = defaultdict(Counter)
    for r in rows:
        per_split[r["split"]][r["label"]] += 1
    print("\nClass distribution per split:")
    print(f"  {'split':<8} {'no_stroke':>10} {'small':>8} {'large':>8} {'total':>8}")
    for split in ("train", "val", "test", "unknown"):
        if split not in per_split:
            continue
        c = per_split[split]
        tot = sum(c.values())
        print(f"  {split:<8} {c[0]:>10} {c[1]:>8} {c[2]:>8} {tot:>8}")
    _ = names  # documentation of label meaning


if __name__ == "__main__":
    main()
