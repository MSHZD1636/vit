"""Build a 3D lesion-size binary label table for AISD (small vs large lesion).

Slices are grouped per patient and ordered along the axial axis (natural sort of
the filename). Two sampling modes:

* ``slab``  (default): slide a window of ``--depth`` consecutive slices with step
  ``--stride``; each slab's lesion volume = total lesion voxels over its slices.
* ``volume``: one sample per patient using all its slices; lesion volume = total
  lesion voxels over the whole case (clean per-case lesion size).

Only lesion-bearing sub-volumes are kept (those with lesion voxels > 0); there is
no "no stroke" class. Each kept sub-volume is labelled:

    0 - small_lesion (lesion voxels <= threshold)
    1 - large_lesion (lesion voxels >  threshold)

The threshold is either fixed (``--small-max-voxels``) or data-adaptive
(``--auto-threshold``: the median lesion volume of lesion-bearing TRAIN samples).

The patient-level train/val/test assignment comes from ``splits_final.pkl`` so
sub-volumes of one patient never leak across splits.

Output CSV columns: ``patient_id,split,sample_id,slice_paths,lesion_voxels,label``.

Example
-------
    python -m classification_3d.build_labels_3d \
        --images-dir /path/to/aisd/image \
        --masks-dir  /path/to/aisd/mask \
        --splits-pkl data/splits_final.pkl \
        --out-csv    classification_3d/labels_3d.csv \
        --mode slab --depth 16 --stride 8 --auto-threshold
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import numpy as np

# Reuse the slice-level helpers (single source of truth).
from classification.build_labels import (build_split_lookup, extract_patient_id,
                                          find_images, index_masks, lesion_area)


def natural_key(path):
    """Sort key that orders 'slice2' before 'slice10' (axial order)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def group_by_patient(images, id_regex):
    groups = defaultdict(list)
    for rel, ap in images:
        pid = extract_patient_id(rel, id_regex)
        if pid is not None:
            groups[pid].append((rel, ap))
    for pid in groups:
        groups[pid].sort(key=lambda t: natural_key(t[0]))
    return groups


def slab_starts(n, depth, stride):
    """Start indices of full-size slabs, always covering the final slices."""
    if n <= depth:
        return [0]
    starts = list(range(0, n - depth + 1, stride))
    if starts[-1] != n - depth:
        starts.append(n - depth)
    return starts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--splits-pkl", default="data/splits_final.pkl")
    ap.add_argument("--out-csv", default="classification_3d/labels_3d.csv")
    ap.add_argument("--id-regex", default=r"(\d{7})")
    ap.add_argument("--lesion-values", type=int, nargs="*", default=None,
                    help="Mask values counted as lesion. Default: any non-zero value.")
    ap.add_argument("--mode", choices=["slab", "volume"], default="slab")
    ap.add_argument("--depth", type=int, default=16,
                    help="Slab size (number of consecutive slices). Use a multiple of 4.")
    ap.add_argument("--stride", type=int, default=8, help="Slab step (slab mode only).")
    ap.add_argument("--small-max-voxels", type=int, default=2000,
                    help="Lesion-bearing sub-volumes with lesion voxels <= this are "
                         "'small', above are 'large'.")
    ap.add_argument("--auto-threshold", action="store_true",
                    help="Ignore --small-max-voxels; split small/large at the MEDIAN "
                         "lesion volume of lesion-bearing TRAIN sub-volumes.")
    ap.add_argument("--keep-unknown-split", action="store_true")
    args = ap.parse_args()

    split_lut = build_split_lookup(args.splits_pkl)
    images = find_images(args.images_dir)
    mask_by_rel, mask_by_base = index_masks(args.masks_dir)
    groups = group_by_patient(images, args.id_regex)
    print(f"Found {len(images)} slices across {len(groups)} patients; "
          f"{len(mask_by_rel)} masks.")

    def area_for(rel, img_path):
        mp = mask_by_rel.get(rel) or mask_by_base.get(os.path.basename(img_path))
        return 0 if mp is None else lesion_area(mp, args.lesion_values)

    # First pass: build every lesion-bearing sub-volume (label filled in pass 2).
    samples = []
    dropped_empty = 0
    for pid, items in groups.items():
        split = split_lut.get(pid, "unknown")
        if split == "unknown" and not args.keep_unknown_split:
            continue
        areas = [area_for(rel, ap) for rel, ap in items]
        paths = [ap for _, ap in items]
        if not paths:
            continue

        if args.mode == "volume":
            spans = [(0, len(paths))]
        else:
            spans = [(s, s + args.depth) for s in slab_starts(len(paths), args.depth, args.stride)]

        for k, (s, e) in enumerate(spans):
            vox = int(sum(areas[s:e]))
            if vox <= 0:                      # only lesion-bearing sub-volumes
                dropped_empty += 1
                continue
            samples.append({
                "patient_id": pid,
                "split": split,
                "sample_id": f"{pid}_{k:03d}" if args.mode == "slab" else f"{pid}_full",
                "slice_paths": ";".join(paths[s:e]),
                "lesion_voxels": vox,
            })

    # Decide the small/large boundary.
    if args.auto_threshold:
        train_vox = [r["lesion_voxels"] for r in samples if r["split"] == "train"]
        threshold = int(np.median(train_vox)) if train_vox else args.small_max_voxels
        print(f"Auto threshold (median train lesion volume) = {threshold} voxels.")
    else:
        threshold = args.small_max_voxels

    for r in samples:
        r["label"] = 0 if r["lesion_voxels"] <= threshold else 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fields = ["patient_id", "split", "sample_id", "slice_paths", "lesion_voxels", "label"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(samples)

    print(f"\nWrote {len(samples)} lesion-bearing 3D samples ({args.mode} mode) "
          f"to {args.out_csv}")
    if dropped_empty:
        print(f"  ({dropped_empty} sub-volumes without lesion were dropped)")
    per_split = defaultdict(Counter)
    for r in samples:
        per_split[r["split"]][r["label"]] += 1
    print("\nClass distribution per split:")
    print(f"  {'split':<8} {'small':>8} {'large':>8} {'total':>8}")
    for split in ("train", "val", "test", "unknown"):
        if split not in per_split:
            continue
        c = per_split[split]
        print(f"  {split:<8} {c[0]:>8} {c[1]:>8} {sum(c.values()):>8}")


if __name__ == "__main__":
    main()
