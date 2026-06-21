"""把 AISD 的 DICOM 序列 + PNG 掩膜切片转成对齐的 3D nii.gz。

仓库的 Task402_AIS.py 假设每个病人已有 CT.nii.gz / GT_hard*.nii.gz，本脚本负责
从原始 DICOM + PNG 掩膜生成它们。

目录假设（按你的实际结构用 --id-regex / 目录布局调整）：
    <dicom_root>/<pid>/*.dcm
    <mask_root>/<pid>/*.png
输出：
    <out_root>/<pid>/CT.nii.gz
    <out_root>/<pid>/GT_hard.nii.gz

用法：
    python reproduce/aisd_dicom_to_nii.py \
        --dicom-root /path/to/aisd/dicom \
        --mask-root  /path/to/aisd/mask \
        --out-root   /path/to/AISD_nii \
        --lesion-values 2 3 4 5
"""
import argparse
import glob
import os
import re

import numpy as np
import SimpleITK as sitk
from PIL import Image


def natural_key(p):
    s = os.path.basename(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_dicom_series(folder):
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(folder)
    if ids:  # 标准 DICOM 序列
        files = reader.GetGDCMSeriesFileNames(folder, ids[0])
    else:    # 退化为按文件名自然排序
        files = sorted(glob.glob(os.path.join(folder, "*.dcm")), key=natural_key)
    reader.SetFileNames(files)
    return reader.Execute(), len(files)


def stack_masks(folder, lesion_values, n_expected, flip_z):
    files = sorted(glob.glob(os.path.join(folder, "*.png")), key=natural_key)
    if not files:
        return None
    slices = []
    for f in files:
        arr = np.array(Image.open(f).convert("L"))
        if lesion_values:
            bin_ = np.isin(arr, np.asarray(lesion_values)).astype(np.uint8)
        else:
            bin_ = (arr > 0).astype(np.uint8)
        slices.append(bin_)
    vol = np.stack(slices, axis=0)          # (Z, H, W)
    if flip_z:
        vol = vol[::-1]
    if n_expected and vol.shape[0] != n_expected:
        print(f"  [warn] {folder}: 掩膜层数 {vol.shape[0]} != CT 层数 {n_expected}")
    return vol


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lesion-values", type=int, nargs="*", default=[2, 3, 4, 5],
                    help="掩膜中算作急性病灶的取值；留空表示所有非零。")
    ap.add_argument("--flip-z", action="store_true",
                    help="若掩膜层序与 DICOM 相反则加此项。")
    args = ap.parse_args()

    pids = sorted(d for d in os.listdir(args.dicom_root)
                  if os.path.isdir(os.path.join(args.dicom_root, d)))
    print(f"发现 {len(pids)} 个病人。")
    for pid in pids:
        ct, n = read_dicom_series(os.path.join(args.dicom_root, pid))
        out_dir = os.path.join(args.out_root, pid)
        os.makedirs(out_dir, exist_ok=True)
        sitk.WriteImage(ct, os.path.join(out_dir, "CT.nii.gz"))

        mdir = os.path.join(args.mask_root, pid)
        vol = stack_masks(mdir, args.lesion_values, n, args.flip_z) if os.path.isdir(mdir) else None
        if vol is None:                       # 无标注则全 0
            vol = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.uint8)
        seg = sitk.GetImageFromArray(vol.astype(np.uint8))
        seg.CopyInformation(ct)               # 对齐 spacing/origin/direction
        sitk.WriteImage(seg, os.path.join(out_dir, "GT_hard.nii.gz"))
        print(f"  {pid}: CT {sitk.GetArrayFromImage(ct).shape}  写出完成")


if __name__ == "__main__":
    main()
