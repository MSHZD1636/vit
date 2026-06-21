"""把分割掩膜叠加到 CT 上，输出彩色 PNG（红=预测病灶，绿色描边=GT 可选）。

这是复现的最后一步：实现"掩膜覆盖病灶区域的图像输出"。

用法：
    python reproduce/overlay_seg.py \
        --ct   /path/to/AISD_nii/<pid>/CT.nii.gz \
        --pred /path/to/results/.../validation_output/<pid>.nii.gz \
        --gt   /path/to/AISD_nii/<pid>/GT_hard.nii.gz \
        --out-dir /path/to/overlay/<pid> \
        --only-lesion
"""
import argparse
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image


def window(ct_slice, wl=40, ww=80):
    """脑窗：窗位 40 / 窗宽 80（HU），映射到 0-255 灰度。"""
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    s = np.clip((ct_slice - lo) / (hi - lo), 0, 1)
    return (s * 255).astype(np.uint8)


def overlay(gray, pred, gt=None, alpha=0.45):
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    m = pred.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * red[m]      # 红色覆盖预测病灶
    if gt is not None:                                   # GT 边界画绿线
        from scipy.ndimage import binary_erosion
        gt_b = gt.astype(bool)
        edge = gt_b & ~binary_erosion(gt_b)
        rgb[edge] = [0, 255, 0]
    return rgb.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, help="CT.nii.gz 或 imagesTs 的 <id>_0000.nii.gz")
    ap.add_argument("--pred", required=True, help="预测掩膜 <id>.nii.gz")
    ap.add_argument("--gt", default=None, help="可选：GT_hard.nii.gz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wl", type=float, default=40)
    ap.add_argument("--ww", type=float, default=80)
    ap.add_argument("--only-lesion", action="store_true", help="只导出含预测病灶的层")
    args = ap.parse_args()

    ct = sitk.GetArrayFromImage(sitk.ReadImage(args.ct))       # (Z,H,W) HU
    pr = sitk.GetArrayFromImage(sitk.ReadImage(args.pred))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(args.gt)) if args.gt else None
    os.makedirs(args.out_dir, exist_ok=True)

    n = 0
    for z in range(ct.shape[0]):
        if args.only_lesion and pr[z].sum() == 0:
            continue
        g = window(ct[z], args.wl, args.ww)
        rgb = overlay(g, pr[z], None if gt is None else gt[z])
        Image.fromarray(rgb).save(os.path.join(args.out_dir, f"slice_{z:03d}.png"))
        n += 1
    print(f"叠加图 {n} 张已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
