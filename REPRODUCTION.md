# Cl-SegNet 急性缺血性卒中分割 —— AISD 数据集完整复现指南

本文档给出从 **DICOM 原始数据** 到 **最终病灶掩膜叠加可视化** 的全流程复现步骤，
对应论文 *"Hybrid CNN-Transformer Network with Circular Feature Interaction for
Acute Ischemic Stroke Lesion Segmentation on Non-contrast CT Scans"*（Cl-SegNet）。

整个项目基于 **nnU-Net** 框架，核心网络 = `Coformer3D`（CNN-Transformer 混合编码器）
+ `Generic_UNet` 编解码器 + `Circle` 圆形特征交互。

> 提示：本仓库内还有两个**分类**子模块（`classification/` 2D 切片三分类、
> `classification_3d/` 3D 大/小病灶二分类），与本分割复现相互独立，互不影响。

---

## 0. 数据流总览

```
AISD 原始数据                         本文档脚本                 nnU-Net 标准流程
─────────────                       ───────────              ──────────────────
DICOM 序列 (每病人一个文件夹)  ──┐
PNG 掩膜切片 (每病人一组)      ──┴─► aisd_dicom_to_nii.py ─► <pid>/CT.nii.gz
                                                            <pid>/GT_hard.nii.gz
                                                                   │
                                          Task402_AIS.py ◄─────────┘
                                                 │
                                                 ▼
                                   nnUNet_raw_data/Task402_AIS/{imagesTr,labelsTr}
                                                 │  nnUNet_plan_and_preprocess
                                                 ▼
                                   nnUNet_preprocessed/Task402_AIS/  (+ splits_final.pkl)
                                                 │  run_training.py (训练)
                                                 ▼
                                   RESULTS_FOLDER/.../AISD_IN_LeakyReLU/  (模型权重)
                                                 │  run_training.py -val (推理)
                                                 ▼
                                   validation_output/<pid>.nii.gz  (预测掩膜)
                                                 │  overlay_seg.py
                                                 ▼
                                   叠加可视化 PNG（红色覆盖病灶区域）
```

---

## 1. 环境安装

论文环境：CUDA 11.4 / Python 3.8 / PyTorch 1.11 / Torchvision 0.12 /
batchgenerators 0.21 / SimpleITK 2.1.1 / scipy 1.8。

```bash
conda create -n clseg python=3.8 -y
conda activate clseg

# 按你的 CUDA 安装 PyTorch（示例为 CUDA 11.4 对应的 1.11）
pip install torch==1.11.0 torchvision==0.12.0

# 安装 nnU-Net（本仓库内的定制版）和 ClSeg
cd nnUNet && pip install -e . && cd ..
cd ClSeg_package && pip install -e . && cd ..

# 其余依赖
pip install SimpleITK==2.1.1 scipy==1.8.0 batchgenerators==0.21 \
            einops timm pydicom pillow matplotlib scikit-learn nibabel
```

---

## 2. 配置路径（必做）

nnU-Net 通过环境变量定位数据，但本仓库在 `nnUNet/nnunet/paths.py` 第 28–30 行
**硬编码**了路径，必须改成你自己的目录：

```python
# nnUNet/nnunet/paths.py  (28-30 行)
os.environ['nnUNet_raw_data_base'] = "/your/path/nnUNet_raw_data_base"
os.environ['nnUNet_preprocessed']  = "/your/path/nnUNet_preprocessed"
os.environ['RESULTS_FOLDER']       = "/your/path/results"
```

三个目录含义：
- `nnUNet_raw_data_base`：原始（转换后）数据集根目录。
- `nnUNet_preprocessed`：预处理结果。
- `RESULTS_FOLDER`：训练得到的模型与日志。

新建目录：

```bash
mkdir -p /your/path/nnUNet_raw_data_base/nnUNet_raw_data
mkdir -p /your/path/nnUNet_preprocessed
mkdir -p /your/path/results
```

---

## 3. 数据准备

### 3.1 AISD 原始数据

从 https://github.com/griffinliang/aisd 下载，你已获取三类数据：
- **DICOM**：每个病人一个文件夹，内含该次 NCCT 扫描的全部 `.dcm` 切片。
- **PNG 图像**：每个切片一张窗宽窗位后的灰度图（仅用于查看/分类，分割训练用 DICOM）。
- **PNG 掩膜**：每个切片一张标注图，**多值**：
  `1=陈旧梗死, 2=清晰急性梗死, 3=模糊急性梗死, 4=不可见急性梗死, 5=梗死`。

> 急性缺血灶分割的硬标签 `GT_hard` 应只包含**急性**病灶。本文默认把掩膜值
> `{2,3,4,5}` 视为病灶（排除 1 陈旧梗死），二值化为 1。可按需调整。

### 3.2 DICOM → CT.nii.gz，PNG 掩膜 → GT_hard.nii.gz

仓库未提供该转换脚本（`Task402_AIS.py` 假设 nii 已存在）。把下面脚本存为
`tools/aisd_dicom_to_nii.py` 并运行。它会为每个病人生成
`<out>/<pid>/CT.nii.gz` 与 `<out>/<pid>/GT_hard.nii.gz`，几何信息对齐。

```python
# tools/aisd_dicom_to_nii.py
"""把 AISD 的 DICOM 序列 + PNG 掩膜切片转成对齐的 3D nii.gz。

目录假设（按你的实际结构用 --id-regex / --dicom-glob / --mask-glob 调整）：
    <dicom_root>/<pid>/*.dcm
    <mask_root>/<pid>/*.png
输出：
    <out_root>/<pid>/CT.nii.gz
    <out_root>/<pid>/GT_hard.nii.gz
"""
import argparse, os, re, glob
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
    ap = argparse.ArgumentParser()
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
```

运行：

```bash
python tools/aisd_dicom_to_nii.py \
    --dicom-root /path/to/aisd/dicom \
    --mask-root  /path/to/aisd/mask \
    --out-root   /path/to/AISD_nii \
    --lesion-values 2 3 4 5
```

> 若发现掩膜与 CT 上下颠倒（病灶位置对不上），加 `--flip-z` 重跑。
> 建议先抽 1–2 例用 ITK-SNAP / 3D Slicer 打开 CT.nii.gz + GT_hard.nii.gz 核对对齐。

### 3.3 转成 nnU-Net 数据集（Task402）

编辑 `nnUNet/nnunet/dataset_conversion/Task402_AIS.py`：
- 第 69 行 `train_dir` 改成上一步的 `--out-root`（即 `/path/to/AISD_nii/`）。
- 第 71 行 `output_folder` 改成
  `<nnUNet_raw_data_base>/nnUNet_raw_data/Task402_AIS`。

该脚本会遍历 `train_dir/<pid>/CT.nii.gz` 与 `GT_hard*.nii.gz`，复制为
`imagesTr/<pid>_0000.nii.gz`、`labelsTr/<pid>.nii.gz`，并生成 `dataset.json`
（单模态 CT、标签 0/1）与一份随机 `splits_final.pkl`。运行：

```bash
python nnUNet/nnunet/dataset_conversion/Task402_AIS.py
```

> 强度归一化：`Task402_AIS.py` 内含 `zscore_norm`（脑窗 `[0,60] HU`）但**默认未应用**，
> CT 保持原始 HU，由 nnU-Net 的 CT 归一化（前景分位裁剪 + z-score）统一处理。
> 若要严格复现论文的 `[0,60]` 脑窗，可在 3.2 写出 CT 前先做窗位裁剪。

---

## 4. nnU-Net 预处理（计划 + 预处理）

```bash
nnUNet_plan_and_preprocess -t 402 --verify_dataset_integrity
```

完成后在 `nnUNet_preprocessed/Task402_AIS/` 下生成 `nnUNetPlansv2.1_plans_3D.pkl`
等计划文件和预处理数据（3d_fullres）。

---

## 5. 使用论文的数据划分（推荐）

仓库根目录 `data/splits_final.pkl` 是论文的固定划分：
**train 300 / val 41 / test 52（共 393）**。用它覆盖自动生成的随机划分：

```bash
cp data/splits_final.pkl /your/path/nnUNet_preprocessed/Task402_AIS/splits_final.pkl
```

> ⚠️ **划分与防泄漏说明**：`run_training.py` 默认 `-fold all`，nnU-Net 在 `all` 折下
> 会用**全部**已预处理样本训练。若要严格按 train/val/test 评估、避免测试集泄漏，
> 推荐的干净做法二选一：
> 1. **仅预处理 train+val**：3.3 时只把 train+val 的 341 例放入 `imagesTr/labelsTr`，
>    把 52 例 test 放入 `imagesTs/`（仅 CT，不含标签），训练后用第 7.2 节
>    `nnUNet_predict` 单独推理 test 集再评估；或
> 2. 用整数折 + 标准 5 折 `splits_final.pkl`（train/val 列表），而非 `all`。
>
> `test` 的 52 个病人 ID 已列在 `Task402_AIS.py` 顶部 `test_ids`，可据此分目录。

---

## 6. 训练

```bash
cd ClSeg_package/ClSeg/run
python run_training.py \
    -network_trainer nnUNetTrainerV2_AISD \
    -gpu 0 \
    -task 402 \
    -outpath AISD
```

关键设置（来自 `nnUNetTrainerV2_AISD`）：
- 网络：`3d_fullres`；归一化 `-norm_cfg IN`（默认）；激活 `LeakyReLU`。
- 训练轮数 `max_num_epochs = 160`，初始 `lr = 1e-2`，`weight_decay = 3e-5`，深监督。
- 输出目录：`RESULTS_FOLDER/3d_fullres_<plans>/Task402_AIS/AISD_IN_LeakyReLU/`
  （`-outpath` 会与 `norm_cfg_activation` 拼接）。

> 断点续训加 `-c`；显存不足可在 `nnUNetTrainerV2_AISD.initialize` 里调小 batch，
> 或改 `-norm_cfg BN`（脚本里 BN 会把 batch_size 设为 8）。

> 也可直接用论文提供的预训练权重（README 第 2.1 节，百度网盘，密码 `4phx`），
> 放到上述输出目录后跳过训练，直接做第 7 节推理。

---

## 7. 推理（生成预测掩膜 nii.gz）

### 7.1 在验证/测试集上推理（复现论文评估）

```bash
cd ClSeg_package/ClSeg/run
python run_training.py \
    -network_trainer nnUNetTrainerV2_AISD \
    -gpu 0 \
    -task 402 \
    -outpath AISD \
    -val --val_folder validation_output
```

预测掩膜保存为 `.../AISD_IN_LeakyReLU/validation_output/<pid>.nii.gz`，
同目录的 `summary.json` 给出 **Dice** 等指标（自动评估）。

### 7.2 对新数据 / 测试集单独推理

```bash
nnUNet_predict \
    -i /your/path/nnUNet_raw_data_base/nnUNet_raw_data/Task402_AIS/imagesTs \
    -o /your/path/predictions_test \
    -t 402 -m 3d_fullres \
    -tr nnUNetTrainerV2_AISD \
    -f all
```

> 输入文件名须符合 nnU-Net 规范：`<caseid>_0000.nii.gz`（单模态）。

---

## 8. 定量评估

- 7.1 流程会在 `validation_output/summary.json` 写入每例及平均 Dice。
- 对 7.2 的独立预测，可用 nnU-Net 评估工具对比预测与 GT：

```bash
nnUNet_evaluate_folder \
    -ref /path/to/test_GT_labels \
    -pred /your/path/predictions_test \
    -l 1
```

输出含 Dice、Jaccard、Hausdorff 等（论文主要报告 Dice / HD95 / ASSD 等病灶分割指标）。

---

## 9. 最终目标：病灶掩膜叠加可视化

把预测掩膜以**半透明红色覆盖在 CT 病灶区域**上，逐层输出 PNG。
将下面脚本存为 `tools/overlay_seg.py`。

```python
# tools/overlay_seg.py
"""把分割掩膜叠加到 CT 上，输出彩色 PNG（红=预测，绿色描边=GT 可选）。"""
import argparse, os
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
    red = np.zeros_like(rgb); red[..., 0] = 255
    m = pred.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * red[m]      # 红色覆盖预测病灶
    if gt is not None:                                   # GT 边界画绿线
        from scipy.ndimage import binary_erosion
        edge = gt.astype(bool) & ~binary_erosion(gt.astype(bool))
        rgb[edge] = [0, 255, 0]
    return rgb.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True, help="CT.nii.gz 或 imagesTs 的 <id>_0000.nii.gz")
    ap.add_argument("--pred", required=True, help="预测掩膜 <id>.nii.gz")
    ap.add_argument("--gt", default=None, help="可选：GT_hard.nii.gz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wl", type=float, default=40); ap.add_argument("--ww", type=float, default=80)
    ap.add_argument("--only-lesion", action="store_true", help="只导出含预测病灶的层")
    args = ap.parse_args()

    ct = sitk.GetArrayFromImage(sitk.ReadImage(args.ct))       # (Z,H,W) HU
    pr = sitk.GetArrayFromImage(sitk.ReadImage(args.pred))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(args.gt)) if args.gt else None
    os.makedirs(args.out_dir, exist_ok=True)

    for z in range(ct.shape[0]):
        if args.only_lesion and pr[z].sum() == 0:
            continue
        g = window(ct[z], args.wl, args.ww)
        rgb = overlay(g, pr[z], None if gt is None else gt[z])
        Image.fromarray(rgb).save(os.path.join(args.out_dir, f"slice_{z:03d}.png"))
    print(f"叠加图已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
```

运行（对某一例）：

```bash
python tools/overlay_seg.py \
    --ct   /path/to/AISD_nii/0073410/CT.nii.gz \
    --pred /your/path/.../validation_output/0073410.nii.gz \
    --gt   /path/to/AISD_nii/0073410/GT_hard.nii.gz \
    --out-dir /your/path/overlay/0073410 \
    --only-lesion
```

得到一组 PNG：**灰度 CT 上红色半透明区域即模型分割出的急性缺血病灶**，
绿色描边为金标准（便于对比）。这就是"掩膜覆盖病灶区域的图像输出"。

---

## 10. 常见问题

| 现象 | 处理 |
|------|------|
| `nnUNet_*` 命令找不到 | 确认 `pip install -e .` 成功、conda 环境已激活 |
| 路径相关报错 / 找不到数据 | 检查 `paths.py` 三个路径是否改对、目录是否已建 |
| 掩膜与 CT 对不齐 / 上下颠倒 | `aisd_dicom_to_nii.py` 加 `--flip-z`；核对层数一致 |
| 显存 OOM | 调小 batch、用 `-norm_cfg BN`、或减小 patch（改 plans） |
| 想排除陈旧梗死 | 转换时 `--lesion-values 2 3 4 5`（默认即排除 1） |
| 测试集泄漏疑虑 | 按第 5 节"干净做法"，test 单独放 imagesTs 并用 7.2 推理 |

---

## 附：流程速查

```bash
# 1. 安装
cd nnUNet && pip install -e . && cd ../ClSeg_package && pip install -e . && cd ..
# 2. 改 nnUNet/nnunet/paths.py 的三个路径
# 3. DICOM/PNG -> nii
python tools/aisd_dicom_to_nii.py --dicom-root <dcm> --mask-root <mask> --out-root <AISD_nii>
#    改 Task402_AIS.py 的 train_dir / output_folder，然后
python nnUNet/nnunet/dataset_conversion/Task402_AIS.py
# 4. 预处理
nnUNet_plan_and_preprocess -t 402 --verify_dataset_integrity
# 5. 用论文划分
cp data/splits_final.pkl <nnUNet_preprocessed>/Task402_AIS/splits_final.pkl
# 6. 训练
cd ClSeg_package/ClSeg/run
python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu 0 -task 402 -outpath AISD
# 7. 推理
python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu 0 -task 402 -outpath AISD -val --val_folder validation_output
# 8. 叠加可视化
python tools/overlay_seg.py --ct <CT.nii.gz> --pred <pred.nii.gz> --gt <GT_hard.nii.gz> --out-dir <overlay> --only-lesion
```
