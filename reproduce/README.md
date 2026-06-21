# Cl-SegNet 急性缺血性卒中分割 —— AISD 数据集完整复现

本文件夹包含**复现所需的全部代码与步骤**，对应论文 *"Hybrid CNN-Transformer
Network with Circular Feature Interaction for Acute Ischemic Stroke Lesion
Segmentation on Non-contrast CT Scans"*（Cl-SegNet），从 **DICOM 原始数据** 一路到
**最终病灶掩膜叠加可视化**。

整个项目基于 **nnU-Net**：核心网络 = `Coformer3D`（CNN-Transformer 混合编码器）
+ `Generic_UNet` 编解码器 + `Circle` 圆形特征交互。

## 本文件夹内容

| 文件 | 作用 |
|------|------|
| `aisd_dicom_to_nii.py` | DICOM 序列 + PNG 多值掩膜 → 对齐的 `CT.nii.gz` / `GT_hard.nii.gz`（仓库缺失，必需） |
| `overlay_seg.py` | 把预测掩膜以半透明红色叠加到 CT 病灶区域，输出 PNG（复现的最终产物） |
| `run_pipeline.sh` | 一键流程速查脚本（改好路径后顺序执行各步） |
| `README.md` | 本说明 |

> 其余环节复用仓库已有代码：`nnUNet/`（含 `dataset_conversion/Task402_AIS.py`）、
> `ClSeg_package/`（含 `run/run_training.py`、`nnUNetTrainerV2_AISD`）、
> 根目录 `data/splits_final.pkl`（论文划分）。
>
> 本仓库还有两个**分类**子模块（`classification/`、`classification_3d/`），
> 与本分割复现相互独立。

---

## 0. 数据流总览

```
DICOM 序列 + PNG 掩膜  ─► reproduce/aisd_dicom_to_nii.py ─► <pid>/CT.nii.gz, GT_hard.nii.gz
                                                                  │
                                        Task402_AIS.py ◄──────────┘
                                              ▼
                        nnUNet_raw_data/Task402_AIS/{imagesTr,labelsTr}
                                              │  nnUNet_plan_and_preprocess
                                              ▼
                        nnUNet_preprocessed/Task402_AIS/ (+ splits_final.pkl)
                                              │  run_training.py（训练）
                                              ▼
                        RESULTS_FOLDER/.../AISD_IN_LeakyReLU/（模型）
                                              │  run_training.py -val（推理）
                                              ▼
                        validation_output/<pid>.nii.gz（预测掩膜）
                                              │  reproduce/overlay_seg.py
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
pip install torch==1.11.0 torchvision==0.12.0          # 按你的 CUDA 调整

cd nnUNet && pip install -e . && cd ..
cd ClSeg_package && pip install -e . && cd ..

pip install SimpleITK==2.1.1 scipy==1.8.0 batchgenerators==0.21 \
            einops timm pydicom pillow matplotlib scikit-learn nibabel
```

---

## 2. 配置路径（必做）

nnU-Net 在 `nnUNet/nnunet/paths.py` 第 28–30 行**硬编码**了路径，改成你自己的：

```python
os.environ['nnUNet_raw_data_base'] = "/your/path/nnUNet_raw_data_base"
os.environ['nnUNet_preprocessed']  = "/your/path/nnUNet_preprocessed"
os.environ['RESULTS_FOLDER']       = "/your/path/results"
```

```bash
mkdir -p /your/path/nnUNet_raw_data_base/nnUNet_raw_data
mkdir -p /your/path/nnUNet_preprocessed
mkdir -p /your/path/results
```

---

## 3. 数据准备

### 3.1 AISD 原始数据

从 https://github.com/griffinliang/aisd 下载：**DICOM**（每病人一个文件夹）、
**PNG 图像**（查看/分类用）、**PNG 掩膜**（多值：
`1=陈旧梗死, 2=清晰急性梗死, 3=模糊急性梗死, 4=不可见急性梗死, 5=梗死`）。

> 急性病灶分割的硬标签默认取掩膜值 `{2,3,4,5}`（排除 1 陈旧梗死）二值化为 1。

### 3.2 DICOM → CT.nii.gz，PNG 掩膜 → GT_hard.nii.gz

```bash
python reproduce/aisd_dicom_to_nii.py \
    --dicom-root /path/to/aisd/dicom \
    --mask-root  /path/to/aisd/mask \
    --out-root   /path/to/AISD_nii \
    --lesion-values 2 3 4 5
```

为每个病人生成 `<AISD_nii>/<pid>/CT.nii.gz` 与 `GT_hard.nii.gz`（几何对齐）。
若掩膜与 CT 上下颠倒，加 `--flip-z`。建议先抽 1–2 例用 ITK-SNAP / 3D Slicer 核对对齐。

### 3.3 转成 nnU-Net 数据集（Task402）

编辑 `nnUNet/nnunet/dataset_conversion/Task402_AIS.py`：
- 第 69 行 `train_dir` → 上一步的 `--out-root`（`/path/to/AISD_nii/`）；
- 第 71 行 `output_folder` → `<nnUNet_raw_data_base>/nnUNet_raw_data/Task402_AIS`。

```bash
python nnUNet/nnunet/dataset_conversion/Task402_AIS.py
```

生成 `imagesTr/<pid>_0000.nii.gz`、`labelsTr/<pid>.nii.gz`、`dataset.json`。

> 强度归一化：`Task402_AIS.py` 内含脑窗 `[0,60]` 的 `zscore_norm` 但默认未应用，
> CT 保持 HU 由 nnU-Net 的 CT 归一化统一处理；要严格复现 `[0,60]` 脑窗可在 3.2 写出前裁剪。

---

## 4. 预处理

```bash
nnUNet_plan_and_preprocess -t 402 --verify_dataset_integrity
```

---

## 5. 使用论文划分（推荐）

`data/splits_final.pkl` 是论文固定划分（train 300 / val 41 / test 52）：

```bash
cp data/splits_final.pkl /your/path/nnUNet_preprocessed/Task402_AIS/splits_final.pkl
```

> ⚠️ **防泄漏**：`run_training.py` 默认 `-fold all` 会用全部样本训练。要严格按
> train/val/test 评估，推荐：3.3 时只把 train+val（341 例）放入 `imagesTr/labelsTr`，
> 52 例 test 放入 `imagesTs/`（仅 CT），训练后用第 7.2 节单独推理 test。
> test 的 52 个病人 ID 见 `Task402_AIS.py` 顶部 `test_ids`。

---

## 6. 训练

```bash
cd ClSeg_package/ClSeg/run
python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu 0 -task 402 -outpath AISD
```

关键设置（`nnUNetTrainerV2_AISD`）：3d_fullres、`norm_cfg IN`、`LeakyReLU`、
`max_num_epochs=160`、`lr=1e-2`、`weight_decay=3e-5`、深监督。输出目录
`RESULTS_FOLDER/3d_fullres_<plans>/Task402_AIS/AISD_IN_LeakyReLU/`。

> 续训加 `-c`；OOM 可改 `-norm_cfg BN`（batch 设为 8）或调小 patch。
> 也可直接用论文预训练权重（主 README 第 2.1 节，百度网盘密码 `4phx`）跳过训练。

---

## 7. 推理（生成预测掩膜 nii.gz）

### 7.1 验证/测试集（复现评估）

```bash
cd ClSeg_package/ClSeg/run
python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu 0 -task 402 -outpath AISD \
    -val --val_folder validation_output
```

预测保存为 `.../AISD_IN_LeakyReLU/validation_output/<pid>.nii.gz`，
同目录 `summary.json` 给出 **Dice** 等指标。

### 7.2 对新数据 / 测试集单独推理

```bash
nnUNet_predict \
    -i <nnUNet_raw_data_base>/nnUNet_raw_data/Task402_AIS/imagesTs \
    -o /your/path/predictions_test \
    -t 402 -m 3d_fullres -tr nnUNetTrainerV2_AISD -f all
```

> 输入文件名须为 `<caseid>_0000.nii.gz`。

---

## 8. 定量评估

```bash
nnUNet_evaluate_folder -ref /path/to/test_GT_labels -pred /your/path/predictions_test -l 1
```

输出 Dice / Jaccard / Hausdorff 等。

---

## 9. 最终目标：病灶掩膜叠加可视化

```bash
python reproduce/overlay_seg.py \
    --ct   /path/to/AISD_nii/0073410/CT.nii.gz \
    --pred /your/path/.../validation_output/0073410.nii.gz \
    --gt   /path/to/AISD_nii/0073410/GT_hard.nii.gz \
    --out-dir /your/path/overlay/0073410 \
    --only-lesion
```

得到一组 PNG：**灰度 CT 上红色半透明区域 = 模型分割出的急性缺血病灶**，
绿色描边为金标准。这就是"掩膜覆盖病灶区域的图像输出"。

---

## 10. 一键速查

改好 `reproduce/run_pipeline.sh` 顶部的路径变量后：

```bash
bash reproduce/run_pipeline.sh
```

## 常见问题

| 现象 | 处理 |
|------|------|
| `nnUNet_*` 找不到 | 确认 `pip install -e .` 成功、环境已激活 |
| 找不到数据/路径报错 | 检查 `paths.py` 三路径、目录是否已建 |
| 掩膜与 CT 不齐/颠倒 | `aisd_dicom_to_nii.py` 加 `--flip-z`，核对层数一致 |
| 显存 OOM | 调小 batch、`-norm_cfg BN`、或减小 patch |
| 排除陈旧梗死 | 转换时 `--lesion-values 2 3 4 5`（默认即排除 1） |
| 测试集泄漏疑虑 | 按第 5 节，test 单独放 imagesTs 并用 7.2 推理 |
