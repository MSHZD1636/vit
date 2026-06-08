# AISD 3D 病灶大小二分类（大病灶 vs 小病灶）— 复用 CNN-Transformer 混合编码器

在不改动 2D 切片三分类（`classification/`）的前提下，新增的 **3D 病灶大小二分类**管线。
**骨干网络直接复用论文 Cl-SegNet 的 CNN-Transformer 混合编码器 `Coformer3D`**
（`ClSeg/network_architecture/hybridformer.py`），取其多尺度特征金字塔，
做全局平均池化 + 拼接 + 全连接，输出 2 类。

## 两个类别

| label | 含义 |
|------|------|
| 0 | small_lesion（含病灶子体块，病灶体素量 <= 阈值） |
| 1 | large_lesion（含病灶子体块，病灶体素量 > 阈值） |

> **只在含病灶的子体块上做大小分类**，不含病灶的子体块直接丢弃（没有"无卒中"类）。

## 采样方式（3D）

把每个病人的切片按轴向排序，两种模式：

- **`slab`（默认）**：用大小 `--depth`、步长 `--stride` 的滑窗切成 3D 子体块，
  子体块病灶体素量 = 窗口内各层病灶像素之和。样本多、深度固定，适合训练。
- **`volume`**：每个病人一个整卷样本，病灶体素量 = 全例病灶体素总和（最贴近"病灶大小"的临床含义，但样本少、深度可变）。

病灶大/小的分界阈值二选一：
- `--auto-threshold`：以训练集**含病灶子体块病灶体素量的中位数**为分界（推荐，数据自适应、天然类别平衡）。
- `--small-max-voxels N`：手动指定体素阈值（默认 2000）。

> 注：slab 模式下，同一个大病灶在不同窗口里落入的体素量不同，部分边缘窗口会被判为
> "small"，这相当于一种数据扰动；若想要**干净的每病例大小标签**，用 `--mode volume`。

## 依赖

```
torch  einops  timm  numpy  pillow
```
> `Coformer3D` 只依赖 `torch/einops/timm`，不牵连 nnU-Net。
> `model_3d.py` 会自动把仓库内 `ClSeg_package` 加入 `sys.path`，**无需** `pip install -e`。

## 用法

### 1. 生成 3D 大小标签表

```bash
python -m classification_3d.build_labels_3d \
    --images-dir /path/to/aisd/image \
    --masks-dir  /path/to/aisd/mask \
    --splits-pkl data/splits_final.pkl \
    --out-csv    classification_3d/labels_3d.csv \
    --mode slab --depth 16 --stride 8 --auto-threshold
```

输出 `labels_3d.csv`（列：`patient_id,split,sample_id,slice_paths,lesion_voxels,label`），
并打印各 split 的 small/large 样本分布。`--depth` 建议用 4 的倍数（编码器在深度上会下采样 2 次）。

### 2. 训练（Coformer3D 骨干）

```bash
python -m classification_3d.train_3d \
    --labels-csv classification_3d/labels_3d.csv \
    --out-dir    classification_3d/runs/coformer \
    --depth 16 --image-size 160 --batch-size 4 --epochs 80 --amp --gpu 0
```

- AdamW + 余弦退火 + AMP，默认类别频率反比加权交叉熵。
- 按验证集 macro-F1 保存 `best.pth`，并写 `val_report.txt` / `history.json`。
- `--single-scale` 只用最深一层特征；默认用全部 4 个尺度拼接（更贴近完整编码器）。

### 3. 测试评估

```bash
python -m classification_3d.evaluate_3d \
    --labels-csv classification_3d/labels_3d.csv \
    --checkpoint classification_3d/runs/coformer/best.pth \
    --split test --depth 16 --image-size 160 \
    --save-report classification_3d/runs/coformer/test_report.txt
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `build_labels_3d.py` | 按病人分组、轴向排序，生成 slab/整卷的**大/小病灶**标签（复用 `classification.build_labels` 的工具函数） |
| `volume_dataset.py` | 把切片列表堆叠成 `(1, D, H, W)` 体数据，深度自适应 pad/采样 + 3D 翻转增广 |
| `model_3d.py` | `CoformerClassifier3D`：复用 `Coformer3D` 编码器 + 多尺度池化 + FC(2) |
| `train_3d.py` / `evaluate_3d.py` | 训练 / 评估（复用 `classification.metrics`） |

## 显存提示

3D + Transformer 较吃显存。若 OOM：调小 `--batch-size`、`--image-size`（如 128）、
`--depth`（如 8），或加 `--single-scale`。

## 与 2D 分类的关系

- `classification/`：2D 切片级 **三分类**（no/small/large），ResNet-18，**保持不变**。
- `classification_3d/`：3D 子体块级 **大/小病灶二分类**，复用论文混合编码器。
两者共享 `classification/metrics.py` 与 `classification/build_labels.py` 的工具函数，
以及同一份 `data/splits_final.pkl` 患者级划分。
