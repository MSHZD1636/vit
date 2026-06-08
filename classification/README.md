# AISD 卒中切片三分类 (no_stroke / small_lesion / large_lesion)

在原 Cl-SegNet **分割**仓库基础上新增的 **2D 切片级分类**管线。
与分割任务的关系和设计取舍：

- **复用程度**：原模型是 3D 分割编解码网络（`Coformer3D` + `Generic_UNet`），
  其编码器理论上可作分类骨干，但整套 nnU-Net 训练/数据管线是为分割写死的。
  本管线按需求采用**切片级 2D + 轻量标准 CNN（ResNet-18）**，自包含、易训练，
  不依赖 nnU-Net / ClSeg 的训练框架。残差块风格参照仓库内
  `ClSeg/network_architecture/ISNet/resnet18.py`，并加上全局池化 + 全连接分类头。
- **为什么切片级**：AISD 是急性缺血性卒中数据集，**每个病人都有病灶**，
  因此“无卒中”类在患者级别几乎无样本；而在切片级别，大量 PNG 切片掩膜为空，
  三类样本都充足，也天然契合下载到的 PNG 图像 / 掩膜数据。

## 三个类别

| label | 含义 | 判定规则 |
|------|------|---------|
| 0 | no_stroke   | 该切片掩膜病灶像素数 == 0 |
| 1 | small_lesion | 0 < 病灶像素数 <= 阈值 |
| 2 | large_lesion | 病灶像素数 > 阈值 |

> AISD 掩膜为多值标注（1=陈旧梗死, 2=清晰急性梗死, 3=模糊急性梗死,
> 4=不可见急性梗死, 5=梗死）。默认把**所有非零像素**视为病灶；
> 如需忽略陈旧梗死，加 `--lesion-values 2 3 4 5`。

## 依赖

```
torch  torchvision  numpy  pillow
```

## 用法

### 1. 生成标签表（从掩膜自动派生 + 沿用患者级划分）

`splits_final.pkl` 提供患者级 train/val/test 划分，确保同一病人的切片不跨集，避免泄漏。

```bash
python -m classification.build_labels \
    --images-dir /path/to/aisd/image \
    --masks-dir  /path/to/aisd/mask \
    --splits-pkl data/splits_final.pkl \
    --out-csv    classification/labels.csv \
    --auto-threshold          # 用 train 集病灶面积中位数自动定 small/large 边界
```

输出 `labels.csv`（列：`patient_id,split,image_path,mask_path,lesion_area,label`），
并打印各 split 的三类样本分布，便于检查类别不平衡。

阈值二选一：
- `--auto-threshold`：以训练集中“有病灶切片”的病灶面积**中位数**为 small/large 分界（推荐，数据自适应）。
- `--small-max-area N`：手动指定像素阈值（默认 500）。

> 切片与掩膜按相对路径/文件名配对，患者 ID 默认用正则 `(\d{7})` 从路径中提取
> （与 splits 里的 7 位 ID 对应）。若你的目录命名不同，用 `--id-regex` 调整。

### 2. 训练

```bash
python -m classification.train \
    --labels-csv classification/labels.csv \
    --out-dir    classification/runs/resnet18 \
    --epochs 60 --batch-size 64 --image-size 256 --amp --gpu 0
```

- 默认按类别频率反比加权交叉熵，缓解类别不平衡（`--no-class-weights` 关闭）。
- 按验证集 **macro-F1** 保存 `best.pth`，同时保存 `last.pth`、`history.json`、`val_report.txt`。

### 3. 测试评估

```bash
python -m classification.evaluate \
    --labels-csv classification/labels.csv \
    --checkpoint classification/runs/resnet18/best.pth \
    --split test \
    --save-report classification/runs/resnet18/test_report.txt
```

输出每类 precision/recall/F1、总体 accuracy、macro-F1 和混淆矩阵。

## 文件说明

| 文件 | 作用 |
|------|------|
| `build_labels.py` | 扫描图像/掩膜，按病灶面积生成三分类标签 CSV |
| `dataset.py` | 读取 CSV 的 `Dataset`，灰度切片 + 轻量增广 + 类别权重 |
| `model.py` | 自包含 ResNet-18 分类器（1 通道输入，3 类输出） |
| `train.py` | 训练循环（AdamW + 余弦退火 + AMP + 加权 CE） |
| `evaluate.py` | 在指定 split 上评估并输出报告 |
| `metrics.py` | 混淆矩阵 / 每类 P-R-F1 / macro-F1（无需 sklearn） |

## 后续可选扩展

- 若坚持**复用原论文的 CNN-Transformer 混合编码器**做骨干，可把 `model.py`
  换成 `ClSeg.network_architecture.cl_seg_aisd.Generic_UNet` 的编码器部分
  （`trans_encoder` + `conv_blocks_context` + 瓶颈 `fuse`）再接全局池化 + FC。
  代价是需安装 nnU-Net/ClSeg 环境且为 3D 结构（需改 2D 或堆叠切片成体数据）。
- 可改为患者级二/三分类：按全例病灶总体积聚合切片标签。
