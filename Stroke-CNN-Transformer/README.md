# Stroke-CNN-Transformer

以论文基线 **nnU-Net / Cl-SegNet（Coformer3D 混合 CNN-Transformer）** 为基础，构建的
急性缺血性卒中（AIS）NCCT 3D 病灶分割网络。代码在 `model.py`，**自包含、仅依赖 `torch`**，
不引入 nnU-Net / ClSeg。

体素级输出：最终经 `Sigmoid` 生成取值 `[0,1]` 的病灶概率图。

```python
from importlib.machinery import SourceFileLoader
m = SourceFileLoader("sct", "Stroke-CNN-Transformer/model.py").load_module()
net = m.build_model(in_channels=1, num_classes=1)
# 输入 (B,1,D,H,W)，要求 D 可被 4 整除、H/W 可被 32 整除
y = net(torch.randn(1, 1, 16, 160, 160))   # -> (1,1,16,160,160)，值域[0,1]
```
> 文件夹名含连字符不便做 Python 包导入，故用上面的 `SourceFileLoader` 或直接
> `python Stroke-CNN-Transformer/model.py` 跑内置 shape 自测。

---

## 1. 整体结构

```
输入 CT  (B,1,D,H,W)
    │
    ├───────────────┐                          双编码器并行
    ▼               ▼
 CNN 分支        Transformer 分支
(局部细节)        (全局上下文)
 4 个 stage        4 个 stage (Hybridformer)
    │               │
    └──► GRN 门控融合 (逐 stage) ◄──┘   f1,f2,f3,f4
              │
       Stage2/3/4 加 MLBDA (双侧差异感知)
              │
        ┌─────┴───────── U-Net 解码器 ─────────┐
        │ up4→cat f3→dec3→BDL                   │
        │ up3→cat f2→dec2                       │
        │ up2→cat f1→dec1                       │
        │ up1→dec0                              │
        └──► 1×1×1 Conv → Sigmoid → 概率图[0,1] ┘
```

四个 stage 的空间分辨率（in-plane）：**H/4 → H/8 → H/16 → H/32**，
通道数 `dims = (64, 128, 256, 320)`。下采样步长（D,H,W）：
`(1,4,4) → (1,2,2) → (2,2,2) → (2,2,2)`，两个分支共用同一调度，保证逐 stage 形状一致、可融合。

以 `16×160×160` 输入为例的各 stage 形状：

| stage | 分辨率 | 形状 (C,D,H,W) |
|------|--------|----------------|
| f1 | H/4  | 64×16×40×40 |
| f2 | H/8  | 128×16×20×20 |
| f3 | H/16 | 256×8×10×10 |
| f4 | H/32 | 320×4×5×5 |

---

## 2. 保留的四项核心优势

### 2.1 双编码器并行结构
`CNNBranch` 与 `TransformerBranch` 对同一输入**并行**编码：
- **CNN 分支**：每个 stage = 步长下采样 + 残差 `ConvBlock3D`（两层 Conv-IN-LeakyReLU），
  擅长**局部细节**。
- **Transformer 分支**：`PatchEmbed` + `HybridformerBlock` 堆叠，擅长**全局上下文**。

两分支在每个 stage 通过 GRN 融合，优势互补。

### 2.2 4-stage 层次化下采样
见上表，分辨率依次降到 H/4, H/8, H/16, H/32，在不同尺度提取特征。

### 2.3 门控残差网络 GRN（特征传递/融合）
`GRN(a, c)`：以 CNN 分支 `a`（局部）为主、Transformer 分支 `c`（全局）为上下文：

```
eta = ELU(W_a·a + W_c·c)
glu = sigmoid(W_g·eta) ⊙ (W_v·eta)      # 门控：自适应选择"直接传递" vs "非线性变换"
out = InstanceNorm(a + glu)
```

门 `sigmoid(W_g·eta)` 在 0~1 间自适应决定让多少非线性变换通过、多少走残差直传，
提高特征流动灵活性。每个 stage 一个 GRN。

### 2.4 高效 Hybridformer Block
`HybridformerBlock`：用**平均池化的 token mixer** 取代标准 Transformer 的 Q-K-V 自注意力，
大幅降低计算量，适合小规模医学数据：

```
x = x + TokenMixer(LN(x))      # 池化式注意力（LTB 或 MSPA）
x = x + DWConv7³(x)            # 深度可分离卷积（局部）
x = x + MLP(LN(x))            # 1×1×1 逐点 MLP
```
`LTBAttention` = 单尺度平均池化（PoolFormer 风格，`pool(x)-x`）。

---

## 3. 两项关键改进

### 3.1 MLBDA — 多层级双侧差异感知（Stage 2/3/4）
大脑近似左右对称，卒中会打破对称性。`MLBDA` 把特征沿**左右轴翻转**取差异，
再用**轻量 SE** 把这种不对称转成通道再加权：

```
x_flip = flip(x, 左右轴)
attn   = SE(x - x_flip)        # 轻量 SE(ratio=8)，从不对称中学注意力
out    = x + x * attn          # 残差调制
```
在 Stage2、Stage3、Stage4 三个层级各加一个，构成"多层级"差异感知。

### 3.2 MSPA — 多尺度金字塔注意力（Stage 3/4，替换 LTB）
在较深的 Stage3、Stage4，用 `MSPA` 替换单尺度 LTB token mixer：
三个**并行**平均池化分支（窗口 3/5/7）建模不同尺度的病灶，再 1×1×1 卷积融合：

```
outs = [AvgPool_k(x) for k in (3,5,7)]
out  = Conv1×1(concat(outs)) - x
```
浅层（Stage1/2）保留 LTB，深层（Stage3/4）用 MSPA，兼顾效率与多尺度建模。

---

## 4. 解码器与跳跃连接

U-Net 风格，从最底层 Stage4（H/32）开始用**转置卷积**逐级上采样，
每级与对应编码器 stage 做**跳跃连接**（concat 融合）：

| 解码步 | 上采样 (转置卷积步长) | 跳跃连接 | 输出 |
|------|----------------------|---------|------|
| Stage4→3 | (2,2,2) 320→256 | cat f3 | dec3 → **BDL** |
| Stage3→2 | (2,2,2) 256→128 | cat f2 | dec2 |
| Stage2→1 | (1,2,2) 128→64  | cat f1 | dec1 |
| Stage1→full | (1,4,4) 64→32 | — | dec0 |

- **BDL（双侧差异学习）**：在解码器底层（Stage4→Stage3）加入，作为最终精细化差异建模，
  与编码器侧 MLBDA 共同构成完整的双侧差异感知体系：
  `res=x; x = x * SE(x - flip(x)); out = res + x`。
- **输出头**：末端 `1×1×1` 卷积映射到单通道 + `Sigmoid` → 体素级病灶概率图 `[0,1]`。

---

## 5. 模块 → 代码对照

| 论文要素 | 代码（`model.py`） |
|---------|-------------------|
| CNN 分支 | `CNNBranch` / `ConvBlock3D` |
| Transformer 分支 | `TransformerBranch` / `HybridformerBlock` |
| 高效注意力（池化替代 QKV） | `LTBAttention` |
| GRN 门控特征传递 | `GRN` |
| MLBDA 多层级双侧差异感知 | `MLBDA`（stage 2/3/4） |
| MSPA 多尺度金字塔注意力 | `MSPA`（stage 3/4 替换 LTB） |
| 解码器 BDL | `BDL` |
| 跳跃连接 + 转置卷积上采样 | `StrokeCNNTransformer.forward` 解码段 |
| Sigmoid 概率输出 | `head` + `torch.sigmoid` |

---

## 6. 训练建议（与 AIS 分割任务衔接）

- **输入尺寸**：`D` 被 4 整除、`H/W` 被 32 整除（如 `16×160×160`）。
- **损失**：二值分割推荐 `Dice + BCE`（病灶占比小，Dice 缓解类别不平衡）。
- **可选深监督**：`build_model(deep_supervision=True)` 时训练阶段额外返回 `dec2/dec3`
  的辅助概率图，可对低分辨率 GT 加权监督。
- **数据/预处理/评估**：直接复用本仓库 `reproduce/` 的 nnU-Net 流程
  （DICOM→nii、`Task402`、`nnUNet_plan_and_preprocess`、Dice 评估、`overlay_seg.py` 叠加可视化）。
  若要纳入 nnU-Net 训练器，可将本模型按 `nnUNetTrainerV2_AISD.initialize_network`
  的方式替换 `self.network`（注意 nnU-Net 期望多类 softmax 输出时需适配，单通道
  Sigmoid 适合二值分割的自定义训练循环）。

## 7. 参数量与自测

```bash
python Stroke-CNN-Transformer/model.py
# 打印参数量(M)、输入/输出形状；输出应为 (1,1,16,160,160)，值域[0,1]
```

依赖：仅 `torch`（建议 ≥1.11，与论文环境一致）。
