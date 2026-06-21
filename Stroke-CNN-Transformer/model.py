"""Stroke-CNN-Transformer: a dual-encoder CNN-Transformer network for acute
ischemic stroke lesion segmentation on non-contrast CT (3D).

Built on the Cl-SegNet / Coformer3D baseline, this model keeps the four core
advantages of the hybrid CNN-Transformer design and adds two stroke-specific
improvements. It is self-contained (only depends on ``torch``); nothing is
imported from nnU-Net / ClSeg.

Core advantages kept
--------------------
1. Dual parallel encoders: a CNN branch (local detail) and a Transformer branch
   (global context) run side by side and are fused stage-by-stage.
2. 4-stage hierarchical downsampling to H/4, H/8, H/16, H/32.
3. Gated Residual Network (GRN) for adaptive feature passing / fusion.
4. Efficient Hybridformer block: average-pooling token mixer instead of Q-K-V.

Two improvements
----------------
A. MLBDA - Multi-Level Bilateral Difference Awareness at encoder Stage 2/3/4:
   lightweight SE on the left-right (hemisphere) difference.
B. MSPA - Multi-Scale Pyramid Attention at encoder Stage 3/4: three parallel
   pooling-attention branches, replacing the single-scale LTB token mixer.

Decoder
-------
U-Net style transposed-conv upsampling with skip connections; a Bilateral
Difference Learning (BDL) module at the bottom (Stage4 -> Stage3) stage; a final
1x1x1 conv + Sigmoid producing a voxel-wise lesion probability map in [0, 1].

See README.md for a full explanation.
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Basic blocks
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    """Stochastic depth (per-sample drop of residual branches)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor()


class LayerNorm3d(nn.Module):
    """Channels-first LayerNorm over the channel dimension of (B,C,D,H,W)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]


class ConvNormAct(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = nn.InstanceNorm3d(cout, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvBlock3D(nn.Module):
    """Two conv-norm-act layers with a residual connection (CNN-branch stage)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = ConvNormAct(cin, cout)
        self.conv2 = ConvNormAct(cout, cout)
        self.proj = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.proj(x)


class SEModule3D(nn.Module):
    """Squeeze-and-Excitation: returns channel gating weights (B,C,1,1,1)."""

    def __init__(self, dim, ratio=8):
        super().__init__()
        hidden = max(dim // ratio, 4)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.squeeze(x).view(b, c)
        return self.excite(y).view(b, c, 1, 1, 1)


# --------------------------------------------------------------------------- #
# Improvement A: Multi-Level Bilateral Difference Awareness (MLBDA)
# --------------------------------------------------------------------------- #
class MLBDA(nn.Module):
    """Lightweight bilateral (left-right) asymmetry awareness.

    The brain is roughly symmetric; a stroke breaks that symmetry. We flip the
    feature map along the left-right axis, take the difference, and use a
    lightweight SE block to turn that asymmetry into a channel re-weighting.
    Applied at encoder Stage 2/3/4.
    """

    def __init__(self, dim, lr_axis=-1, ratio=8):
        super().__init__()
        self.lr_axis = lr_axis
        self.se = SEModule3D(dim, ratio)

    def forward(self, x):
        x_flip = torch.flip(x, dims=[self.lr_axis])
        attn = self.se(x - x_flip)        # channel weights from the asymmetry
        return x + x * attn               # residual modulation


# --------------------------------------------------------------------------- #
# Bilateral Difference Learning (BDL) - decoder bottom refinement
# --------------------------------------------------------------------------- #
class BDL(nn.Module):
    """Stronger bilateral difference module used once in the decoder bottom.

    Mirrors the baseline's flip-SE interaction: modulate features by the SE of
    the left-right difference, then add back the residual.
    """

    def __init__(self, dim, lr_axis=-1, ratio=4):
        super().__init__()
        self.lr_axis = lr_axis
        self.se = SEModule3D(dim, ratio)

    def forward(self, x):
        res = x
        x_flip = torch.flip(x, dims=[self.lr_axis])
        x = x * self.se(x - x_flip)
        return res + x


# --------------------------------------------------------------------------- #
# Token mixers for the Hybridformer block
# --------------------------------------------------------------------------- #
class LTBAttention(nn.Module):
    """Local Token-mixing Block: average pooling replaces Q-K-V (single scale)."""

    def __init__(self, dim, window_size=3):
        super().__init__()
        self.pool = nn.AvgPool3d(window_size, stride=1, padding=window_size // 2)

    def forward(self, x):
        return self.pool(x) - x          # pooled context minus identity (PoolFormer style)


class MSPA(nn.Module):
    """Improvement B: Multi-Scale Pyramid Attention.

    Three parallel pooling-attention branches at different window sizes capture
    lesions of different sizes; their fused response replaces the single-scale
    LTB token mixer at Stage 3/4.
    """

    def __init__(self, dim, scales: Sequence[int] = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.AvgPool3d(k, stride=1, padding=k // 2) for k in scales])
        self.fuse = nn.Conv3d(dim * len(scales), dim, 1)

    def forward(self, x):
        outs = [pool(x) for pool in self.branches]
        return self.fuse(torch.cat(outs, dim=1)) - x


class HybridformerBlock(nn.Module):
    """Efficient hybrid block: token mixer + depthwise conv + pointwise MLP."""

    def __init__(self, dim, token_mixer="ltb", mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        if token_mixer == "mspa":
            self.mixer = MSPA(dim)
        else:
            self.mixer = LTBAttention(dim)
        self.dwconv = nn.Sequential(
            nn.Conv3d(dim, dim, 7, padding=3, groups=dim),
            nn.InstanceNorm3d(dim, affine=True),
            nn.GELU(),
        )
        self.norm2 = LayerNorm3d(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, hidden, 1), nn.GELU(), nn.Conv3d(hidden, dim, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.dwconv(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# Down / up sampling
# --------------------------------------------------------------------------- #
class DownSample(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.proj = nn.Conv3d(cin, cout, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.proj(x)


# --------------------------------------------------------------------------- #
# Improvement core: Gated Residual Network (GRN) for fusing the two branches
# --------------------------------------------------------------------------- #
class GRN(nn.Module):
    """Gated Residual Network (3D, conv version).

    Adaptively chooses between passing the primary feature through directly and
    applying a non-linear transform of (primary + context), via a learned gate
    (GLU). Here ``a`` = CNN (local) branch is the primary, ``c`` = Transformer
    (global) branch is the context.

        eta = ELU(W_a a + W_c c)
        glu = sigmoid(W_g eta) * (W_v eta)        # gated transform
        out = InstanceNorm(a + glu)
    """

    def __init__(self, dim):
        super().__init__()
        self.fc_a = nn.Conv3d(dim, dim, 1)
        self.fc_c = nn.Conv3d(dim, dim, 1)
        self.elu = nn.ELU(inplace=True)
        self.gate = nn.Conv3d(dim, dim, 1)
        self.value = nn.Conv3d(dim, dim, 1)
        self.norm = nn.InstanceNorm3d(dim, affine=True)

    def forward(self, a, c):
        eta = self.elu(self.fc_a(a) + self.fc_c(c))
        glu = torch.sigmoid(self.gate(eta)) * self.value(eta)
        return self.norm(a + glu)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class CNNBranch(nn.Module):
    """Local-detail branch: 4 strided conv stages -> H/4, H/8, H/16, H/32."""

    def __init__(self, in_ch, dims, strides):
        super().__init__()
        self.downs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        cin = in_ch
        for i, (d, s) in enumerate(zip(dims, strides)):
            self.downs.append(DownSample(cin, d, s))
            self.blocks.append(ConvBlock3D(d, d))
            cin = d

    def forward(self, x):
        feats = []
        for down, blk in zip(self.downs, self.blocks):
            x = blk(down(x))
            feats.append(x)
        return feats


class TransformerBranch(nn.Module):
    """Global-context branch: patch embed + 4 Hybridformer stages.

    Stages 3 and 4 use the MSPA token mixer (improvement B); stages 1 and 2 use
    the single-scale LTB mixer.
    """

    def __init__(self, in_ch, dims, strides, depths, drop_path=0.1):
        super().__init__()
        mixers = ["ltb", "ltb", "mspa", "mspa"]
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        cin = in_ch
        idx = 0
        for i, (d, s, n) in enumerate(zip(dims, strides, depths)):
            self.downs.append(DownSample(cin, d, s))
            blocks = [HybridformerBlock(d, token_mixer=mixers[i], drop_path=dpr[idx + j])
                      for j in range(n)]
            self.stages.append(nn.Sequential(*blocks))
            idx += n
            cin = d

    def forward(self, x):
        feats = []
        for down, stage in zip(self.downs, self.stages):
            x = stage(down(x))
            feats.append(x)
        return feats


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class StrokeCNNTransformer(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 dims: Sequence[int] = (64, 128, 256, 320),
                 depths: Sequence[int] = (2, 2, 2, 2),
                 decoder_base: int = 32,
                 lr_axis: int = -1,
                 deep_supervision: bool = False):
        super().__init__()
        # in-plane /4 at stage1, then /2 each -> H/4, H/8, H/16, H/32
        strides = [(1, 4, 4), (1, 2, 2), (2, 2, 2), (2, 2, 2)]
        self.lr_axis = lr_axis
        self.deep_supervision = deep_supervision

        # ----- dual parallel encoders -----
        self.cnn = CNNBranch(in_channels, dims, strides)
        self.trans = TransformerBranch(in_channels, dims, strides, depths)

        # ----- GRN fusion at every stage -----
        self.fuse = nn.ModuleList([GRN(d) for d in dims])

        # ----- MLBDA at Stage 2/3/4 (index 1,2,3) -----
        self.mlbda = nn.ModuleDict({
            str(i): MLBDA(dims[i], lr_axis=lr_axis) for i in (1, 2, 3)})

        # ----- U-Net decoder -----
        self.up4 = nn.ConvTranspose3d(dims[3], dims[2], (2, 2, 2), (2, 2, 2))
        self.dec3 = ConvBlock3D(dims[2] * 2, dims[2])
        self.bdl = BDL(dims[2], lr_axis=lr_axis)                 # decoder bottom

        self.up3 = nn.ConvTranspose3d(dims[2], dims[1], (2, 2, 2), (2, 2, 2))
        self.dec2 = ConvBlock3D(dims[1] * 2, dims[1])

        self.up2 = nn.ConvTranspose3d(dims[1], dims[0], (1, 2, 2), (1, 2, 2))
        self.dec1 = ConvBlock3D(dims[0] * 2, dims[0])

        self.up1 = nn.ConvTranspose3d(dims[0], decoder_base, (1, 4, 4), (1, 4, 4))
        self.dec0 = ConvBlock3D(decoder_base, decoder_base)

        # ----- segmentation head -----
        self.head = nn.Conv3d(decoder_base, num_classes, kernel_size=1)
        if deep_supervision:
            self.aux3 = nn.Conv3d(dims[2], num_classes, 1)
            self.aux2 = nn.Conv3d(dims[1], num_classes, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        cnn_feats = self.cnn(x)            # 4 x (B, C_i, ...)
        trans_feats = self.trans(x)        # 4 x (B, C_i, ...)

        # stage-wise GRN fusion (CNN = primary/local, Transformer = context/global)
        feats = [self.fuse[i](cnn_feats[i], trans_feats[i]) for i in range(4)]

        # multi-level bilateral difference awareness at stage 2/3/4
        for i in (1, 2, 3):
            feats[i] = self.mlbda[str(i)](feats[i])

        f1, f2, f3, f4 = feats             # H/4, H/8, H/16, H/32

        # decoder with skip connections
        d3 = self.dec3(torch.cat([self.up4(f4), f3], dim=1))
        d3 = self.bdl(d3)                  # bilateral difference learning (bottom)
        d2 = self.dec2(torch.cat([self.up3(d3), f2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), f1], dim=1))
        d0 = self.dec0(self.up1(d1))

        logits = self.head(d0)
        prob = torch.sigmoid(logits)       # voxel-wise lesion probability [0, 1]

        if self.deep_supervision and self.training:
            return prob, torch.sigmoid(self.aux2(d2)), torch.sigmoid(self.aux3(d3))
        return prob


def build_model(in_channels=1, num_classes=1, **kwargs):
    return StrokeCNNTransformer(in_channels=in_channels, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Shape smoke test (run where torch is installed).
    net = build_model()
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")
    x = torch.randn(1, 1, 16, 160, 160)   # (B, C, D, H, W); D%4==0, H/W%32==0
    with torch.no_grad():
        y = net(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))      # expect (1, 1, 16, 160, 160), values in [0,1]
