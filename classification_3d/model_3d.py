"""3D binary classifier that reuses the paper's CNN-Transformer hybrid encoder.

The backbone is ``Coformer3D`` (from ``ClSeg/network_architecture/hybridformer.py``),
the exact hybrid encoder used by the Cl-SegNet segmentation network: each block
combines a pooling token-mixer, a depthwise Conv3d and an MLP. We take its
multi-scale feature pyramid, global-average-pool every scale, concatenate, and
attach a linear head -> 2 logits (no_stroke / stroke).

Only ``torch``, ``einops`` and ``timm`` are required (Coformer3D pulls in no
nnU-Net code). The Cl-SegNet package is added to ``sys.path`` so it imports
without a separate ``pip install -e``.
"""
import os
import sys

import torch
import torch.nn as nn

# Make the in-repo ClSeg package importable without installation.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLSEG_PKG = os.path.join(_REPO_ROOT, "ClSeg_package")
if _CLSEG_PKG not in sys.path:
    sys.path.insert(0, _CLSEG_PKG)

from ClSeg.network_architecture.hybridformer import Coformer3D, _init_weights  # noqa: E402

# Same encoder configuration as cl_seg_aisd.Generic_UNet.
DEFAULT_SCALE_FACTOR = [(1, 0.5, 0.5), (1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
DEFAULT_EMBED_DIM = [64, 128, 256, 320]
DEFAULT_DEPTHS = [3, 3, 6, 3]


class CoformerClassifier3D(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, embed_dim=DEFAULT_EMBED_DIM,
                 depths=DEFAULT_DEPTHS, scale_factor=DEFAULT_SCALE_FACTOR,
                 patch_size=(1, 4, 4), use_multiscale=True, dropout=0.2):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.encoder = Coformer3D(
            patch_size=patch_size, in_chans=in_chans, scale_factor=scale_factor,
            embed_dim=embed_dim, depths=depths,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        feat_dim = sum(embed_dim) if use_multiscale else embed_dim[-1]
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)
        self.apply(_init_weights)

    def forward(self, x):
        feats = self.encoder(x)  # tuple of (B, C_i, D_i, H_i, W_i)
        if self.use_multiscale:
            pooled = [self.pool(f).flatten(1) for f in feats]
            z = torch.cat(pooled, dim=1)
        else:
            z = self.pool(feats[-1]).flatten(1)
        z = self.norm(z)
        z = self.dropout(z)
        return self.fc(z)


def build_model(in_chans=1, num_classes=2, **kwargs):
    return CoformerClassifier3D(in_chans=in_chans, num_classes=num_classes, **kwargs)
