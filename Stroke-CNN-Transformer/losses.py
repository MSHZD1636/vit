"""Losses for binary lesion segmentation.

The model outputs voxel-wise probabilities in [0, 1] (Sigmoid). We combine soft
Dice (handles the heavy foreground/background imbalance of stroke lesions) with
binary cross-entropy. Deep-supervision outputs (if any) are weighted.
"""
import torch
import torch.nn as nn

EPS = 1e-6


class SoftDiceLoss(nn.Module):
    def forward(self, prob, target):
        prob = prob.flatten(1)
        target = target.flatten(1).float()
        inter = (prob * target).sum(1)
        denom = prob.sum(1) + target.sum(1)
        dice = (2 * inter + EPS) / (denom + EPS)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss()

    def forward(self, prob, target):
        prob = prob.clamp(EPS, 1 - EPS)
        target = target.float()
        bce = nn.functional.binary_cross_entropy(prob, target)
        dice = self.dice(prob, target)
        return self.bce_weight * bce + self.dice_weight * dice


class DeepSupervisionLoss(nn.Module):
    """Wrap a base loss; if the model returns a tuple, weight aux outputs.

    Weights halve for each lower-resolution output and are normalised. Aux
    targets are obtained by trilinear-downsampling the full-resolution target.
    """

    def __init__(self, base=None):
        super().__init__()
        self.base = base or BCEDiceLoss()

    def forward(self, outputs, target):
        if not isinstance(outputs, (tuple, list)):
            return self.base(outputs, target)
        weights = [1 / (2 ** i) for i in range(len(outputs))]
        s = sum(weights)
        weights = [w / s for w in weights]
        loss = 0.0
        for w, out in zip(weights, outputs):
            if out.shape[2:] != target.shape[2:]:
                t = nn.functional.interpolate(target.float(), size=out.shape[2:],
                                              mode="trilinear", align_corners=False)
                t = (t > 0.5).float()
            else:
                t = target
            loss = loss + w * self.base(out, t)
        return loss
