"""Lightweight ResNet-18 classifier for 2D CT slices.

Self-contained (no nnU-Net / ClSeg import chain). The residual block mirrors the
style of ``ClSeg/network_architecture/ISNet/resnet18.py`` already in the repo,
extended with a global-average-pool + linear classification head producing the
three stroke classes (no_stroke / small_lesion / large_lesion).
"""
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x) + self.shortcut(x)
        return F.relu(out)


class ResNet18Classifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_channels=64, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inchannel = base_channels
        c = base_channels
        self.layer1 = self._make_layer(c, 2, stride=1)
        self.layer2 = self._make_layer(c * 2, 2, stride=2)
        self.layer3 = self._make_layer(c * 4, 2, stride=2)
        self.layer4 = self._make_layer(c * 8, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * 8, num_classes)

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResBlock(self.inchannel, channels, s))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(in_channels=1, num_classes=3, **kwargs):
    return ResNet18Classifier(in_channels=in_channels, num_classes=num_classes, **kwargs)
