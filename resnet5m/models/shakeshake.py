"""Shake-Shake ResNet (26 2xNd family) for CIFAR.

Gastaldi, "Shake-Shake regularization", arXiv:1705.07485 (2017).
Structure follows the reference PyTorch port by hysts (github.com/hysts/pytorch_shake_shake):
each block has two residual branches  ReLU-conv3x3-BN-ReLU-conv3x3-BN  whose outputs are mixed
with random per-sample weights during training (shake-shake-image), and averaged at test time.
When the shape changes, the skip connection is two 1x1 convs on the (strided) input and on the
input shifted by one pixel, concatenated and batch-normalised.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import shake_shake


def _branch(cin, cout, stride):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
    )


class ShiftedProjection(nn.Module):
    """Projection skip connection used when a block changes width and/or resolution."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(cin, cout // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(cin, cout - cout // 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(x)
        y1 = x[:, :, ::self.stride, ::self.stride]
        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))[:, :, ::self.stride, ::self.stride]
        return self.bn(torch.cat([self.conv1(y1), self.conv2(y2)], dim=1))


class ShakeBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.branch1 = _branch(cin, cout, stride)
        self.branch2 = _branch(cin, cout, stride)
        self.shortcut = ShiftedProjection(cin, cout, stride) if (cin != cout or stride != 1) else None

    def forward(self, x):
        y = shake_shake(self.branch1(x), self.branch2(x), self.training)
        return y + (x if self.shortcut is None else self.shortcut(x))


class ShakeShakeNet(nn.Module):
    """depth = 6n + 2 (n blocks per stage, 26 -> 4); widths = per-branch channels of the 3 stages,
    e.g. (32, 64, 128) is the paper's 2x32d."""

    def __init__(self, depth=26, widths=(40, 80, 160), stem=16, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "Shake-Shake depth must be 6n+2"
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem))
        blocks, cin = [], stem
        for i, w in enumerate(widths):
            for j in range(n):
                blocks.append(ShakeBlock(cin, w, 2 if (i > 0 and j == 0) else 1))
                cin = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.relu(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
