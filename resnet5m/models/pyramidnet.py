"""PyramidNet (optionally with ShakeDrop) for CIFAR.

Han et al., "Deep Pyramidal Residual Networks", CVPR 2017.
Yamada et al., "ShakeDrop Regularization for Deep Residual Learning", arXiv:1802.02375 (2018).
Follows the reference implementation github.com/dyhan0920/PyramidNet-PyTorch: the width grows
linearly by alpha over all blocks, and the skip connection is an (average-pooled) identity whose
extra channels are zero-padded, so it adds no parameters.
"""
import torch.nn as nn
import torch.nn.functional as F

from .layers import branch_regularizer, linear_drop_rates


class PyramidBasic(nn.Module):
    expansion = 1

    def __init__(self, cin, cmid, stride, reg, p_drop):
        super().__init__()
        cout = cmid * self.expansion
        self.branch = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin, cmid, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cmid),
            nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.reg = branch_regularizer(reg, p_drop)
        self.stride, self.pad = stride, cout - cin

    def forward(self, x):
        y = self.reg(self.branch(x))
        s = F.avg_pool2d(x, 2) if self.stride == 2 else x
        if self.pad > 0:
            s = F.pad(s, (0, 0, 0, 0, 0, self.pad))
        return y + s


class PyramidBottleneck(PyramidBasic):
    expansion = 4

    def __init__(self, cin, cmid, stride, reg, p_drop):
        nn.Module.__init__(self)
        cout = cmid * self.expansion
        self.branch = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin, cmid, 1, bias=False),
            nn.BatchNorm2d(cmid),
            nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cmid, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cmid),
            nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.reg = branch_regularizer(reg, p_drop)
        self.stride, self.pad = stride, cout - cin


class PyramidNet(nn.Module):
    """depth = 6n + 2 (basic) or 9n + 2 (bottleneck). `round_to` rounds widths to a multiple
    (8 makes tensor cores happier; 1 reproduces the paper)."""

    def __init__(self, depth=110, alpha=84, bottleneck=False, num_classes=10,
                 reg="none", p_drop=0.0, round_to=1):
        super().__init__()
        block = PyramidBottleneck if bottleneck else PyramidBasic
        per_block = 9 if bottleneck else 6
        assert (depth - 2) % per_block == 0
        n = (depth - 2) // per_block
        total = 3 * n
        rates = linear_drop_rates(total, p_drop)

        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16))
        blocks, width, cin = [], 16.0, 16
        for k in range(total):
            width += alpha / total
            cmid = max(round_to, int(round(width / round_to)) * round_to)
            stride = 2 if k in (n, 2 * n) else 1
            blocks.append(block(cin, cmid, stride, reg, rates[k]))
            cin = cmid * block.expansion
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.relu(self.bn(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
