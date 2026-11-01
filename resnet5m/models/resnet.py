"""Classic post-activation ResNet with BasicBlocks, configurable stages and widths.

He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
CIFAR variant (3x3 stem, no max-pool) as in github.com/kuangliu/pytorch-cifar, the repository
suggested in the project description. Block:  y = ReLU(S(x) + F(x)),
F(x) = BN(conv3x3(ReLU(BN(conv3x3(x))))),  S(x) = identity or BN(conv1x1(x)).
"""
import torch.nn as nn
import torch.nn.functional as F

from .layers import SqueezeExcite, branch_regularizer, linear_drop_rates


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, se=False, reg="none", p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SqueezeExcite(cout) if se else nn.Identity()
        self.reg = branch_regularizer(reg, p_drop)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return F.relu(self.shortcut(x) + self.reg(self.se(o)))


class ResNet(nn.Module):
    """blocks[i] BasicBlocks of width widths[i] per stage; every stage after the first halves
    the resolution. ResNet-18 = blocks (2,2,2,2), widths (64,128,256,512)."""

    def __init__(self, blocks=(2, 2, 2), widths=(96, 192, 320), num_classes=10,
                 se=False, reg="none", p_drop=0.0, zero_init_residual=True):
        super().__init__()
        rates = linear_drop_rates(sum(blocks), p_drop)
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin, k = [], widths[0], 0
        for i, (b, w) in enumerate(zip(blocks, widths)):
            for j in range(b):
                layers.append(BasicBlock(cin, w, 2 if (i > 0 and j == 0) else 1, se, reg, rates[k]))
                cin, k = w, k + 1
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_residual:  # Goyal et al. 2017: each block starts as the identity
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
