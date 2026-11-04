"""Pre-activation Wide ResNet with configurable depth and per-stage widths.

Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016 (pre-activation blocks).

Each residual block computes  y = S(x) + F(x)  with
    F(x) = conv3x3(ReLU(BN(conv3x3(ReLU(BN(x))))))
and S(x) the identity, or a 1x1 projection when the shape changes. Optional extras per block:
squeeze-excitation on F(x), dropout between the convs, and ShakeDrop / stochastic depth on F(x).
"""
import torch.nn as nn
import torch.nn.functional as F

from .layers import SqueezeExcite, branch_regularizer, linear_drop_rates


class PreActBlock(nn.Module):
    def __init__(self, cin, cout, stride, dropout=0.0, se=False, reg="none", p_drop=0.0, branch_bn=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SqueezeExcite(cout) if se else nn.Identity()
        # ShakeDrop only converges when the branch ends with BN (Yamada et al. add it to WRN);
        # without it our WRN-28 + ShakeDrop run stalled at 77% validation accuracy.
        if branch_bn is None:
            branch_bn = reg == "shakedrop"
        self.bn3 = nn.BatchNorm2d(cout) if branch_bn else nn.Identity()
        self.reg = branch_regularizer(reg, p_drop)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        s = x if self.shortcut is None else self.shortcut(o)
        o = self.conv1(o)
        o = self.conv2(self.dropout(F.relu(self.bn2(o))))
        return s + self.reg(self.bn3(self.se(o)))


class WideResNet(nn.Module):
    """depth = 6n + 4 (n blocks per stage); widths = output channels of the three stages;
    strides = stride of each stage's first block. Strides change the compute (feature-map sizes),
    not the parameter count: (1, 2, 1) keeps the last stage at 16x16 for ~1.8x the FLOPs."""

    def __init__(self, depth=28, widths=(64, 128, 224), strides=(1, 2, 2), stem=16, num_classes=10,
                 dropout=0.0, se=False, reg="none", p_drop=0.0, branch_bn=None):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        assert len(strides) == len(widths), "one stride per stage"
        n = (depth - 4) // 6
        rates = linear_drop_rates(n * len(widths), p_drop)
        self.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        blocks, cin, k = [], stem, 0
        for i, w in enumerate(widths):
            for j in range(n):
                stride = strides[i] if j == 0 else 1
                blocks.append(PreActBlock(cin, w, stride, dropout, se, reg, rates[k], branch_bn))
                cin, k = w, k + 1
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.relu(self.bn(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
