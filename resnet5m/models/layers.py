"""Shared building blocks: stochastic residual regularizers and squeeze-excitation.

ShakeShake: Gastaldi, "Shake-Shake regularization", arXiv:1705.07485 (2017).
ShakeDrop:  Yamada et al., "ShakeDrop Regularization for Deep Residual Learning", arXiv:1802.02375 (2018).
DropPath:   Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
SE:         Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_sample_uniform(x, low, high):
    shape = (x.size(0),) + (1,) * (x.dim() - 1)
    return torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(low, high)


class _ShakeShakeFn(torch.autograd.Function):
    """Forward: alpha*x1 + (1-alpha)*x2 with per-sample alpha ~ U(0,1).
    Backward: gradients mixed with an independent per-sample beta ~ U(0,1)."""

    @staticmethod
    def forward(ctx, x1, x2):
        alpha = _per_sample_uniform(x1, 0.0, 1.0)
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad):
        beta = _per_sample_uniform(grad, 0.0, 1.0)
        return beta * grad, (1 - beta) * grad


def shake_shake(x1, x2, training):
    if training:
        return _ShakeShakeFn.apply(x1, x2)
    return 0.5 * (x1 + x2)


class _ShakeDropFn(torch.autograd.Function):
    """Forward: (b + alpha - b*alpha) * x, backward: (b + beta - b*beta) * grad,
    with b ~ Bernoulli(p_keep), alpha ~ U(-1, 1), beta ~ U(0, 1), all per sample."""

    @staticmethod
    def forward(ctx, x, p_keep):
        gate = torch.bernoulli(torch.full((x.size(0),) + (1,) * (x.dim() - 1), p_keep, device=x.device)).to(x.dtype)
        ctx.save_for_backward(gate)
        alpha = _per_sample_uniform(x, -1.0, 1.0)
        return (gate + alpha - gate * alpha) * x

    @staticmethod
    def backward(ctx, grad):
        (gate,) = ctx.saved_tensors
        beta = _per_sample_uniform(grad, 0.0, 1.0)
        return (gate + beta - gate * beta) * grad, None


class ShakeDrop(nn.Module):
    """ShakeDrop on a residual branch; `p_drop` is the probability of shaking (1 - p_keep).
    At test time the branch is scaled by its expected multiplier, p_keep."""

    def __init__(self, p_drop=0.0):
        super().__init__()
        self.p_keep = 1.0 - p_drop

    def forward(self, x):
        if self.p_keep >= 1.0:
            return x
        if self.training:
            return _ShakeDropFn.apply(x, self.p_keep)
        return self.p_keep * x

    def extra_repr(self):
        return f"p_drop={1 - self.p_keep:.3f}"


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample, rescale survivors."""

    def __init__(self, p_drop=0.0):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if not self.training or self.p_drop == 0.0:
            return x
        keep = 1.0 - self.p_drop
        mask = torch.empty((x.size(0),) + (1,) * (x.dim() - 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"p_drop={self.p_drop:.3f}"


def branch_regularizer(kind, p_drop):
    """Regularizer applied to a residual branch output: 'shakedrop', 'droppath' or none."""
    if kind == "shakedrop":
        return ShakeDrop(p_drop)
    if kind == "droppath":
        return DropPath(p_drop)
    return nn.Identity()


def linear_drop_rates(n_blocks, p_final):
    """Linear decay rule (stochastic depth / ShakeDrop): block l of L gets p_final * l / L."""
    return [p_final * (i + 1) / n_blocks for i in range(n_blocks)]


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s
