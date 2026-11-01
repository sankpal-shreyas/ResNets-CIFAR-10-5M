"""Sharpness-Aware Minimization as a perturb/restore helper around any base optimizer.

Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization", ICML 2021 (adaptive=True).
Adapted from github.com/davda54/sam (MIT). Written as a helper rather than an optimizer subclass
so SAM can be switched on for only the late phase of training (Zhou et al., ICLR 2025).

Per step:  loss(w).backward(); sam.perturb()  # w <- w + e(w)
           loss(w + e).backward() with BN statistics frozen; sam.restore(); optimizer.step()
"""
import torch
import torch.nn as nn


class SAM:
    def __init__(self, params, rho=0.05, adaptive=False, eta=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.rho, self.adaptive, self.eta = rho, adaptive, eta
        self._eps = None

    @torch.no_grad()
    def perturb(self):
        params = [p for p in self.params if p.grad is not None]
        grads = [p.grad for p in params]
        if self.adaptive:  # scale by |w| + eta per element (ASAM)
            scale = [p.abs().add_(self.eta) for p in params]
            scaled = torch._foreach_mul(grads, scale)
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(scaled)))
            eps = torch._foreach_mul(scaled, scale)
        else:
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
            eps = [g.clone() for g in grads]
        torch._foreach_mul_(eps, self.rho / (norm + 1e-12))
        torch._foreach_add_(params, eps)
        self._eps = (params, eps)

    @torch.no_grad()
    def restore(self):
        params, eps = self._eps
        torch._foreach_sub_(params, eps)
        self._eps = None


def set_bn_update(model, enabled):
    """Freeze/unfreeze BatchNorm running statistics (momentum 0 leaves them unchanged)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if enabled:
                if hasattr(m, "_saved_momentum"):
                    m.momentum = m._saved_momentum
            else:
                m._saved_momentum = m.momentum
                m.momentum = 0.0
