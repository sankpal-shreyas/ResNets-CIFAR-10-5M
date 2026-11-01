"""Batch-level augmentation on the GPU and soft-target losses.

Cutout: DeVries & Taylor, arXiv:1708.04552 (2017).
MixUp:  Zhang et al., ICLR 2018.
CutMix: Yun et al., ICCV 2019.
"""
import numpy as np
import torch
import torch.nn.functional as F


def cutout(x, size):
    """Zero a size x size square per image (centre uniform over the image, may be clipped).
    Applied after normalisation, so 0 is the dataset mean."""
    if size <= 0:
        return x
    n, _, h, w = x.shape
    cy = torch.randint(0, h, (n, 1, 1), device=x.device)
    cx = torch.randint(0, w, (n, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    half = size // 2
    inside = (ys >= cy - half) & (ys < cy - half + size) & (xs >= cx - half) & (xs < cx - half + size)
    return x * (~inside).unsqueeze(1).to(x.dtype)


def one_hot(labels, num_classes=10, smoothing=0.0):
    off = smoothing / num_classes
    t = torch.full((labels.size(0), num_classes), off, device=labels.device)
    return t.scatter_(1, labels.view(-1, 1), 1.0 - smoothing + off)


def mix_batch(x, targets, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5):
    """Apply MixUp or CutMix (one per batch, chosen with `switch_prob` for CutMix when both are
    enabled) with probability `prob`. `targets` are dense [N, C] probabilities. Returns the mixed
    batch and mixed targets; the same mixed batch is what a teacher sees during distillation."""
    use_mix, use_cut = mixup_alpha > 0, cutmix_alpha > 0
    if not (use_mix or use_cut) or np.random.rand() >= prob:
        return x, targets
    cut = use_cut and (not use_mix or np.random.rand() < switch_prob)
    perm = torch.randperm(x.size(0), device=x.device)
    if cut:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        h, w = x.shape[2:]
        rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = np.clip(cy - rh // 2, 0, h), np.clip(cy + rh // 2, 0, h)
        x0, x1 = np.clip(cx - rw // 2, 0, w), np.clip(cx + rw // 2, 0, w)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)  # exact area ratio after clipping
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * targets + (1 - lam) * targets[perm]


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()


def kd_loss(student_logits, teacher_probs_t, temperature):
    """KL(teacher || student) at temperature T, scaled by T^2 (Hinton et al. 2015).
    `teacher_probs_t` are the teacher (ensemble) probabilities already computed at temperature T."""
    log_p_s = F.log_softmax(student_logits.float() / temperature, dim=1)
    kl = (teacher_probs_t * (torch.log(teacher_probs_t.clamp_min(1e-8)) - log_p_s)).sum(1).mean()
    return kl * temperature ** 2
