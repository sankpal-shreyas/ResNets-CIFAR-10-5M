"""Exponential moving average of model weights (Polyak averaging), including BN statistics."""
import copy

import torch


class EMA:
    def __init__(self, model, decay=0.9995):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.bind(model)

    def bind(self, model):
        """Cache matching tensor lists. The optimizer and SAM update weights in place, so the
        cached views stay valid; call again only if the model's tensors are re-created."""
        src, dst = model.state_dict(), self.module.state_dict()
        self._float = ([dst[k] for k, v in src.items() if v.dtype.is_floating_point],
                       [v for v in src.values() if v.dtype.is_floating_point])
        self._other = [(dst[k], v) for k, v in src.items() if not v.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, step):
        # warm-up: short memory early in training so the average is not dominated by init
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_lerp_(self._float[0], self._float[1], 1.0 - d)
        if step % 100 == 0:  # integer buffers (BN batch counters) do not affect inference
            for dst, src in self._other:
                dst.copy_(src)
