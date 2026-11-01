"""Teacher ensemble for knowledge distillation, replayed as a CUDA graph.

The teachers are small networks run on 128-image batches, so launching their kernels one at a
time from Python leaves the GPU partly idle. Capturing the whole ensemble forward pass once and
replaying it removes that overhead; the computation is unchanged.
"""
import torch
import torch.nn.functional as F


class TeacherEnsemble:
    """Mean over teachers of softmax(logits / T) for a batch of normalised images."""

    def __init__(self, teachers, temperature, amp_dtype, cuda_graph=True):
        self.teachers, self.temperature, self.amp_dtype = teachers, temperature, amp_dtype
        self.cuda_graph = cuda_graph
        self.graph = self.static_x = self.static_out = None

    @torch.no_grad()
    def _forward(self, x):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None,
                            cache_enabled=False):
            p = sum(F.softmax(t(x).float() / self.temperature, dim=1) for t in self.teachers)
        return p / len(self.teachers)

    def _capture(self, x):
        self.static_x = torch.empty_like(x, memory_format=torch.channels_last).copy_(x)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up (cuDNN autotuning) outside the graph
            for _ in range(3):
                self._forward(self.static_x)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self._forward(self.static_x)

    def __call__(self, x):
        if not self.cuda_graph:
            return self._forward(x)
        if self.graph is None or x.shape != self.static_x.shape:
            self._capture(x)
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_out.clone()
