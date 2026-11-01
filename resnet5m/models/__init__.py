"""Model registry. Every model is a CNN with residual (skip) connections."""
from .pyramidnet import PyramidNet
from .resnet import ResNet
from .shakeshake import ShakeShakeNet
from .wrn import WideResNet

MODELS = {
    "wrn": WideResNet,
    "shakeshake": ShakeShakeNet,
    "pyramidnet": PyramidNet,
    "resnet": ResNet,
}

PARAM_LIMIT = 5_000_000


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_params(model):
    """Number of trainable parameters (what the 5M budget counts: weights, BN affine, head)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
