"""Experiment configuration: defaults <- YAML file <- command-line overrides (--set a.b=value)."""
import copy
import os

import yaml

DEFAULTS = {
    "name": None,                 # run name -> runs/<name>/
    "seed": 0,
    "out_dir": "runs",
    "data": {
        "root": "data",
        "val_size": 5000,         # stratified hold-out from the 50k training images; 0 = train on all
        "split_seed": 0,
        "policy": "ta",           # per-sample auto-augment: ta | aa | ra | none
        "photometric": False,     # extra colour / contrast / sharpness / JPEG jitter (data.photometric_ops)
        "pad": 4,                 # random-crop padding (reflect)
        "cutout": 16,             # cutout square size on the GPU, 0 = off
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "mix_prob": 1.0,          # probability a batch is mixed (when an alpha > 0)
        "mix_switch_prob": 0.5,   # P(CutMix) when both are enabled
        "repeats": 1,             # augmentation multiplicity per batch
        "num_workers": 8,
    },
    "model": {"name": "wrn", "depth": 28, "widths": [64, 128, 224]},
    "enforce_param_limit": True,  # students must have <= 5,000,000 trainable params
    "train": {
        "epochs": 200,
        "batch_size": 128,
        "lr": 0.1,                # peak learning rate
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 5e-4,
        "no_bn_bias_decay": True, # no weight decay on BN affine params and biases
        "warmup_epochs": 5,
        "final_lr_ratio": 0.0,    # cosine decays to lr * final_lr_ratio
        "label_smoothing": 0.1,
        "ema_decay": 0.9995,      # per-step EMA of weights, 0 = off
        "sam_rho": 0.0,           # 0 = plain SGD
        "sam_adaptive": False,
        "sam_start_frac": 0.0,    # enable SAM after this fraction of training (late-phase SAM)
        "grad_clip": 0.0,
        "amp": "auto",            # auto (bf16 on Ampere+, fp16 + loss scaling on V100) | bf16 | fp16 | none
        "compile": False,
        "ckpt_every": 10,
    },
    "kd": {
        "teachers": [],           # checkpoint paths (model_final.pt) of teachers trained from scratch
        "temperature": 1.0,
        "alpha": 0.0,             # weight of the hard-label loss; (1 - alpha) on the distillation loss
        "cuda_graph": True,       # replay the teachers' forward pass as a CUDA graph (same outputs, faster)
    },
    "eval": {"tta": ["none", "flip", "flip_shift"]},
}


def _merge(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_file(path):
    """Read a YAML config; `base: other.yaml` (relative path) inherits from another file."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = _load_file(os.path.join(os.path.dirname(path), base))
        if "model" in cfg:
            parent["model"] = cfg.pop("model")
        cfg = _merge(parent, cfg)
    return cfg


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        file_cfg = _load_file(path)
        if "model" in file_cfg:  # architectures take different kwargs: replace, don't merge
            cfg["model"] = file_cfg.pop("model")
        _merge(cfg, file_cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg
