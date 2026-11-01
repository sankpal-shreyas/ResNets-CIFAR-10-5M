"""Inference with optional test-time augmentation (TTA) and checkpoint (de)serialisation.

TTA views (probabilities averaged over views):
  'none'        the image itself
  'flip'        image + horizontal mirror
  'flip_shift'  mirror x {no shift, 1-px shifts up-left / down-right} (airbench-style, 6 views)
"""
import torch
import torch.nn.functional as F

from .models import build_model


def _views(x, tta):
    views = [x]
    if tta in ("flip", "flip_shift"):
        views.append(x.flip(-1))
    if tta == "flip_shift":
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        for v in (padded[:, :, 0:-2, 0:-2], padded[:, :, 2:, 2:]):
            views += [v, v.flip(-1)]
    elif tta not in ("none", "flip"):
        raise ValueError(f"unknown tta '{tta}'")
    return views


def autocast_dtype(kind="auto"):
    """Mixed-precision dtype: bf16 on Ampere and newer GPUs, fp16 on older ones (V100 has no
    bf16 tensor cores); None disables autocast."""
    if kind == "none":
        return None
    if kind == "auto":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[kind]


@torch.no_grad()
def predict_probs(model, images_u8, normalize, tta="none", batch_size=1000):
    """Class probabilities [N, C] (float32, on the model's device) for uint8 NCHW images."""
    model.eval()
    dtype = autocast_dtype()
    out = []
    for i in range(0, len(images_u8), batch_size):
        views = _views(normalize(images_u8[i:i + batch_size]), tta)
        with torch.autocast("cuda", dtype=dtype):
            p = sum(F.softmax(model(v).float(), dim=1) for v in views)
        out.append(p / len(views))
    return torch.cat(out)


@torch.no_grad()
def accuracy(model, images_u8, labels, normalize, tta="none"):
    probs = predict_probs(model, images_u8, normalize, tta)
    return (probs.argmax(1) == labels).float().mean().item()


def save_model(path, model_cfg, state_dict, **extra):
    torch.save({"model_cfg": model_cfg, "state_dict": state_dict, **extra}, path)


def load_model(path, device="cuda"):
    """Rebuild a model from a checkpoint written by save_model (train.py's model_final.pt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_cfg"])
    model = build_model(cfg.pop("name"), **cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ckpt
