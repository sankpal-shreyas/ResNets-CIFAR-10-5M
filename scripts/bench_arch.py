"""GPU speed of every architecture in the depth vs width queues, on synthetic data.

Times what train.py does per step (bf16 autocast, channels_last, forward + backward + fused SGD step,
batch 128) and evaluation (forward only, batch 1000), so the shapes can be compared on speed as well
as accuracy: at equal accuracy a faster shape trains for more epochs in the same GPU-hours. Data
loading is excluded (it depends on the CPU, not the architecture).

Usage: python scripts/bench_arch.py [--queues queues/hpc_depth_width.txt ...] [--out results/...]
Writes results/arch_bench_<gpu>.csv (one row per architecture) and prints a table.
"""
import argparse
import os
import re
import shlex
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resnet5m.config import load_config  # noqa: E402
from resnet5m.inference import autocast_dtype  # noqa: E402
from resnet5m.models import build_model, count_params  # noqa: E402


def queued_models(paths):
    """Unique (name, model config) pairs from queue files, first occurrence wins."""
    seen, out = set(), []
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cfg_path, rest = line.split(" --set ", 1)
            cfg = load_config(cfg_path, shlex.split(rest))
            key = repr(sorted(cfg["model"].items()))
            if key not in seen:
                seen.add(key)
                out.append((re.sub(r"_s\d+$", "", cfg["name"]), cfg["model"]))
    return out


def timed(fn, steps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def bench(model_cfg, batch=128, steps=30, warmup=25, eval_batch=1000, repeats=3):
    cfg = dict(model_cfg)
    model = build_model(cfg.pop("name"), **cfg).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, fused=True)
    dtype = autocast_dtype()
    x = torch.randn(batch, 3, 32, 32, device="cuda").contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (batch,), device="cuda")
    xe = torch.randn(eval_batch, 3, 32, 32, device="cuda").contiguous(memory_format=torch.channels_last)

    def train_step():
        with torch.autocast("cuda", dtype=dtype):
            loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def eval_step():
        with torch.autocast("cuda", dtype=dtype):
            model(xe)

    # best of `repeats` timings: laptop GPUs change clocks with temperature and load
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_train = min(timed(train_step, steps, warmup if i == 0 else 3) for i in range(repeats))
    mem = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    t_eval = min(timed(eval_step, max(5, steps // 3), 3) for _ in range(repeats))
    return {"params": count_params(model), "train_img_per_s": batch / t_train,
            "eval_img_per_s": eval_batch / t_eval, "train_peak_mem_mb": mem}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queues", nargs="+", default=["queues/hpc_depth_width.txt"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0)
    rows = []
    for name, mcfg in queued_models(args.queues):
        r = {"name": name, "depth": mcfg.get("depth"), "widths": str(mcfg.get("widths")), **bench(mcfg, steps=args.steps)}
        print(f"{name:14s} depth {r['depth']:2d} {r['widths']:17s} {r['params']:>10,}  "
              f"train {r['train_img_per_s']:7.0f} img/s  eval {r['eval_img_per_s']:7.0f} img/s  "
              f"peak {r['train_peak_mem_mb']:5.0f} MB", flush=True)
        rows.append(r)
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df["gpu"] = gpu
    base = df.loc[df["name"] == "dwA_d28_w64", "train_img_per_s"]
    if len(base):
        df["train_speed_vs_wrn28"] = df["train_img_per_s"] / base.iloc[0]
    out = args.out or os.path.join("results", "arch_bench_" + re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_") + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({gpu})")


if __name__ == "__main__":
    main()
