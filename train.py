"""Train a residual network on CIFAR-10, optionally distilling from teachers.

Usage:
    python train.py --config configs/wrn28_ta.yaml
    python train.py --config configs/wrn28_ta.yaml --set train.epochs=50 name=quick_test
    python train.py --config configs/wrn28_ta.yaml --resume      # continue an interrupted run

Outputs in runs/<name>/:
    config.yaml       fully resolved configuration
    log.csv           per-epoch lr, train loss/acc, validation accuracy (raw and EMA weights)
    model_final.pt    final weights (EMA if enabled) + model config, loadable by predict.py
    summary.json      parameter count, training time, final validation accuracy per TTA mode
    last.pt           resumable state (removed once training finishes)
"""
import argparse
import csv
import json
import math
import os
import random
import time

# One BLAS/OpenMP thread per process: all heavy compute is on the GPU, and on Windows every
# spawned DataLoader worker would otherwise reserve per-thread OpenBLAS buffers (~1.9 GB each).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402  (must come after the thread settings above)
import torch
import yaml

from resnet5m.config import load_config
from resnet5m.data import GpuNormalize, load_cifar10_train, make_train_loader, stratified_split
from resnet5m.distill import TeacherEnsemble
from resnet5m.ema import EMA
from resnet5m.inference import accuracy, autocast_dtype, load_model, predict_probs, save_model
from resnet5m.mix import cutout, kd_loss, mix_batch, one_hot, soft_cross_entropy
from resnet5m.models import PARAM_LIMIT, build_model, count_params
from resnet5m.sam import SAM, set_bn_update
from resnet5m.winperf import disable_power_throttling

LOG_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_acc", "val_acc_ema", "sam", "epoch_time", "img_per_s"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def param_groups(model, weight_decay, no_bn_bias_decay):
    """Weight decay on conv/linear weights only (BN affine params and biases are 1-D)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if no_bn_bias_decay and p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step, total, warmup, final_ratio):
    """Linear warm-up, then cosine decay from 1 to final_ratio (per optimisation step)."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    cfg["name"] = cfg["name"] or os.path.splitext(os.path.basename(args.config))[0]
    run_dir = os.path.join(cfg["out_dir"], cfg["name"])
    last_path = os.path.join(run_dir, "last.pt")
    if os.path.exists(run_dir) and not args.resume:
        raise SystemExit(f"{run_dir} already exists: pass --resume or choose another name")
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        raise SystemExit(f"{run_dir} is already finished (summary.json exists); nothing to do")
    os.makedirs(run_dir, exist_ok=True)
    dc, tc, kc = cfg["data"], cfg["train"], cfg["kd"]

    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    disable_power_throttling()  # Windows laptops: keep the kernel-launching thread at full speed

    # ---- data: 50k CIFAR-10 training images, stratified hold-out for validation ----
    images, labels = load_cifar10_train(dc["root"])
    tr_idx, va_idx = stratified_split(labels, dc["val_size"], dc["split_seed"])
    loader = make_train_loader(images[tr_idx], labels[tr_idx], tc["batch_size"], dc["policy"], dc["pad"],
                               dc["num_workers"], dc["repeats"], cfg["seed"], dc["photometric"])
    normalize = GpuNormalize(device)
    val_x = torch.from_numpy(images[va_idx]).to(device)
    val_y = torch.from_numpy(labels[va_idx]).to(device)
    has_val = len(va_idx) > 0
    print(f"train {len(tr_idx)} / val {len(va_idx)} images, {len(loader)} steps per epoch")

    # ---- model ----
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name"), **mcfg).to(device).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"model {cfg['model']}: {n_params:,} trainable parameters")
    if cfg["enforce_param_limit"] and n_params > PARAM_LIMIT:
        raise SystemExit(f"{n_params:,} trainable parameters exceeds the {PARAM_LIMIT:,} budget")
    with open(os.path.join(run_dir, "model.txt"), "w") as f:
        f.write(f"{model}\n\ntrainable parameters: {n_params:,}\n")

    teachers = [load_model(p, device)[0] for p in kc["teachers"]]
    for t in teachers:
        t.requires_grad_(False)
    if teachers:
        print(f"distilling from {len(teachers)} teacher(s), T={kc['temperature']}, alpha={kc['alpha']}")

    # ---- optimisation ----
    opt = torch.optim.SGD(param_groups(model, tc["weight_decay"], tc["no_bn_bias_decay"]), lr=tc["lr"],
                          momentum=tc["momentum"], nesterov=tc["nesterov"], fused=True)
    total_steps = tc["epochs"] * len(loader)
    warmup_steps = tc["warmup_epochs"] * len(loader)
    sam = SAM(model.parameters(), tc["sam_rho"], tc["sam_adaptive"]) if tc["sam_rho"] > 0 else None
    sam_start_epoch = int(round(tc["sam_start_frac"] * tc["epochs"]))
    ema = EMA(model, tc["ema_decay"]) if tc["ema_decay"] > 0 else None
    fwd = torch.compile(model) if tc["compile"] else model
    amp_dtype = autocast_dtype(tc["amp"])
    # fp16 (GPUs without bf16, e.g. V100) needs loss scaling; the scaler is a no-op otherwise
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    if sam is not None and scaler.is_enabled():
        raise SystemExit("SAM is implemented for bf16/fp32 only: set train.amp=none on this GPU")
    print(f"mixed precision: {amp_dtype}")
    teacher = TeacherEnsemble(teachers, kc["temperature"], amp_dtype, kc["cuda_graph"]) if teachers else None

    start_epoch, step, elapsed = 0, 0, 0.0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema:
            ema.module.load_state_dict(ck["ema"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step, elapsed = ck["epoch"], ck["step"], ck["elapsed"]
        print(f"resumed from epoch {start_epoch}")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log_path = os.path.join(run_dir, "log.csv")
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    def save_last(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.module.state_dict() if ema else None, "scaler": scaler.state_dict(),
                    "epoch": epoch, "step": step, "elapsed": elapsed}, last_path)

    def compute_loss(x, targets, t_probs):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = fwd(x)
        loss = soft_cross_entropy(out, targets)
        if t_probs is not None:
            loss = kc["alpha"] * loss + (1 - kc["alpha"]) * kd_loss(out, t_probs, kc["temperature"])
        return out, loss

    # ---- training loop ----
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, tc["epochs"]):
            model.train()
            use_sam = sam is not None and epoch >= sam_start_epoch
            t0 = time.time()
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            seen = 0
            for x, y in loader:
                lr = tc["lr"] * lr_factor(step, total_steps, warmup_steps, tc["final_lr_ratio"])
                for g in opt.param_groups:
                    g["lr"] = lr
                x = cutout(normalize(x), dc["cutout"])
                targets = one_hot(y.to(device, non_blocking=True), 10, tc["label_smoothing"])
                x, targets = mix_batch(x, targets, dc["mixup_alpha"], dc["cutmix_alpha"],
                                       dc["mix_prob"], dc["mix_switch_prob"])
                # teachers see exactly the student's (augmented, mixed) batch
                t_probs = teacher(x) if teacher else None

                out, loss = compute_loss(x, targets, t_probs)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if use_sam:  # second pass at the adversarially perturbed weights
                    sam.perturb()
                    opt.zero_grad(set_to_none=True)
                    set_bn_update(model, False)
                    compute_loss(x, targets, t_probs)[1].backward()
                    set_bn_update(model, True)
                    sam.restore()
                if tc["grad_clip"] > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scaler.step(opt)
                scaler.update()
                step += 1
                if ema:
                    ema.update(step)
                loss_sum += loss.detach() * x.size(0)
                correct += (out.argmax(1) == targets.argmax(1)).sum()
                seen += x.size(0)

            epoch_time = time.time() - t0
            elapsed += epoch_time
            row = {"epoch": epoch + 1, "lr": f"{lr:.5f}", "train_loss": f"{(loss_sum / seen).item():.4f}",
                   "train_acc": f"{(correct / seen).item():.4f}", "sam": int(use_sam),
                   "epoch_time": f"{epoch_time:.1f}", "img_per_s": f"{seen / epoch_time:.0f}"}
            if has_val:
                row["val_acc"] = f"{accuracy(model, val_x, val_y, normalize):.4f}"
                if ema:
                    row["val_acc_ema"] = f"{accuracy(ema.module, val_x, val_y, normalize):.4f}"
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, LOG_FIELDS).writerow(row)
            print(" ".join(f"{k}={row[k]}" for k in LOG_FIELDS if k in row), flush=True)
            if (epoch + 1) % tc["ckpt_every"] == 0:
                save_last(epoch + 1)
    except KeyboardInterrupt:
        save_last(epoch)
        raise SystemExit(f"interrupted: state saved to {last_path}; rerun with --resume")

    # ---- final evaluation and export ----
    final = ema.module if ema else model
    summary = {"name": cfg["name"], "model": cfg["model"], "params": n_params, "epochs": tc["epochs"],
               "train_hours": round(elapsed / 3600, 3), "weights": "ema" if ema else "raw"}
    if has_val:
        summary["val_acc_raw_weights"] = accuracy(model, val_x, val_y, normalize)
        for tta in cfg["eval"]["tta"]:
            summary[f"val_acc_{tta}"] = accuracy(final, val_x, val_y, normalize, tta)
        # per-image validation probabilities: paired run comparisons, ensembles, confusion matrices
        probs = predict_probs(final, val_x, normalize, "flip")
        np.save(os.path.join(run_dir, "val_probs.npy"), probs.cpu().numpy().astype(np.float16))
    save_model(os.path.join(run_dir, "model_final.pt"), cfg["model"], final.state_dict(),
               params=n_params, summary=summary)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(last_path):
        os.remove(last_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
