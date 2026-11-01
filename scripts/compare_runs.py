"""Paired comparison of runs on the same validation images (uses runs/<name>/val_probs.npy).

For two runs: accuracies, discordant counts and an exact McNemar test (is the difference real or
validation noise?). For two or more runs: the accuracy of their probability-averaged ensemble,
which shows whether models are diverse enough to be useful together (e.g. as a teacher ensemble).

Usage: python scripts/compare_runs.py runA runB [runC ...]
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resnet5m.data import load_cifar10_train, stratified_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    images, labels = load_cifar10_train("data")
    _, val_idx = stratified_split(labels, args.val_size, args.split_seed)
    y = labels[val_idx]
    for r in args.runs:  # older runs did not save validation probabilities: compute them
        path = os.path.join(args.root, r, "val_probs.npy")
        if not os.path.exists(path):
            import torch
            from resnet5m.data import GpuNormalize
            from resnet5m.inference import load_model, predict_probs
            model, _ = load_model(os.path.join(args.root, r, "model_final.pt"))
            p = predict_probs(model, torch.from_numpy(images[val_idx]).cuda(), GpuNormalize("cuda"), "flip")
            np.save(path, p.cpu().numpy().astype(np.float16))
    probs = {r: np.load(os.path.join(args.root, r, "val_probs.npy")).astype(np.float32) for r in args.runs}
    correct = {r: p.argmax(1) == y for r, p in probs.items()}
    for r in args.runs:
        print(f"{r:40s} acc={correct[r].mean():.4f}")

    if len(args.runs) == 2:
        a, b = (correct[r] for r in args.runs)
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        print(f"only A right: {n10}, only B right: {n01}, McNemar exact p = {p:.3f}"
              f"  ({'significant' if p < 0.05 else 'not significant'} at 5%)")
    ens = np.mean([probs[r] for r in args.runs], axis=0)
    print(f"ensemble of {len(args.runs)} (mean probs): acc={(ens.argmax(1) == y).mean():.4f}")


if __name__ == "__main__":
    main()
