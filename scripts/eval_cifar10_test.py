"""Evaluate runs on the official CIFAR-10 test batch (10,000 labelled images, never trained on).

It is the only held-out set for the all-data (*_full) models, and a second one, twice the size of
the validation split, for the rest. For each run with a model_final.pt this writes
runs/<name>/cifar10_test.json (accuracy per TTA) and cifar10_test_probs.npz (float16 probabilities,
one array per TTA), then prints a table. Runs that already have results are skipped unless --force.

Usage: python scripts/eval_cifar10_test.py [run ...] [--force]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resnet5m.data import GpuNormalize, load_cifar10_test  # noqa: E402
from resnet5m.inference import load_model, predict_probs  # noqa: E402

TTAS = ("none", "flip", "flip_shift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names (default: every run with a model_final.pt)")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--force", action="store_true", help="re-evaluate runs that already have results")
    args = ap.parse_args()

    names = args.runs or sorted(os.path.basename(os.path.dirname(p))
                                for p in glob.glob(os.path.join(args.root, "*", "model_final.pt")))
    images, labels = load_cifar10_test("data")
    x, y = torch.from_numpy(images).cuda(), torch.from_numpy(labels).cuda()
    norm = GpuNormalize("cuda")
    rows = []
    for name in names:
        d = os.path.join(args.root, name)
        out = os.path.join(d, "cifar10_test.json")
        if os.path.exists(out) and not args.force:
            with open(out) as f:
                rows.append((name, json.load(f)))
            continue
        try:
            model, _ = load_model(os.path.join(d, "model_final.pt"))
        except RuntimeError as e:  # checkpoints from before a model change (b3: no branch BN)
            print(f"skip {name}: {str(e).splitlines()[0]}", flush=True)
            continue
        probs = {t: predict_probs(model, x, norm, t) for t in TTAS}
        acc = {t: (p.argmax(1) == y).float().mean().item() for t, p in probs.items()}
        np.savez_compressed(os.path.join(d, "cifar10_test_probs.npz"),
                            **{t: p.cpu().numpy().astype(np.float16) for t, p in probs.items()})
        with open(out, "w") as f:
            json.dump(acc, f, indent=2)
        rows.append((name, acc))
        del model, probs
        torch.cuda.empty_cache()
        print(f"{name:40s} " + " ".join(f"{acc[t] * 100:6.2f}" for t in TTAS), flush=True)
    print(f"\n{'run':40s} " + " ".join(f"{t:>10s}" for t in TTAS))
    for name, acc in rows:
        print(f"{name:40s} " + " ".join(f"{acc[t] * 100:10.2f}" for t in TTAS))


if __name__ == "__main__":
    main()
