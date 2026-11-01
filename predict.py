"""Predict the hidden test set and write a Kaggle submission.

Usage:
    python predict.py --ckpt runs/<name>/model_final.pt --tta flip_shift --out submission.csv

Prints the model's trainable-parameter count (must be <= 5,000,000) and checks the file
against sample_submission.csv (same ids, same order, labels in 0..9).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from resnet5m.data import CLASSES, GpuNormalize, load_test_images
from resnet5m.inference import load_model, predict_probs
from resnet5m.models import PARAM_LIMIT, count_params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tta", default="flip_shift", choices=["none", "flip", "flip_shift"])
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--root", default="data")
    ap.add_argument("--save-probs", action="store_true", help="also save class probabilities (.npy)")
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt)
    n_params = count_params(model)
    print(f"model: {ckpt['model_cfg']}")
    print(f"trainable parameters: {n_params:,} (limit {PARAM_LIMIT:,})")
    assert n_params <= PARAM_LIMIT, "model exceeds the parameter budget"
    if "summary" in ckpt:
        print("validation summary:", {k: v for k, v in ckpt["summary"].items() if k.startswith("val_acc")})

    ids, images = load_test_images(args.root)
    probs = predict_probs(model, torch.from_numpy(images).cuda(), GpuNormalize("cuda"), args.tta)
    labels = probs.argmax(1).cpu().numpy()

    sample = pd.read_csv(os.path.join(args.root, "kaggle", "sample_submission.csv"))
    assert np.array_equal(sample["id"].to_numpy(), ids), "test ids do not match sample_submission.csv"
    sub = pd.DataFrame({"id": ids, "label": labels.astype(int)})
    assert sub["label"].between(0, 9).all() and len(sub) == len(sample)
    sub.to_csv(args.out, index=False)
    if args.save_probs:
        np.save(os.path.splitext(args.out)[0] + "_probs.npy", probs.cpu().numpy())

    counts = np.bincount(labels, minlength=10)
    print(f"wrote {args.out}: {len(sub)} predictions (TTA={args.tta})")
    print("predicted class counts:", dict(zip(CLASSES, counts.tolist())))


if __name__ == "__main__":
    main()
