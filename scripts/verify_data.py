"""Sanity-check the Kaggle competition data.

1. Training set: check that the Kaggle train labels and a sample of train PNGs are
   identical to the standard CIFAR-10 training set (torchvision copy), so either source
   can be used interchangeably.
2. Test set: report aggregate statistics only (shape, per-channel mean/std, sharpness) and
   compare them to the training set to see whether the hidden set looks shifted. The test
   images are never matched against any labeled dataset (that would amount to a label lookup).
"""
import glob
import os

import numpy as np
import pandas as pd
import torchvision
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_png(path):
    im = Image.open(path)
    return np.asarray(im.convert("RGB")), im.mode, im.size


def main():
    ref = torchvision.datasets.CIFAR10(os.path.join(ROOT, "cifar10"), train=True, download=True)
    ref_x, ref_y = ref.data, np.array(ref.targets)

    labels = pd.read_csv(os.path.join(ROOT, "kaggle", "train_labels.csv"))
    assert len(labels) == 50000
    same_labels = (labels["label"].values == ref_y).mean()
    print(f"train labels identical to CIFAR-10 train order: {same_labels * 100:.2f}%")
    print("class counts:", np.bincount(labels["label"].values).tolist())

    paths = sorted(glob.glob(os.path.join(ROOT, "kaggle", "train_images", "*.png")))
    n_same = 0
    for p in paths:
        idx = int(os.path.basename(p)[6:11])
        img, _, _ = load_png(p)
        n_same += int(np.array_equal(img, ref_x[idx]))
    print(f"sampled train PNGs pixel-identical to CIFAR-10 train: {n_same}/{len(paths)}")

    tpaths = sorted(glob.glob(os.path.join(ROOT, "kaggle", "test_images", "*.png")))
    if tpaths:
        imgs, modes, sizes = [], set(), set()
        for p in tpaths:
            img, mode, size = load_png(p)
            imgs.append(img); modes.add(mode); sizes.add(size)
        t = np.stack(imgs).astype(np.float64) / 255
        r = ref_x.astype(np.float64) / 255
        print(f"test images: {len(tpaths)}  modes={modes}  sizes={sizes}")
        print("per-channel mean  train:", r.mean((0, 1, 2)).round(4), " test:", t.mean((0, 1, 2)).round(4))
        print("per-channel std   train:", r.std((0, 1, 2)).round(4), " test:", t.std((0, 1, 2)).round(4))
        # spatial high-frequency energy (mean abs difference of neighbouring pixels): blur/noise indicator
        hf = lambda a: np.abs(np.diff(a, axis=2)).mean()
        print(f"neighbour-pixel |diff|  train: {hf(r[:5000]):.4f}  test: {hf(t):.4f}")


if __name__ == "__main__":
    main()
