"""Download the competition data with kagglehub into data/kaggle/.

First tries the official one-call download
    kagglehub.competition_download('cifar-10-res-net-architecture-challenge-dl')
As of 2026-10-30 Kaggle has not generated the bundled archive for this competition
(the API answers 404 / "no gcs url"), so the script falls back to downloading the
individual files with kagglehub, with polite rate limiting and exponential backoff.
It is resumable: files that already exist are skipped.

Credentials: kagglehub reads the API token from ~/.kaggle/access_token.

Usage:
    python scripts/download_kaggle_data.py                         # CSVs + 2,000 test PNGs
    python scripts/download_kaggle_data.py --what train-sample -n 100  # random train PNGs (for verify_data.py)
    python scripts/download_kaggle_data.py --what train            # all 50,000 train PNGs (slow: rate-limited)
"""
import argparse
import os
import random
import shutil
import time

import kagglehub

SLUG = "cifar-10-res-net-architecture-challenge-dl"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kaggle")


def try_full_download():
    """Return True if the whole competition could be downloaded in one call."""
    try:
        path = kagglehub.competition_download(SLUG)
    except Exception as e:
        print(f"full download unavailable ({str(e).splitlines()[0][:80]}); falling back to per-file")
        return False
    shutil.copytree(path, ROOT, dirs_exist_ok=True)
    print("full download ok:", path)
    return True


def fetch(name, retries=150):
    """Download competition file `name` (e.g. 'test_images/test_0001.png') to ROOT/name.
    Kaggle's API quota appears to be a rolling daily window, so on errors we back off up to
    10 minutes per attempt and keep trying (150 attempts ~ 1 day)."""
    dest = os.path.join(ROOT, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            cached = kagglehub.competition_download(SLUG, path=name)
            shutil.copyfile(cached, dest)
            return "ok"
        except Exception as e:  # 429s and transient network errors
            wait = min(600, 10 * 2 ** attempt)
            print(f"  {name}: {str(e).splitlines()[0][:70]} -> sleeping {wait}s", flush=True)
            time.sleep(wait)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["test", "train-sample", "train"], default="test")
    ap.add_argument("-n", type=int, default=100, help="number of random train images for --what train-sample")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    ap.add_argument("--no-full", action="store_true", help="skip the one-call full download attempt")
    args = ap.parse_args()

    if not args.no_full and try_full_download():
        return

    if args.what == "test":
        names = ["test.csv", "sample_submission.csv", "train_labels.csv"]
        names += [f"test_images/test_{i:04d}.png" for i in range(2000)]
    elif args.what == "train-sample":
        rng = random.Random(0)
        names = [f"train_images/train_{i:05d}.png" for i in sorted(rng.sample(range(50000), args.n))]
    else:
        names = [f"train_images/train_{i:05d}.png" for i in range(50000)]

    t0, counts = time.time(), {"ok": 0, "skip": 0, "fail": 0}
    for k, name in enumerate(names):
        status = fetch(name)
        counts[status] += 1
        if status == "ok":
            time.sleep(args.delay)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(names)} {counts} {time.time() - t0:.0f}s", flush=True)
    print("done", counts, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
