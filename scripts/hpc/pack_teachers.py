"""Pack the teacher checkpoints a queue file needs into p1_teachers.zip for upload to OnDemand.

Reads every kd.teachers=[...] list in the queue file(s) and stores those runs/<name>/model_final.pt
files (uncompressed: checkpoints do not compress) under the same paths.

Usage (repository root):  python scripts/hpc/pack_teachers.py queues/hpc_aug_multiplicity.txt
On the cluster:           unzip -o ~/p1_teachers.zip -d /scratch/$USER/dl-p1/code
"""
import argparse
import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queues", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "p1_teachers.zip"))
    args = ap.parse_args()
    paths = []
    for q in args.queues:
        for line in open(q):
            if line.lstrip().startswith("#"):
                continue
            for group in re.findall(r"kd\.teachers=\[([^\]]*)\]", line):
                paths += [p for p in group.split(",") if p and p not in paths]
    missing = [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit("missing teacher checkpoints:\n  " + "\n  ".join(missing))
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for p in paths:
            z.write(os.path.join(ROOT, p), p)
            print(f"  {p}  ({os.path.getsize(os.path.join(ROOT, p)) / 1e6:.1f} MB)")
    print(f"wrote {args.out}: {len(paths)} checkpoints, {os.path.getsize(args.out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
