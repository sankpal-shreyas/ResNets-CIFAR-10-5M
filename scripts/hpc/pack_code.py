"""Pack the code (no data, runs, venvs or git history) into p1_code.zip for upload to OnDemand.

Usage (from the repository root):  python scripts/hpc/pack_code.py [--out p1_code.zip]
On the cluster:  unzip -o ~/p1_code.zip -d /scratch/$USER/dl-p1/code
"""
import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INCLUDE = ["train.py", "predict.py", "requirements.txt", "README.md", "resnet5m", "configs", "queues", "scripts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "p1_code.zip"))
    args = ap.parse_args()
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = os.path.join(ROOT, item)
            files = [path] if os.path.isfile(path) else [
                os.path.join(d, f) for d, _, fs in os.walk(path) if "__pycache__" not in d for f in fs]
            for f in sorted(files):
                if f.endswith(".pyc"):
                    continue
                arc = os.path.relpath(f, ROOT).replace(os.sep, "/")
                info = zipfile.ZipInfo.from_file(f, arc)
                if f.endswith(".sh"):  # keep shell scripts executable after unzip
                    info.external_attr = 0o755 << 16
                with open(f, "rb") as fh:
                    z.writestr(info, fh.read(), zipfile.ZIP_DEFLATED)
                n += 1
    print(f"wrote {args.out}: {n} files, {os.path.getsize(args.out) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
