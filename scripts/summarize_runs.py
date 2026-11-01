"""Print a table of all runs in runs/: parameters, schedule, time and validation accuracy.

Usage: python scripts/summarize_runs.py [--sort val_acc_flip_shift] [--csv results.csv]
Unfinished runs show their latest logged epoch and validation accuracy.
"""
import argparse
import glob
import json
import os

import pandas as pd


def collect(root="runs"):
    rows = []
    for run in sorted(glob.glob(os.path.join(root, "*", "log.csv"))):
        d = os.path.dirname(run)
        log = pd.read_csv(run)
        row = {"name": os.path.basename(d), "epochs_done": int(log["epoch"].max()) if len(log) else 0}
        if "val_acc_ema" in log and log["val_acc_ema"].notna().any():
            row["last_val_ema"] = log["val_acc_ema"].iloc[-1]
            row["best_val_ema"] = log["val_acc_ema"].max()
        summary_path = os.path.join(d, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            row.update({k: s[k] for k in ("params", "epochs", "train_hours") if k in s})
            row.update({k: v for k, v in s.items() if k.startswith("val_acc")})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--sort", default="val_acc_flip_shift")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    df = collect(args.root)
    if df.empty:
        print("no runs found"); return
    if args.sort in df:
        df = df.sort_values(args.sort, ascending=False, na_position="last")
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
