#!/bin/bash
# Train one config on the NYU burst cluster (normally submitted by submit_jobs.sh).
# Resumable: submitting the same run again continues from runs/<name>/last.pt.
# Usage: sbatch --account=<acct> --partition=<gpu partition> scripts/hpc/train.sbatch \
#               <config.yaml> [key=value overrides...]
#SBATCH --job-name=p1-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
set -euo pipefail
WORK=${WORK:-/scratch/$USER/dl-p1}
cd "$WORK/code"
mkdir -p runs/logs
CONFIG=$1; shift
NAME=""
for a in "$@"; do case $a in name=*) NAME=${a#name=} ;; esac; done
# a standby job (see submit_jobs.sh) whose run an earlier job has already finished
if [ -n "$NAME" ] && [ -f "runs/$NAME/summary.json" ]; then echo "$NAME is already finished"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv
"$WORK/venv/bin/python" train.py --config "$CONFIG" \
  --set data.root="$WORK/data" data.num_workers="${SLURM_CPUS_PER_TASK:-8}" "$@" --resume
