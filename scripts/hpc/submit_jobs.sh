#!/bin/bash
# Submit one Slurm job per line of a queue file (default queues/hpc.txt), using train.sbatch.
# Lines look like the local queues:  <config.yaml> --set name=<run> [key=value ...]
# Usage: bash scripts/hpc/submit_jobs.sh [account] [gpu-partition] [queue file]
# Defaults: the course account and the 1 x A100 40GB partition (Fall 2026 course allocation).
set -euo pipefail
cd "$(dirname "$0")/../.."
ACCOUNT=${1:-cs_gy_6923-2026fa}
PARTITION=${2:-c12m85-a100-1}
QUEUE=${3:-queues/hpc.txt}
# OnDemand's Jupyter session is itself a Slurm job started with --get-user-env, so its shell has
# SLURM_GET_USER_ENV=1. Jobs submitted from it inherit the variable, and slurmd then rebuilds a
# login environment on the freshly booted cloud node; that fails and leaves the job
# "user env retrieval failed requeued held" (and the user cannot release it). Drop the variables.
unset SLURM_GET_USER_ENV SLURM_EXPORT_ENV

# job time limit: 24 h, or the partition's maximum if that is shorter (jobs are resumable)
limit=$(sinfo -h -p "$PARTITION" -o %l 2>/dev/null | head -1)
case "$limit" in
  ""|infinite|UNLIMITED|*-*) TIME=24:00:00 ;;
  *) TIME=$limit ;;
esac
mkdir -p runs/logs  # Slurm does not create the directory of the --output file

active=$(squeue -h -u "${USER:-$(whoami)}" -o %j 2>/dev/null || true)
while read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  name=$(grep -oE 'name=[^ ]+' <<< "$line" | cut -d= -f2)
  if [ -f "runs/$name/summary.json" ]; then echo "skip $name (finished)"; continue; fi
  if grep -qx "$name" <<< "$active"; then echo "skip $name (already queued/running)"; continue; fi
  config=${line%% *}
  overrides=${line#*--set }
  # --export=ALL: the job gets this shell's environment (minus the variables unset above)
  # shellcheck disable=SC2086
  sbatch --export=ALL --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" --job-name="$name" \
    --output="runs/logs/%x-%j.out" scripts/hpc/train.sbatch "$config" $overrides
done < "$QUEUE"
echo "submitted; watch with:  squeue -u \$USER     logs: runs/logs/<name>-<jobid>.out"
