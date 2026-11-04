#!/bin/bash
# Submit one Slurm job per line of a queue file (default queues/hpc.txt), using train.sbatch.
# Lines look like the local queues:  <config.yaml> --set name=<run> [key=value ...]
# A line may start with  after=<run>[,<run>...]  (e.g. a distillation run whose teachers are trained
# by earlier lines): the job then waits until those runs' jobs have finished successfully.
# Every run also gets RETRIES standby jobs (default 3). A standby job starts only after all of the
# run's earlier jobs have ended unsuccessfully (a cloud node that never came up, the time limit, a
# crash) and resumes from runs/<name>/last.pt; once the run finishes, Slurm cancels the rest.
# Running the script again skips finished runs and tops unfinished ones up to 1 + RETRIES jobs.
# Usage: [RETRIES=n] bash scripts/hpc/submit_jobs.sh [account] [gpu-partition] [queue file]
# Defaults: the course account and the 1 x A100 40GB partition (Fall 2026 course allocation).
set -euo pipefail
cd "$(dirname "$0")/../.."
ACCOUNT=${1:-cs_gy_6923-2026fa}
PARTITION=${2:-c12m85-a100-1}
QUEUE=${3:-queues/hpc.txt}
RETRIES=${RETRIES:-3}
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

declare -A jobids  # run name -> ids of its queued or running jobs, oldest first
while read -r jid jname; do jobids[$jname]="${jobids[$jname]:+${jobids[$jname]} }$jid"; done \
  < <(squeue -h -u "${USER:-$(whoami)}" -o "%i %j" 2>/dev/null | sort -n)
while read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  after=""
  if [[ "$line" == after=* ]]; then after=${line%% *}; after=${after#after=}; line=${line#* }; fi
  name=$(grep -oE 'name=[^ ]+' <<< "$line" | cut -d= -f2)
  if [ -f "runs/$name/summary.json" ]; then echo "skip $name (finished)"; continue; fi
  read -ra ids <<< "${jobids[$name]:-}"
  if (( ${#ids[@]} > RETRIES )); then echo "skip $name (already queued/running)"; continue; fi
  waits=""
  if [ -n "$after" ]; then
    for a in ${after//,/ }; do
      [ -f "runs/$a/summary.json" ] && continue
      if [ -z "${jobids[$a]:-}" ]; then echo "skip $name (waits for $a, which is neither finished nor queued)"; continue 2; fi
      waits+=":${jobids[$a]%% *}"  # the job of $a that runs next, not its standby jobs
    done
  fi
  config=${line%% *}
  overrides=${line#*--set }
  queued=${#ids[@]}
  new=()
  while (( ${#ids[@]} <= RETRIES )); do
    dep=${waits:+afterok$waits}
    # a standby job: starts only once every earlier job of the run has ended unsuccessfully
    if (( ${#ids[@]} )); then dep+=${dep:+,}afternotok:$(IFS=:; echo "${ids[*]}"); fi
    opts=()
    # --kill-on-invalid-dep: a job whose dependency can no longer be met (a run it waits for failed,
    # or an earlier job finished its own run) is cancelled instead of pending forever
    [ -n "$dep" ] && opts=(--dependency="$dep" --kill-on-invalid-dep=yes)
    # --export=ALL: the job gets this shell's environment (minus the variables unset above)
    # shellcheck disable=SC2086
    id=$(sbatch --parsable --export=ALL --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" \
      --job-name="$name" --output="runs/logs/%x-%j.out" ${opts[@]+"${opts[@]}"} \
      scripts/hpc/train.sbatch "$config" $overrides)
    ids+=("${id%%;*}")
    new+=("${id%%;*}")
  done
  jobids[$name]="${ids[*]}"
  if (( queued )); then
    echo "$name (job ${ids[*]:0:queued}): added standby job(s) ${new[*]}"
  else
    echo "submitted $name as job ${new[0]}${new[1]:+ (standby jobs ${new[*]:1})}${after:+ (starts after $after)}"
  fi
done < "$QUEUE"
echo "watch with:  squeue -u \$USER     logs: runs/logs/<name>-<jobid>.out"
