#!/bin/bash
# Show what is needed to submit training jobs on the NYU burst cluster, and suggest the
# submit command. Run in a terminal of a burst Jupyter session (or a burst shell).
echo "== host: $(hostname)   user: $USER   $(date)"

echo; echo "== Slurm accounts you may use (account | partitions it is limited to, if any)"
sacctmgr -nP show assoc where user="$USER" format=account,partition 2>&1 | sort -u

echo; echo "== partitions with GPUs (partition | gres | time limit | cpus | memory MB)"
sinfo -h -o "%P|%G|%l|%c|%m" 2>/dev/null | sort -u | grep -i gpu || echo "(none visible)"

echo; echo "== storage"
df -h "/scratch/$USER" "$HOME" 2>&1 | sed 's/^/   /'

echo; echo "== internet (needed once by setup_env.sh)"
for u in https://pypi.org https://download.pytorch.org/whl/cu126/torch/ https://www.cs.toronto.edu; do
  printf "   %-30s HTTP %s\n" "$u" "$(curl -s -o /dev/null -m 10 -w '%{http_code}' $u)"
done

# Suggest: first course account; GPU partition preferring A100 > H100 > L4 > V100 (by name/gres)
acct=$(sacctmgr -nP show assoc where user="$USER" format=account 2>/dev/null | grep -v '^$' | sort -u | head -1)
gpu_parts=$(sinfo -h -o "%P|%G" 2>/dev/null | grep -i gpu | cut -d'|' -f1 | tr -d '*' | sort -u)
part=""
for pat in a100 h100 l4 v100 gpu; do
  part=$(grep -i "$pat" <<< "$gpu_parts" | head -1)
  [ -n "$part" ] && break
done
echo
echo "== suggested (check the account is your Deep Learning course account):"
echo "   cd /scratch/\$USER/dl-p1/code && bash scripts/hpc/submit_jobs.sh ${acct:-<account>} ${part:-<gpu-partition>}"
