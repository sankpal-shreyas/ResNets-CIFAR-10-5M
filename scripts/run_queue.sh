#!/bin/bash
# Run train.py once per line of a queue file, sequentially. Each line is
#   <config.yaml> --set name=<run name> [key=value ...]
# Runs whose runs/<name>/summary.json exists are skipped and unfinished ones resume, so the
# queue can be restarted at any time. Lines starting with # are ignored.
# Usage: bash scripts/run_queue.sh queues/phase1.txt
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
mkdir -p runs/logs
while read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  name=$(grep -oE 'name=[^ ]+' <<< "$line" | cut -d= -f2)
  if [ -f "runs/$name/summary.json" ]; then echo "skip   $name (done)"; continue; fi
  echo "start  $name  $(date '+%a %H:%M')"
  # shellcheck disable=SC2086
  $PY train.py --config $line --resume > "runs/logs/$name.log" 2>&1
  status=$?
  echo "end    $name  $(date '+%a %H:%M')  exit=$status  $(grep -oE '"val_acc_flip_shift": [0-9.]+' runs/$name/summary.json 2>/dev/null)"
done < "$1"
