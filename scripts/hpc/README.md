# Training on NYU HPC Cloud Bursting (Open OnDemand only)

Course allocation (Fall 2026): Slurm account `cs_gy_6923-2026fa`, 300 GPU hours. Partitions:
`interactive` (Jupyter), `n2c48m24` (CPU), `g2-standard-12/24` (1/2 x L4),
`c12m85-a100-1` / `c24m170-a100-2` (1/2 x A100 40 GB). We use `c12m85-a100-1` for training.
NYU VPN is required off campus. Do not use the Torch cluster for coursework.

**GPU instances with low GPU utilisation for 20 minutes are shut down automatically**, so all
setup runs in a CPU Jupyter session; GPU nodes only ever run training jobs.

The jobs are listed in `queues/hpc.txt`: two large distillation teachers and three <=5M students
on long schedules, all on the same 45k/5k split as the laptop runs.

## 1. Start a CPU Jupyter session
https://ood.burst.hpc.nyu.edu → *Interactive Apps* → *Jupyter*: account `cs_gy_6923-2026fa`,
partition `interactive`, ~3 hours → *Launch* → *Connect to Jupyter*.

## 2. Upload the code
JupyterLab file browser → *Upload* → `p1_code.zip` from the project folder.

## 3. Unpack, check, set up (one time, ~15-40 min)
*File → New → Terminal*:

```bash
mkdir -p /scratch/$USER/dl-p1 && cd /scratch/$USER/dl-p1
unzip -o ~/p1_code.zip -d code && cd code     # no unzip? python3 -m zipfile -e ~/p1_code.zip .
bash scripts/hpc/check_access.sh
bash scripts/hpc/setup_env.sh
```

`setup_env.sh` installs Python 3.12 + PyTorch and downloads CIFAR-10 into `/scratch/$USER/dl-p1`.
If the CIFAR-10 download stalls, upload `data/cifar10/cifar-10-python.tar.gz` from the laptop
into `/scratch/$USER/dl-p1/data/cifar10/` and run the script again.

## 4. Submit the training jobs (1 x A100 each)
```bash
bash scripts/hpc/submit_jobs.sh              # = cs_gy_6923-2026fa on c12m85-a100-1
squeue -u $USER                              # R = running, PD = waiting, PD (Dependency) = standby
tail -f runs/logs/<job-name>-<jobid>.out     # training log
sacct -u $USER --format=JobName%30,State,Elapsed   # finished / failed jobs
```

Jobs keep running after the Jupyter session ends. Each run is submitted as one job plus three
standby jobs (`RETRIES=<n> bash scripts/hpc/submit_jobs.sh ...` for another number). A standby job
starts only when all of the run's earlier jobs have ended unsuccessfully, and continues from the
last checkpoint (`runs/<name>/last.pt`, saved every 10 epochs); once the run finishes, Slurm cancels
its remaining standby jobs, so `sacct` lists them as `CANCELLED`. This covers the 24-hour time
limit, crashes, and cloud nodes that never come up (about half of our jobs: `sacct` shows the job
`CANCELLED` after about 12 minutes with no `batch` step). Running `submit_jobs.sh` again skips
finished runs, tops unfinished ones back up to four jobs, and resubmits runs that have none left.
To stop a run, cancel all of its jobs with `scancel -n <name>`: cancelling only the running job
starts a standby one.

A queue line starting with `after=<run>[,<run>...]` (e.g. a student distilled from teachers trained
by earlier lines) is submitted with a Slurm dependency: it shows as `PD (Dependency)` until those
runs' current jobs finish, and is cancelled automatically if one of them fails, even if a standby
job then finishes that run (run `submit_jobs.sh` again once it has).

## 5. Bring results back
When `runs/<name>/summary.json` exists, download from `runs/<name>/` (file browser, right-click →
*Download*): `model_final.pt`, `summary.json`, `log.csv`, `val_probs.npy`, into the same
`runs/<name>/` folder on the laptop.
