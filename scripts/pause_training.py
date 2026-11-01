"""Pause or resume all local train.py processes (and their DataLoader workers) on Windows.

Suspending keeps all progress (no checkpoint needed) but the paused run still holds its GPU
memory and RAM. To free those or reboot, stop the run instead: unfinished runs continue from
their last checkpoint when the queue (scripts/run_queue.sh) or train.py --resume is run again.

Usage: python scripts/pause_training.py pause | resume | status
"""
import ctypes
import json
import subprocess
import sys

PROCESS_SUSPEND_RESUME = 0x0800


def training_processes():
    """(pid, description) for train.py processes and their child worker processes."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json")
    procs = json.loads(subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                      capture_output=True, text=True).stdout or "[]")
    procs = [procs] if isinstance(procs, dict) else procs
    trainers = {p["ProcessId"] for p in procs if "train.py" in (p["CommandLine"] or "")}
    return [(p["ProcessId"], "train.py" if p["ProcessId"] in trainers else "worker")
            for p in procs if p["ProcessId"] in trainers or p["ParentProcessId"] in trainers]


def apply(action):
    ntdll, k32 = ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32")
    fn = ntdll.NtSuspendProcess if action == "pause" else ntdll.NtResumeProcess
    procs = training_processes()
    for pid, kind in procs:
        handle = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        status = fn(handle) if handle else -1
        k32.CloseHandle(handle)
        print(f"{action:6s} {kind:8s} pid {pid}: {'ok' if status == 0 else 'failed'}")
    if not procs:
        print("no train.py processes found")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd in ("pause", "resume"):
        apply(cmd)
    else:
        for pid, kind in training_processes():
            print(f"{kind:8s} pid {pid}")
