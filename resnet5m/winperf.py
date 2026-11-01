"""Keep a training process at full CPU speed on Windows laptops.

Windows 11 may put processes it considers background into "efficiency mode" (EcoQoS): low
clocks and efficiency cores. Training is bound by the single Python thread that launches GPU
kernels, so this halves throughput. We opt the process out of execution-speed throttling and
raise its priority. No-op on other operating systems.
"""
import ctypes
import os

_PROCESS_POWER_THROTTLING = 4            # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLE_EXECUTION_SPEED = 0x1
_THROTTLE_IGNORE_TIMER_RESOLUTION = 0x4
_ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
_PROCESS_SET_INFORMATION = 0x0200


class _ThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def disable_power_throttling(pid=None):
    """Opt process `pid` (default: this process) out of EcoQoS. Returns True on success."""
    if os.name != "nt":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
    if not handle:
        return False
    state = _ThrottlingState(1, _THROTTLE_EXECUTION_SPEED | _THROTTLE_IGNORE_TIMER_RESOLUTION, 0)
    ok = k32.SetProcessInformation(handle, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state))
    k32.SetPriorityClass(handle, _ABOVE_NORMAL_PRIORITY_CLASS)
    if pid is not None:
        k32.CloseHandle(handle)
    return bool(ok)


if __name__ == "__main__":  # python -m resnet5m.winperf <pid> ... : apply to running processes
    import sys
    for p in sys.argv[1:]:
        print(p, disable_power_throttling(int(p)))
