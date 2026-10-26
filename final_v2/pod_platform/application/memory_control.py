"""Serialize large research jobs and spill retained arrays without changing data."""

from __future__ import annotations

import contextlib
import fcntl
import gc
import json
import mmap
import os
import signal
import threading
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def research_job_lock(state):
    """Fail clearly instead of silently launching a second full dataset process."""
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "heavy_research.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another data-heavy research job holds the lock; finish it before starting this job"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def release_pages(array):
    if isinstance(array, np.memmap):
        array.flush()
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and hasattr(mapping, "madvise"):
            mapping.madvise(mmap.MADV_DONTNEED)


def spill_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = np.lib.format.open_memmap(
        path, mode="w+", dtype=array.dtype, shape=array.shape
    )
    stored[:] = array
    release_pages(stored)
    return stored


def spill_schedule(path, schedule):
    if not schedule:
        return schedule
    days = sorted(schedule)
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(days), len(schedule[days[0]])),
    )
    for row, day in enumerate(days):
        values[row] = schedule[day]
    result = {day: values[row] for row, day in enumerate(days)}
    release_pages(values)
    return result


def memory_sample():
    status = Path("/proc/self/status")
    result = {"pid": os.getpid()}
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                result[key.rstrip(":") + "_kib"] = int(value)
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["system_available_kib"] = int(line.split()[1])
    current_path = Path("/sys/fs/cgroup/memory.current")
    limit_path = Path("/sys/fs/cgroup/memory.max")
    if current_path.exists() and limit_path.exists():
        limit = limit_path.read_text().strip()
        if limit.isdigit():
            result["cgroup_available_kib"] = (
                max(0, int(limit) - int(current_path.read_text())) // 1024
            )
    return result


def collect_research_memory(arrays=()):
    for array in arrays:
        release_pages(array)
    gc.collect()


@contextlib.contextmanager
def memory_watchdog(max_rss_mib=1600, minimum_available_mib=800):
    """Stop this worker before its resident memory can exhaust the desktop.

    Only the current research worker receives SIGINT. No other user process is
    killed. The last sample remains on disk through the campaign checkpoints.
    """
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.5):
            sample = memory_sample()
            resident = sample.get("VmRSS_kib", 0) / 1024
            available = (
                min(
                    sample.get("system_available_kib", 1e12),
                    sample.get("cgroup_available_kib", 1e12),
                )
                / 1024
            )
            if resident > max_rss_mib or available < minimum_available_mib:
                print(
                    f"Memory guard stopping research worker: RSS {resident:.0f} MiB, available {available:.0f} MiB",
                    flush=True,
                )
                os.kill(os.getpid(), signal.SIGINT)
                return

    watcher = threading.Thread(
        target=monitor, name="research_memory_guard", daemon=True
    )
    watcher.start()
    try:
        yield
    finally:
        stopped.set()
        watcher.join(timeout=1)
