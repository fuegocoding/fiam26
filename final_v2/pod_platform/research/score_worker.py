"""Hard resource boundaries for untrusted numerical strategy expressions."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import resource
import threading
import time
from pathlib import Path

import numpy as np


def _worker(connection, function, extra_memory_mib):
    if isinstance(function, dict):
        compiler = importlib.import_module(
            "pod_platform.research.strategy_code"
        ).compile_strategy
        function = compiler(function["source"], function["features"])
    try:
        status = Path("/proc/self/status").read_text().splitlines()
        virtual_kib = next(
            int(line.split()[1])
            for line in status
            if line.startswith("VmSize:")
        )
        limit = virtual_kib * 1024 + extra_memory_mib * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        connection.send("ready")
        while True:
            message = connection.recv()
            if message is None:
                return
            features, history, daily = message
            for array in [*features.values(), *history.values(), daily]:
                array.setflags(write=False)
            try:
                with np.errstate(all="ignore"):
                    value = np.asarray(
                        function(features, history, daily), dtype=float
                    )
                if value.shape != (daily.shape[1],) or np.isinf(value).any():
                    raise ValueError(
                        "score must return one finite-or-missing value per security"
                    )
                connection.send((True, value))
            except (Exception, MemoryError) as exc:
                connection.send(
                    (
                        False,
                        {"type": type(exc).__name__, "reason": str(exc)[:400]},
                    )
                )
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()
        os._exit(0)


class ScoreWorker:
    """One numerical subprocess per strategy, bounded per call and in memory.

    A fresh interpreter receives only the declared past-data packet. It does
    not inherit the research process's datasets, threads or provider clients.
    Globals expose the finite numpy facade, with a hard address-space allowance
    beyond interpreter startup. Timeout/worker failure rejects the strategy.
    """

    def __init__(self, function, *, timeout_seconds=3, extra_memory_mib=192):
        self.timeout_seconds = timeout_seconds
        context = mp.get_context("spawn")
        if hasattr(function, "_strategy_source"):
            function = {
                "source": function._strategy_source,
                "features": function._strategy_feature_names,
            }
        self.parent, child = context.Pipe()
        self.process = context.Process(
            target=_worker,
            args=(child, function, extra_memory_mib),
            daemon=True,
        )
        self.process.start()
        child.close()
        try:
            if not self.parent.poll(15) or self.parent.recv() != "ready":
                raise ValueError("Numerical worker startup failed")
        except (EOFError, OSError, ValueError):
            self.close()
            raise ValueError("Numerical worker startup failed") from None

    def score(self, features, history, daily):
        packet = (features, history, daily)
        size = sum(
            a.nbytes for a in [*features.values(), *history.values(), daily]
        )
        if size > 64 * 1024**2:
            raise ValueError("Strategy packet exceeds 64 MiB resource bound")
        done, errors = threading.Event(), []
        deadline = time.monotonic() + self.timeout_seconds

        def send():
            try:
                self.parent.send(packet)
            except (EOFError, BrokenPipeError, OSError) as exc:
                errors.append(type(exc).__name__)
            finally:
                done.set()

        sender = threading.Thread(target=send, daemon=True)
        sender.start()
        if not done.wait(self.timeout_seconds):
            self.process.terminate()
            self.process.join(timeout=0.5)
            self.parent.close()
            sender.join(timeout=0.5)
            raise ValueError("Strategy exceeded numerical worker send timeout")
        if errors:
            raise ValueError(
                "Strategy worker stopped within its hard resource boundary"
            )
        try:
            if not self.parent.poll(max(0, deadline - time.monotonic())):
                self.close()
                raise ValueError("Strategy exceeded numerical worker timeout")
            success, result = self.parent.recv()
        except (EOFError, BrokenPipeError, OSError):
            raise ValueError(
                "Strategy worker stopped within its hard resource boundary"
            ) from None
        if not success:
            raise ValueError(
                f"Strategy worker rejected expression: {result['type']}: {result['reason']}"
            )
        return result

    def close(self):
        try:
            if self.process.is_alive():
                self.parent.send(None)
        except (OSError, BrokenPipeError):
            pass
        self.process.join(timeout=0.2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=0.5)
        self.parent.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
