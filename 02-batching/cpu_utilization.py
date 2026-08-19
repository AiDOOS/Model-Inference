"""
Module 02 - CPU utilization sampler, shared by all four approaches.

Throughput and latency tell you WHAT happened. On CPU-only hardware,
"how much of the machine was actually working while that happened" is the
missing number - it's the direct analog of "GPU utilization %" on a GPU
box, and it's the actual evidence for "batching uses the hardware better,"
not just "batching produces bigger numbers."

Usage:
    sampler = CpuUtilizationSampler(interval_s=0.5)
    sampler.start()
    ... run the benchmark ...
    samples = sampler.stop()   # list of {"t": elapsed_s, "cpu_percent": float}
"""

import threading
import time
from typing import Dict, List

import psutil


class CpuUtilizationSampler:
    """Samples system-wide CPU utilization on a background thread at a
    fixed interval, independent of whatever the main thread is doing.

    psutil.cpu_percent() with no `interval` argument reports usage since the
    LAST call - so the first sample in any run is meaningless (it covers an
    unknown span before start() was called) and is dropped.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._samples: List[Dict] = []
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)  # prime it; see docstring
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval_s)
            cpu_percent = psutil.cpu_percent(interval=None)
            self._samples.append({
                "t": time.perf_counter() - self._start_time,
                "cpu_percent": cpu_percent,
            })

    def start(self) -> None:
        self._samples = []
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        return self._samples

    @staticmethod
    def average(samples: List[Dict]) -> float:
        if not samples:
            return float("nan")
        return sum(s["cpu_percent"] for s in samples) / len(samples)
