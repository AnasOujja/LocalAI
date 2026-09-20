"""Utilities for tracking process memory usage (RSS) during training."""
from __future__ import annotations

import threading
import time

import psutil


class PeakMemoryTracker:
    """Samples the current process' RSS in a background thread and keeps
    track of the peak value observed, in bytes."""

    def __init__(self, interval: float = 0.05):
        self._process = psutil.Process()
        self._interval = interval
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                self._peak_bytes = max(self._peak_bytes, rss)
            except psutil.Error:
                pass
            time.sleep(self._interval)

    def __enter__(self):
        self._peak_bytes = self._process.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 ** 2)
