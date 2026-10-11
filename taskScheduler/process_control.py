from __future__ import annotations

import contextlib
import os
import signal
import subprocess  # nosec B404 - runs local gpann_modular binary
import time
from typing import List, Optional


LINE_BUFFERED_PREFIX = ["stdbuf", "-oL", "-eL"]


class ManagedProcess:
    def __init__(self, command: List[str], log_filepath: str) -> None:
        self.command = command
        self.log_filepath = log_filepath
        self.started_at = time.monotonic()
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)
        self.log_file = open(log_filepath, "w")
        self.process = subprocess.Popen(  # nosec B603
            LINE_BUFFERED_PREFIX + command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def is_running(self) -> bool:
        return self.process.poll() is None

    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def read_log(self) -> str:
        self.log_file.flush()
        if os.path.exists(self.log_filepath):
            with open(self.log_filepath, "r", errors="replace") as f:
                return f.read()
        return ""

    def signal_process_group(self, signal_number: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal_number)

    def terminate(self, notice_seconds: float = 0.0) -> None:
        if not self.is_running():
            self._close_log()
            return

        if notice_seconds > 0:
            self.signal_process_group(signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=notice_seconds)

        if self.is_running():
            self.signal_process_group(signal.SIGKILL)
            self.process.wait()

        self._close_log()

    def _close_log(self) -> None:
        """Closes internal file handles safely."""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()