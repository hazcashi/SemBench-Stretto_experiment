"""Owned subprocesses only: readiness checks, bounded waits and cleanup."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


class Services:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.children = []

    def start(self, name, command, port, timeout, expected_model=None):
        import requests
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))  # Refuse to reuse an unknown server.
        log_path = self.log_dir / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != "nt"))
        except BaseException:
            handle.close()
            raise
        self.children.append((process, handle))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited ({process.returncode}); see {log_path}")
            try:
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=2)
                response.raise_for_status()
                status = response.json()
                if expected_model and status.get("model_name") != expected_model:
                    raise RuntimeError(f"Wrong model on port {port}")
                return status
            except requests.RequestException:
                time.sleep(.5)
        raise TimeoutError(f"{name} not ready after {timeout}s; see {log_path}")

    def close(self):
        for process, handle in reversed(self.children):
            try:
                if process.poll() is None:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
            finally:
                handle.close()
        self.children.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
