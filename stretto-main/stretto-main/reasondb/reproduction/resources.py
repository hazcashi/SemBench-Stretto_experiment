"""Process identity-aware resource sampling, including Linux faults and I/O."""
import os
import time

from reasondb.reproduction.telemetry import cgroup_memory, proc_counters, counter_delta
from pathlib import Path


class ResourceSampler:
    def __init__(self):
        self.previous = {}

    def sample(self):
        import psutil
        root = psutil.Process()
        rows = []
        for proc in [root, *root.children(recursive=True)]:
            try:
                identity = (proc.pid, proc.create_time())
                command = proc.cmdline()
                role = "runner" if proc.pid == os.getpid() else "child"
                if command and Path(command[0]).name.startswith("nsys"):
                    role = "nsys"
                elif "--role" in command:
                    role = command[command.index("--role") + 1]
                elif any("text_embed_server" in part for part in command):
                    role = "embedding"
                elif any("nsys" in part for part in command):
                    role = "nsys"
                memory = proc.memory_info()
                counters = proc_counters(proc.pid)
                old = self.previous.get(identity)
                row = {"pid": proc.pid, "created_at": identity[1], "role": role,
                       "rss_bytes": memory.rss, "counters": counters,
                       "delta": counter_delta(old, counters) if old else {"status": "first_sample"}}
                try:
                    full = proc.memory_full_info()
                    row.update(uss_bytes=getattr(full, "uss", None), pss_bytes=getattr(full, "pss", None))
                except (psutil.Error, OSError):
                    row.update(uss_bytes=None, pss_bytes=None)
                rows.append(row)
                self.previous[identity] = counters
            except (psutil.Error, OSError):
                continue
        alive = {(r["pid"], r["created_at"]) for r in rows}
        self.previous = {k: v for k, v in self.previous.items() if k in alive}
        host = psutil.virtual_memory()
        stage_file = os.environ.get("STRETTO_STAGE_FILE")
        stage = Path(stage_file).read_text().strip() if stage_file and Path(stage_file).exists() else "unknown"
        return {"schema_version": 1, "at": time.time(), "stage": stage, "processes": rows,
                "process_tree_rss_bytes": sum(r["rss_bytes"] for r in rows),
                "host_memory": {"total_bytes": host.total, "available_bytes": host.available, "used_bytes": host.used},
                "cgroup_memory": cgroup_memory()}
