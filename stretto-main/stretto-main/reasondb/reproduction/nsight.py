"""Optional Nsight CLI capture. Never run Kineto and Nsight in the same process."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import re


FLAGS = ("--cuda-um-cpu-page-faults", "--cuda-um-gpu-page-faults", "--capture-range-end")


def check_nsys():
    executable = shutil.which("nsys")
    if executable is None:
        raise RuntimeError("profile-nsys requires the separate Nsight image (STRETTO_NSYS_IMAGE); nsys was not found")
    version = subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    help_text = subprocess.check_output([executable, "profile", "--help"], text=True, stderr=subprocess.STDOUT)
    missing = [flag for flag in FLAGS if flag not in help_text]
    if missing:
        raise RuntimeError(f"Nsight CLI {version} lacks required flags: {missing}")
    environment = subprocess.run([executable, "status", "--environment"], text=True, capture_output=True)
    package_hash = Path("/opt/stretto/nsight-package.sha256")
    return {"version": version, "package_sha256": package_hash.read_text().strip() if package_hash.exists() else None,
            "environment_returncode": environment.returncode,
            "environment": environment.stdout + environment.stderr}


def command(command, directory, role):
    destination = Path(directory) / "nsight"
    destination.mkdir(parents=True, exist_ok=True)
    return ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi", "--capture-range-end=repeat:2", "--kill=none",
            "--cuda-um-cpu-page-faults=true", "--cuda-um-gpu-page-faults=true",
            "--export=sqlite", f"--output={destination / role}", *command]


def analyze(directory):
    """A missing table is UNKNOWN, not zero faults. Preserve raw reports for the GUI."""
    directory = Path(directory)
    limitations = []
    for log in (directory.parent / "services").rglob("*.log"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(r"(?i)(page.?fault|unified.?memory|\bUVM\b)", line) and re.search(
                    r"(?i)(not support|unavailable|disabled|permission|requires.*root|failed|error)", line):
                limitations.append({"log": str(log), "message": line})
    results = []
    for path in (directory / "nsight").glob("*.sqlite"):
        item = {"file": str(path), "scope": "sampled benchmark batches", "tables": []}
        try:
            with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
                names = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                item["cuda_kernel_rows"] = None
                for name in names:
                    upper = name.upper()
                    if upper in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
                        safe = '"' + name.replace('"', '""') + '"'
                        item["cuda_kernel_rows"] = (item["cuda_kernel_rows"] or 0) + connection.execute(
                            f"SELECT COUNT(*) FROM {safe}").fetchone()[0]
                    if "UNIFIED_MEMORY" in upper or "PAGE_FAULT" in upper:
                        safe = '"' + name.replace('"', '""') + '"'
                        count = connection.execute(f"SELECT COUNT(*) FROM {safe}").fetchone()[0]
                        columns = [r[1] for r in connection.execute(f"PRAGMA table_info({safe})")]
                        table = {"name": name, "event_rows": count,
                                 "status": "observed" if count else "no_events_in_table"}
                        # GPU fault groups may represent multiple faults. Never equate
                        # a generic UM counter's `value` column with a fault count.
                        for column in columns:
                            if column.lower() in ("numberofpagefaults", "numberoffaults"):
                                quoted = '"' + column.replace('"', '""') + '"'
                                table["reported_fault_count"] = connection.execute(
                                    f"SELECT COALESCE(SUM({quoted}), 0) FROM {safe}").fetchone()[0]
                        item["tables"].append(table)
                item["status"] = "tables_exported" if item["tables"] else "no_um_tables; zero faults not established"
        except (sqlite3.Error, OSError) as exc:
            item.update(status="unavailable", reason=str(exc))
        results.append(item)
    report = {"schema_version": 1, "status": "reports_exported" if results else "unavailable",
              "scope": "sampled benchmark batches only", "reports": results,
              "raw_reports": [{"file": str(p), "bytes": p.stat().st_size}
                              for p in (directory / "nsight").glob("*.nsys-rep")],
              "collector_limitations": limitations,
              "note": "No Unified Memory allocation is introduced. Inspect CLI logs for collector warnings; missing events do not prove collector availability."}
    (directory / "cuda_page_faults.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
