"""Optional profiling sidecars. No torch dependency until a backend opts in.

Wall intervals can overlap (prefetch threads, nested generation/forward spans).
They are observations, never an additive end-to-end breakdown.
"""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
try:
    import resource as _resource
except ImportError:
    _resource = None
import threading
import time
import uuid


SCHEMA = 1
ACTIVE = None


def proc_stat(text):
    # comm can contain spaces and parentheses. Fields after the LAST ')' start at 3.
    fields = text[text.rfind(")") + 2:].split()
    return {"minor_faults": int(fields[7]), "major_faults": int(fields[9]),
            "start_ticks": int(fields[19])}


def proc_counters(pid=None, root="/proc"):
    try:
        directory = Path(root) / str(pid or os.getpid())
        result = proc_stat((directory / "stat").read_text())
        io = dict(line.split(":", 1) for line in (directory / "io").read_text().splitlines())
        result.update({key: int(io[key]) for key in ("read_bytes", "write_bytes", "rchar", "wchar")})
        return {"status": "measured", **result}
    except PermissionError:
        return {"status": "permission_denied"}
    except (OSError, ValueError, KeyError):
        return {"status": "unavailable"}


def counter_delta(before, after):
    if before.get("status") != "measured" or after.get("status") != "measured":
        return {"status": after.get("status", "unavailable")}
    if before.get("start_ticks") != after.get("start_ticks"):
        return {"status": "process_restarted"}
    return {"status": "measured", **{k: max(0, after[k] - before[k])
            for k in ("minor_faults", "major_faults", "read_bytes", "write_bytes", "rchar", "wchar")}}


def thread_faults():
    if _resource is None or not hasattr(_resource, "RUSAGE_THREAD"):
        return None
    value = _resource.getrusage(_resource.RUSAGE_THREAD)
    return value.ru_minflt, value.ru_majflt


def cgroup_memory(root="/sys/fs/cgroup"):
    root = Path(root)
    try:
        # Resolve the process's cgroup, including hosts without a cgroup namespace.
        relative = ""
        if str(root) == "/sys/fs/cgroup":
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    relative = line.split("::", 1)[1].lstrip("/")
                elif "memory" in line.split(":", 2)[1].split(","):
                    relative = "memory/" + line.split(":", 2)[2].lstrip("/")
        candidates = [root / relative, root, root / "memory"]
        for directory in candidates:
            current = directory / "memory.current"
            limit = directory / "memory.max"
            if not current.exists():
                current = directory / "memory.usage_in_bytes"
                limit = directory / "memory.limit_in_bytes"
            if current.exists():
                cap = limit.read_text().strip()
                return {"status": "measured", "used_bytes": int(current.read_text()),
                        "limit_bytes": None if cap == "max" else int(cap)}
        return {"status": "unavailable"}
    except PermissionError:
        return {"status": "permission_denied"}
    except (OSError, ValueError):
        return {"status": "unavailable"}


def request_headers():
    if not os.environ.get("STRETTO_STAGE_FILE"):
        return {}
    from reasondb.utils.timing import current_phase
    return {"X-Stretto-Phase": current_phase() or "prepare"}


def set_stage(stage):
    path = os.environ.get("STRETTO_STAGE_FILE")
    if path:
        target = Path(path)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(stage, encoding="utf-8")
        temporary.replace(target)


class TraceQuota:
    def __init__(self):
        self.claimed = set()

    def claim(self, stage, route):
        route = "vanilla" if route == "direct" else route
        if stage != "benchmark" or route not in ("vanilla", "kv") or route in self.claimed:
            return False
        self.claimed.add(route)
        return True


class Probe:
    def __init__(self, directory, role, model, torch, engine="none", detailed=False):
        self.directory, self.role, self.model = Path(directory), role, model
        self.directory.mkdir(parents=True, exist_ok=True)
        self.torch, self.engine, self.detailed = torch, engine, detailed
        self.lock = threading.Lock()
        self.local = threading.local()
        self.quota = TraceQuota()
        self.context = {}
        self.pending = []
        self.peaks = {"allocated_bytes": 0, "reserved_bytes": 0}
        self.forward_times = {"prefill_s": 0., "decode_s": 0.}
        self.forward_observed = False

    def write(self, kind, **data):
        row = {"schema_version": SCHEMA, "kind": kind, "at": time.time(),
               "pid": os.getpid(), "role": self.role, "model": self.model,
               **self.context, "batch_id": getattr(self.local, "batch_id", None), **data}
        with self.lock:
            with (self.directory / f"{self.role}_phases.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def limitation(self, feature, exc):
        self.write("limitation", feature=feature, status="unavailable", reason=str(exc))

    def begin_request(self, phase, path):
        stage_file = os.environ.get("STRETTO_STAGE_FILE")
        stage = Path(stage_file).read_text().strip() if stage_file and Path(stage_file).exists() else "unknown"
        self.context = {"request_id": uuid.uuid4().hex, "stage": stage, "phase": phase, "endpoint": path}
        self.peaks = {"allocated_bytes": self.torch.cuda.memory_allocated(),
                      "reserved_bytes": self.torch.cuda.memory_reserved()}
        self.torch.cuda.reset_peak_memory_stats()
        self.forward_times = {"prefill_s": 0., "decode_s": 0.}
        self.forward_observed = False
        self.before = proc_counters()
        self.capture_peaks()

    def capture_peaks(self):
        if self.torch.cuda.is_available():
            allocated = self.torch.cuda.max_memory_allocated()
            reserved = self.torch.cuda.max_memory_reserved()
            with self.lock:
                self.peaks["allocated_bytes"] = max(self.peaks["allocated_bytes"], allocated)
                self.peaks["reserved_bytes"] = max(self.peaks["reserved_bytes"], reserved)

    def finish_request(self):
        self.flush()
        self.capture_peaks()
        self.write("request_resources", counters=proc_counters(),
                   delta=counter_delta(self.before, proc_counters()), peaks=self.peaks.copy())

    def flush(self):
        with self.lock:
            pending, self.pending = self.pending, []
        if not pending:
            return
        # One synchronization per batch/request, never per token.
        available = True
        try:
            self.torch.cuda.synchronize()
        except Exception as exc:
            available = False
            self.limitation("cuda_event_timing", exc)
        for start, end, row in pending:
            seconds = None
            if available:
                try:
                    seconds = start.elapsed_time(end) / 1000
                except Exception as exc:
                    self.limitation("cuda_event_timing", exc)
            if seconds is not None:
                name = row["name"]
                if name in ("vanilla_prefill", "cached_query_prefill", "decode"):
                    self.forward_observed = True
                    self.forward_times["decode_s" if name == "decode" else "prefill_s"] += seconds
            self.write("span", **row, gpu_elapsed_s=seconds,
                       gpu_timing_status="measured" if seconds is not None else "unavailable")

    @contextmanager
    def span(self, name, gpu=False, **metadata):
        if not self.detailed:
            yield
            return
        start = time.perf_counter()
        context = self.context.copy()
        batch_id = getattr(self.local, "batch_id", None)
        faults = thread_faults()
        events = None
        annotation = None
        try:
            if gpu and self.torch.cuda.is_available():
                events = (self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True))
                events[0].record()
            if self.engine == "nsys":
                self.torch.cuda.nvtx.range_push(name)
                annotation = "nvtx"
            elif self.engine == "torch":
                annotation = self.torch.profiler.record_function(name)
                annotation.__enter__()
        except Exception as exc:
            events = None
            self.limitation("span_annotation", exc)
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            cpu_s = time.perf_counter() - start
            if events:
                events[1].record()
            if annotation == "nvtx":
                self.torch.cuda.nvtx.range_pop()
            elif annotation is not None:
                annotation.__exit__(None, None, None)
            after = thread_faults()
            row = {"name": name, "start_monotonic_s": start, "cpu_wall_s": cpu_s,
                   "thread_id": threading.get_native_id(), "intervals_overlap": True,
                   "thread_minor_faults": after[0] - faults[0] if faults and after else None,
                   "thread_major_faults": after[1] - faults[1] if faults and after else None,
                   **context, "batch_id": batch_id, "error_type": error, **metadata}
            self.capture_peaks()
            if self.torch.cuda.is_available():
                row.update(allocated_bytes=self.torch.cuda.memory_allocated(),
                           reserved_bytes=self.torch.cuda.memory_reserved(),
                           peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
            if events:
                # Capture context now; prefetch may finish on another thread/batch.
                with self.lock:
                    self.pending.append((*events, row))
            else:
                self.write("span", **row, gpu_elapsed_s=None)

    @contextmanager
    def batch(self, route, index):
        self.local.batch_id = f"{self.context.get('request_id', 'standalone')}:{route}:{index}"
        trace = None
        selected = self.engine != "none" and self.quota.claim(self.context.get("stage"), route)
        normalized = "vanilla" if route == "direct" else route
        trace_path = self.directory / "traces" / f"{self.role}_{normalized}.json"
        started = False
        if selected:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self.engine == "torch":
                    trace = self.torch.profiler.profile(
                        activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=True)
                    trace.__enter__()
                else:
                    self.torch.cuda.profiler.start()
                started = True
            except Exception as exc:
                self.limitation(f"{self.engine}_trace", exc)
        try:
            with self.span("batch", gpu=True, route=route):
                yield
        finally:
            self.flush()
            if started:
                try:
                    if trace is not None:
                        trace.__exit__(None, None, None)
                        trace.export_chrome_trace(str(trace_path))
                        with trace_path.open(encoding="utf-8") as handle:
                            exported = json.load(handle)
                        gpu_events = sum(event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                                         for event in exported.get("traceEvents", []))
                        del exported
                        status = "captured" if gpu_events else "cpu_only"
                        if not gpu_events:
                            self.limitation("torch_cuda_trace", "Export contains no CUDA kernel/memcpy/memset events")
                    else:
                        self.torch.cuda.profiler.stop()
                        status = "range_completed"
                    self.write("trace", engine=self.engine, route=normalized,
                               status=status,
                               path=str(trace_path) if trace is not None else "nsight capture range")
                except Exception as exc:
                    self.limitation(f"{self.engine}_trace_export", exc)
            self.local.batch_id = None


@contextmanager
def span(name, gpu=False, **metadata):
    if ACTIVE is None:
        yield
    else:
        with ACTIVE.span(name, gpu, **metadata):
            yield


@contextmanager
def batch(route, index):
    if ACTIVE is None:
        yield
    else:
        with ACTIVE.batch(route, index):
            yield


def reset_peaks(device=None):
    import torch
    if ACTIVE is not None:
        ACTIVE.capture_peaks()
    torch.cuda.reset_peak_memory_stats(device)


def synchronize():
    import torch
    with span("gpu_sync_wait"):
        torch.cuda.synchronize()


def transfer(tensor, device, **kwargs):
    import torch
    target = torch.device(device)
    source = tensor.device
    if source.type == target.type and (target.index is None or source.index == target.index):
        return tensor.to(device, **kwargs)
    direction = f"{source.type}_to_{target.type}"
    with span(direction, gpu=True, tensor_bytes=tensor.numel() * tensor.element_size()):
        return tensor.to(device, **kwargs)


def load_cache(path, **kwargs):
    import torch
    if ACTIVE is None or not ACTIVE.detailed:
        return torch.load(path, **kwargs)
    before = proc_counters() if ACTIVE and ACTIVE.detailed else None
    with span("cache_file_load", file_bytes=Path(path).stat().st_size,
              io_scope="process counters; may include concurrent prefetch/readahead"):
        result = torch.load(path, **kwargs)
    if before is not None:
        ACTIVE.write("cache_io", file_bytes=Path(path).stat().st_size,
                     delta=counter_delta(before, proc_counters()))
    return result


def save_cache(value, path):
    import torch
    with span("cache_file_save"):
        torch.save(value, path)


def prefill(model, **kwargs):
    ids = kwargs.get("input_ids")
    with span("cache_build_prefill", gpu=True, input_tokens=int(ids.numel()) if ids is not None else None):
        return model(**kwargs)


def generated_tokens(sequences, input_length, eos_ids, pad_id):
    if eos_ids is None:
        eos_ids = []
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    counts = []
    for row in sequences:
        count = 0
        for token in row[input_length:]:
            # A PAD id emitted before EOS is a real generated token. Generation
            # inserts padding only after a sequence finishes; stop at its first EOS.
            count += 1
            if token in eos_ids:
                break
        counts.append(count)
    return counts


def generate(model, **kwargs):
    if ACTIVE is None or not ACTIVE.detailed:
        return model.generate(**kwargs)
    probe = ACTIVE
    forward_count = 0
    stack = []
    cached = kwargs.get("past_key_values") is not None
    def before(_module, args, inputs):
        nonlocal forward_count
        ids = inputs.get("input_ids", args[0] if args else None)
        name = ("cached_query_prefill" if cached else "vanilla_prefill") if forward_count == 0 else "decode"
        scope = probe.span(name, gpu=True, input_tokens=int(ids.numel()) if ids is not None else None)
        scope.__enter__()
        stack.append(scope)
        forward_count += 1
    def after(_module, _args, _kwargs, _output):
        if stack:
            stack.pop().__exit__(None, None, None)
    hooks = [model.register_forward_pre_hook(before, with_kwargs=True),
             model.register_forward_hook(after, with_kwargs=True, always_call=True)]
    started = time.perf_counter()
    try:
        with probe.span("generation", gpu=True):
            result = model.generate(**kwargs)
        generation_cpu_s = time.perf_counter() - started
        input_length = kwargs["input_ids"].shape[1]
        config = getattr(model, "generation_config", None)
        eos = kwargs.get("eos_token_id", getattr(config, "eos_token_id", None))
        counts = generated_tokens(result.sequences.detach().cpu().tolist(), input_length,
                                  eos, kwargs.get("pad_token_id", getattr(config, "pad_token_id", None)))
        probe.write("generation", cpu_wall_s=generation_cpu_s, forward_count=forward_count,
                    decode_forward_count=max(0, forward_count - 1), output_tokens=sum(counts),
                    output_tokens_per_item=counts, cached_prefix=cached)
        if forward_count == 0:
            probe.limitation("forward_hooks", "generate did not invoke the model forward hooks")
        return result
    finally:
        for hook in hooks:
            hook.remove()
        while stack:
            stack.pop().__exit__(None, None, None)


def summarize(directory):
    """Keep sum-of-intervals separate from end-to-end time; stream large sidecars."""
    directory = Path(directory)
    groups = {}
    batches, requests = {}, []
    limitations, traces = [], []
    fields = ["role", "stage", "phase", "name", "observations", "cpu_wall_sum_s", "gpu_elapsed_sum_s", "tensor_bytes", "input_tokens"]
    for path in directory.glob("*_phases.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                batch_id = row.get("batch_id")
                item_batch = None
                if batch_id:
                    key_batch = (row["role"], batch_id)
                    item_batch = batches.setdefault(key_batch, {
                        **{k: row.get(k) for k in ("role", "stage", "phase", "request_id", "batch_id")},
                        "generation_cpu_s": None, "output_tokens": None, "decode_forward_count": None,
                        "prefill_gpu_s": None, "decode_gpu_s": None, "h2d_bytes": 0, "d2h_bytes": 0,
                        "h2d_gpu_s": None, "d2h_gpu_s": None, "peak_allocated_bytes": None,
                        "peak_reserved_bytes": None})
                if row["kind"] == "generation" and item_batch is not None:
                    item_batch.update(generation_cpu_s=row["cpu_wall_s"], output_tokens=row["output_tokens"],
                                      decode_forward_count=row["decode_forward_count"])
                    if row["decode_forward_count"] == 0 and item_batch["prefill_gpu_s"] is not None:
                        item_batch["decode_gpu_s"] = 0.
                if row["kind"] == "request_resources":
                    requests.append({**{k: row.get(k) for k in ("role", "stage", "phase", "request_id")},
                                     "delta": row["delta"], "peaks": row["peaks"]})
                if row["kind"] == "limitation":
                    limitations.append(row)
                elif row["kind"] == "trace":
                    traces.append(row)
                elif row["kind"] == "span":
                    if item_batch is not None:
                        name = row["name"]
                        target = {"vanilla_prefill": "prefill_gpu_s", "cached_query_prefill": "prefill_gpu_s",
                                  "decode": "decode_gpu_s", "cpu_to_cuda": "h2d_gpu_s", "cuda_to_cpu": "d2h_gpu_s"}.get(name)
                        if target and row.get("gpu_elapsed_s") is not None:
                            item_batch[target] = (item_batch[target] or 0) + row["gpu_elapsed_s"]
                        if name in ("cpu_to_cuda", "cuda_to_cpu"):
                            item_batch["h2d_bytes" if name == "cpu_to_cuda" else "d2h_bytes"] += row.get("tensor_bytes", 0)
                        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
                            if row.get(field) is not None:
                                item_batch[field] = max(item_batch[field] or 0, row[field])
                    key = tuple(row.get(k) for k in fields[:4])
                    item = groups.setdefault(key, {**dict(zip(fields[:4], key)), "observations": 0,
                        "cpu_wall_sum_s": 0., "gpu_elapsed_sum_s": None, "tensor_bytes": 0, "input_tokens": 0})
                    item["observations"] += 1
                    item["cpu_wall_sum_s"] += row["cpu_wall_s"]
                    if row.get("gpu_elapsed_s") is not None:
                        item["gpu_elapsed_sum_s"] = (item["gpu_elapsed_sum_s"] or 0) + row["gpu_elapsed_s"]
                    item["tensor_bytes"] += row.get("tensor_bytes") or 0
                    item["input_tokens"] += row.get("input_tokens") or 0
    # Forward GPU records are flushed after the generation metadata.
    for item in batches.values():
        if item["decode_forward_count"] == 0 and item["prefill_gpu_s"] is not None:
            item["decode_gpu_s"] = 0.
    memory = {"sampled_gpu_peak_used_bytes": None, "processes": {}}
    resources = directory / "resources.jsonl"
    if resources.exists():
        with resources.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                for proc in sample.get("processes", []):
                    identity = f"{proc['pid']}:{proc['created_at']}"
                    entry = memory["processes"].setdefault(identity, {"role": proc["role"],
                        "rss_peak_bytes": 0, "uss_peak_bytes": None, "pss_peak_bytes": None,
                        "minor_fault_delta_sum": None, "major_fault_delta_sum": None,
                        "read_bytes_delta_sum": None, "write_bytes_delta_sum": None,
                        "counter_scope": "between samples; excludes activity before first/after last sample"})
                    for source, target in (("rss_bytes", "rss_peak_bytes"), ("uss_bytes", "uss_peak_bytes"), ("pss_bytes", "pss_peak_bytes")):
                        if proc.get(source) is not None:
                            entry[target] = max(entry[target] or 0, proc[source])
                    delta = proc.get("delta", {})
                    if delta.get("status") == "measured":
                        for source, target in (("minor_faults", "minor_fault_delta_sum"), ("major_faults", "major_fault_delta_sum"),
                                               ("read_bytes", "read_bytes_delta_sum"), ("write_bytes", "write_bytes_delta_sum")):
                            entry[target] = (entry[target] or 0) + delta[source]
                for values in csv.reader(sample.get("gpu_csv", "").splitlines()):
                    if len(values) == 5:
                        used = int(float(values[2].strip()) * 1024**2)
                        memory["sampled_gpu_peak_used_bytes"] = max(memory["sampled_gpu_peak_used_bytes"] or 0, used)
    report = {"schema_version": SCHEMA, "intervals_overlap": True,
              "timing_semantics": "interval sums, not additive wall-clock or pure GPU kernel time",
              "spans": list(groups.values()), "traces": traces, "limitations": limitations,
              "memory_and_faults": memory, "requests": requests,
              "trace_coverage": [{"role": r, "route": route, "status": next((
                  t["status"] for t in traces if t["role"] == r and t["route"] == route), "not_captured")}
                  for r in ("small", "large") for route in ("vanilla", "kv")]}
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups.values())
    batch_fields = ["role", "stage", "phase", "request_id", "batch_id", "generation_cpu_s", "output_tokens",
                    "decode_forward_count", "prefill_gpu_s", "decode_gpu_s", "h2d_bytes", "d2h_bytes",
                    "h2d_gpu_s", "d2h_gpu_s", "peak_allocated_bytes", "peak_reserved_bytes"]
    with (directory / "batches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=batch_fields)
        writer.writeheader()
        writer.writerows(batches.values())
    return report
