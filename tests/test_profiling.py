"""CPU contract tests plus an opt-in, weight-download-free CUDA regression."""
from contextlib import nullcontext
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" if (ROOT / "source").exists() else ROOT / "stretto-main/stretto-main"
sys.path.insert(0, str(SOURCE))
from reasondb.reproduction import telemetry as t
from reasondb.reproduction import nsight


def stat_text(minor=10, major=2, started=77):
    fields = ["0"] * 24
    fields[0], fields[7], fields[9], fields[19] = "S", str(minor), str(major), str(started)
    return "42 (worker (with spaces)) " + " ".join(fields)


def test_proc_parser_io_and_restart(tmp_path):
    proc = tmp_path / "42"
    proc.mkdir()
    (proc / "stat").write_text(stat_text())
    (proc / "io").write_text("read_bytes: 0\nwrite_bytes: 20\nrchar: 100\nwchar: 40\n")
    before = t.proc_counters(42, tmp_path)
    (proc / "stat").write_text(stat_text(15, 3))
    after = t.proc_counters(42, tmp_path)
    delta = t.counter_delta(before, after)
    assert delta["minor_faults"] == 5 and delta["major_faults"] == 1
    assert delta["read_bytes"] == 0  # Successful logical reads need not hit disk.
    after["start_ticks"] = 88
    assert t.counter_delta(before, after)["status"] == "process_restarted"
    assert t.proc_counters(999, tmp_path)["status"] == "unavailable"


def test_permissions_are_not_zero(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(Path, "read_text", denied)
    assert t.proc_counters()["status"] == "permission_denied"
    assert t.counter_delta({}, {"status": "permission_denied"}) == {"status": "permission_denied"}


def test_cgroup_unlimited_and_missing(tmp_path):
    assert t.cgroup_memory(tmp_path)["status"] == "unavailable"
    (tmp_path / "memory.current").write_text("1024")
    (tmp_path / "memory.max").write_text("max")
    assert t.cgroup_memory(tmp_path) == {"status": "measured", "used_bytes": 1024, "limit_bytes": None}


def test_trace_quota_excludes_smoke_and_reference():
    quota = t.TraceQuota()
    assert not quota.claim("smoke", "vanilla")
    assert quota.claim("benchmark", "direct")
    assert not quota.claim("benchmark", "vanilla")
    assert quota.claim("benchmark", "kv")
    assert not quota.claim("benchmark", "kv")
    assert not quota.claim("reference", "kv")
    assert not quota.claim("benchmark", "cache_build")


def test_generated_tokens_count_eos_exclude_padding():
    assert t.generated_tokens([[10, 11, 7, 2, 2], [10, 11, 8, 9, 2]], 2, 2, 2) == [2, 3]
    assert t.generated_tokens([[10, 7, 2, 0, 0]], 1, [2, 3], 0) == [2]
    assert t.generated_tokens([[10, 7, 0, 0]], 1, [2, 3], 0) == [3]


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))
    def numel(self):
        return self.shape[0] * self.shape[1]
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class FakeCuda:
    def __init__(self):
        self.syncs = 0
        self.allocated, self.reserved = 100, 200
    def is_available(self):
        return True
    def memory_allocated(self):
        return self.allocated
    def memory_reserved(self):
        return self.reserved
    max_memory_allocated = memory_allocated
    max_memory_reserved = memory_reserved
    def reset_peak_memory_stats(self):
        pass
    def synchronize(self):
        self.syncs += 1
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def elapsed_time(self, other):
            return 1.0


class FakeModel:
    def __init__(self, steps=2, fail=False):
        self.pre, self.post = [], []
        self.steps, self.fail = steps, fail
        self.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    def register_forward_pre_hook(self, hook, **kwargs):
        self.pre.append(hook)
        return SimpleNamespace(remove=lambda: self.pre.remove(hook))
    def register_forward_hook(self, hook, **kwargs):
        self.post.append(hook)
        return SimpleNamespace(remove=lambda: self.post.remove(hook))
    def generate(self, **kwargs):
        for i in range(self.steps):
            inputs = {"input_ids": kwargs["input_ids"] if i == 0 else Tensor([[7]])}
            for hook in self.pre:
                hook(self, (), inputs)
            if self.fail:
                raise RuntimeError("forward failed")
            for hook in self.post:
                hook(self, (), inputs, None)
        return SimpleNamespace(sequences=Tensor([kwargs["input_ids"].values[0] + [7] * self.steps]))


def make_probe(tmp_path, monkeypatch):
    cuda = FakeCuda()
    probe = t.Probe(tmp_path, "small", "fake", SimpleNamespace(cuda=cuda), detailed=True)
    probe.begin_request("execution", "/text_qa")
    probe.context["stage"] = "benchmark"
    monkeypatch.setattr(t, "ACTIVE", probe)
    return probe, cuda


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("cached,steps", [(False, 1), (False, 3), (True, 1), (True, 3)])
def test_forward_phases_single_token_prompt_and_output(tmp_path, monkeypatch, cached, steps):
    probe, cuda = make_probe(tmp_path, monkeypatch)
    model = FakeModel(steps)
    args = {"input_ids": Tensor([[10]])}  # First forward is prefill even for length 1.
    if cached:
        args["past_key_values"] = object()
    expected = model.generate(**args).sequences.values
    with probe.batch("kv" if cached else "vanilla", 0):
        actual = t.generate(model, **args)
    assert actual.sequences.values == expected
    assert cuda.syncs == 1  # Not synchronized per forward/token.
    data = rows(tmp_path / "small_phases.jsonl")
    names = [r["name"] for r in data if r["kind"] == "span"]
    assert names.count("cached_query_prefill" if cached else "vanilla_prefill") == 1
    assert names.count("decode") == steps - 1
    assert not model.pre and not model.post
    summary = t.summarize(tmp_path)
    assert summary["intervals_overlap"] is True
    import csv
    record = next(csv.DictReader((tmp_path / "batches.csv").open()))
    assert int(record["output_tokens"]) == steps
    assert float(record["decode_gpu_s"]) == pytest.approx((steps - 1) * .001)


def test_exception_removes_hooks_and_retains_spans(tmp_path, monkeypatch):
    probe, _ = make_probe(tmp_path, monkeypatch)
    model = FakeModel(fail=True)
    with pytest.raises(RuntimeError, match="forward failed"):
        with probe.batch("vanilla", 0):
            t.generate(model, input_ids=Tensor([[10]]))
    assert not model.pre and not model.post
    assert any(r.get("name") == "vanilla_prefill" for r in rows(tmp_path / "small_phases.jsonl"))


def test_trace_failure_retains_numeric_metrics_and_limitation(tmp_path, monkeypatch):
    probe, _ = make_probe(tmp_path, monkeypatch)
    probe.engine = "torch"
    def unavailable(**kwargs):
        raise RuntimeError("CUPTI denied")
    probe.torch.profiler = SimpleNamespace(profile=unavailable, record_function=lambda _: nullcontext(),
                                         ProfilerActivity=SimpleNamespace(CPU=0, CUDA=1))
    with probe.batch("kv", 0):
        t.generate(FakeModel(1), input_ids=Tensor([[10]]), past_key_values=object())
    data = rows(tmp_path / "small_phases.jsonl")
    assert any(r["kind"] == "limitation" and "CUPTI denied" in r["reason"] for r in data)
    assert any(r.get("name") == "cached_query_prefill" for r in data)
    assert not any(r["kind"] == "trace" for r in data)


@pytest.mark.parametrize("gpu_events", [False, True])
def test_exported_trace_must_contain_gpu_activity(tmp_path, monkeypatch, gpu_events):
    probe, _ = make_probe(tmp_path, monkeypatch)
    probe.engine = "torch"
    class Export:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def export_chrome_trace(self, filename):
            Path(filename).write_text(json.dumps({"traceEvents": [{"cat": "kernel" if gpu_events else "cpu_op"}]}))
    probe.torch.profiler = SimpleNamespace(profile=lambda **kwargs: Export(),
        record_function=lambda _: nullcontext(), ProfilerActivity=SimpleNamespace(CPU=0, CUDA=1))
    with probe.batch("kv", 0):
        t.generate(FakeModel(1), input_ids=Tensor([[10]]), past_key_values=object())
    data = rows(tmp_path / "small_phases.jsonl")
    trace = next(row for row in data if row["kind"] == "trace")
    assert trace["status"] == ("captured" if gpu_events else "cpu_only")
    assert (tmp_path / "traces/small_kv.json").exists()


def test_same_device_transfer_not_counted(tmp_path, monkeypatch):
    probe, _ = make_probe(tmp_path, monkeypatch)
    class Device:
        def __init__(self, value):
            self.type, _, index = str(value).partition(":")
            self.index = int(index) if index else None
    class CopyTensor:
        device = Device("cuda:0")
        def to(self, device, **kwargs):
            return self
        def numel(self):
            return 4
        def element_size(self):
            return 2
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=Device))
    tensor = CopyTensor()
    assert t.transfer(tensor, "cuda:0") is tensor
    assert not (tmp_path / "small_phases.jsonl").exists()
    with probe.batch("kv", 0):
        t.transfer(tensor, "cpu")
    transfers = [r for r in rows(tmp_path / "small_phases.jsonl") if r.get("name") == "cuda_to_cpu"]
    assert len(transfers) == 1 and transfers[0]["tensor_bytes"] == 8


def test_peak_preserved_before_inner_reset(tmp_path, monkeypatch):
    probe, cuda = make_probe(tmp_path, monkeypatch)
    cuda.allocated = 900
    probe.capture_peaks()
    cuda.allocated = 100
    probe.finish_request()
    assert probe.peaks["allocated_bytes"] == 900


def test_failed_gpu_timing_preserves_cpu_observation(tmp_path, monkeypatch):
    probe, cuda = make_probe(tmp_path, monkeypatch)
    def failure():
        raise RuntimeError("CUDA timing unavailable")
    cuda.synchronize = failure
    with probe.batch("vanilla", 0):
        t.generate(FakeModel(1), input_ids=Tensor([[10]]))
    data = rows(tmp_path / "small_phases.jsonl")
    assert any(r["kind"] == "limitation" for r in data)
    observation = next(r for r in data if r.get("name") == "vanilla_prefill")
    assert observation["cpu_wall_s"] >= 0
    assert observation["gpu_elapsed_s"] is None
    assert observation["gpu_timing_status"] == "unavailable"


def test_missing_nsys_and_feature_flags(monkeypatch):
    monkeypatch.setattr(nsight.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="STRETTO_NSYS_IMAGE"):
        nsight.check_nsys()
    monkeypatch.setattr(nsight.shutil, "which", lambda _: "/bin/nsys")
    monkeypatch.setattr(nsight.subprocess, "check_output", lambda *args, **kw: "old version")
    with pytest.raises(RuntimeError, match="lacks required flags"):
        nsight.check_nsys()


def test_nsight_separate_engine_and_bounded_ranges(tmp_path):
    command = nsight.command(["python", "-m", "backend"], tmp_path, "small")
    assert "--capture-range-end=repeat:2" in command
    assert "--cuda-um-gpu-page-faults=true" in command
    assert "--sample=none" in command
    assert "--kill=none" in command


def test_um_absent_and_zero_tables_are_different(tmp_path):
    target = tmp_path / "nsight"
    target.mkdir()
    with sqlite3.connect(target / "unknown.sqlite") as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    with sqlite3.connect(target / "zero.sqlite") as connection:
        connection.execute("CREATE TABLE CUDA_UNIFIED_MEMORY_CPU_PAGE_FAULT_EVENTS (id INTEGER)")
    result = nsight.analyze(tmp_path)
    assert any("zero faults not established" in r["status"] for r in result["reports"])
    assert any(r["tables"] and r["tables"][0]["status"] == "no_events_in_table" for r in result["reports"])


def test_resource_sample_identity_and_roles(monkeypatch):
    from reasondb.reproduction.resources import ResourceSampler
    sampler = ResourceSampler()
    first, second = sampler.sample(), sampler.sample()
    assert any(p["role"] == "runner" for p in first["processes"])
    assert first["host_memory"]["total_bytes"] > 0
    assert all(p["rss_bytes"] > 0 for p in second["processes"])


def test_cuda_real_tiny_llama_preserves_generation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required; run on the rented GPU")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    config = transformers.LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        bos_token_id=1, eos_token_id=None, pad_token_id=0, attn_implementation="eager")
    model = transformers.LlamaForCausalLM(config).eval().cuda()
    inputs = torch.tensor([[1, 8, 9, 10]], device="cuda")
    with torch.inference_mode():
        cache = model(inputs[:, :2], use_cache=True).past_key_values
        for cached in (False, True):
            import copy
            args = dict(input_ids=inputs, max_new_tokens=3, do_sample=False,
                        pad_token_id=0, return_dict_in_generate=True)
            if cached:
                args["past_key_values"] = copy.deepcopy(cache)
            monkeypatch.setattr(t, "ACTIVE", None)
            expected = model.generate(**copy.deepcopy(args)).sequences
            probe = t.Probe(tmp_path, "small", "tiny-random", torch, detailed=True)
            probe.begin_request("execution", "/text_qa")
            monkeypatch.setattr(t, "ACTIVE", probe)
            with probe.batch("kv" if cached else "vanilla", 0):
                actual = t.generate(model, **copy.deepcopy(args)).sequences
            assert torch.equal(actual, expected)
            assert probe.forward_observed and probe.forward_times["prefill_s"] >= 0
