"""Reproduction CLI. All expensive stages run in the image, on the rented host."""
import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from reasondb.reproduction.profile import load_profile
from reasondb.reproduction.processes import Services
from reasondb.reproduction.subset import create_subset, TABLES
from reasondb.reproduction.telemetry import set_stage, summarize

REPO = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+"\n", encoding="utf-8")


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def check_access(profile):
    from huggingface_hub import hf_hub_download
    from openai import OpenAI
    for key in ("HF_TOKEN", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Set {key} in the untracked .env before renting")
    for role in ("small", "large"):
        model = profile[role]
        for filename in ("config.json", "tokenizer.json"):
            hf_hub_download(model["model_name"], filename, revision=model["revision"])
    # Small real calls prove permissions rather than only checking key presence.
    with OpenAI(timeout=30, max_retries=0) as client:
        for model in ("gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"):
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply OK"}], max_tokens=2)
        client.embeddings.create(model="text-embedding-3-small", input="access check")
    return {"status": "passed", "checks": ["HF gated config/tokenizer", "OpenAI planner/metadata/embeddings"]}


def doctor(profile):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Docker must expose exactly one CUDA GPU (--gpus device=0)")
    if torch.cuda.get_device_properties(0).total_memory < 44 * 1024**3:
        raise RuntimeError("This dual-model experiment requires approximately 48GB VRAM")
    # Availability alone does not prove that the installed CUDA extensions launch.
    from flash_attn import flash_attn_func
    if (torch.ones(1, device="cuda") + 1).item() != 2:
        raise RuntimeError("CUDA tensor smoke failed")
    qk = torch.zeros((1, 8, 2, 128), dtype=torch.bfloat16, device="cuda")
    value = torch.ones_like(qk)
    output = flash_attn_func(qk, qk, value, causal=True)
    torch.cuda.synchronize()
    if not torch.allclose(output, value):
        raise RuntimeError("FlashAttention BF16/head_dim=128 smoke failed")
    del qk, value, output
    root = Path(os.environ["HF_HOME"])
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1024**3
    if free < profile["min_free_disk_gib"]:
        raise RuntimeError(f"Only {free:.1f}GiB free; need {profile['min_free_disk_gib']}GiB")
    access = check_access(profile)
    return {"status": "passed", "gpu": torch.cuda.get_device_name(0), "free_disk_gib": free,
            "compute_capability": torch.cuda.get_device_capability(0),
            "cuda_tensor_smoke": "passed", "flash_attention_smoke": "passed",
            "cuda": torch.version.cuda, "torch": torch.__version__, "access": access,
            "nvidia_smi": command_output(["nvidia-smi"])}


def download_models(profile):
    from huggingface_hub import snapshot_download
    patterns = ["*.json", "tokenizer*", "*.safetensors", "*.model", "*.txt"]
    for role in ("small", "large"):
        model = profile[role]
        snapshot_download(model["model_name"], revision=model["revision"], allow_patterns=patterns)
    snapshot_download("BAAI/bge-small-en-v1.5", revision=profile["embedding_revision"], allow_patterns=patterns)


def backend_command(profile_path, role, trace_dir=None):
    command = [sys.executable, "-m", "reasondb.reproduction.backend", "--profile", str(profile_path), "--role", role]
    if trace_dir:
        engine = os.environ.get("STRETTO_TRACE_ENGINE", "torch")
        command += ["--trace-dir", str(trace_dir), "--trace-engine", engine]
        if engine == "nsys":
            from reasondb.reproduction.nsight import command as nsight_command
            command = nsight_command(command, Path(trace_dir).parent, role)
    if os.environ.get("STRETTO_DETAILED_PROFILE") == "1":
        command += ["--detailed"]
    return command


def smoke_model(model, cache_dir):
    import requests
    response = requests.post(f"http://127.0.0.1:{model['port']}/text_qa", json={
        "column_name": "reports.report", "texts": ["The home team won the basketball game."],
        "questions": ["Did the home team win?"],
        "boolean": True, "cache_dir": str(cache_dir), "vanilla": True,
        "effective_compression_ratio": 0.0, "materialized_compression_ratio": 0.0,
        "keep_in_memory": False,
    }, timeout=180)
    response.raise_for_status()
    result = response.json()
    # The upstream boolean endpoint explicitly instructs the model to emit 1/0.
    # Validate the known true answer using that wire protocol, not Yes/No text.
    if not result.get("answers") or str(result["answers"][0]).strip() != "1":
        raise RuntimeError(f"{model['model_name']} failed the known-answer smoke test")
    return result


class Sampler:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        from reasondb.reproduction.resources import ResourceSampler
        sampler = ResourceSampler()
        with self.path.open("w") as handle:
            while not self.stop.is_set():
                try:
                    row = sampler.sample()
                    try:
                        row["gpu_csv"] = command_output(["nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
                    except Exception as exc:
                        row["gpu_sampling_error"] = type(exc).__name__
                    handle.write(json.dumps(row)+"\n")
                    handle.flush()
                except Exception as exc:
                    handle.write(json.dumps({"sampling_error": type(exc).__name__})+"\n")
                self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)


def run_queries(profile, subset_dir, run_dir, *, nl_smoke=False):
    set_stage("setup")
    import asyncio
    import torch
    from reasondb.database.database import Database
    from reasondb.database.indentifier import InPlaceColumn
    from reasondb.executor import Executor
    from reasondb.monitor.collector import Collector
    from reasondb.optimizer.gd_optimizer import GradientDescentOptimizer, OptimizationConfig, GlobalOptimizationMode
    from reasondb.optimizer.label_optimizer import LabelOptimizer
    from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
    from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
    from reasondb.query_plan.physical_operator import CostType
    from reasondb.query_plan.query import Query
    from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
    from reasondb.reasoning.llm import GPT4o
    from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
    from reasondb.utils.logging import FileLogger
    from reasondb.evaluation.benchmarks.rotowire import ROTOWIRE_QUERIES
    from reasondb.evaluation.row_signature import RowSignature
    from reasondb.evaluation.metrics.metrics_manager import confusion_counts, precision_recall_f1
    from reasondb.reproduction.profile import build_configurator

    random.seed(profile["seed"])
    torch.manual_seed(profile["seed"])
    database = Database(identifier_for_caching=profile["cache_namespace"])
    for table in TABLES:
        database.add_table(path=subset_dir / f"{table}.csv", name=table,
                           text_columns=[InPlaceColumn("reports.report")] if table == "reports" else ())
    configurator = build_configurator(profile)
    reasoner = SelfCorrectionReasoner(llm=GPT4o(), configurator=configurator,
                                     logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
                                     few_shot_database=DUMMY_FEW_SHOT_DATABASE)
    optimizer = GradientDescentOptimizer(OptimizationConfig(
        cost_type=CostType.RUNTIME, global_optimization_mode=GlobalOptimizationMode.GLOBAL,
        sample_size=profile["sample_size"], device=torch.device("cpu")))
    # FileLogger requires paths underneath relative `logging/`.
    logger = FileLogger(Path("logging") / run_dir.name)
    executor = Executor(name="optim_global", database=database, reasoner=reasoner,
                        configurator=configurator, optimizer=optimizer, logger=logger)
    collector = Collector(jsonl_path=run_dir / "events.jsonl", run_id=run_dir.name).install()
    guarantees = (PrecisionGuarantee(profile["precision"]), RecallGuarantee(profile["recall"]))
    rows = []
    try:
        with executor:
            # This additional NL query is not part of upstream ROTOWIRE_QUERIES.
            # Its generated plan can require levels the upstream GD optimizer does
            # not support. Keep it explicitly opt-in; never catch its failure and
            # report a successful smoke. Fixed benchmark plans still use the real
            # configurator, optimizer, and local model backends below.
            if nl_smoke:
                query = Query("Return the reports where the home team won the game.")
                set_stage("smoke")
                started = time.perf_counter()
                result, cost = asyncio.run(executor.execute_query(query, logger / "smoke", *guarantees))
                frame = asyncio.run(executor.extract_data(result, logger / "smoke-output"))
                frame.to_csv(run_dir / "smoke.csv", index=False)
                rows.append({"stage": "smoke", "query": query.query, "wall_s": time.perf_counter()-started,
                             "rows": len(frame), "cost": cost.to_json()})
                executor.clean_up()
            # Preserve upstream fixed benchmark logical plans, including column labels.
            # Dataset rows are reindexed in this subset: do not use upstream human labels.
            started = time.perf_counter()
            set_stage("benchmark")
            benchmark = executor.execute_benchmark(ROTOWIRE_QUERIES, *guarantees,
                                                   skip_nl_translation=True, results_cache_dir=None)
            elapsed = time.perf_counter()-started
            benchmark_rows = []
            for index, query in enumerate(ROTOWIRE_QUERIES):
                frame = benchmark.results[query.query]
                frame.to_csv(run_dir / f"benchmark_{index}.csv", index=False)
                row = {"stage": "benchmark", "query": query.query, "wall_s": elapsed,
                       "rows": len(frame), "cost": benchmark.costs[query.query].to_json(),
                       "metrics_status": "silver reference pending",
                       "tuned_pipelines": benchmark.tuned_pipelines[query.query]}
                rows.append(row)
                benchmark_rows.append(row)
            write_json(run_dir / "queries.json", rows)
            # Independent uncompressed 8B reference, using upstream LabelOptimizer.
            original_optimizer = executor.optimizer
            label_optimizer = LabelOptimizer()
            label_optimizer.set_database(database)
            executor.optimizer = label_optimizer
            executor.name, executor.role = "silver_8b", "label"
            try:
                set_stage("reference")
                labels = executor.execute_benchmark(ROTOWIRE_QUERIES, skip_nl_translation=True,
                                                    results_cache_dir=None)
            finally:
                executor.optimizer = original_optimizer
            for index, query in enumerate(ROTOWIRE_QUERIES):
                frame = benchmark.results[query.query]
                label = labels.results[query.query]
                label.to_csv(run_dir / f"silver_{index}.csv", index=False)
                pred_sig, label_sig = RowSignature.of(frame), RowSignature.of(label)
                metrics = precision_recall_f1(*confusion_counts(pred_sig.hashes, label_sig.hashes))
                benchmark_rows[index].update({"metrics_status": "scored",
                                             "metrics_against": "8B vanilla silver (not human labels)", **metrics})
    finally:
        collector.close()
        write_json(run_dir / "telemetry_summary.json", collector.snapshot_aggregates())
        write_json(run_dir / "queries.json", rows)
        with (run_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "query", "wall_s", "rows", "precision", "recall", "f1_score"])
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in writer.fieldnames} for row in rows)


def run(profile, profile_path, run_dir, tracing=False, trace_engine="torch", *, nl_smoke=False):
    import torch
    from huggingface_hub import __version__ as hf_version
    cache = Path(os.environ["REASONDB_CACHE_DIR"])
    profiling_dir = run_dir / "profiling"
    profiling_dir.mkdir(exist_ok=True)
    previous_environment = {key: os.environ.get(key) for key in (
        "STRETTO_STAGE_FILE", "STRETTO_TRACE_ENGINE", "STRETTO_DETAILED_PROFILE")}
    os.environ["STRETTO_STAGE_FILE"] = str(profiling_dir / "stage.txt")
    os.environ["STRETTO_TRACE_ENGINE"] = trace_engine
    os.environ["STRETTO_DETAILED_PROFILE"] = "1" if tracing else "0"
    set_stage("preflight")
    manifest = {"run_id": run_dir.name, "started_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "profile": profile, "python": platform.python_version(),
                "torch": torch.__version__, "cuda": torch.version.cuda, "huggingface_hub": hf_version,
                "image": os.environ.get("STRETTO_IMAGE_REF"), "image_id": os.environ.get("STRETTO_IMAGE_ID"),
                "source_revision": os.environ.get("STRETTO_SOURCE_REVISION"),
                "profile_enabled": tracing, "sampling_interval_s": 1,
                "result_cache_enabled": False,
                "nl_smoke_enabled": nl_smoke,
                "benchmark_plan_source": "upstream ROTOWIRE_QUERIES fixed logical plans",
                "inference_max_new_tokens": {"boolean": 4, "extract": 64},
                "request_concurrency": 1,
                "gpu_model_dtype": "bfloat16",
                "flash_attention_build": os.environ.get("STRETTO_FLASH_BUILD"),
                "optimizer_device": "cpu",
                "cache_existed_before_run": cache.exists() and any(cache.iterdir()),
                "trace_engine": trace_engine if tracing else "none",
                "prefill_decode_separation": "forward CUDA events and CPU spans" if tracing else "disabled in baseline",
                "profiling_overhead": "not yet measured; compare independent runs",
                "trace_scope": "first benchmark batch per model and route, including optimizer samples"}
    started = time.perf_counter()
    try:
        write_json(run_dir / "manifest.json", manifest)
        if tracing and trace_engine == "nsys":
            from reasondb.reproduction.nsight import check_nsys
            write_json(profiling_dir / "nsight_environment.json", check_nsys())
        write_json(run_dir / "doctor.json", doctor(profile))
        (run_dir / "pip-freeze.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"]))
        download_models(profile)
        subset_dir = Path(os.environ["STRETTO_DATA_ROOT"]) / profile["cache_namespace"] / "subset"
        manifest["subset"] = create_subset(REPO / "reasondb/evaluation/benchmarks/files", subset_dir, profile["reports_limit"])
        trace_dir = profiling_dir / "traces" if tracing else None
        if trace_dir:
            trace_dir.mkdir()
        with Sampler(run_dir / "resources.jsonl"):
            # Validate individually before paying for both models' GPU residency.
            for role in ("small", "large"):
                model = profile[role]
                set_stage(f"individual_{role}")
                with Services(run_dir / "services" / f"individual_{role}") as services:
                    status = services.start(role, backend_command(profile_path, role), model["port"],
                                            profile["readiness_timeout_s"], model["model_name"])
                    write_json(run_dir / f"{role}_individual_status.json", status)
                    write_json(run_dir / f"{role}_individual.json", smoke_model(model, cache))
            with Services(run_dir / "services" / "concurrent") as services:
                set_stage("concurrent_smoke")
                for role in ("small", "large"):
                    model = profile[role]
                    status = services.start(role, backend_command(profile_path, role, trace_dir), model["port"],
                                            profile["readiness_timeout_s"], model["model_name"])
                    write_json(run_dir / f"{role}_concurrent_status.json", status)
                services.start("embedding", [sys.executable, "-m", "reasondb.backends.text_embed_server"],
                               5324, profile["readiness_timeout_s"])
                for role in ("small", "large"):
                    write_json(run_dir / f"{role}_concurrent.json", smoke_model(profile[role], cache))
                run_queries(profile, subset_dir, run_dir, nl_smoke=nl_smoke)
        backend_errors = []
        for path in run_dir.glob("*_requests.jsonl"):
            for line in path.read_text().splitlines():
                request = json.loads(line)
                if request["status"] >= 400 or (request.get("inference_stats") or {}).get("n_errors", 0):
                    backend_errors.append({"log": path.name, "path": request["path"], "status": request["status"]})
        manifest["backend_errors"] = backend_errors
        if backend_errors:
            raise RuntimeError("Backend errors occurred; results are retained but this run is not a validated baseline")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        manifest["total_wall_s"] = time.perf_counter()-started
        manifest["cache_bytes_after_run"] = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) if cache.exists() else 0
        write_json(run_dir / "manifest.json", manifest)
        if (run_dir / "resources.jsonl").exists():
            shutil.copyfile(run_dir / "resources.jsonl", profiling_dir / "resources.jsonl")
        try:
            summarize(profiling_dir)
            if tracing and trace_engine == "nsys":
                from reasondb.reproduction.nsight import analyze
                analyze(profiling_dir)
        except Exception as exc:
            (profiling_dir / "summary_limitation.txt").write_text(str(exc), encoding="utf-8")
        finally:
            for key, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("doctor", "run", "profile", "profile-nsys", "inspect", "access"))
    parser.add_argument("--config", default="/opt/stretto/configs/single_gpu.json")
    parser.add_argument("--nl-smoke", action="store_true",
                        help="Opt into the extra NL smoke before the fixed benchmark; generated multi-level plans may be unsupported")
    args = parser.parse_args()
    if args.nl_smoke and args.action not in ("run", "profile", "profile-nsys"):
        parser.error("--nl-smoke requires run, profile, or profile-nsys")
    profile = load_profile(args.config)
    if args.action == "inspect":
        from reasondb.reproduction.profile import build_configurator
        configurator = build_configurator(profile)
        identifiers = [op.get_operation_identifier() for op in configurator.physical_operators]
        if any(name in identifier for identifier in identifiers for name in ("70B", "Image", "Audio", "llava")):
            raise RuntimeError("Non-text or 70B operator in single-GPU suite")
        print(json.dumps({"profile": profile, "operators": identifiers}, indent=2))
        return
    root = Path(os.environ["STRETTO_RESULTS_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    run_dir.mkdir(exist_ok=False)
    os.environ["STRETTO_RUN_DIR"] = str(run_dir)
    os.environ["STRETTO_EMBED_REVISION"] = profile["embedding_revision"]
    print(f"Results: {run_dir}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.action in ("doctor", "access"):
        try:
            report = check_access(profile) if args.action == "access" else doctor(profile)
            write_json(run_dir / "doctor.json", report)
            print(json.dumps(report, indent=2))
        except BaseException:
            (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    else:
        run(profile, args.config, run_dir, tracing=args.action in ("profile", "profile-nsys"),
            trace_engine="nsys" if args.action == "profile-nsys" else "torch", nl_smoke=args.nl_smoke)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
