"""All upstream RotowireRandom shapes, live inference and durable per-query results.

Only model roles and the reference model change. Query generation, overlap sampling,
configuration, global optimization and scoring use upstream implementations.
"""
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

from reasondb.reproduction.subset import TABLES, sha256
from reasondb.reproduction.telemetry import set_stage


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def query_id(query):
    return hashlib.sha256(json.dumps(query.to_json(), sort_keys=True).encode()).hexdigest()


def check_backend_errors(run_dir):
    # Reject partial 2xx responses too, before writing a reusable checkpoint.
    # Otherwise a later attempt could skip a scored-but-invalid prior result.
    for path in Path(run_dir).glob("*_requests.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["status"] >= 400 or (row.get("inference_stats") or {}).get("n_errors", 0):
                    raise RuntimeError(f"Backend error in {path.name}; query is not checkpointed")


def load_checkpoint(directory, expected_id):
    """Never skip work based only on a directory or a partially written CSV."""
    directory = Path(directory)
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    row = json.loads(marker.read_text(encoding="utf-8"))
    if row.get("query_id") != expected_id or row.get("metrics_status") != "scored":
        raise RuntimeError(f"Mismatched/incomplete query checkpoint: {directory}")
    if set(row.get("output_sha256", {})) != {"benchmark.csv", "silver.csv"}:
        raise RuntimeError(f"Incomplete output checkpoint: {directory}")
    for name, digest in row["output_sha256"].items():
        if sha256(directory / name) != digest:
            raise RuntimeError(f"Damaged output checkpoint: {directory / name}")
    return row


def generate_queries(payload):
    from reasondb.evaluation.benchmark import FilterStats
    from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
    queries = RotowireRandom.generate_random_queries(
        "test", num_queries_per_shape=RotowireRandom.num_queries_per_shape,
        keep_per_shape=RotowireRandom.queries_kept_per_shape,
        filter_stats=FilterStats.from_payload(payload))
    if len(queries) != 60 or len({q.query for q in queries}) != 60:
        raise RuntimeError(f"Upstream generator produced {len(queries)} queries; require 60 unique queries")
    return queries


def save_progress(run_dir, state, rows, *, status, total=60, error=None):
    summary = {"status": status, "expected_queries": total, "scored_queries": len(rows),
               "resumed_queries": sum(bool(r.get("resumed")) for r in rows),
               "state_dir": str(state), "reference": "8B vanilla silver, not human labels",
               "query_set_sha256": sha256(state / "queries.json") if (state / "queries.json").exists() else None}
    summary["benchmark_wall_sum_s"] = sum(r.get("wall_s", 0) for r in rows)
    summary["reference_wall_sum_s"] = sum(r.get("reference_wall_s", 0) for r in rows)
    summary["timing_scope"] = "per-query sums across source runs; excludes service startup and phase-0 preparation"
    if (state / "preparation.json").exists():
        summary["preparation"] = json.loads((state / "preparation.json").read_text())
    if error:
        summary["error"] = error
    atomic_json(run_dir / "queries.json", rows)
    atomic_json(run_dir / "workload_summary.json", summary)
    atomic_json(state / "progress.json", {**summary, "latest_run": str(run_dir)})
    fields = ["query_index", "query_id", "wall_s", "reference_wall_s", "rows",
              "configuring_s", "candidate_sampling_s", "optimizer_solve_s", "execution_s",
              "precision", "recall", "f1_score", "source_run_id", "resumed"]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_full_workload(profile, data_dir, run_dir):
    import torch
    from reasondb.backends.simulate_store import SimulateStore
    from reasondb.database.database import Database
    from reasondb.database.indentifier import InPlaceColumn
    from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
    from reasondb.evaluation.filter_stats import compute_filter_stats
    from reasondb.evaluation.metrics.metrics_manager import confusion_counts, precision_recall_f1
    from reasondb.evaluation.row_signature import RowSignature
    from reasondb.executor import Executor
    from reasondb.monitor.collector import Collector
    from reasondb.optimizer.gd_optimizer import GradientDescentOptimizer, OptimizationConfig, GlobalOptimizationMode
    from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
    from reasondb.optimizer.label_optimizer import LabelOptimizer
    from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
    from reasondb.query_plan.physical_operator import CostType
    from reasondb.query_plan.query import Queries
    from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
    from reasondb.reasoning.llm import GPT4o
    from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
    from reasondb.reproduction.profile import build_configurator
    from reasondb.utils.logging import FileLogger

    if SimulateStore.get_simulate() is not None or SimulateStore.get_precompute() is not None:
        raise RuntimeError("Full workload requires live backends and owns its prompt-pin store")

    class PromptPins(SimulateStore):
        # Reuse upstream prompt pinning without accumulating all per-item answers.
        # No simulate store is installed: every non-checkpointed query calls models.
        def record_text_qa(self, *args, **kwargs):
            pass

    state = Path(profile["experiment_dir"]) / "state"
    state.mkdir(parents=True, exist_ok=True)
    identity = {"profile": profile, "tables": {name: sha256(Path(data_dir) / f"{name}.csv") for name in TABLES}}
    identity_path = state / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Experiment source/config/data changed; use a new experiment directory")
    atomic_json(identity_path, identity)
    pin_path = state / "prompt_pins.json"
    pins = PromptPins()
    if pin_path.exists():
        pins.__dict__.update(SimulateStore.load(pin_path).__dict__)

    def save_pins():
        temporary = pin_path.with_suffix(".tmp")
        pins.save(temporary)
        temporary.replace(pin_path)

    random.seed(profile["seed"])
    torch.manual_seed(profile["seed"])
    database = Database(identifier_for_caching=profile["cache_namespace"])
    for table in TABLES:
        database.add_table(path=Path(data_dir) / f"{table}.csv", name=table,
                           text_columns=[InPlaceColumn("reports.report")] if table == "reports" else ())
    configurator = build_configurator(profile)
    reasoner = SelfCorrectionReasoner(llm=GPT4o(), configurator=configurator,
                                    logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
                                    few_shot_database=DUMMY_FEW_SHOT_DATABASE)
    optimizer = GradientDescentOptimizer(OptimizationConfig(
        cost_type=CostType.RUNTIME, global_optimization_mode=GlobalOptimizationMode.GLOBAL,
        sample_size=profile["sample_size"], device=torch.device("cpu")))
    logger = FileLogger(Path("logging") / run_dir.name)
    executor = Executor(name="optim_global", database=database, reasoner=reasoner,
                        configurator=configurator, optimizer=optimizer, logger=logger)
    reference_optimizer = LabelOptimizer()
    reference_optimizer.set_database(database)
    collector = Collector(jsonl_path=run_dir / "events.jsonl", run_id=run_dir.name).install()
    rows = []
    current_index = None
    sidecar = Path(os.environ["STRETTO_STAGE_FILE"]).with_name("current_query.json")
    SimulateStore.set_precompute(pins)
    try:
        stats_path = state / "filter_stats.json"
        if not stats_path.exists():
            print("Phase 0: live 8B filter statistics for upstream query generation", flush=True)
            set_stage("filter_stats")
            executor.optimizer, executor.name, executor.role = reference_optimizer, "silver_8b", "filter_stats"
            started = time.perf_counter()
            payload = compute_filter_stats(RotowireRandom("test", database, Queries()), executor)
            check_backend_errors(run_dir)
            atomic_json(state / "preparation.json", {"filter_stats_wall_s": time.perf_counter() - started,
                        "source_run_id": run_dir.name, "scope": "live filter statistics including operator/cache preparation"})
            payload["reference_model"] = profile["large"]
            save_pins()
            atomic_json(stats_path, payload)
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        queries_path = state / "queries.json"
        if queries_path.exists():
            queries = Queries.load(state)
        else:
            queries = generate_queries(payload)
            queries.dump(state)
        if len(queries) != 60 or len({q.query for q in queries}) != 60:
            raise RuntimeError("Pinned workload must contain all 60 distinct upstream queries")
        # Freeze a copy with each attempt, alongside source/data identity and filter stats.
        atomic_json(run_dir / "workload.json", {"query_set": str(queries_path),
                    "query_set_sha256": sha256(queries_path), "identity": identity,
                    "generator": "RotowireRandom, seed=42, 10 draws/shape, keep 5, 12 shapes",
                    "original_paper_query_identity": "not asserted; overlap stats use 8B instead of 70B"})
        queries.dump(run_dir / "workload")
        guarantees = (PrecisionGuarantee(profile["precision"]), RecallGuarantee(profile["recall"]))
        executor.optimizer, executor.name, executor.role = optimizer, "optim_global", "sweep"
        save_progress(run_dir, state, rows, status="running")
        with executor:
            for index, query in enumerate(queries):
                current_index = index
                qid = query_id(query)
                directory = state / f"query-{index:03d}"
                previous = load_checkpoint(directory, qid)
                if previous:
                    rows.append({**previous, "resumed": True})
                    print(f"[{index+1}/60] resume scored query {qid[:12]}", flush=True)
                    save_progress(run_dir, state, rows, status="running")
                    continue
                directory.mkdir(exist_ok=True)
                atomic_json(sidecar, {"query_index": index, "query_id": qid})
                print(f"[{index+1}/60] benchmark {qid[:12]}: {query.query}", flush=True)
                # Keep logs distinct even though execute_benchmark sees one query at a time.
                executor.logger = logger / f"full-query-{index:03d}"
                executor.optimizer, executor.name, executor.role = optimizer, "optim_global", "sweep"
                random.seed(profile["seed"] + index)
                torch.manual_seed(profile["seed"] + index)
                set_stage("benchmark")
                started = time.perf_counter()
                benchmark = executor.execute_benchmark(Queries(deepcopy(query)), *guarantees,
                                                       skip_nl_translation=True, results_cache_dir=None)
                elapsed = time.perf_counter() - started
                frame = benchmark.results[query.query]
                frame.to_csv(directory / "benchmark.csv", index=False)
                set_stage("reference")
                executor.optimizer, executor.name, executor.role = reference_optimizer, "silver_8b", "label"
                started = time.perf_counter()
                labels = executor.execute_benchmark(Queries(deepcopy(query)), skip_nl_translation=True,
                                                    results_cache_dir=None)
                reference_wall = time.perf_counter() - started
                check_backend_errors(run_dir)
                label = labels.results[query.query]
                label.to_csv(directory / "silver.csv", index=False)
                pred_sig, label_sig = RowSignature.of(frame), RowSignature.of(label)
                metrics = precision_recall_f1(*confusion_counts(pred_sig.hashes, label_sig.hashes))
                cost = benchmark.costs[query.query].to_json()
                components = cost.get("component_times", {})
                row = {"stage": "benchmark", "query_index": index, "query_id": qid, "query": query.query,
                       "wall_s": elapsed, "reference_wall_s": reference_wall, "rows": len(frame),
                       "cost": cost,
                       "configuring_s": components.get("configuring"),
                       "candidate_sampling_s": components.get("profiling"),
                       "optimizer_solve_s": components.get("optimizer_solve"),
                       "execution_s": components.get("execution"),
                       "tuned_pipelines": benchmark.tuned_pipelines[query.query],
                       "reference_pipelines": labels.tuned_pipelines[query.query],
                       "metrics_status": "scored", "metrics_against": "8B vanilla silver (not human labels)",
                       "source_run_id": run_dir.name, "resumed": False, **metrics,
                       "output_dir": str(directory),
                       "output_sha256": {n: sha256(directory / n) for n in ("benchmark.csv", "silver.csv")}}
                save_pins()
                atomic_json(directory / "completed.json", row)
                rows.append(row)
                save_progress(run_dir, state, rows, status="running")
                print(f"[{index+1}/60] scored rows={len(frame)} F1={metrics['f1_score']:.4f}", flush=True)
                del benchmark, labels, frame, label
        save_progress(run_dir, state, rows, status="completed")
    except BaseException as exc:
        atomic_json(state / "last_failure.json", {"query_index": current_index,
                    "run": str(run_dir), "traceback": traceback.format_exc()})
        save_progress(run_dir, state, rows, status="failed", error=type(exc).__name__)
        raise
    finally:
        try:
            save_pins()
        finally:
            SimulateStore.set_precompute(None)
            collector.close()
            atomic_json(run_dir / "telemetry_summary.json", collector.snapshot_aggregates())
            if sidecar.exists():
                sidecar.unlink()
