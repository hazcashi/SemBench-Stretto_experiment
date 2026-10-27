"""Offline full-workload contracts. Synthetic overlap stats are test fixtures only."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" if (ROOT / "source").exists() else ROOT / "stretto-main/stretto-main"
sys.path.insert(0, str(SOURCE))
from reasondb.reproduction import full_workload as full
from reasondb.reproduction.profile import load_profile
from reasondb.reproduction.subset import copy_full_dataset


def test_full_dataset_keeps_all_rows_and_bytes(tmp_path):
    source = SOURCE / "reasondb/evaluation/benchmarks/files"
    manifest = copy_full_dataset(source, tmp_path / "data")
    assert manifest["tables"]["reports"]["rows"] == 728
    for table in full.TABLES:
        assert (source / f"{table}.csv").read_bytes() == (tmp_path / "data" / f"{table}.csv").read_bytes()


def test_partial_data_cannot_be_called_full(tmp_path):
    for name in full.TABLES:
        (tmp_path / f"{name}.csv").write_text("game_id\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="728"):
        copy_full_dataset(tmp_path, tmp_path / "copy")


def test_profile_requires_full_scope(tmp_path):
    path = ROOT / "configs/rotowire_full60.json"
    if not path.exists():
        # Older image test layouts have only single_gpu.json; explicit test fixture.
        path = Path("/project/configs/rotowire_full60.json")
    config = json.loads(path.read_text())
    profile = load_profile(path)
    assert profile["reports_limit"] == 728 and profile["expected_queries"] == 60
    config["reports_limit"] = 100
    bad = tmp_path / "partial.json"
    full.atomic_json(bad, config)
    with pytest.raises(ValueError, match="728"):
        load_profile(bad)


def synthetic_overlap():
    pytest.importorskip("torch")
    from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
    from reasondb.query_plan.logical_plan import LogicalFilter
    keys = {}
    for key, operators in RotowireRandom.get_operator_options().items():
        expressions = [o.expression for o in operators[LogicalFilter]]
        keys[key] = {"overlap_matrix": [[1] for _ in expressions],
                     "predicate_to_matrix_id": {s: i for i, s in enumerate(expressions)},
                     "matrix_id_to_predicate": {str(i): s for i, s in enumerate(expressions)},
                     "row_ids": [[0]]}
    return {"keys": keys}


def test_all_60_upstream_queries_are_deterministic_and_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from reasondb.query_plan.query import Queries
    first, second = full.generate_queries(synthetic_overlap()), full.generate_queries(synthetic_overlap())
    assert len(first) == 60
    assert [full.query_id(q) for q in first] == [full.query_id(q) for q in second]
    first.dump(tmp_path)
    assert [full.query_id(q) for q in Queries.load(tmp_path)] == [full.query_id(q) for q in first]
    # Both team/player pools and the 2/3/4 semantic-operator shapes are present.
    assert any("teams_to_games" in q.query for q in first)
    assert any("players_to_games" in q.query for q in first)
    counts = {len(q.get_gt_logical_plan().plan_steps) - 2 for q in first}
    assert counts == {2, 3, 4}


def test_all_shapes_fit_upstream_dependency_levels():
    pytest.importorskip("torch")
    from reasondb.database.indentifier import ConcreteTableIdentifier, VirtualTableIdentifier, VirtualColumnIdentifier
    from reasondb.optimizer.dependency_graph import DependencyGraph
    roots = []
    for table in full.TABLES:
        with (SOURCE / "reasondb/evaluation/benchmarks/files" / f"{table}.csv").open(encoding="utf-8-sig") as handle:
            columns = next(csv.reader(handle))
        roots.append(SimpleNamespace(identifier=VirtualTableIdentifier(table),
                     concrete_identifier=ConcreteTableIdentifier(table),
                     columns=[VirtualColumnIdentifier(f"{table}.{c}") for c in columns]))
    for query in full.generate_queries(synthetic_overlap()):
        steps = [SimpleNamespace(logical_plan_step=s, index=i)
                 for i, s in enumerate(query.get_gt_logical_plan().plan_steps)]
        graph = DependencyGraph.from_unoptimized_physical_plan(
            SimpleNamespace(plan_steps=steps), SimpleNamespace(root_tables=roots))
        cascades, _ = graph.get_parallel_pipelines(SimpleNamespace(step_bounds=(0, 2)))
        assert all(len(c) == 1 for c in cascades), query.query


def test_incomplete_checkpoint_never_skips_and_corruption_is_loud(tmp_path):
    (tmp_path / "benchmark.csv").write_text("points\n12\n")
    (tmp_path / "silver.csv").write_text("points\n12\n")
    assert full.load_checkpoint(tmp_path, "qid") is None
    record = {"query_id": "qid", "metrics_status": "scored",
              "output_sha256": {n: full.sha256(tmp_path / n) for n in ("benchmark.csv", "silver.csv")}}
    full.atomic_json(tmp_path / "completed.json", record)
    assert full.load_checkpoint(tmp_path, "qid") == record
    with pytest.raises(RuntimeError, match="Mismatched"):
        full.load_checkpoint(tmp_path, "other-query")
    (tmp_path / "benchmark.csv").write_text("points\n99\n")
    with pytest.raises(RuntimeError, match="Damaged"):
        full.load_checkpoint(tmp_path, "qid")


@pytest.mark.parametrize("status,n_errors", [(500, 0), (200, 1)])
def test_partial_backend_response_cannot_be_checkpointed(tmp_path, status, n_errors):
    row = {"status": status, "inference_stats": {"n_errors": n_errors}}
    (tmp_path / "small_requests.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="not checkpointed"):
        full.check_backend_errors(tmp_path)


def test_progress_does_not_claim_completion_after_failure(tmp_path):
    run, state = tmp_path / "run", tmp_path / "state"
    run.mkdir()
    state.mkdir()
    full.atomic_json(state / "queries.json", [])
    full.save_progress(run, state, [{"query_index": 0, "resumed": True}], status="failed", error="example")
    summary = json.loads((run / "workload_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["scored_queries"] == 1 and summary["expected_queries"] == 60
    assert summary["resumed_queries"] == 1


def test_bottleneck_summary_separates_queries(tmp_path):
    from reasondb.reproduction.telemetry import summarize
    spans = [{"kind": "span", "role": "small", "stage": "benchmark", "phase": "execution",
              "name": "vanilla_prefill", "query_index": i, "query_id": f"qid-{i}",
              "cpu_wall_s": i + 1, "gpu_elapsed_s": .5} for i in (0, 1)]
    (tmp_path / "small_phases.jsonl").write_text("\n".join(json.dumps(r) for r in spans))
    report = summarize(tmp_path)
    assert report["spans"][0]["cpu_wall_sum_s"] == 3
    with (tmp_path / "per_query_spans.csv").open() as handle:
        per_query = list(csv.DictReader(handle))
    assert [(r["query_id"], float(r["cpu_wall_sum_s"])) for r in per_query] == [("qid-0", 1), ("qid-1", 2)]


def test_bundle_verification_rejects_changed_source(tmp_path):
    spec = importlib.util.spec_from_file_location("full_launcher", ROOT / "scripts/run_full_rotowire.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    (tmp_path / "source").mkdir()
    path = tmp_path / "source/example.py"
    path.write_text("original")
    files = {"source/example.py": full.sha256(path)}
    full.atomic_json(tmp_path / "bundle_manifest.json", {
        "files": files, "tree_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()})
    launcher.verify_bundle(tmp_path)
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        launcher.verify_bundle(tmp_path)


def test_live_workload_routes_all_60_and_resumes_only_scored_queries(monkeypatch, tmp_path):
    """Real orchestration/generation/scoring with API, model and DB boundaries replaced."""
    pytest.importorskip("torch")
    import pandas as pd
    from reasondb.backends.simulate_store import SimulateStore
    from reasondb.database import database
    from reasondb.evaluation import filter_stats
    from reasondb import executor
    from reasondb.monitor import collector
    from reasondb.optimizer import gd_optimizer, label_optimizer
    from reasondb.reasoning.reasoners import self_correction
    from reasondb.reasoning import llm
    from reasondb.reproduction import profile
    from reasondb.utils import logging

    calls, stats_calls = [], []
    failure = {"enabled": True}
    class Logger:
        def __truediv__(self, other):
            return self
    class FakeExecutor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute_benchmark(self, queries, *guarantees, **kwargs):
            assert len(queries) == 1
            assert kwargs == {"skip_nl_translation": True, "results_cache_dir": None}
            query = next(iter(queries))
            calls.append((self.name, full.query_id(query)))
            if self.name == "optim_global":
                assert len(guarantees) == 2
                if failure["enabled"] and len(calls) == 5:
                    raise RuntimeError("injected backend failure")
            else:
                assert self.name == "silver_8b" and not guarantees
                assert self.optimizer.database is self.database
            # Prompt-only recording must not retain model answers or enable replay.
            assert SimulateStore.get_simulate() is None
            SimulateStore.get_precompute().record_text_qa("ignored")
            return SimpleNamespace(results={query.query: pd.DataFrame({"value": [12]})},
                costs={query.query: SimpleNamespace(to_json=lambda: {"execution": 1})},
                tuned_pipelines={query.query: ["offline fixture"]})
    class Reference:
        def set_database(self, value):
            self.database = value
    class Collector:
        def __init__(self, **kwargs):
            pass
        def install(self):
            return self
        def close(self):
            pass
        def snapshot_aggregates(self):
            return {}
    monkeypatch.setattr(database, "Database", lambda **kw: SimpleNamespace(add_table=lambda **kw: None))
    monkeypatch.setattr(executor, "Executor", FakeExecutor)
    monkeypatch.setattr(profile, "build_configurator", lambda p: object())
    monkeypatch.setattr(llm, "GPT4o", lambda: object())
    monkeypatch.setattr(self_correction, "SelfCorrectionReasoner", lambda **kw: object())
    monkeypatch.setattr(gd_optimizer, "GradientDescentOptimizer", lambda *a: object())
    monkeypatch.setattr(label_optimizer, "LabelOptimizer", Reference)
    monkeypatch.setattr(logging, "FileLogger", lambda p: Logger())
    monkeypatch.setattr(collector, "Collector", Collector)
    def stats(*args, **kwargs):
        stats_calls.append(True)
        return synthetic_overlap()
    monkeypatch.setattr(filter_stats, "compute_filter_stats", stats)
    config = dict(seed=42, cache_namespace="offline", precision=.9, recall=.9, sample_size=100,
                  experiment_dir=str(tmp_path / "experiment"), large={"model_name": "fixture-8B"})
    data = tmp_path / "data"
    data.mkdir()
    for table in full.TABLES:
        (data / f"{table}.csv").write_text("test\n1\n")
    first, second = tmp_path / "first", tmp_path / "second"
    for run in (first, second):
        (run / "profiling").mkdir(parents=True)
    monkeypatch.setenv("STRETTO_STAGE_FILE", str(first / "profiling/stage.txt"))
    with pytest.raises(RuntimeError, match="injected"):
        full.run_full_workload(config, data, first)
    assert json.loads((first / "workload_summary.json").read_text())["scored_queries"] == 2
    assert SimulateStore.get_precompute() is None
    failure["enabled"] = False
    calls.clear()
    monkeypatch.setenv("STRETTO_STAGE_FILE", str(second / "profiling/stage.txt"))
    full.run_full_workload(config, data, second)
    summary = json.loads((second / "workload_summary.json").read_text())
    assert summary["status"] == "completed" and summary["scored_queries"] == 60
    assert summary["resumed_queries"] == 2
    assert len(calls) == 58 * 2  # Every remaining query does live benchmark AND reference.
    assert len(stats_calls) == 1
    assert SimulateStore.get_precompute() is None and SimulateStore.get_simulate() is None
    state = Path(config["experiment_dir"]) / "state"
    assert json.loads((state / "prompt_pins.json").read_text())["text_qa"] == {}
