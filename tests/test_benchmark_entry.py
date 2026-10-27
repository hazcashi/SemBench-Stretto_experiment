"""Offline routing and dependency checks; these do not validate GPU execution."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source' if (ROOT / 'source').exists() else ROOT / 'stretto-main/stretto-main'
sys.path.insert(0, str(SOURCE))
pytest.importorskip('torch')
pytest.importorskip('duckdb')
import pandas as pd
from reasondb.reproduction import runner
from reasondb.evaluation.benchmarks.rotowire import ROTOWIRE_QUERIES


@pytest.mark.parametrize('nl_smoke,fail_smoke', [(False, False), (True, False), (True, True)])
def test_query_entry_preserves_benchmark_and_silver(monkeypatch, tmp_path, nl_smoke, fail_smoke):
    """Exercise real run_queries control flow; replace all API/model/DB boundaries."""
    from reasondb.database import database
    from reasondb import executor
    from reasondb.optimizer import gd_optimizer, label_optimizer
    from reasondb.monitor import collector
    from reasondb.reasoning.reasoners import self_correction
    from reasondb.reasoning import llm
    from reasondb.reproduction import profile
    from reasondb.utils import logging

    calls, stages = [], []
    class Logger:
        def __truediv__(self, other):
            return self
    class FakeExecutor:
        def __init__(self, **kwargs):
            self.optimizer = kwargs['optimizer']
            self.name = kwargs['name']
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            calls.append('exit')
        async def execute_query(self, query, logger, *guarantees):
            calls.append('nl')
            assert len(guarantees) == 2
            if fail_smoke:
                raise RuntimeError('unsupported generated plan')
            return object(), SimpleNamespace(to_json=lambda: {})
        async def extract_data(self, *args):
            return pd.DataFrame({'report': ['example']})
        def clean_up(self):
            calls.append('cleanup')
        def execute_benchmark(self, queries, *guarantees, **kwargs):
            calls.append('silver' if self.name == 'silver_8b' else 'benchmark')
            assert queries is ROTOWIRE_QUERIES
            assert kwargs == {'skip_nl_translation': True, 'results_cache_dir': None}
            assert len(guarantees) == (0 if self.name == 'silver_8b' else 2)
            return SimpleNamespace(
                results={q.query: pd.DataFrame({'points': ['12'], 'assists': ['3']}) for q in queries},
                costs={q.query: SimpleNamespace(to_json=lambda: {}) for q in queries},
                tuned_pipelines={q.query: ['test plan'] for q in queries})
    class Collector:
        def __init__(self, **kwargs):
            pass
        def install(self):
            return self
        def close(self):
            calls.append('collector_closed')
        def snapshot_aggregates(self):
            return {}
    monkeypatch.setattr(database, 'Database', lambda **kw: SimpleNamespace(add_table=lambda **kw: None))
    monkeypatch.setattr(executor, 'Executor', FakeExecutor)
    monkeypatch.setattr(profile, 'build_configurator', lambda p: object())
    monkeypatch.setattr(llm, 'GPT4o', lambda: object())
    monkeypatch.setattr(self_correction, 'SelfCorrectionReasoner', lambda **kw: object())
    monkeypatch.setattr(gd_optimizer, 'GradientDescentOptimizer', lambda *a: object())
    monkeypatch.setattr(label_optimizer, 'LabelOptimizer', lambda: SimpleNamespace(set_database=lambda d: None))
    monkeypatch.setattr(logging, 'FileLogger', lambda p: Logger())
    monkeypatch.setattr(collector, 'Collector', Collector)
    monkeypatch.setattr(runner, 'set_stage', stages.append)
    config = dict(seed=42, cache_namespace='offline', precision=.9, recall=.9, sample_size=100)
    # Omit the keyword on the default path to test the actual default, not just False.
    options = {'nl_smoke': True} if nl_smoke else {}
    if fail_smoke:
        with pytest.raises(RuntimeError, match='unsupported generated plan'):
            runner.run_queries(config, tmp_path, tmp_path, **options)
        assert calls == ['nl', 'exit', 'collector_closed']
        assert json.loads((tmp_path / 'queries.json').read_text()) == []
    else:
        runner.run_queries(config, tmp_path, tmp_path, **options)
        assert calls == (['nl', 'cleanup'] if nl_smoke else []) + ['benchmark', 'silver', 'exit', 'collector_closed']
        assert stages == ['setup'] + (['smoke'] if nl_smoke else []) + ['benchmark', 'reference']
        rows = json.loads((tmp_path / 'queries.json').read_text())
        assert rows[-1]['metrics_status'] == 'scored'
        assert (tmp_path / 'smoke.csv').exists() == nl_smoke
        assert (tmp_path / 'benchmark_0.csv').exists()
        assert (tmp_path / 'silver_0.csv').exists()


def test_upstream_fixed_rotowire_extracts_form_independent_level_zero_cascades():
    """Use the real dependency algorithm and upstream logical query, no LLM config.

    Points and assists both read original report/name columns. Unlike filtering
    an extracted winner, assists does not consume the points output column.
    This checks dependency splitting, not physical plan generation/execution.
    """
    from reasondb.database.indentifier import ConcreteTableIdentifier, VirtualTableIdentifier, VirtualColumnIdentifier
    from reasondb.optimizer.dependency_graph import DependencyGraph
    roots = []
    for name in runner.TABLES:
        with (SOURCE / 'reasondb/evaluation/benchmarks/files' / f'{name}.csv').open(encoding='utf-8') as handle:
            columns = next(csv.reader(handle))
        roots.append(SimpleNamespace(identifier=VirtualTableIdentifier(name),
                                     concrete_identifier=ConcreteTableIdentifier(name),
                                     columns=[VirtualColumnIdentifier(f'{name}.{c}') for c in columns]))
    for query in ROTOWIRE_QUERIES:
        plan = deepcopy(query.get_gt_logical_plan())
        steps = [SimpleNamespace(logical_plan_step=s, index=i) for i, s in enumerate(plan.plan_steps)]
        graph = DependencyGraph.from_unoptimized_physical_plan(
            SimpleNamespace(plan_steps=steps), SimpleNamespace(root_tables=roots))
        assert 2 not in graph.dependencies[3]  # assists does not depend on points
        cascades, bounds = graph.get_parallel_pipelines(SimpleNamespace(step_bounds=(0, 2)))
        assert bounds == (2, 4)
        assert [[s.index for s in c] for c in cascades] == [[2], [3]]
