"""Execute predicates on joined/extracted columns without models or API calls."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source' if (ROOT / 'source').exists() else ROOT / 'stretto-main/stretto-main'
sys.path.insert(0, str(SOURCE))
duckdb = pytest.importorskip('duckdb')
pytest.importorskip('torch')
from reasondb.database.indentifier import ConcreteColumn, ConcreteTableIdentifier, DataType, VirtualTableIdentifier, IndexColumn
from reasondb.database.sql import SqlQuery, JoinConditions, JoinConditionConjuction, JoinConditionAvailableData, FilterConditionAvailableData
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.query_plan.llm_parameters import LlmParameterTemplate


@pytest.fixture
def joined():
    db = duckdb.connect()
    db.execute('CREATE TABLE _extracts (id INTEGER, _winner VARCHAR)')
    db.execute("INSERT INTO _extracts VALUES (1, 'Hawks'), (2, 'Bucks')")
    db.execute('CREATE TABLE teams_to_games (id INTEGER, name VARCHAR)')
    db.execute("INSERT INTO teams_to_games VALUES (1, 'Hawks'), (2, 'Pistons')")
    db.execute('ALTER TABLE teams_to_games ADD COLUMN _index_teams_to_games INTEGER')
    db.execute('UPDATE teams_to_games SET _index_teams_to_games = id')
    columns = [ConcreteColumn('_extracts._winner', DataType.STRING, 'winning_team'),
               ConcreteColumn('teams_to_games.name', DataType.STRING, 'name')]
    left, right = map(ConcreteTableIdentifier, ('_extracts', 'teams_to_games'))
    query = SqlQuery(db, JoinConditions(
        JoinConditionConjuction(left, 'inner', []),
        JoinConditionConjuction(right, 'inner', [JoinConditionAvailableData(
            '<left_table>.id = <right_table>.id', left, right)])), columns, [IndexColumn('teams_to_games')])
    state = SimpleNamespace(get_concrete_column_from_virtual=lambda col: {
        'winning_team': columns[0], 'name': columns[1]}[col.column_name])
    yield db, query, state
    db.close()


def observation(state, expression):
    return asyncio.run(TraditionalFilter(1.0, 0.0).get_observation(
        database_state=state, inputs=[VirtualTableIdentifier('joined')],
        output=VirtualTableIdentifier('filtered'), output_columns=[],
        llm_parameters={'filter_condition': LlmParameterTemplate(expression)},
        data_sample=[], logical_plan_step=None, logger=None))


@pytest.mark.parametrize('expression', [
    '{joined.winning_team} = {joined.name}',
    '{winning_team} = {name} AND {name} != \'Bucks\'',
])
def test_filter_after_join_preserves_each_physical_column(joined, expression):
    db, query, state = joined
    obs = observation(state, expression)
    assert db.execute(obs.get_sql([query]).to_positive_str(False)).fetchall() == [(1, 'Hawks', 'Hawks')]
    # Virtual/materialized queries use the logical aliases, not hidden names.
    db.execute('CREATE VIEW joined AS SELECT _winner AS winning_team, name FROM _extracts JOIN teams_to_games USING(id)')
    predicate = obs.filter_condition_virtual.cond_text(False)
    assert db.execute('SELECT winning_team, name FROM joined WHERE ' + predicate).fetchall() == [('Hawks', 'Hawks')]


def test_later_table_alias_renaming_preserves_filter_and_observation(joined):
    db, query, state = joined
    obs = observation(state, "{joined.winning_team} = {joined.name} AND '_extracts.name' = '_extracts.name'")
    original = obs.filter_condition.cond_text(False)
    filtered = obs.get_sql([query])
    filtered.rename_tables(['_extracts', 'teams_to_games'], 'left')
    sql = filtered.to_positive_str(False)
    assert "'_extracts.name' = '_extracts.name'" in sql
    assert db.execute(sql).fetchall() == [(1, 'Hawks', 'Hawks')]
    assert obs.filter_condition.cond_text(False) == original


def test_single_table_and_legacy_gold_mixing():
    state = SimpleNamespace(get_concrete_column_from_virtual=lambda col:
                            ConcreteColumn('teams.' + col.column_name, DataType.STRING))
    obs = observation(state, '{joined.name} != \'Pistons\'')
    with duckdb.connect() as db:
        db.execute("CREATE TABLE teams AS SELECT 'Hawks' AS name")
        assert db.execute('SELECT name FROM teams WHERE ' + obs.filter_condition.cond_text(False)).fetchall() == [('Hawks',)]
    for cond in (obs.filter_condition, FilterConditionAvailableData('<input_table>.name IS NOT NULL', ConcreteTableIdentifier('teams'))):
        cond.not_allow_discard_fraction = 0.25
        assert 'OR __random__ < 0.25' in cond.cond_text(True)
