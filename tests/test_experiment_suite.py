import importlib.util
import json
from pathlib import Path
import sys

import pytest

SUITE_PATH = Path(__file__).resolve().parents[1] / 'scripts/run_experiment_suite.py'
if not SUITE_PATH.is_file():
    # Existing published image: launch scripts are deployed alongside .env here.
    SUITE_PATH = Path('/workspace/stretto-run/run_experiment_suite.py')
SPEC = importlib.util.spec_from_file_location('suite', SUITE_PATH)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


@pytest.mark.parametrize('profile_only', [False, True])
def test_stage_selection_keeps_checks_and_both_sizes_without_nl_smoke(profile_only):
    command = ['python', 'launcher.py', '--env-file', '/private/.env']
    stages = suite.build_stages('python', command, Path('/100.json'), Path('/728.json'),
                                profile_only=profile_only)
    expected = ['100-profile', '728-profile'] if profile_only else ['100-run', '100-profile', '728-run', '728-profile']
    assert [name for name, _, _ in stages] == ['inspect', 'doctor', 'tiny-cuda'] + expected
    for name, argv, experiment in stages:
        assert '--nl-smoke' not in argv
        if experiment:
            assert argv[-1] == ('/100.json' if name.startswith('100-') else '/728.json')
            assert argv[-3] == name.split('-')[1]


@pytest.mark.parametrize('status,errors,failure', [('failed', [], False), ('completed', ['backend'], False), ('completed', [], True)])
def test_result_rejects_failure(tmp_path, status, errors, failure):
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': status, 'backend_errors': errors}))
    if failure:
        (tmp_path / 'failure.txt').touch()
    with pytest.raises(RuntimeError):
        suite.validate_result(tmp_path)


def test_zero_exit_still_requires_successful_manifest(tmp_path):
    log = tmp_path / 'stage.log'
    with pytest.raises(RuntimeError, match='No Results'):
        suite.run_stage([sys.executable, '-c', 'print("no results")'], log, True)
    assert 'no results' in log.read_text()


def test_success_log_and_manifest(tmp_path):
    (tmp_path / 'manifest.json').write_text('{"status":"completed","backend_errors":[]}')
    result = suite.run_stage([sys.executable, '-c', f'print({"Results: " + str(tmp_path)!r})'], tmp_path / 'stage.log', True)
    assert result == str(tmp_path)


def test_nonzero_stage_prevents_later_work_and_preserves_state(tmp_path):
    stages = [('first', [sys.executable, '-c', 'print("failed"); raise SystemExit(4)'], False),
              ('later', [sys.executable, '-c', 'raise AssertionError("must not run")'], False)]
    with pytest.raises(RuntimeError, match='exited 4'):
        suite.execute(stages, tmp_path)
    state = json.loads((tmp_path / 'suite.json').read_text())
    assert state['status'] == 'failed'
    assert len(state['stages']) == 1
    assert state['stages'][0]['status'] == 'failed'
    assert not (tmp_path / 'later.log').exists()
