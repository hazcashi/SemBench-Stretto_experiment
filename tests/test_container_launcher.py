"""CPU-only checks for secret parsing and SSH environment restoration."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('container_launcher', Path(__file__).resolve().parents[1] / 'scripts/reproduce_in_container.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_credentials_bom_crlf_and_no_shell_evaluation(tmp_path):
    path = tmp_path / '.env'
    path.write_bytes(b'\xef\xbb\xbfHF_TOKEN="hf_test"\r\nOPENAI_API_KEY=\'$(never-execute)\'\r\nPATH=bad\r\nSTRETTO_DATA_DIR=./data\r\n')
    assert launcher.load_credentials(path) == {'HF_TOKEN': 'hf_test', 'OPENAI_API_KEY': '$(never-execute)'}


@pytest.mark.parametrize('line', ['HF_TOKEN=', 'HF_TOKEN="secret-value', 'HF_TOKEN=secret-value # comment', 'HF_TOKEN=secret-value\nHF_TOKEN=another-secret'])
def test_invalid_credentials_do_not_echo_secrets(tmp_path, line):
    path = tmp_path / '.env'
    path.write_text(line + '\nOPENAI_API_KEY=test\n', encoding='utf-8')
    with pytest.raises(ValueError) as err:
        launcher.load_credentials(path)
    assert 'secret-value' not in str(err.value)
    assert 'another-secret' not in str(err.value)


def test_environment_restores_runtime_preserves_gpu_assignment():
    parent = {'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': 'GPU-assigned',
              'PYTHONPATH': '/wrong/source', 'PYTHONHOME': '/wrong/python', 'HF_HOME': '/wrong/cache'}
    env = launcher.runtime_environment(parent, {'HF_TOKEN': 'test'})
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-assigned'
    assert env['HF_HOME'] == '/data/hf'
    assert env['PATH'].startswith('/opt/venv/bin:')
    assert 'PYTHONPATH' not in env and 'PYTHONHOME' not in env
    assert parent['HF_HOME'] == '/wrong/cache'
    assert launcher.runtime_environment({}, {})['CUDA_VISIBLE_DEVICES'] == '0'
