"""Run the packaged Stretto environment inside a Vast SSH container (no Docker daemon).

Copy this standalone launcher into /workspace/stretto-run. It only reads the two
credential fields from the adjacent .env; shell expressions are never executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


SOURCE = Path('/opt/stretto/source')
PYTHON = Path('/opt/venv/bin/python')
EXPECTED_IMAGE = 'ghcr.io/hazcashi/sembench-stretto_experiment:436207bd211943709930577ec44895d8a1db0cd5'
CREDENTIALS = ('HF_TOKEN', 'OPENAI_API_KEY')
DEFAULTS = {
    'HF_HOME': '/data/hf',
    'REASONDB_CACHE_DIR': '/data/cache',
    'STRETTO_DATA_ROOT': '/data',
    'STRETTO_RESULTS_ROOT': '/results',
    'PYTHONUNBUFFERED': '1',
    'PYTHONHASHSEED': '42',
    'KV_CACHE_PIN_GB': '0',
    'TOKENIZERS_PARALLELISM': 'false',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OMP_NUM_THREADS': '4',
}


def load_credentials(path):
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key = key.strip()
        if key not in CREDENTIALS:
            continue
        if not sep or key in values:
            raise ValueError(f'Missing assignment or duplicate credential field: {key}')
        value = value.strip()
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f'Unmatched quotes in field: {key}')
            value = value[1:-1]
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'Empty value or whitespace in field: {key}; omit inline comments')
        values[key] = value
    missing = [key for key in CREDENTIALS if not values.get(key)]
    if missing:
        raise ValueError('Fill credential fields: ' + ', '.join(missing))
    return values


def runtime_environment(parent, credentials):
    env = dict(parent)
    # Vast's SSH setup may replace PATH or omit Dockerfile environment variables.
    env.update(DEFAULTS)
    env.update(credentials)
    env['PATH'] = '/opt/venv/bin:/usr/local/cuda/bin:' + env.get('PATH', '/usr/bin:/bin')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    # Preserve a provider's UUID/index mapping; otherwise use the sole rented GPU.
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'access', 'doctor', 'run', 'profile', 'profile-nsys'))
    parser.add_argument('--env-file', type=Path, default=Path(__file__).resolve().with_name('.env'))
    parser.add_argument('--config', type=Path, default=Path('/opt/stretto/configs/single_gpu.json'))
    parser.add_argument('--nl-smoke', action='store_true',
                        help='Opt into additional NL smoke (not part of the fixed Rotowire benchmark)')
    args = parser.parse_args()
    if args.nl_smoke and args.action not in ('run', 'profile', 'profile-nsys'):
        parser.error('--nl-smoke requires run, profile, or profile-nsys')
    if not PYTHON.is_file() or not (SOURCE / 'reasondb/reproduction/runner.py').is_file():
        parser.error('Packaged Stretto runtime missing: expected /opt/venv and /opt/stretto/source. Check the rented image.')
    config = args.config.resolve()
    if not config.is_file():
        parser.error('Config file not found')
    try:
        credentials = {} if args.action == 'inspect' else load_credentials(args.env_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    env = runtime_environment(os.environ, credentials)
    installation = Path('/opt/stretto/installation.json')
    if Path('/opt/stretto/bootstrap-owner').exists() and not installation.is_file():
        parser.error('CUDA template installation is incomplete; finish bootstrap first')
    for directory in ('/data/hf', '/data/cache', '/results/logging', '/results/terminal', '/results/host-info'):
        Path(directory).mkdir(parents=True, exist_ok=True)
    # The published image directs upstream relative logs to /results/logging.
    if not (SOURCE / 'logging').is_symlink() or (SOURCE / 'logging').resolve() != Path('/results/logging'):
        parser.error('Unexpected source/logging layout; expected symlink to /results/logging')
    hashes = {}
    for relative in ('reasondb/reproduction/runner.py', 'reasondb/reproduction/backend.py',
                     'reasondb/backends/kv_cache_text_qa_server.py',
                     'reasondb/database/sql.py', 'reasondb/operators/filter/traditional_filter.py'):
        p = SOURCE / relative
        if p.is_file():
            hashes[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    context = {
        'deployment': 'vast_direct_container',
        'expected_template_image': EXPECTED_IMAGE,
        'image_identity_status': 'not_verified_from_host_daemon',
        'source_sha256': hashes,
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'nl_smoke_enabled': args.nl_smoke,
        'python': str(PYTHON), 'source': str(SOURCE),
        'credential_values_recorded': False,
    }
    if installation.is_file():
        record = json.loads(installation.read_text(encoding='utf-8'))
        if record.get('status') != 'installation_checks_passed':
            parser.error('CUDA template installation checks have not passed')
        context['deployment'] = record['deployment']
        context['expected_template_image'] = record['declared_template_image']
        context['installation'] = record
        env['STRETTO_SOURCE_REVISION'] = record['source_revision']
        env['STRETTO_FLASH_BUILD'] = record['flash_attention_build']
    # Per-invocation file preserves evidence across subsequent runs/configs.
    import uuid
    evidence = Path('/results/host-info') / f'container-launch-{uuid.uuid4().hex[:8]}.json'
    evidence.write_text(json.dumps(context, indent=2) + '\n', encoding='utf-8')
    print(f'Deployment record: {evidence}', flush=True)
    os.chdir(SOURCE)
    os.execve(str(PYTHON), [str(PYTHON), '-m', 'reasondb.reproduction.runner',
                          args.action, '--config', str(config)] + (['--nl-smoke'] if args.nl_smoke else []), env)


if __name__ == '__main__':
    main()
