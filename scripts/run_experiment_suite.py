"""Sequential 100-report and 728-report run/profile using the prebuilt runtime."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid


def validate_result(directory):
    if directory is None:
        raise RuntimeError('No Results path in current log')
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('status') != 'completed' or manifest.get('backend_errors')
            or (directory / 'failure.txt').exists()):
        raise RuntimeError(f'Incomplete/failed experiment: {directory}')


def run_stage(command, log_path, experiment=False):
    result = None
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                if line.startswith('Results: '):
                    result = line.removeprefix('Results: ').strip()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code != 0:
        raise RuntimeError(f'Stage exited {code}; inspect {log_path}')
    if experiment:
        validate_result(result)
    return result


def execute(stages, directory):
    state = {'status': 'running', 'stages': []}
    status_file = directory / 'suite.json'

    def save():
        status_file.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')

    try:
        for name, command, experiment in stages:
            item = {'name': name, 'status': 'running', 'log': str(directory / f'{name}.log')}
            state['stages'].append(item)
            save()
            print(f'\nStarting {name}', flush=True)
            item['results'] = run_stage(command, Path(item['log']), experiment)
            item['status'] = 'completed'
            save()
        state['status'] = 'completed'
    except BaseException as exc:
        if state['stages']:
            state['stages'][-1]['status'] = 'failed'
        state['status'] = 'failed'
        state['error'] = str(exc)
        raise
    finally:
        save()


def build_stages(python, command, small, full, *, profile_only=False):
    stages = [(action, command + [action, '--config', str(small)], False)
              for action in ('inspect', 'doctor')]
    stages.append(('tiny-cuda', [str(python), '-m', 'pytest',
                  '/opt/stretto/tests/test_profiling.py', '-k', 'cuda_real_tiny_llama', '-q'], False))
    for size, config in ((100, small), (728, full)):
        for action in (('profile',) if profile_only else ('run', 'profile')):
            stages.append((f'{size}-{action}', command + [action, '--config', str(config)], True))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path(__file__).resolve().with_name('.env'))
    parser.add_argument('--profile-only', action='store_true',
                        help='Run 100 and 728 fixed benchmark profiles; omit baseline run rounds (PyTorch, not Nsight)')
    args = parser.parse_args()
    python = Path('/opt/venv/bin/python')
    helper = Path(__file__).resolve().with_name('reproduce_in_container.py')
    if not python.is_file() or not helper.is_file():
        parser.error('Use the Stretto prebuilt image and upload both launcher scripts')
    from reproduce_in_container import load_credentials
    load_credentials(args.env_file)  # Validate without logging secret values.
    root = Path('/results/suites')
    root.mkdir(parents=True, exist_ok=True)
    # Hold across all stages. Prevent two suites competing for the same GPU/cache.
    import fcntl
    with (root / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another experiment suite is running')
        directory = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        directory.mkdir()
        base = json.loads(Path('/opt/stretto/configs/single_gpu.json').read_text())
        base['reports_limit'] = 100
        small = directory / '100.json'
        small.write_text(json.dumps(base, indent=2) + '\n')
        base.update(name='llama8b_3b_rotowire_full_v1', reports_limit=728)
        full = directory / '728.json'
        full.write_text(json.dumps(base, indent=2) + '\n')
        command = [str(python), str(helper), '--env-file', str(args.env_file.resolve())]
        stages = build_stages(python, command, small, full, profile_only=args.profile_only)
        print(f'Suite directory: {directory}', flush=True)
        execute(stages, directory)
        print(f'All stages completed: {directory}/suite.json', flush=True)


if __name__ == '__main__':
    main()
