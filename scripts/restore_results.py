"""Restore and verify the checked-in experiment archive. Python stdlib only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(bundle, destination, verify_only=False):
    manifest = json.loads((bundle / 'archive-manifest.json').read_text(encoding='utf-8-sig'))
    expected = json.loads((bundle / 'remote-inventory.json').read_text(encoding='utf-8-sig'))['files']
    archive = bundle / manifest['archive']
    if sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 mismatch; fetch the archive again')
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        if len(names) != len(set(names)) or set(names) != {'results/' + n for n in expected}:
            raise ValueError('Archive file list differs from remote inventory')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe archive path')
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Archive path escapes destination')
            key = name.removeprefix('results/')
            if target.exists() and sha256(target) != expected[key]['sha256']:
                raise ValueError(f'Existing file differs; refusing to overwrite: {target}')
            if verify_only and not target.is_file():
                raise ValueError(f'Missing restored file: {target}')
        created = 0
        for entry in entries:
            target = destination.joinpath(*PurePosixPath(entry.filename).parts)
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restore-tmp')
            # Exclusive create preserves any incomplete output from an earlier attempt.
            with temporary.open('xb') as output, z.open(entry) as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            key = entry.filename.removeprefix('results/')
            if sha256(temporary) != expected[key]['sha256'] or temporary.stat().st_size != expected[key]['bytes']:
                raise ValueError(f'Restored file failed verification: {temporary}')
            os.replace(temporary, target)
            created += 1
    print(f'Verified {len(expected)} result files; restored {created}. Location: {destination / "results"}')


def main():
    bundle = Path(__file__).resolve().parents[1] / 'stretto-results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=bundle,
                        help='Parent directory of the restored results/ folder')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    restore(bundle, args.destination, args.verify_only)


if __name__ == '__main__':
    main()
