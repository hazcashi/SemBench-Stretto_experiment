"""Build a self-contained source overlay for the already published Vast runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stretto-main/stretto-main"
TABLES = ("players", "teams", "reports", "players_to_games", "teams_to_games")


def package(destination):
    files = {}
    for path in (SOURCE / "reasondb").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(SOURCE)
        if "ground_truth" in relative.parts:
            continue
        if "files" in relative.parts and path.name not in {f"{t}.csv" for t in TABLES}:
            continue
        # No local caches, credentials, weights or private/user data belong here.
        if path.suffix.lower() not in (".py", ".json", ".txt", ".md", ".csv", ".yaml", ".yml", ".jinja2"):
            continue
        files["source/" + relative.as_posix()] = path
    provenance = json.loads((ROOT / "configs/source_provenance.json").read_text(encoding="utf-8"))
    for name in TABLES:
        relative = f"reasondb/evaluation/benchmarks/files/{name}.csv"
        path = SOURCE / relative
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != provenance["files"][relative]["sha256"]:
            raise RuntimeError(f"Upstream dataset mismatch: {relative}")
    for relative in ("scripts/run_full_rotowire.py", "scripts/reproduce_in_container.py",
                     "configs/rotowire_full60.json", "configs/source_provenance.json",
                     "docs/full_rotowire_vast.md", "docker/image.txt"):
        files[relative] = ROOT / relative
    for path in (ROOT / "tests").glob("test_*.py"):
        files["tests/" + path.name] = path
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"format": 1, "base_image": (ROOT / "docker/image.txt").read_text().strip(),
                "source_revision": source_revision,
                "source_identity": "bundled file hashes are authoritative, including uncommitted changes",
                "upstream_commit": provenance["compared_upstream_commit"],
                "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}}
    tree_hash = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    manifest["tree_sha256"] = tree_hash
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.writestr("stretto-full-rotowire/" + name, path.read_bytes())
        archive.writestr("stretto-full-rotowire/bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Bundle: {destination.resolve()}")
    print(f"Source tree SHA256: {tree_hash}")
    print(f"ZIP SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist/stretto-full-rotowire.zip"))
    package(parser.parse_args().output)
