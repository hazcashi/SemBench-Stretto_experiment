"""One command in the Vast prebuilt Stretto SSH image; no rebuild or pip install."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f"Missing/invalid bundle file: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Bundle content changed: {relative}; build/upload a fresh bundle")
    observed = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    if observed != manifest["tree_sha256"]:
        raise RuntimeError("Bundle manifest tree hash mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path,
                        default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument("--experiment-dir", type=Path,
                        help="Same directory resumes verified scored queries; a new directory starts fresh")
    parser.add_argument("--action", choices=("run", "profile"), default="profile")
    parser.add_argument("--check", action="store_true", help="Offline bundle/routing/query-shape checks only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = verify_bundle(root)
    source = root / "source"
    python = Path("/opt/venv/bin/python")
    if not python.is_file():
        parser.error("Use the prebuilt Stretto Vast SSH image: /opt/venv/bin/python is missing")
    from reproduce_in_container import runtime_environment, load_credentials
    env = runtime_environment(os.environ, {})
    env["PYTHONPATH"] = str(source)
    env["STRETTO_IMAGE_REF"] = manifest["base_image"]
    env["STRETTO_IMAGE_ID"] = "provider image digest not independently verified"
    env["STRETTO_SOURCE_REVISION"] = manifest["source_revision"]
    env["STRETTO_SOURCE_TREE_SHA256"] = manifest["tree_sha256"]
    env["STRETTO_FLASH_BUILD"] = "prebuilt runtime; actual versions recorded in pip-freeze"
    for path in (Path("/results/logging"), Path("/results/full-rotowire60")):
        path.mkdir(parents=True, exist_ok=True)
    logs = source / "logging"
    if not logs.exists() and not logs.is_symlink():
        logs.symlink_to("/results/logging", target_is_directory=True)
    if not logs.is_symlink() or logs.resolve() != Path("/results/logging"):
        parser.error("Bundle source/logging must point to /results/logging")
    experiment = (args.experiment_dir or Path("/results/full-rotowire60") /
                  f"{manifest['tree_sha256'][:12]}-{args.action}").resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    # One GPU/port set is shared by all attempts, even in different directories.
    import fcntl
    with Path("/results/full-rotowire60/.gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another full Rotowire workload is running")
        config = json.loads((root / "configs/rotowire_full60.json").read_text(encoding="utf-8"))
        config.update(experiment_dir=str(experiment), source_tree_sha256=manifest["tree_sha256"],
                      measurement_mode=args.action)
        config_path = experiment / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            parser.error("Experiment config/source differs; choose another --experiment-dir")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        context = {"at": datetime.now(timezone.utc).isoformat(), "bundle": manifest,
                   "action": args.action, "declared_image_identity": "user must select the pinned image in Vast",
                   "experiment_dir": str(experiment), "credential_values_recorded": False}
        (experiment / "deployment.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
        print(f"Experiment: {experiment}", flush=True)
        print("Scope: ALL 728 reports, ALL 60 RotowireRandom queries; 8B reference, not the original 70B experiment", flush=True)
        subprocess.run([str(python), "-m", "reasondb.reproduction.runner", "inspect",
                        "--config", str(config_path)], cwd=source, env=env, check=True)
        # These checks never contact APIs or download model weights.
        subprocess.run([str(python), "-m", "pytest", str(root / "tests/test_full_workload.py"),
                        "-q"], cwd=source, env=env, check=True)
        if args.check:
            print("Offline checks passed; GPU/model/API execution has not started.", flush=True)
            return
        env.update(load_credentials(args.env_file))
        print("Starting live inference; completed queries resume from verified checkpoints.", flush=True)
        code = subprocess.call([str(python), "-m", "reasondb.reproduction.runner", args.action,
                                "--config", str(config_path)], cwd=source, env=env)
        if code:
            print(f"Failed (exit {code}). Logs/checkpoints retained. Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
