"""Fetch only public upstream Rotowire CSVs, verifying the audited content hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", default="docs/source_provenance.json")
    parser.add_argument("--destination", default="stretto-main/stretto-main/reasondb/evaluation/benchmarks/files")
    args = parser.parse_args()
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    for table in ("players", "teams", "reports", "players_to_games", "teams_to_games"):
        relative = f"reasondb/evaluation/benchmarks/files/{table}.csv"
        expected = provenance["files"][relative]["sha256"]
        if not provenance["files"][relative]["matches_upstream"]:
            raise RuntimeError(f"Unaudited dataset: {relative}")
        target = destination / f"{table}.csv"
        if target.exists() and hashlib.sha256(target.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected:
            continue
        url = f"https://raw.githubusercontent.com/DataManagementLab/stretto/{provenance['compared_upstream_commit']}/{relative}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=60) as response:
                    content = response.read().replace(b"\r\n", b"\n")
                if hashlib.sha256(content).hexdigest() != expected:
                    raise RuntimeError(f"SHA256 mismatch: {relative}")
                target.write_bytes(content)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
        print(f"Verified {table}.csv")


if __name__ == "__main__":
    main()
