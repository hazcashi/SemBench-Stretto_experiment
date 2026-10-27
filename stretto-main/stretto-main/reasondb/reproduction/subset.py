"""Deterministic relational Rotowire subset, without reindexing source files."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

TABLES = ("players", "teams", "reports", "players_to_games", "teams_to_games")


def copy_full_dataset(source, destination):
    """Preserve every row and byte of all five upstream tables, including row IDs."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"selection": "all upstream rows, byte-identical copies", "tables": {}}
    for table in TABLES:
        src, dst = source / f"{table}.csv", destination / f"{table}.csv"
        with src.open(encoding="utf-8-sig", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if table == "reports" and count != 728:
            raise ValueError(f"Full Rotowire requires 728 reports, found {count}")
        shutil.copyfile(src, dst)
        manifest["tables"][table] = {"rows": count, "sha256": sha256(dst),
                                     "source_sha256": sha256(src)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def create_subset(source, destination, limit=100):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tables, fields = {}, {}
    for table in TABLES:
        with (source / f"{table}.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields[table] = reader.fieldnames
            tables[table] = list(reader)
    tables["reports"] = tables["reports"][:limit]
    if not tables["reports"]:
        raise ValueError("Rotowire reports is empty")
    games = {r["game_id"] for r in tables["reports"]}
    for entity in ("players", "teams"):
        relation = f"{entity}_to_games"
        tables[relation] = [r for r in tables[relation] if r["game_id"] in games]
        names = {r["name"] for r in tables[relation]}
        tables[entity] = [r for r in tables[entity] if r["name"] in names]
        if names - {r["name"] for r in tables[entity]}:
            raise ValueError(f"Dangling {entity} relation")
    manifest = {"limit": limit, "selection": "first reports in source CSV order", "tables": {}}
    for table in TABLES:
        path = destination / f"{table}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fields[table], lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables[table])
        manifest["tables"][table] = {"rows": len(tables[table]), "sha256": sha256(path),
                                     "source_sha256": sha256(source / path.name)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest
