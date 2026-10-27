import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" if (ROOT / "source").exists() else ROOT / "stretto-main/stretto-main"
sys.path.insert(0, str(SOURCE))

from reasondb.reproduction.profile import load_profile
from reasondb.reproduction.processes import Services
from reasondb.reproduction.subset import create_subset


def test_profile_routes_preserve_upstream():
    from reasondb.config.model_registry import ModelRegistry
    profile = load_profile(ROOT / "configs/single_gpu.json")
    registry = ModelRegistry.get()
    assert profile["large"]["model_name"].endswith("3.1-8B-Instruct")
    assert profile["small"]["model_name"].endswith("3.2-3B-Instruct")
    assert registry.spec_by_key("llama8B").size_class == "small"
    assert registry.spec_by_key("llama70B").port == 5012
    assert registry.method_config()["kv3B08"] == (profile["small"]["model_name"], .8)
    assert registry.method_config()["kv8B08"] == (profile["large"]["model_name"], .8)


@pytest.mark.parametrize("answer,valid", [("1", True), ("0", False), ("10", False),
                                          ("yes", False), ("Not sure", False), ("", False)])
def test_model_smoke_uses_boolean_wire_protocol(monkeypatch, tmp_path, answer, valid):
    requests = pytest.importorskip("requests")
    from reasondb.reproduction.runner import smoke_model
    from types import SimpleNamespace
    def post(url, json, timeout):
        assert json["boolean"] is True
        assert "Yes or No" not in json["questions"][0]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"answers": [answer]})
    monkeypatch.setattr(requests, "post", post)
    if valid:
        assert smoke_model({"port": 5014, "model_name": "stub"}, tmp_path)["answers"] == ["1"]
    else:
        with pytest.raises(RuntimeError, match="known-answer"):
            smoke_model({"port": 5014, "model_name": "stub"}, tmp_path)


def test_subset_determinism_and_relations(tmp_path):
    source = SOURCE / "reasondb/evaluation/benchmarks/files"
    first = create_subset(source, tmp_path / "a", 100)
    second = create_subset(source, tmp_path / "b", 100)
    assert first == second
    assert first["tables"]["reports"]["rows"] == 100
    def read(name):
        with (tmp_path / "a" / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    games = {r["game_id"] for r in read("reports")}
    for entity in ("players", "teams"):
        names = {r["name"] for r in read(entity)}
        relation = read(f"{entity}_to_games")
        assert relation
        assert all(r["game_id"] in games and r["name"] in names for r in relation)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_readiness_and_cleanup(tmp_path):
    port = free_port()
    script = tmp_path / "server.py"
    script.write_text('''from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{"status":"alive","model_name":"test"}')
HTTPServer(("127.0.0.1", '''+str(port)+'''), Handler).serve_forever()
''')
    with Services(tmp_path / "logs") as services:
        status = services.start("test", [sys.executable, str(script)], port, 10, "test")
        process = services.children[0][0]
        assert status["status"] == "alive"
        assert process.poll() is None
    assert process.poll() is not None


def test_early_exit_and_cleanup(tmp_path):
    services = Services(tmp_path)
    with pytest.raises(RuntimeError, match="exited"):
        with services:
            services.start("fail", [sys.executable, "-c", "raise SystemExit(7)"], free_port(), 5)
    assert services.children == []


def test_port_conflict_never_reuses_unknown_server(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with Services(tmp_path) as services, pytest.raises(OSError):
            services.start("conflict", [sys.executable, "-c", "pass"], listener.getsockname()[1], 1)
        assert services.children == []


def test_readiness_timeout_cleans_child(tmp_path):
    services = Services(tmp_path)
    with pytest.raises(TimeoutError):
        with services:
            services.start("timeout", [sys.executable, "-c", "import time; time.sleep(30)"], free_port(), .1)
    assert services.children == []


@pytest.mark.skipif(os.name == "nt", reason="Linux TIME_WAIT semantics")
def test_restart_port_after_server_close(tmp_path):
    import threading
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.listen()

    def serve_once():
        connection, _ = listener.accept()
        with connection:
            connection.recv(4096)
            connection.sendall(b"HTTP/1.0 200 OK\r\nContent-Length:2\r\n\r\nOK")
            connection.shutdown(socket.SHUT_WR)
            while connection.recv(4096):
                pass
        listener.close()

    thread = threading.Thread(target=serve_once)
    thread.start()
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.sendall(b"GET / HTTP/1.0\r\n\r\n")
        while client.recv(4096):
            pass
    thread.join(timeout=5)
    assert not thread.is_alive()
    # The readiness/cleanup test starts a real HTTP server; force the same port.
    script = tmp_path / "server.py"
    script.write_text('''from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{"status":"alive","model_name":"test"}')
HTTPServer(("127.0.0.1", '''+str(port)+'''), Handler).serve_forever()
''')
    with Services(tmp_path / "logs") as services:
        assert services.start("restarted", [sys.executable, str(script)], port, 10, "test")["status"] == "alive"


def test_failed_run_preserves_manifest(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from reasondb.reproduction import runner
    monkeypatch.setenv("REASONDB_CACHE_DIR", str(tmp_path / "cache"))
    run_dir = tmp_path / "failed-run"
    run_dir.mkdir()
    def fail_access(_profile):
        raise RuntimeError("injected unavailable access")
    monkeypatch.setattr(runner, "doctor", fail_access)
    profile = load_profile(ROOT / "configs/single_gpu.json")
    with pytest.raises(RuntimeError, match="unavailable access"):
        runner.run(profile, ROOT / "configs/single_gpu.json", run_dir)
    report = json.loads((run_dir / "manifest.json").read_text())
    assert report["status"] == "failed"
    assert report["nl_smoke_enabled"] is False
    assert report["benchmark_plan_source"] == "upstream ROTOWIRE_QUERIES fixed logical plans"
    assert report["error_type"] == "RuntimeError"
    assert report["cache_bytes_after_run"] == 0
    assert "injected unavailable access" in (run_dir / "failure.txt").read_text()


def test_access_fails_without_secrets_before_network(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("huggingface_hub")
    from reasondb.reproduction.runner import check_access
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Set HF_TOKEN"):
        check_access(load_profile(ROOT / "configs/single_gpu.json"))


def test_toolbox_isolation_and_quality_order():
    pytest.importorskip("torch")
    from reasondb.reproduction.profile import build_configurator
    from reasondb.interface.default_operator_toolbox import build_toolbox
    profile = load_profile(ROOT / "configs/single_gpu.json")
    configurator = build_configurator(profile)
    identifiers = [op.get_operation_identifier() for op in configurator.physical_operators]
    assert all(not any(s in identifier for s in ("Image", "Audio", "70B", "llava")) for identifier in identifiers)
    assert any("3.2-3B-Instruct-cr0.8" in identifier for identifier in identifiers)
    assert any("3.1-8B-Instruct-cr0.0-vanilla" in identifier for identifier in identifiers)
    for ops in (configurator.physical_operators.filter_operators, configurator.physical_operators.extract_operators):
        candidates = [op for op in ops if "TextQa" in type(op).__name__]
        # Stretto picks the highest-quality candidate as its labeling/fallback ceiling.
        best = max(candidates, key=lambda op: op.quality)
        assert "3.1-8B-Instruct-cr0.0-vanilla" in best.get_operation_identifier()
    defaults = [op.get_operation_identifier() for op in build_toolbox()]
    assert any("70B" in identifier for identifier in defaults)
    assert any("Image" in identifier for identifier in defaults)
