"""Launch the original KV server with the explicit single-GPU profile."""
import argparse
import json
import os
from pathlib import Path
import time
import signal

from reasondb.reproduction.profile import load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--role", choices=("small", "large"), required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-engine", choices=("none", "torch", "nsys"), default="none")
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    model = profile[args.role]
    import torch
    from flask import request, g
    from reasondb.backends import kv_cache_text_qa_server as server
    from reasondb.reproduction import telemetry

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    probe = telemetry.Probe(Path(os.environ["STRETTO_RUN_DIR"]) / "profiling",
                            args.role, model["model_name"], torch,
                            engine=args.trace_engine, detailed=args.detailed)
    telemetry.ACTIVE = probe
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    server.PINNED_KV_STORE.configure(0)
    class SingleGpuWrapper(server.KvTextQaModelWrapper):
        def compute_direct_qa_response(self, questions, contexts, boolean_question, batch_size=1):
            # The upstream direct endpoint otherwise defaults to batches of 128.
            return super().compute_direct_qa_response(questions, contexts, boolean_question, batch_size=1)

    server.model_wrapper = SingleGpuWrapper(
        model["model_name"], 0, compression_ratios=(0.8,), batch_sizes=(1,),
        press_name=profile["press_name"], vanilla_batch_size=1,
        model_revision=model["revision"], strict_errors=True,
    )
    placement = getattr(server.model_wrapper.pipe.model, "hf_device_map", {})
    if any(str(v) not in ("0", "cuda:0") for v in placement.values()):
        raise RuntimeError(f"Unexpected CPU/disk/multi-GPU placement: {placement}")
    stats_path = Path(os.environ["STRETTO_RUN_DIR"]) / f"{args.role}_requests.jsonl"

    @server.app.before_request
    def before_request():
        g.started = time.perf_counter()
        if request.path != "/status":
            probe.begin_request(request.headers.get("X-Stretto-Phase", "diagnostic"), request.path)
            telemetry.reset_peaks()

    @server.app.after_request
    def after_request(response):
        if request.path == "/status":
            payload = response.get_json()
            payload["revision"] = model["revision"]
            payload["attention_implementation"] = getattr(server.model_wrapper.pipe.model.config, "_attn_implementation", None)
            payload["device_map"] = {k: str(v) for k, v in placement.items()}
            response.set_data(json.dumps(payload))
            return response
        elapsed = time.perf_counter() - g.started
        probe.finish_request()
        g.probe_finished = True
        row = {"path": request.path, "status": response.status_code,
               "model": model["model_name"], "revision": model["revision"],
               "elapsed_s": elapsed,
               "allocated_bytes": torch.cuda.memory_allocated(),
               "reserved_bytes": torch.cuda.memory_reserved(),
               "peak_allocated_bytes": probe.peaks["allocated_bytes"],
               "peak_reserved_bytes": probe.peaks["reserved_bytes"],
               "prefill_s": probe.forward_times["prefill_s"] if probe.forward_observed else None,
               "decode_s": probe.forward_times["decode_s"] if probe.forward_observed else None,
               "timing_semantics": "GPU forward interval sums; cache-build prefill separate",
               "profile_enabled": args.detailed, "trace_engine": args.trace_engine,
               **probe.context,
               "inference_stats": (response.get_json(silent=True) or {}).get("stats")}
        with stats_path.open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        return response

    @server.app.teardown_request
    def finish_failed_request(error):
        if error is not None and request.path != "/status":
            if not getattr(g, "probe_finished", False):
                probe.finish_request()
            probe.write("request_error", error_type=type(error).__name__, reason=str(error))

    server.app.run(host="127.0.0.1", port=model["port"], debug=False, threaded=False)


if __name__ == "__main__":
    main()
