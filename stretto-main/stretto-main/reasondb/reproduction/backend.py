"""Launch the original KV server with the explicit single-GPU profile."""
import argparse
import json
import os
from pathlib import Path
import time

from reasondb.reproduction.profile import load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--role", choices=("small", "large"), required=True)
    parser.add_argument("--trace-dir")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    model = profile[args.role]
    import torch
    from flask import request, g
    from reasondb.backends import kv_cache_text_qa_server as server

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
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
    traced = False

    @server.app.before_request
    def before_request():
        nonlocal traced
        g.started = time.perf_counter()
        g.trace = None
        if request.path != "/status":
            torch.cuda.reset_peak_memory_stats()
        if args.trace_dir and not traced and request.path == "/text_qa":
            traced = True
            try:
                g.trace = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True,
                )
                g.trace.__enter__()
            except Exception as exc:
                g.trace = None
                Path(args.trace_dir, f"{args.role}_limitation.txt").write_text(str(exc))

    @server.app.after_request
    def after_request(response):
        if request.path == "/status":
            payload = response.get_json()
            payload["revision"] = model["revision"]
            payload["attention_implementation"] = getattr(server.model_wrapper.pipe.model.config, "_attn_implementation", None)
            payload["device_map"] = {k: str(v) for k, v in placement.items()}
            response.set_data(json.dumps(payload))
            return response
        if g.trace is not None:
            try:
                g.trace.__exit__(None, None, None)
                g.trace.export_chrome_trace(str(Path(args.trace_dir) / f"{args.role}.json"))
            except Exception as exc:
                Path(args.trace_dir, f"{args.role}_limitation.txt").write_text(str(exc))
        row = {"path": request.path, "status": response.status_code,
               "model": model["model_name"], "revision": model["revision"],
               "elapsed_s": time.perf_counter()-g.started,
               "allocated_bytes": torch.cuda.memory_allocated(),
               "reserved_bytes": torch.cuda.memory_reserved(),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "prefill_s": None, "decode_s": None,
               "profile_enabled": bool(args.trace_dir),
               "inference_stats": (response.get_json(silent=True) or {}).get("stats")}
        with stats_path.open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        return response

    server.app.run(host="127.0.0.1", port=model["port"], debug=False, threaded=False)


if __name__ == "__main__":
    main()
