# Stretto Text-Only Reproduction & GPU Profiling — Codex Instructions

## Mission
Reproduce the **text-only** execution path of the upstream Stretto project on **one rented NVIDIA GPU with approximately 48 GB VRAM**, then establish reproducible application-level and GPU-level profiling. This is a **small undergraduate research baseline**, not a request to reproduce every paper experiment or implement speculative optimizations now.

Upstream repository: https://github.com/DataManagementLab/stretto
Upstream README: https://github.com/DataManagementLab/stretto/blob/main/README.md

Read the actual repository, README, launch scripts, model registry, model backends, optimizer, workers, benchmark producers, and cache generation code before modifying anything. Treat the repository source as ground truth; this document is a project brief, not a claim that every proposed model is already supported.

## Goals and scope
1. Bring up a reproducible **text-only** Stretto baseline with local model inference.
2. Replace the paper's original **70B large + 8B small** text model pairing with **8B large + 3B (preferred) or <=4B small** for a single 48GB GPU.
3. Preserve Stretto's query planning, operator selection, cache semantics, execution, and metrics as much as practical.
4. Successfully run one end-to-end small text-only query, then a small text benchmark, and only then expand.
5. Capture a usable profile of query/operator timing, LLM prefill/decode timing if instrumentable, GPU memory, model/cache footprint, CPU<->GPU transfers, and idle gaps.
6. Build a Docker-based reproducible deployment for a rented Linux NVIDIA host; keep the workflow friendly to development on Windows + VS Code.

## Important constraints
- Start with Stretto's original backend; **do not replace it with vLLM** for the baseline.
- **Do not add MoE** or expert prefetching now; these are possible *future* research directions.
- The objective is reproducibility and identifying bottlenecks, **not speedups yet**.
- Single GPU means **GPU index 0** unless the runtime remaps it. Inspect launch scripts for implicit multi-GPU assumptions.
- Target models: `meta-llama/Llama-3.1-8B-Instruct` as the large model; `meta-llama/Llama-3.2-3B-Instruct` as candidate small model. Verify gated model access, licensing and Hugging Face auth. If 3B compatibility is costly, investigate a supported smaller model and document the trade-off before swapping model families.
- Both models may fit in 48GB in principle, but account for FP16/BF16 weights, separate CUDA contexts, activation workspaces, KV caches, embedding server, batch sizes, and allocator overhead. Measure peak, do not assume fit.
- Query planning currently requires `OPENAI_API_KEY` per upstream README. **Local text inference does not imply a fully offline query-planning pipeline.** First identify the exact API call boundary. Do not silently bypass it, invent results, or claim fully local operation. If possible, provide a clearly separated optional future offline-planner path.
- Prefer text datasets such as Rotowire/Movies and a query path that never starts image/audio backends. Do not assume every benchmark variant is text-only; inspect it.
- Preserve repository submodules (`kvpress`, `palimpzest`) and record their pinned commits. Do not push model weights, HF tokens, datasets, generated caches, or secret `.env` files to Git.
- We may rent an RTX A6000 48GB, RTX 6000 Ada 48GB, L40S 48GB, or similar; deployment must not depend on a single GPU SKU if avoidable.
- Storage may need hundreds of GB for complete KV-cache sweeps. Begin with **minimal caches / minimal workload** and estimate actual disk footprint before generation.

## Start with a repository audit (do this first)
Inspect and report:
- Existing setup, runtime and dependency versions (Python 3.12 upstream), CUDA / FlashAttention needs, and submodule install steps.
- `reasondb/config/model_registry.py` (actual keys, model names, families, size classes, method prefixes, ports, defaults).
- `scripts/start_servers_text.sh`, `reasondb/backends/kv_cache_text_qa_server.py`, text embedding server and related server startup code.
- `scripts/run_coordinator.py`, `scripts/run_worker.py`, producer definitions, and local mode usage.
- `scripts/generate_kv_cache.py` and actual KV-cache method mappings / compression compatibility.
- Where model-size classifications, costs, profiling samples, operator availability, and precision/recall expectations affect the optimizer.
- What is stored in `benchmark_results/`, and how existing telemetry captures events.
- Identify the smallest legitimate end-to-end *text-only* example and its command, including whether the plan uses a cloud LLM.

Before coding, produce `docs/reproduction_plan.md` with: architecture/call graph, changes required, risks, exact test stages, and GPU/disk/RAM estimates. Use verified file paths and line references where possible. Clearly distinguish **verified current behavior** from **proposed changes**.

## Implementation phases
### Phase 0 — Inspect without destabilizing upstream
- Record upstream remote and commit SHA, submodule SHAs, Python/CUDA/GPU driver information.
- Add a `docs/upstream_baseline.md` documenting unmodified configurations.
- Prefer new config files, launch scripts, and a small compatibility layer over intrusive refactoring.

### Phase 1 — Model-size remapping
- Add/configure 8B as **large** and 3B as **small** while retaining unmodified upstream presets for comparison.
- Update all coupled places (registry, method prefixes, ports, cache generation, launch scripts, cost / accuracy estimation configs, benchmark selectors).
- Avoid fragile textual search/replace of `70B` with `8B`: these labels may encode semantics, cache compression levels, or filenames.
- Test that each model works *individually* before starting both.
- Then test both models concurrently on the same GPU, track VRAM and port conflicts, and confirm the correct backend is selected for every operator.
- Re-profile runtime/quality assumptions for the new pairing. Do not reuse the original 70B-vs-8B quality/cost measurements as if they described 8B-vs-3B.

### Phase 2 — Text-only baseline
- Run a small text-only demo, then one small benchmark (prefer Rotowire or Movies if verified feasible).
- Support coordinator + text worker, or upstream `--local` path if validated; keep all required text/embedding servers running.
- Document explicit commands and expected output paths.
- Make evaluation repeatable: fixed dataset subset, precision/recall targets, seeds where applicable, max tokens, batch size, compression settings, concurrency, model versions, and run IDs.
- Save logs and record failures; report partial results honestly.

### Phase 3 — Profiling
Produce three levels of evidence:
1. **End-to-end/application:** query planning time, optimization time, per-operator latency, backend request latency, execution time, total wall-clock and throughput where meaningful. Prefer Stretto telemetry first; add monotonic timestamp instrumentation only where needed.
2. **Inference/memory:** separate prefill vs decode latency and token counts if reliably observable, CPU RAM, GPU allocated/reserved/peak memory, KV cache resident/disk bytes, load and transfer timing.
3. **GPU trace:** NVIDIA Nsight Systems or PyTorch Profiler on the actual **LLM backend process**, not just coordinator. Annotate phases with NVTX where possible. If rented host forbids relevant profiling permissions/counters, explain precisely what works and what does not; do not request excessively broad container privileges as a blanket fix.

Each result should record: git commit, image digest/tag, GPU name, driver, CUDA runtime, Python/PyTorch versions, models and revisions, workload, concurrency, KV compression settings, run timestamp, cache cold/warm status, and sampling overhead. Store compact summaries as JSON/CSV and traces outside Git.

### Phase 4 — Docker deployment
- Create a GPU-ready Linux/amd64 Dockerfile and `.dockerignore`, preferably pinning base image and key dependencies.
- Check Python 3.12 support and correct compatible PyTorch / CUDA / FlashAttention versions before pinning. Avoid hardcoding compatibility guesses.
- Use bind mounts or Docker volumes for source (if appropriate), HF cache, Stretto dataset/KV cache, logs, profiler outputs. Docker image must not embed gated model weights or API secrets.
- Provide `docker compose` or clear `docker run` commands for text inference services + executor/coordinator, with ports and health/readiness checks as feasible.
- Docker GPU access requires **host NVIDIA driver + NVIDIA Container Toolkit**. Validate with a CUDA smoke test on the rented host.
- If building locally on Windows, use Docker Desktop Linux containers (`linux/amd64`) or GitHub Actions to build and publish to `ghcr.io/<owner>/<repo>:<tag>` (GHCR). **A normal GitHub repo stores the Dockerfile/code; GHCR stores built images.** A GPU is not required simply to package Python code, but CUDA extensions may need a supported compiler/toolchain and architecture flags; validate the resulting image on the target GPU.
- Do not assume the rented provider gives a full VM or Docker daemon; document any provider-specific restrictions before depending on `docker compose` or privileged profiling.
- Prepare a short deployment guide for both `git clone && docker build` and `docker pull ghcr.io/...` methods. If repository/image is private, document authentication without committing credentials.
- Optional CI: GitHub Actions smoke test + build/push image on release or manual trigger. Do not trigger costly GPU execution in CI.

## Deliverables
- `docs/reproduction_plan.md` — verified architecture audit and migration plan.
- `docs/upstream_baseline.md` — upstream commit, environment, baseline configurations.
- `docs/deployment.md` — from empty rented host to benchmark, commands for Docker/GPU/auth/volume setup.
- `docs/profiling.md` — instrumentation locations, exact commands, interpretation caveats, output schema.
- Dockerfile, `.dockerignore`, compose/launch scripts and minimal model config as appropriate.
- Small text-only reproducibility test; and a metrics/report directory convention.
- `docs/experiment_log.md` — attempted runs, errors, fixes, and final validated command.

## Acceptance criteria
1. On the rented GPU, Docker container can run `nvidia-smi` and PyTorch sees CUDA.
2. A single **small** model backend responds, then 8B backend responds, then both coexist without unexpected OOM.
3. At least one true text query completes through the actual Stretto planning/execution chain, with its dependencies (including any planner API) explicitly documented.
4. A text benchmark runs on a deliberately small, documented workload with meaningful outputs and no image/70B dependency.
5. Report per-operator and end-to-end times, GPU peak VRAM, and at least one backend GPU/profile trace (or a clear evidence-based limitation explanation).
6. Reproducible instructions work from a fresh checkout without manual edits to hidden global paths.

## Instructions for Codex behavior
- First **inspect**, then summarize specific files/paths and the minimum changes needed. Do not claim success without commands and observed output.
- Make changes incrementally, one phase at a time; run available CPU-only/static smoke tests before expensive GPU tests.
- Maintain a working task checklist and mark pending GPU-only validation explicitly.
- If blocked by no GPU, API key, Hugging Face access, or missing dependencies, implement only verifiable work and give exact instructions for running the remaining checks on the rented host.
- Avoid premature refactoring, unrequested query optimization, swapping to vLLM, adding MoE, or faking measurements.
- Never commit credentials, user data, trained weights, or large caches.
- Communicate in **Traditional Chinese**, while code, identifiers, commands and documentation filenames can remain English.

## First Codex request
"Read AGENTS.md and audit the checked-out Stretto repository against it. Before making large code changes, write docs/reproduction_plan.md with verified code locations, a minimal text-only runnable path, requirements for single-GPU 8B + 3B, and a staged Docker/profiling implementation checklist. Then implement only Phase 0 and the smallest safe prerequisite for Phase 1. Report what was verified locally and what still requires a rented 48GB GPU."
