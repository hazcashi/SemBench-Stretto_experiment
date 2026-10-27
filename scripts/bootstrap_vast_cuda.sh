#!/usr/bin/env bash
# Run inside the NVIDIA CUDA Vast SSH container, not on the Windows host.
set -Eeuo pipefail
trap 'printf "Setup stopped at line %s. Keep the log; do not run the benchmark yet.\n" "$LINENO" >&2' ERR
REV=436207bd211943709930577ec44895d8a1db0cd5
ROOT=/opt/stretto
CHECKOUT=$ROOT/checkout
export PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH
unset PYTHONPATH PYTHONHOME
export DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# df inside an overlay may show the host disk, not the paid container quota.
QUOTA=${1:-}
if [[ ! "$QUOTA" =~ ^[0-9]+$ ]] || (( QUOTA < 250 )); then
    echo 'Usage: bash bootstrap_vast_cuda.sh ACTUAL_VAST_DISK_GB' >&2
    echo 'Read the allocated Disk capacity on your instance card. Require >=250 GB; recommend 300 GB. Do not enter host SSD capacity.' >&2
    exit 2
fi
[[ $(id -u) == 0 ]] || { echo 'Run as root inside the rented container.' >&2; exit 2; }
[[ $(uname -m) == x86_64 ]] || { echo 'This environment requires x86_64.' >&2; exit 2; }
python3.12 - <<'PY'
from pathlib import Path
import shutil
release = Path('/etc/os-release').read_text()
assert 'ID=ubuntu' in release and 'VERSION_ID="24.04"' in release, 'Expected Ubuntu 24.04'
assert shutil.disk_usage('/opt').free >= 150 * 1024**3, 'Need >=150 GiB free before installing'
PY
nvcc --version | grep -q 'release 12.8,' || { echo 'CUDA 12.8 devel toolkit required (nvcc).' >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Own only this installation. Never overwrite a provider/user virtualenv.
if [[ -e /opt/venv && ! -f "$ROOT/bootstrap-owner" ]]; then
    echo '/opt/venv already exists without our ownership marker; stop and inspect.' >&2
    exit 2
fi
mkdir -p "$ROOT" /data/hf /data/cache /results/logging /results/terminal /results/host-info
if [[ -f "$ROOT/bootstrap-owner" ]]; then
    [[ $(cat "$ROOT/bootstrap-owner") == "$REV" ]] || { echo 'Different bootstrap revision.' >&2; exit 2; }
else
    printf '%s\n' "$REV" > "$ROOT/bootstrap-owner"
fi
printf '%s\n' '{"status":"installation_in_progress"}' > "$ROOT/installation.json"
apt-get update
apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-dev git build-essential ca-certificates libgomp1 tmux
if [[ ! -d "$CHECKOUT" ]]; then
    git clone --no-checkout https://github.com/hazcashi/SemBench-Stretto_experiment.git "$CHECKOUT"
    git -C "$CHECKOUT" checkout --detach "$REV"
fi
[[ $(git -C "$CHECKOUT" rev-parse HEAD) == "$REV" ]] || { echo 'Checkout revision mismatch.' >&2; exit 2; }
git -C "$CHECKOUT" diff --quiet
git -C "$CHECKOUT" diff --cached --quiet
git -C "$CHECKOUT" submodule update --init --recursive
git -C "$CHECKOUT" submodule status --recursive | tee /results/host-info/submodules.txt

[[ -x /opt/venv/bin/python ]] || python3.12 -m venv /opt/venv
python -m pip install --upgrade pip==25.2
python -m pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cu128 -r "$CHECKOUT/docker/requirements.lock"
export MAX_JOBS=1 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9'
export FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE
export CFLAGS=-DFLASHATTENTION_DISABLE_BACKWARD NVCC_PREPEND_FLAGS=-DFLASHATTENTION_DISABLE_BACKWARD
if [[ ! -f "$ROOT/flash-build-complete" ]]; then
    python -m pip install --no-deps --no-build-isolation 'https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/flash_attn-2.8.3.post2.tar.gz#sha256=d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b'
    python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.3.post2"'
    printf '%s\n' fa2-2.8.3.post2-sm80-no-backward > "$ROOT/flash-build-complete"
fi
python -m pip install --no-deps "$CHECKOUT/stretto-main/stretto-main/kvpress"
for entry in source configs tests; do
    target="$CHECKOUT/$entry"
    [[ "$entry" != source ]] || target="$CHECKOUT/stretto-main/stretto-main"
    if [[ -e "$ROOT/$entry" || -L "$ROOT/$entry" ]]; then
        [[ $(readlink -f "$ROOT/$entry") == "$target" ]] || { echo "Unexpected $ROOT/$entry" >&2; exit 2; }
    else
        ln -s "$target" "$ROOT/$entry"
    fi
done
cd "$ROOT/source"
python "$CHECKOUT/scripts/fetch_rotowire.py" --provenance "$CHECKOUT/docs/source_provenance.json" --destination reasondb/evaluation/benchmarks/files
python -m pip install --no-deps -e .
if [[ -e logging || -L logging ]]; then
    [[ -L logging && $(readlink -f logging) == /results/logging ]] || { echo 'Unexpected source/logging; stop without overwriting it.' >&2; exit 2; }
else
    ln -s /results/logging logging
fi
python -m pip check
python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL vss;'); c.execute('LOAD vss;')"
python -c 'import torch,flash_attn; from kvpress import ExpectedAttentionPress,KeyRerotationPress,KVzipPress,FinchPress; assert torch.__version__ == "2.9.1+cu128"'
export HF_HOME=/data/hf REASONDB_CACHE_DIR=/data/cache STRETTO_DATA_ROOT=/data STRETTO_RESULTS_ROOT=/results
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false KV_CACHE_PIN_GB=0 OMP_NUM_THREADS=4
python -m reasondb.reproduction.runner inspect
# No model download/API calls. Includes the random tiny-Llama CUDA test when visible.
python -m pytest "$ROOT/tests" -q
python -m pip freeze > /results/host-info/pip-freeze.txt
nvcc --version > /results/host-info/nvcc.txt
python - "$QUOTA" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path('/opt/stretto')
record = {
    'deployment': 'vast_cuda_bootstrap',
    'source_revision': subprocess.check_output(['git', '-C', str(root/'checkout'), 'rev-parse', 'HEAD'], text=True).strip(),
    'declared_template_image': 'vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312-2026-10-21',
    'image_identity_status': 'template_declared_not_host_verified',
    'declared_vast_disk_gb': int(sys.argv[1]),
    'requirements_sha256': hashlib.sha256((root/'checkout/docker/requirements.lock').read_bytes()).hexdigest(),
    'flash_attention_build': (root/'flash-build-complete').read_text().strip(),
    'python': sys.version, 'status': 'installation_checks_passed',
    'benchmark_validation': 'pending',
}
text = json.dumps(record, indent=2) + '\n'
(root/'installation.json').write_text(text)
pathlib.Path('/results/host-info/installation.json').write_text(text)
PY
echo 'Installation checks passed. Next: launcher doctor, then run. Benchmark validation is still pending.'
