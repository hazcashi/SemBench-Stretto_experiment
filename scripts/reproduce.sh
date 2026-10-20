#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
ACTION=${1:-doctor}
case "$ACTION" in doctor|run|profile|profile-nsys|inspect|access) ;; *) echo 'Usage: bash scripts/reproduce.sh {doctor|run|profile|profile-nsys|inspect|access}' >&2; exit 2 ;; esac
command -v docker >/dev/null || { echo 'Install Docker and NVIDIA Container Toolkit on the host first.' >&2; exit 1; }
docker info >/dev/null
env_value() { sed -n "s/^$1=//p" .env | tail -n 1 | tr -d '\r'; }
if [[ ! -f .env ]]; then
  echo 'Copy .env.example to .env and fill HF_TOKEN and OPENAI_API_KEY.' >&2
  exit 1
fi
IMAGE=${STRETTO_IMAGE:-$(env_value STRETTO_IMAGE)}
if [[ "$ACTION" == profile-nsys ]]; then
  IMAGE=${STRETTO_NSYS_IMAGE:-$(env_value STRETTO_NSYS_IMAGE)}
  if [[ -z "$IMAGE" ]]; then
    echo 'Set STRETTO_NSYS_IMAGE to the separately built profiling image; see docs/profiling.md.' >&2
    exit 1
  fi
fi
if [[ -z "$IMAGE" && -f docker/image.txt ]]; then IMAGE=$(tr -d '\r\n' < docker/image.txt); fi
if [[ -z "$IMAGE" ]]; then
  echo 'No published image is pinned yet. Set STRETTO_IMAGE to a built tag or GHCR digest; see docs/deployment.md.' >&2
  exit 1
fi
DATA=${STRETTO_DATA_DIR:-$(env_value STRETTO_DATA_DIR)}
RESULTS=${STRETTO_RESULTS_DIR:-$(env_value STRETTO_RESULTS_DIR)}
DATA=$(realpath -m "${DATA:-./data}")
RESULTS=$(realpath -m "${RESULTS:-./results}")
mkdir -p "$DATA/hf" "$DATA/cache" "$RESULTS/logging"
# Pull immutable published references; explicitly local images need no registry.
if [[ "$IMAGE" == *@sha256:* || "$IMAGE" == ghcr.io/* ]]; then docker pull "$IMAGE"; fi
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
SOURCE_REVISION=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")
EXTRA=()
if [[ "$ACTION" != inspect && "$ACTION" != access ]]; then EXTRA+=(--gpus 'device=0'); fi
docker run --rm --init "${EXTRA[@]}" --shm-size 2g --env-file .env \
  -e STRETTO_IMAGE_REF="$IMAGE" -e STRETTO_IMAGE_ID="$IMAGE_ID" -e STRETTO_SOURCE_REVISION="$SOURCE_REVISION" \
  --mount "type=bind,source=$DATA,target=/data" \
  --mount "type=bind,source=$RESULTS,target=/results" \
  "$IMAGE" "$ACTION"
