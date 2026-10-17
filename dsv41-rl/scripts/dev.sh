#!/usr/bin/env bash
# Run a command in the training container: the sglang dsv41 image already carries the torch,
# tilelang and safetensors versions the reference implementation needs.
#   GPUS=4,5 dsv41-rl/scripts/dev.sh python -m pytest dsv41-rl/tests -q
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
exec docker run --rm --init --ipc=host --ulimit memlock=-1 \
  --gpus "\"device=${GPUS:-4}\"" \
  -v /models:/models:ro \
  -v "$repo":/workspace/feather-train \
  -v "${DSRL_CACHE:-/data3/dsrl-cache}":/root/.cache \
  -w /workspace/feather-train \
  -e PYTHONPATH=/workspace/feather-train/dsv41-rl/src \
  -e DSRL_CKPT -e DSRL_REF_DIR \
  --entrypoint "" \
  "${DSRL_IMAGE:-lmsysorg/sglang:dev-dsv41}" "$@"
