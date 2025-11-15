#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${1:-t-tech/T-lite-it-1.0}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DEFAULT_TARGET_DIR="${SCRIPT_DIR}/../models/T-lite-it-1.0"
TARGET_DIR=${2:-${DEFAULT_TARGET_DIR}}

mkdir -p "${TARGET_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install it via 'pip install huggingface_hub'." >&2
  exit 1
fi

HF_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}
TOKEN_FLAG=()
if [[ -n "${HF_TOKEN}" ]]; then
  TOKEN_FLAG=("--token" "${HF_TOKEN}")
fi

if [[ ${#TOKEN_FLAG[@]} -gt 0 ]]; then
  huggingface-cli download "${MODEL_ID}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False \
    --include "*" \
    ${TOKEN_FLAG[@]}
else
  huggingface-cli download "${MODEL_ID}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False \
    --include "*"
fi

cat <<'MSG'
Model weights downloaded successfully.
Update your .env file to point MODELS_DIR to the folder containing the weights if needed.
MSG
