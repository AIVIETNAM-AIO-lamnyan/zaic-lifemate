#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-zaic-vllm}"

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "Conda not found at: $CONDA_EXE" >&2
  exit 1
fi

mapfile -t CONFIG_VALUES < <(
  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" python - "$ROOT_DIR" <<'PY'
import importlib.util
import sys
from pathlib import Path
from urllib.parse import urlparse

config_path = Path(sys.argv[1]) / "src" / "llm_phase" / "config.py"
spec = importlib.util.spec_from_file_location("lifemate_llm_config", config_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load config: {config_path}")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

print(config.MODEL_ID)
print(config.MAX_CONTEXT_LENGTH)
print(urlparse(config.DEFAULT_VLLM_URL).port or 8000)
PY
)

if [[ "${#CONFIG_VALUES[@]}" -ne 3 ]]; then
  echo "Failed to read MODEL_ID, MAX_CONTEXT_LENGTH, and endpoint port from src/llm_phase/config.py" >&2
  exit 1
fi

CONFIG_MODEL_ID="${CONFIG_VALUES[0]}"
CONFIG_MAX_CONTEXT_LENGTH="${CONFIG_VALUES[1]}"
CONFIG_PORT="${CONFIG_VALUES[2]}"

MODEL_PATH="${VLLM_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$CONFIG_MODEL_ID}"
PORT="${VLLM_PORT:-$CONFIG_PORT}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"

if (( MAX_MODEL_LEN > CONFIG_MAX_CONTEXT_LENGTH )); then
  echo "VLLM_MAX_MODEL_LEN exceeds config MAX_CONTEXT_LENGTH; capping to $CONFIG_MAX_CONTEXT_LENGTH"
  MAX_MODEL_LEN="$CONFIG_MAX_CONTEXT_LENGTH"
fi

echo "Model source: $MODEL_PATH"
echo "Served name:  $SERVED_MODEL_NAME"
echo "Endpoint:     http://localhost:$PORT/v1/chat/completions"
echo "Context:      $MAX_MODEL_LEN (config limit: $CONFIG_MAX_CONTEXT_LENGTH)"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "${VLLM_HOST:-0.0.0.0}" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager
