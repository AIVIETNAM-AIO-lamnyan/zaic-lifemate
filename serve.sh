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

PRIMARY_MODEL="${VLLM_MODEL_PATH:-Qwen/Qwen3.5-4B}"
FALLBACK_MODEL="${VLLM_FALLBACK_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$CONFIG_MODEL_ID}"
PORT="${VLLM_PORT:-$CONFIG_PORT}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"

if (( MAX_MODEL_LEN > CONFIG_MAX_CONTEXT_LENGTH )); then
  echo "VLLM_MAX_MODEL_LEN exceeds config MAX_CONTEXT_LENGTH; capping to $CONFIG_MAX_CONTEXT_LENGTH"
  MAX_MODEL_LEN="$CONFIG_MAX_CONTEXT_LENGTH"
fi

run_vllm() {
  local model_path="$1"
  local lower
  lower="$(printf '%s' "$model_path" | tr '[:upper:]' '[:lower:]')"

  # Qwen3.5 cần trust-remote-code + parser qwen3_coder; các model Qwen cũ dùng hermes.
  local parser_args=(--tool-call-parser hermes)
  if [[ "$lower" == *qwen3.5* || "$lower" == *qwen3-coder* ]]; then
    parser_args=(--trust-remote-code --tool-call-parser qwen3_coder)
  fi

  # Model 4B cần nhiều VRAM hơn model 0.5B.
  local gpu_util="${VLLM_GPU_MEMORY_UTILIZATION:-}"
  if [[ -z "$gpu_util" ]]; then
    if [[ "$lower" == *0.5b* || "$lower" == *0_5b* ]]; then
      gpu_util="0.50"
    else
      gpu_util="0.90"
    fi
  fi

  echo "Model source: $model_path"
  echo "Served name:  $SERVED_MODEL_NAME"
  echo "Endpoint:     http://localhost:$PORT/v1/chat/completions"
  echo "Context:      $MAX_MODEL_LEN (config limit: $CONFIG_MAX_CONTEXT_LENGTH)"
  echo "Parser:       ${parser_args[*]} | gpu-memory-utilization: $gpu_util"

  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" vllm serve "$model_path" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "${VLLM_HOST:-0.0.0.0}" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$gpu_util" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-auto-tool-choice \
    "${parser_args[@]}" \
    --enforce-eager
}

# VLLM_NO_FALLBACK=1 để tắt fallback (ví dụ khi muốn thấy lỗi thật của model chính).
FALLBACK_ENABLED=1
if [[ "${VLLM_NO_FALLBACK:-0}" == "1" || "$PRIMARY_MODEL" == "$FALLBACK_MODEL" ]]; then
  FALLBACK_ENABLED=0
fi

set +e
run_vllm "$PRIMARY_MODEL"
STATUS=$?
set -e

if (( STATUS == 0 || STATUS == 130 || STATUS == 143 )); then
  exit "$STATUS"
fi

if (( FALLBACK_ENABLED == 0 )); then
  echo "Model '$PRIMARY_MODEL' thoát với mã $STATUS; fallback đang tắt." >&2
  exit "$STATUS"
fi

echo "" >&2
echo "Model '$PRIMARY_MODEL' không chạy được (exit $STATUS)." >&2
echo "Chuyển sang model dự phòng '$FALLBACK_MODEL'..." >&2
echo "" >&2

run_vllm "$FALLBACK_MODEL"
