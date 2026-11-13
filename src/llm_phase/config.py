"""Cấu hình cố định cho pipeline LLM theo quy định của Ban tổ chức ZAC 2026 và nhóm.
Định cấu hình context length, temperature, max_turns, model_id, vLLM endpoint và các đường dẫn.
"""
from __future__ import annotations

import os
from pathlib import Path

# Thư mục gốc của repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Thư mục dữ liệu: ưu tiên /data (khi BTC mount vào container), fallback thư mục local
if Path("/data").is_dir():
    DATA_ROOT = Path("/data")
elif (PROJECT_ROOT / "train+public_test" / "train+public_test").is_dir():
    DATA_ROOT = PROJECT_ROOT / "train+public_test" / "train+public_test"
elif (PROJECT_ROOT / "train+public_test" / "train").is_dir():
    DATA_ROOT = PROJECT_ROOT / "train+public_test"
else:
    DATA_ROOT = PROJECT_ROOT / "train+public_test"

# Cho phép ghi đè qua biến môi trường nếu cần
if os.environ.get("DATA_ROOT"):
    DATA_ROOT = Path(os.environ["DATA_ROOT"])
elif os.environ.get("ZAC_DATA_ROOT"):
    DATA_ROOT = Path(os.environ["ZAC_DATA_ROOT"])

# Thư mục lưu kết quả dự đoán (submission)
if Path("/result").is_dir():
    RESULT_DIR = Path("/result")
else:
    RESULT_DIR = PROJECT_ROOT / "result"

if os.environ.get("RESULT_DIR"):
    RESULT_DIR = Path(os.environ["RESULT_DIR"])

RESULT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SUBMISSION_PATH = RESULT_DIR / "submission.jsonl"

# Kích thước Context window tối đa theo quy định thi: 2^14 = 16384 tokens
MAX_CONTEXT_LENGTH: int = 16384 

# Nhiệt độ cố định 0.0 theo đúng quy định của ban tổ chức ZAC 2026
TEMPERATURE: float = 0.0

# Số lượt tương tác tối đa của agent loop
MAX_TURNS: int = 10

# Số token sinh tối đa cho mỗi lượt phản hồi của LLM
MAX_TOKENS: int = 512

# Tên mô hình LLM mục tiêu
MODEL_ID: str = os.environ.get("MODEL_NAME", os.environ.get("MODEL_ID", "qwen3.5:4b"))

# Endpoint URL của vLLM (OpenAI-compatible Chat Completions)
DEFAULT_VLLM_URL: str = os.environ.get(
    "VLLM_URL",
    os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
)

# API key xác thực với vLLM (mặc định 'EMPTY')
DEFAULT_API_KEY: str = os.environ.get("VLLM_API_KEY", "EMPTY")

# Thời gian chờ HTTP tối đa (giây)
REQUEST_TIMEOUT: int = int(os.environ.get("LLM_TIMEOUT", "90"))
