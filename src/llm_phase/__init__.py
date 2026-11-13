"""Package llm_phase: Pipeline suy luận LLM cho LifeMate (ZAC 2026).
Tối ưu hóa chạy qua vLLM endpoint, không phụ thuộc Ollama.
"""
from __future__ import annotations

from .config import (
    DATA_ROOT,
    DEFAULT_API_KEY,
    DEFAULT_SUBMISSION_PATH,
    DEFAULT_VLLM_URL,
    MAX_CONTEXT_LENGTH,
    MAX_TOKENS,
    MAX_TURNS,
    MODEL_ID,
    PROJECT_ROOT,
    REQUEST_TIMEOUT,
    RESULT_DIR,
    TEMPERATURE,
)
from .model import ModelConfig, get_default_model
from .prompt import (
    SYSTEM_PROMPT,
    create_user_message,
    build_messages_for_llm,
    format_user_context,
)
from .get_tool import get_available_tool, get_catalog
from .llm import (
    build_request_payload,
    send_llm_request,
    parse_llm_response,
    call_llm,
)
from .submission import (
    format_submission_entry,
    save_submission_sample,
    SubmissionWriter,
    load_submission,
    evaluate_submission,
)
from .agent import agent_loop, run_sample_pipeline

__all__ = [
    # Cấu hình rule BTC & nhóm
    "MAX_CONTEXT_LENGTH",
    "TEMPERATURE",
    "MAX_TURNS",
    "MAX_TOKENS",
    "MODEL_ID",
    "DEFAULT_VLLM_URL",
    "DEFAULT_API_KEY",
    "REQUEST_TIMEOUT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RESULT_DIR",
    "DEFAULT_SUBMISSION_PATH",
    # Model & Prompt & Tools
    "ModelConfig",
    "get_default_model",
    "SYSTEM_PROMPT",
    "create_user_message",
    "build_messages_for_llm",
    "format_user_context",
    "get_available_tool",
    "get_catalog",
    # LLM request & parse
    "build_request_payload",
    "send_llm_request",
    "parse_llm_response",
    "call_llm",
    # Submission & Evaluation
    "format_submission_entry",
    "save_submission_sample",
    "SubmissionWriter",
    "load_submission",
    "evaluate_submission",
    # Agent loop & Pipeline
    "agent_loop",
    "run_sample_pipeline",
]
