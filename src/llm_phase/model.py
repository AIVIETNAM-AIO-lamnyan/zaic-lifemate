"""Định nghĩa cấu hình mô hình LLM (ModelConfig) cho vLLM / OpenAI-compatible endpoint.
Không sử dụng Ollama cho suy luận.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import (
    DEFAULT_API_KEY,
    DEFAULT_VLLM_URL,
    MAX_CONTEXT_LENGTH,
    MAX_TOKENS,
    MODEL_ID,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)


@dataclass
class ModelConfig:
    """Cấu hình cho Model LLM chạy qua vLLM endpoint.
    
    Attributes:
        name: Tên/ID model (mặc định lấy từ config.MODEL_ID = 'qwen3.5:4b').
        endpoint_url: URL API endpoint gửi request (mặc định vLLM Chat Completions).
        temperature: Nhiệt độ sinh (mặc định 0.0 theo chuẩn ZAC 2026).
        max_context_length: Context length tối đa (2^14 = 16384 tokens).
        max_tokens: Số token sinh tối đa cho mỗi lượt (mặc định 512).
        top_p: Tham số top_p (mặc định 1.0).
        timeout: Thời gian chờ tối đa (giây, mặc định 90).
        api_key: Khóa xác thực (mặc định 'EMPTY' cho vLLM).
        extra_options: Tùy chọn bổ sung cho payload nếu cần.
    """
    name: str = MODEL_ID
    endpoint_url: str = DEFAULT_VLLM_URL
    temperature: float = TEMPERATURE
    max_context_length: int = MAX_CONTEXT_LENGTH
    max_tokens: int = MAX_TOKENS
    top_p: float = 1.0
    timeout: int = REQUEST_TIMEOUT
    api_key: str = DEFAULT_API_KEY
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Tải cấu hình model từ các biến môi trường hoặc dùng giá trị mặc định từ config."""
        model_name = os.environ.get("MODEL_NAME", os.environ.get("MODEL_ID", MODEL_ID))
        endpoint = os.environ.get("VLLM_URL", os.environ.get("LLM_ENDPOINT", DEFAULT_VLLM_URL))
        api_key = os.environ.get("VLLM_API_KEY", DEFAULT_API_KEY)
        timeout = int(os.environ.get("LLM_TIMEOUT", str(REQUEST_TIMEOUT)))
        return cls(
            name=model_name,
            endpoint_url=endpoint,
            temperature=TEMPERATURE,
            max_context_length=MAX_CONTEXT_LENGTH,
            max_tokens=MAX_TOKENS,
            api_key=api_key,
            timeout=timeout,
        )


def get_default_model() -> ModelConfig:
    """Trả về ModelConfig mặc định dựa trên config và môi trường."""
    return ModelConfig.from_env()
