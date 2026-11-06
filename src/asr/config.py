"""Cấu hình của module ASR: danh sách model và đường dẫn dữ liệu.

Model mặc định là **PhoWhisper-large**, được chọn qua benchmark trên toàn bộ
5.036 lượt nói có nhãn của tập train (WER 24,69 %, thấp nhất trong 7 model
mã nguồn mở đã thử). Các model còn lại được giữ trong ``MODELS`` để có thể
đổi model chỉ bằng một tham số, ví dụ ``model="qwen3_asr_1_7b"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Âm thanh của cuộc thi: WAV 16 kHz, mono, PCM 16-bit.
SAMPLE_RATE = 16000

# Thư mục gốc của repo: src/asr/config.py -> lùi 3 cấp.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thư mục chứa các split (train/, public_test/, ...). Mặc định trùng với quy
# ước của src/llm_phase; có thể ghi đè bằng biến môi trường ZAC_DATA_ROOT
# (ví dụ khi đóng gói Docker, dữ liệu được mount ở chỗ khác).
DATA_ROOT = Path(os.environ.get(
    "ZAC_DATA_ROOT",
    PROJECT_ROOT / "train+public_test" / "train+public_test",
))


@dataclass(frozen=True)
class ModelSpec:
    """Mô tả một model ASR.

    Attributes:
        key: Tên ngắn gọn dùng để chọn model (``phowhisper_large``).
        model_id: Id trên Hugging Face Hub, hoặc đường dẫn thư mục chứa
            trọng số khi chạy offline.
        backend: Họ model, quyết định wrapper trong ``backends.BACKENDS``.
        language: Gợi ý ngôn ngữ theo quy ước của từng backend
            (``vietnamese`` cho Whisper, ``Vietnamese`` cho Qwen3-ASR,
            ``vi-VN`` cho Nemotron). ``None``: để model tự nhận diện.
        batch_size: Số đoạn âm thanh tối đa được giải mã cùng lúc.
        options: Tham số riêng của từng backend.
    """

    key: str
    model_id: str
    backend: str
    language: str | None = None
    batch_size: int = 8
    options: dict[str, Any] = field(default_factory=dict)


# Danh sách model đã benchmark, xếp theo WER tăng dần trên tập train.
_SPECS = (
    # PhoWhisper chỉ được fine-tune trên tiếng Việt: không cần gợi ý ngôn ngữ.
    ModelSpec("phowhisper_large", "vinai/PhoWhisper-large", "hf_whisper"),
    ModelSpec("qwen3_asr_1_7b", "Qwen/Qwen3-ASR-1.7B", "qwen_asr",
              language="Vietnamese", options={"max_new_tokens": 256}),
    ModelSpec("whisper_large_v3", "openai/whisper-large-v3", "hf_whisper",
              language="vietnamese"),
    ModelSpec("phowhisper_medium", "vinai/PhoWhisper-medium", "hf_whisper"),
    ModelSpec("whisper_large_v3_turbo", "openai/whisper-large-v3-turbo",
              "hf_whisper", language="vietnamese"),
    ModelSpec("qwen3_asr_0_6b", "Qwen/Qwen3-ASR-0.6B", "qwen_asr",
              language="Vietnamese", options={"max_new_tokens": 256}),
    # Nemotron (RNN-T) chỉ hỗ trợ batch 1 trong transformers.
    ModelSpec("nemotron_asr_0_6b", "nvidia/nemotron-3.5-asr-streaming-0.6b",
              "hf_rnnt", language="vi-VN", batch_size=1),
)

#: Tra cứu model theo ``key``.
MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}

#: Model được dùng khi người gọi không chỉ định.
DEFAULT_MODEL = "phowhisper_large"
