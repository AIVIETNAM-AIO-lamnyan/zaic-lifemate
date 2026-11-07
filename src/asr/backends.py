"""Wrapper thống nhất cho các họ model ASR.

Mỗi họ model có API riêng:

==============  ==========================  ===========================
backend         model                       thư viện
==============  ==========================  ===========================
``hf_whisper``  Whisper, PhoWhisper         ``transformers`` pipeline
``hf_rnnt``     NVIDIA Nemotron ASR         ``transformers`` (RNN-T)
``qwen_asr``    Qwen3-ASR                   ``qwen-asr``
==============  ==========================  ===========================

Tất cả được giấu sau lớp :class:`ASRBackend` với hợp đồng đơn giản:
gọi ``load()`` một lần, sau đó ``transcribe_batch(list[array]) -> list[str]``.
Thêm một họ model mới chỉ cần viết một lớp con và đăng ký vào ``BACKENDS``.

Các thư viện nặng (torch, transformers, qwen_asr) chỉ được import bên trong
hàm, nên môi trường Python chỉ cần cài thư viện của backend thực sự dùng.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import SAMPLE_RATE, ModelSpec


def resolve_device(device: str) -> str:
    """Đổi ``auto`` thành ``cuda:0`` nếu có GPU, ngược lại là ``cpu``."""
    if device != "auto":
        return device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: str, device: str) -> Any:
    """Chọn kiểu số thực cho trọng số model.

    ``auto``: bfloat16 trên GPU hỗ trợ (A30, RTX 30xx trở lên), float16 trên
    GPU cũ hơn, float32 trên CPU (half precision trên CPU chậm hoặc không
    được hỗ trợ).
    """
    import torch
    if dtype != "auto":
        return getattr(torch, dtype)
    if not device.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class ASRBackend(ABC):
    """Giao diện chung của mọi model ASR.

    Args:
        spec: Mô tả model (id, ngôn ngữ, batch size...).
        device: ``auto``, ``cuda:0`` hoặc ``cpu``.
        dtype: ``auto``, ``float16``, ``bfloat16`` hoặc ``float32``.
        sample_rate: Tần số lấy mẫu của âm thanh đầu vào.
    """

    def __init__(self, spec: ModelSpec, device: str = "auto",
                 dtype: str = "auto", sample_rate: int = SAMPLE_RATE) -> None:
        self.spec = spec
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.sample_rate = sample_rate
        self.model: Any = None   # được gán trong load()

    @abstractmethod
    def load(self) -> None:
        """Nạp trọng số vào bộ nhớ (gán ``self.model``)."""

    @abstractmethod
    def transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        """Nhận dạng một batch âm thanh 16 kHz mono, trả về một text/đoạn."""

    def unload(self) -> None:
        """Giải phóng RAM và bộ nhớ GPU (ví dụ để nhường GPU cho LLM)."""
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFWhisperBackend(ASRBackend):
    """Whisper và các bản fine-tune (PhoWhisper) qua ``transformers``."""

    def load(self) -> None:
        """Tạo pipeline ``automatic-speech-recognition``."""
        from transformers import pipeline
        self.model = pipeline("automatic-speech-recognition",
                              model=self.spec.model_id, dtype=self.dtype,
                              device=self.device)

    def transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        """Giải mã greedy để kết quả tất định (chạy lại cho cùng output)."""
        generate_kwargs = {"num_beams": 1, "do_sample": False,
                           **self.spec.options.get("generate_kwargs", {})}
        if self.spec.language:
            generate_kwargs.update(language=self.spec.language,
                                   task="transcribe")
        # Pipeline nhận âm thanh trong bộ nhớ dưới dạng dict raw + tần số.
        inputs = [{"raw": clip, "sampling_rate": self.sample_rate}
                  for clip in batch]
        outputs = self.model(inputs, batch_size=self.spec.batch_size,
                             generate_kwargs=generate_kwargs)
        return [item["text"].strip() for item in outputs]


class HFRNNTBackend(ASRBackend):
    """NVIDIA Nemotron ASR (FastConformer RNN-T).

    Cần ``transformers >= 5.13`` và ``librosa`` (bộ trích đặc trưng).
    """

    def load(self) -> None:
        """Nạp processor (trích đặc trưng) và model RNN-T."""
        from transformers import AutoModelForRNNT, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self.model = AutoModelForRNNT.from_pretrained(
            self.spec.model_id, dtype=self.dtype, device_map=self.device)

    def transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        """Chia ``batch`` thành các nhóm ``spec.batch_size`` rồi giải mã.

        Khác với hai backend còn lại, processor của Nemotron không tự chia
        batch, nên việc chia nhóm được làm ở đây.
        """
        size = max(1, self.spec.batch_size)
        return [text
                for start in range(0, len(batch), size)
                for text in self._decode(batch[start:start + size])]

    def _decode(self, batch: list[np.ndarray]) -> list[str]:
        """Trích đặc trưng, sinh token rồi giải mã thành text."""
        inputs = self.processor(batch, sampling_rate=self.sample_rate,
                                language=self.spec.language or "auto",
                                return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, return_dict_in_generate=True)
        texts = self.processor.batch_decode(outputs.sequences,
                                            skip_special_tokens=True)
        return [text.strip() for text in texts]


class QwenASRBackend(ASRBackend):
    """Qwen3-ASR (0.6B / 1.7B) qua package ``qwen-asr``."""

    def load(self) -> None:
        """Nạp model; batch size tối đa được cấu hình ngay khi khởi tạo."""
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(
            self.spec.model_id, dtype=self.dtype, device_map=self.device,
            max_inference_batch_size=self.spec.batch_size,
            max_new_tokens=self.spec.options.get("max_new_tokens", 256))

    def transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        """``qwen-asr`` nhận âm thanh dạng tuple ``(array, sample_rate)``."""
        audio = [(clip, self.sample_rate) for clip in batch]
        results = self.model.transcribe(
            audio=audio, language=[self.spec.language] * len(batch))
        return [result.text.strip() for result in results]


#: Ánh xạ trường ``backend`` của ``ModelSpec`` sang lớp wrapper tương ứng.
BACKENDS: dict[str, type[ASRBackend]] = {
    "hf_whisper": HFWhisperBackend,
    "hf_rnnt": HFRNNTBackend,
    "qwen_asr": QwenASRBackend,
}


def build_backend(spec: ModelSpec, device: str = "auto",
                  dtype: str = "auto") -> ASRBackend:
    """Khởi tạo wrapper cho ``spec`` (chưa nạp trọng số).

    Raises:
        KeyError: Nếu ``spec.backend`` chưa được đăng ký trong ``BACKENDS``.
    """
    if spec.backend not in BACKENDS:
        raise KeyError(f"Backend '{spec.backend}' chưa được hỗ trợ. "
                       f"Các backend có sẵn: {sorted(BACKENDS)}")
    return BACKENDS[spec.backend](spec, device=device, dtype=dtype)
