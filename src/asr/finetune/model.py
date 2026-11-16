"""Nạp PhoWhisper, gắn LoRA và nhận dạng một tập lượt nói.

**Vì sao LoRA thay vì fine-tune toàn bộ?** Fine-tune toàn bộ 1,54B tham số
với AdamW cần khoảng 20-25 GB VRAM (trọng số + gradient + 2 trạng thái của
optimizer). LoRA giữ nguyên trọng số gốc và chỉ học thêm hai ma trận nhỏ
``A`` (d x r) và ``B`` (r x d) cho mỗi lớp chiếu attention:

    W' = W + (alpha / r) * B @ A          (r = 32 << d = 1280)

Số tham số cần học khoảng 31,5 triệu (384 lớp chiếu x 32 x (1280 + 1280)),
tức khoảng 2 % model. Phần lớn bộ nhớ khi đó là trọng số
gốc (khoảng 3 GB ở bfloat16), vừa với GPU 8 GB.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import torch

from ..audio_io import load_audio
from ..config import SAMPLE_RATE
from .data import Utterance

#: Ngôn ngữ và tác vụ, dùng cho cả nhãn khi học lẫn prompt khi sinh text.
LANGUAGE, TASK = "vi", "transcribe"

#: Lớp chiếu trong attention của Whisper được gắn LoRA.
_ATTENTION = r"(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)"


def pick_device_dtype(device: str = "auto") -> tuple[str, torch.dtype]:
    """``cuda:0`` + bfloat16 nếu có GPU hỗ trợ, ngược lại ``cpu`` + float32."""
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not device.startswith("cuda"):
        return device, torch.float32
    bf16 = torch.cuda.is_bf16_supported()
    return device, torch.bfloat16 if bf16 else torch.float16


def load_whisper(model_id: str | Path, device: str,
                 dtype: torch.dtype) -> tuple[Any, Any]:
    """Nạp ``WhisperForConditionalGeneration`` và ``WhisperProcessor``.

    Tokenizer được đặt prefix ``<|vi|><|transcribe|>`` để mọi nhãn bắt đầu
    giống hệt chuỗi mà model sinh ra lúc nhận dạng tiếng Việt.
    """
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)

    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype).to(device)
    return model, processor


def add_lora(model: Any, rank: int = 32, alpha: int = 64,
             dropout: float = 0.05, decoder_only: bool = False) -> Any:
    """Gắn LoRA vào các lớp chiếu attention, đóng băng phần còn lại.

    Args:
        model: Model Whisper đã nạp.
        rank: Hạng ``r`` của LoRA (lớn hơn: học được nhiều hơn, tốn hơn).
        alpha: Hệ số khuếch đại; thường đặt ``alpha = 2 * rank``.
        dropout: Dropout trên nhánh LoRA, giảm overfit trên tập nhỏ.
        decoder_only: Chỉ gắn LoRA vào decoder. Nhanh và nhẹ hơn vì encoder
            không cần lan truyền ngược, nhưng chỉ học được "cách viết"
            (từ vựng, chính tả, tên riêng), không học thêm "cách nghe"
            (giọng vùng miền, nhiễu) vốn nằm ở encoder.
    """
    from peft import LoraConfig, get_peft_model

    part = "decoder" if decoder_only else "(encoder|decoder)"
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        # Biểu thức chính quy khớp toàn bộ tên module, ví dụ
        # model.decoder.layers.3.encoder_attn.q_proj
        target_modules=rf".*{part}\.layers\.\d+\.{_ATTENTION}")
    return get_peft_model(model, config)


def normalize(text: str) -> str:
    """Chuẩn hoá để tính WER: viết thường, bỏ dấu câu, gộp khoảng trắng.

    Chỉ dùng để chấm điểm; text đưa cho LLM vẫn là output thô.
    """
    text = unicodedata.normalize("NFC", text.lower())
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch
                   for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """WER (%) của toàn tập sau khi chuẩn hoá (bỏ các nhãn rỗng)."""
    import jiwer

    pairs = [(normalize(ref), normalize(hyp))
             for ref, hyp in zip(references, hypotheses) if normalize(ref)]
    refs, hyps = zip(*pairs)
    return 100 * jiwer.wer(list(refs), list(hyps))


@torch.inference_mode()
def transcribe(model: Any, processor: Any, utterances: list[Utterance],
               batch_size: int = 8) -> list[str]:
    """Nhận dạng các lượt nói bằng giải mã greedy (1 beam, tất định)."""
    model.eval()
    device, dtype = model.device, model.dtype

    def run(batch: list[Utterance]) -> list[str]:
        audio = [load_audio(utt.audio_path) for utt in batch]
        features = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt").input_features.to(device, dtype)
        tokens = model.generate(input_features=features, language=LANGUAGE,
                                task=TASK, num_beams=1, do_sample=False,
                                max_new_tokens=225)
        return [text.strip() for text in
                processor.batch_decode(tokens, skip_special_tokens=True)]

    return [text
            for start in range(0, len(utterances), batch_size)
            for text in run(utterances[start:start + batch_size])]
