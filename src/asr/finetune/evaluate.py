"""So WER của các model ASR trên tập dev, qua đúng đường chạy của pipeline.

Khác với lúc train (gọi ``model.generate`` trên model đang có LoRA), script
này dùng ``src.asr.backends``, tức đúng code mà ``get_user_request_from_audio``
sử dụng. Vì vậy nó đo được cả ảnh hưởng của việc gộp LoRA và lưu model.

Tập dev được tạo lại với cùng ``seed`` và ``dev_ratio`` như lúc train, nên
không chứa lượt nói nào model đã học.

Chạy từ thư mục gốc repo::

    python -m src.asr.finetune.evaluate
    python -m src.asr.finetune.evaluate --models phowhisper_large \
        phowhisper_large_zac
"""

from __future__ import annotations

import argparse
import time

from ..api import load_asr_model, unload_asr_model
from ..audio_io import load_audio
from ..config import MODELS
from .data import load_utterances, split_by_sample
from .model import word_error_rate


def main() -> None:
    """Nhận dạng tập dev bằng từng model và in bảng WER."""
    parser = argparse.ArgumentParser(description="So WER trên tập dev")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS),
                        default=["phowhisper_large", "phowhisper_large_zac"])
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, dev = split_by_sample(load_utterances("train"), args.dev_ratio,
                             args.seed)
    clips = [load_audio(utt.audio_path) for utt in dev]
    references = [utt.text for utt in dev]
    print(f"Tập dev: {len(dev)} lượt nói\n")

    rows = []
    for key in args.models:
        backend = load_asr_model(key)
        started = time.perf_counter()
        hypotheses = backend.transcribe_batch(clips)
        seconds = time.perf_counter() - started
        rows.append(f"| {key} | {word_error_rate(references, hypotheses):.2f}"
                    f" | {seconds / len(clips) * 1000:.0f} |")
        unload_asr_model()                   # nhường GPU cho model tiếp theo

    print("| model | WER dev % | ms / lượt nói |\n|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
