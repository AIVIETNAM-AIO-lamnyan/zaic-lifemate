"""Chạy thử module ASR với model thật trên một vài mẫu.

In text ASR cạnh transcript gốc (``audio_text``, chỉ có ở tập train) và thời
gian xử lý từng mẫu. Thời gian nạp model được in riêng ở dòng đầu.

Chạy từ thư mục gốc của repo::

    python -m src.asr.demo                          # 5 mẫu train đầu tiên
    python -m src.asr.demo --split public_test -n 3
    python -m src.asr.demo --ids public_0006 public_0015
    python -m src.asr.demo --model qwen3_asr_1_7b   # đổi model
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from .api import (get_sample, get_user_request_from_audio, infer_split,
                  list_sample_ids, load_asr_model)
from .config import DEFAULT_MODEL, MODELS


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Chạy thử ASR trên một vài mẫu của cuộc thi.")
    parser.add_argument("-n", type=int, default=5,
                        help="số mẫu đầu tiên của split (bỏ qua nếu có --ids)")
    parser.add_argument("--split", default="train",
                        help="split lấy mẫu: train | public_test | ...")
    parser.add_argument("--ids", nargs="+",
                        help="danh sách id cụ thể, ví dụ public_0006")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=sorted(MODELS), help="model ASR")
    parser.add_argument("--device", default="auto",
                        help="auto | cuda:0 | cpu")
    return parser.parse_args()


def format_sample(sample_id: str, sample: dict[str, Any], current: str,
                  history: list[str], seconds: float) -> str:
    """Định dạng kết quả của một mẫu: mỗi lượt gồm dòng ASR và dòng REF.

    Dòng REF (transcript gốc) chỉ hiện khi tập dữ liệu có ``audio_text``.
    """
    refs = [turn.get("audio_text", "") for turn in sample["history"]
            if turn.get("role") == "user"]
    turns = [*zip([f"u{k}" for k in range(1, len(history) + 1)],
                  history, refs),
             ("cur", current, sample.get("audio_text", ""))]
    lines = [f"=== {sample_id}  ({seconds:.2f} s, "
             f"{len(history)} lượt history)"]
    lines += [f"  {name:<4}ASR: {hyp}" + (f"\n      REF: {ref}" if ref else "")
              for name, hyp, ref in turns]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Nạp model, nhận dạng từng mẫu và in kết quả."""
    args = parse_args()
    ids = args.ids or list_sample_ids(args.split)[:args.n]

    started = time.perf_counter()
    load_asr_model(args.model, args.device)
    print(f"Đã nạp {args.model} trong "
          f"{time.perf_counter() - started:.1f} s\n")

    for sample_id in ids:
        # Với --ids, split suy ra từ tiền tố id; ngược lại dùng --split.
        split = None if args.ids else args.split
        started = time.perf_counter()
        current, history = get_user_request_from_audio(
            sample_id, split=split, model=args.model, device=args.device)
        seconds = time.perf_counter() - started
        sample = get_sample(sample_id, split or infer_split(sample_id))
        print(format_sample(sample_id, sample, current, history, seconds))


if __name__ == "__main__":
    main()
