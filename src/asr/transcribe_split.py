"""Nhận dạng mọi mẫu của một split, ghi ra JSON cho đồng đội dùng ngay.

Mỗi mẫu được xử lý đúng như trong pipeline thật (``get_user_request_from_
audio``: cùng model, cùng hotword theo mẫu). Kết quả::

    {"public_0002": {"current_req": "...", "history_req": ["...", ...]},
     ...}

Chạy từ thư mục gốc repo::

    python -m src.asr.transcribe_split --split public_test
    python -m src.asr.transcribe_split --split train \
        --out results/asr/train.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .api import get_user_request_from_audio, list_sample_ids, load_asr_model
from .config import DEFAULT_MODEL, PROJECT_ROOT


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="public_test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="mặc định results/asr/<split>_<model>.json")
    args = parser.parse_args()
    out = args.out or (PROJECT_ROOT / "results" / "asr"
                       / f"{args.split}_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    load_asr_model(args.model)
    ids = list_sample_ids(args.split, args.data_root)
    results, started = {}, time.monotonic()
    for index, sample_id in enumerate(ids, start=1):
        current, history = get_user_request_from_audio(
            sample_id, split=args.split, data_root=args.data_root,
            model=args.model)
        results[sample_id] = {"current_req": current, "history_req": history}
        if index % 50 == 0 or index == len(ids):
            print(f"{index}/{len(ids)} mẫu | "
                  f"{time.monotonic() - started:.0f} giây", flush=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
