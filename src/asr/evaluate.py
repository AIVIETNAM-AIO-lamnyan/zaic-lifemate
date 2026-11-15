"""Kiểm tra module ASR của nhóm cho đúng kết quả đã báo cáo.

Chạy đúng hàm ``get_user_request_from_audio`` (cùng model, cùng hotword từ
lịch sử hội thoại) trên **200 hội thoại validation cố định** của tập train
(``validation_ids.txt``, 665 lượt nói), rồi tính WER theo thang "team"
(``scoring.py``) so với transcript gốc ``audio_text``.

Số mong đợi với model ft06 (``gipformer1.5-zac-ft06``) + hậu xử lý
``postprocess.py``: **WER 6,75 %** (6,86 % khi chưa có hậu xử lý). Lệch
quá 0,05 điểm nghĩa là có gì đó khác với lúc đo (model, phiên bản
sherpa-onnx, cách lấy hotword...) và cần kiểm tra trước khi dùng.

``--compare <csv>`` (tuỳ chọn): so từng câu với transcript của script đánh
giá gốc (``utt_id,hypothesis``, chữ HOA); báo tỉ lệ câu trùng khít, mong đợi
>= 99 %.

Lưu ý: model bản cuối học cả 200 hội thoại này nên WER sẽ thấp hơn nhiều;
khi đó chỉ dùng ``--compare`` để kiểm tra code đọc model đúng.

Chạy từ thư mục gốc repo (CPU, ~4-8 phút)::

    python -m src.asr.evaluate
    python -m src.asr.evaluate --compare beam4_history_hw1.0.csv

Kết quả ghi vào ``results/asr/`` (đã nằm trong .gitignore: chứa transcript
dữ liệu thi, KHÔNG commit).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import get_sample, get_user_request_from_audio, load_asr_model
from .config import DEFAULT_MODEL, PROJECT_ROOT
from .scoring import edit_distance, normalize_team

#: Danh sách 200 hội thoại validation (mã mẫu, không chứa dữ liệu thi).
IDS_FILE = Path(__file__).with_name("validation_ids.txt")
#: WER mong đợi (%) với model ft06 và sai lệch cho phép (điểm).
EXPECTED_WER, TOLERANCE = 6.75, 0.05


def read_ids(path: Path = IDS_FILE) -> list[str]:
    """Mã hội thoại, bỏ dòng trống và dòng chú thích ``#``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines
            if line.strip() and not line.startswith("#")]


def utterances(sample_id: str, data_root: str | None) -> pd.DataFrame:
    """Transcript gốc của mọi lượt user (history trước, lượt cuối sau).

    Mã lượt theo quy ước của bộ đánh giá: ``<id>_u<k>`` cho lượt thứ k trong
    history, ``<id>`` cho lượt cuối.
    """
    sample = get_sample(sample_id, data_root=data_root)
    history = [turn["audio_text"] for turn in sample.get("history", [])
               if turn.get("role") == "user"]
    ids = [f"{sample_id}_u{k}" for k in range(1, len(history) + 1)]
    return pd.DataFrame({"utt_id": [*ids, sample_id],
                         "reference": [*history, sample["audio_text"]]})


def main() -> None:
    """Giải mã 200 hội thoại, tính WER, so với số mong đợi."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--compare", type=Path, default=None,
                        help="CSV utt_id,hypothesis của script đánh giá gốc")
    args = parser.parse_args()
    load_asr_model(args.model)

    frames, started = [], time.monotonic()
    for k, sample_id in enumerate(read_ids(), start=1):
        current, history = get_user_request_from_audio(
            sample_id, data_root=args.data_root, model=args.model)
        table = utterances(sample_id, args.data_root)
        table["hypothesis"] = [*history, current]
        frames.append(table)
        if k % 50 == 0:
            print(f"{k}/200 hội thoại, "
                  f"{(time.monotonic() - started) / 60:.1f} phút", flush=True)
    table = pd.concat(frames, ignore_index=True)

    refs = table["reference"].map(lambda t: normalize_team(t).split())
    hyps = table["hypothesis"].map(lambda t: normalize_team(t).split())
    errors = np.array([edit_distance(r, h) for r, h in zip(refs, hyps)])
    wer = 100 * errors.sum() / refs.map(len).sum()
    report = {"model": args.model, "utterances": len(table),
              "wer": round(float(wer), 2)}
    if args.compare:
        theirs = (pd.read_csv(args.compare, encoding="utf-8-sig", dtype=str)
                  .set_index("utt_id")["hypothesis"].str.lower())
        same = table["hypothesis"].str.lower().to_numpy() == (
            theirs.reindex(table["utt_id"]).fillna("").to_numpy())
        report["identical_to_pipeline"] = round(float(same.mean()) * 100, 2)

    out_dir = PROJECT_ROOT / "results" / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"evaluate_validation_{args.model}.csv",
                 index=False, encoding="utf-8-sig")
    (out_dir / f"evaluate_validation_{args.model}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    gap = abs(report["wer"] - EXPECTED_WER)
    print(f"Mong đợi (model ft06): {EXPECTED_WER} % -> "
          f"{'KHỚP' if gap <= TOLERANCE else f'LỆCH {gap:.2f} điểm'}")


if __name__ == "__main__":
    main()
