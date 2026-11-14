"""Thang chấm WER "team" của nhóm, dùng cho ``evaluate.py``.

Chép nguyên từ ``gipformer_finetune/src/gft/scoring.py`` (vốn trùng khớp
10.072/10.072 câu với ``normalize_team`` của bộ benchmark lam_lai_cuoc_doi
và tái hiện hàm ``render`` của repo Datyth), để con số đội tự chạy được so
trực tiếp với con số trong báo cáo (6,86 % trên 200 hội thoại validation).
Không sửa riêng file này: mọi thay đổi phải làm ở cả hai nơi.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_LANGUAGE_TAG = re.compile(r"<\|?/?[a-z]{2,3}\|?>", re.IGNORECASE)
_TEAM_UNITS = ("", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
               "chín")
_TONE_IDS = {"̀": "1", "́": "2", "̃": "3", "̉": "4",
             "̣": "5"}
_TEAM_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")
_TEAM_TIME = re.compile(r"\b(\d{1,2})[:h](\d{2})\b")
_TEAM_DIGIT_LETTER = re.compile(r"(\d)([^\W\d_])")
_TEAM_LETTER_DIGIT = re.compile(r"([^\W\d_])(\d)")
_TEAM_NUMBER = re.compile(r"\b\d{1,9}\b")
_TEAM_NON_WORD = re.compile(r"[^\w]+")


@lru_cache(maxsize=4096)
def spell_team(number: int) -> str:
    """Đọc ``0 <= number < 10**9`` đúng như hàm ``_spell`` của Datyth."""
    if number == 0:
        return "không"
    out: list[str] = []
    for scale, word in ((10 ** 6, "triệu"), (1000, "nghìn"), (1, "")):
        group, number = divmod(number, scale)
        if not group:
            continue
        hundreds, rest = divmod(group, 100)
        tens, unit = divmod(rest, 10)
        part = []
        if hundreds or out:
            part += [_TEAM_UNITS[hundreds] if hundreds else "không", "trăm"]
        if tens == 0 and unit and (hundreds or out):
            part.append("linh")
        if tens == 1:
            part.append("mười")
        elif tens:
            part += [_TEAM_UNITS[tens], "mươi"]
        if unit:
            part.append("mốt" if unit == 1 and tens > 1
                        else "lăm" if unit == 5 and tens
                        else "tư" if unit == 4 and tens > 1
                        else _TEAM_UNITS[unit])
        out += part + ([word] if word else [])
    return " ".join(out)


@lru_cache(maxsize=65536)
def tone_canon(word: str) -> str:
    """Chuyển dấu thanh của từ thành một chữ số ở cuối ("hòa" -> "hoa2")."""
    decomposed = unicodedata.normalize("NFD", word)
    tone = next((_TONE_IDS[c] for c in decomposed if c in _TONE_IDS), "")
    base = unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if c not in _TONE_IDS))
    return base + tone


def normalize_team(text: str) -> str:
    """Chuẩn hoá theo thang "team" (giống hệt bản trong lam_lai_cuoc_doi)."""
    text = unicodedata.normalize("NFC", _LANGUAGE_TAG.sub(" ", text)).lower()
    text = _TEAM_THOUSANDS.sub("", _TEAM_THOUSANDS.sub("", text))
    text = _TEAM_TIME.sub(
        lambda m: f"{m.group(1)} giờ" + (
            "" if m.group(2) == "00" else f" {int(m.group(2))}"), text)
    text = text.replace("%", " phần trăm ")
    text = _TEAM_DIGIT_LETTER.sub(r"\1 \2", text)
    text = _TEAM_LETTER_DIGIT.sub(r"\1 \2", text)
    text = _TEAM_NUMBER.sub(lambda m: spell_team(int(m.group())), text)
    return " ".join(map(tone_canon, _TEAM_NON_WORD.sub(" ", text).split()))


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Khoảng cách Levenshtein theo từ (quy hoạch động trên một hàng)."""
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """WER gộp cả tập (tổng lỗi / tổng số từ tham chiếu), thang "team"."""
    refs = [normalize_team(text).split() for text in references]
    hyps = [normalize_team(text).split() for text in hypotheses]
    errors = sum(map(edit_distance, refs, hyps))
    return errors / max(1, sum(map(len, refs)))
