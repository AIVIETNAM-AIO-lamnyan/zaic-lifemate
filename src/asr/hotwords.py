"""Hotword theo từng lượt nói, lấy từ lịch sử hội thoại (đúng luật BTC).

Luật cuộc thi cấm giải pháp đọc ``<split>_environment/`` (kể cả
``state.json``), nên hotword chỉ lấy từ input hợp lệ trong ``<split>.jsonl``:
**những gì trợ lý đã nói và đã tra cứu ở các lượt trước**. Người dùng hay
nhắc lại đúng những tên đó ("nhắn cho Kevin Tran", "đặt vé tàu SE7"):

* tên trong ``tool_results`` của các lượt assistant (các trường name, title,
  location, stops, symbol...);
* trong câu trả lời ``text`` của trợ lý: cụm từ viết hoa liền nhau không
  đứng đầu câu ("Kevin Tran") và mã vừa chữ vừa số ("SE7", "VTV1").

Lượt user thứ k chỉ dùng các lượt assistant **đứng trước** nó; lượt cuối dùng
toàn bộ history. Lượt đầu tiên của hội thoại không có hotword.

Kết quả (``gipformer_finetune/DECISIONS.md`` mục 15; điểm hotword chọn trên
dev, đo một lần trên 200 hội thoại validation): WER 7,00 % (greedy) ->
6,86 %, nghe đúng tên riêng 85,0 % -> 86,4 %, không tên nào bị "bịa". Thêm
tên trong ``user_profile`` và mã trong tools catalog không giúp thêm, nên
không dùng.

Gipformer chỉ viết chữ HOA, không dấu câu, số bằng lời đọc, nên mỗi tên được
đổi sang đúng dạng đó (cùng luật với lúc chuẩn bị nhãn fine-tune).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from typing import Any

from .text_numbers import token_readings

#: Các trường trong ``tool_results`` được coi là "tên" (mọi loại, kể cả
#: tiêu đề lịch và nhãn báo thức).
ENTITY_KEYS = frozenset({
    "name", "title", "location", "address", "members", "artist", "channels",
    "seller", "operator", "bank", "stop", "stops", "area", "venue", "label",
    "airline", "provider", "from_name", "symbol"})
#: Chuỗi dài hơn mức này là nội dung (tin nhắn, mô tả), không phải tên.
MAX_ENTITY_CHARS = 60
#: Số cách đọc tối đa giữ cho mỗi tên có chữ số.
MAX_VARIANTS = 4
#: Số từ tối đa của một cụm viết hoa lấy từ câu trả lời của trợ lý.
MAX_PHRASE_WORDS = 5
#: Một từ (chữ, số, dấu + và gạch nối bên trong, vd. "S+", "Follow-up").
_WORD = re.compile(r"[\w+]+(?:-\w+)*")
#: Ranh giới câu: từ viết hoa ngay sau các dấu này chỉ là đầu câu.
_SENTENCE_END = re.compile(r"[.!?:;\n]\s*$")

# Ghép lại các cấu trúc số bị dấu câu chia cắt, theo thứ tự (giống luật
# chuẩn bị nhãn khi fine-tune).
_NUMERIC_RULES = (
    (re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))"), ""),
    (re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))"), ""),
    (re.compile(r"\b(\d{1,2}):(\d{2})\b"), r"\1h\2"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), r"\1 tháng \2 năm \3"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})\b"), r"\1 tháng \2"),
    (re.compile(r"(?<=\d),(?=\d)"), " phẩy "),
    (re.compile(r"%"), " phần trăm "),
)
_NON_WORD = re.compile(r"[^\w]+|_")

# Dấu thanh kiểu cũ ("hoà") -> kiểu mới ("hòa"), kiểu Gipformer đã học.
_TONE_MARKS = ("̀", "́", "̉", "̃", "̣")
_OLD_TO_NEW = {
    unicodedata.normalize("NFC", first + second + mark):
    unicodedata.normalize("NFC", first + mark + second)
    for first, second in (("o", "a"), ("o", "e"), ("u", "y"))
    for mark in _TONE_MARKS
}
_OLD_STYLE = re.compile(
    r"(?<!q)(" + "|".join(map(re.escape, _OLD_TO_NEW)) + r")\b")


def to_tokens(text: str) -> list[str]:
    """Chuỗi gốc -> token chữ thường, không dấu câu, dấu thanh kiểu mới."""
    text = unicodedata.normalize("NFC", text).lower()
    for pattern, replacement in _NUMERIC_RULES:
        text = pattern.sub(replacement, text)
    text = _NON_WORD.sub(" ", text)
    return _OLD_STYLE.sub(lambda m: _OLD_TO_NEW[m.group(1)], text).split()


def spoken_forms(text: str) -> list[str]:
    """Các dạng chữ HOA mà Gipformer viết khi nghe ``text``.

    ``"Tàu SE7"`` -> ``["TÀU SE BẢY"]``; số có nhiều cách đọc thì giữ tối đa
    :data:`MAX_VARIANTS` cách. Tên có ký tự không đọc được -> ``[]``.
    """
    options: list[list[str]] = [[]]
    for token in to_tokens(text):
        readings = (token_readings(token)
                    if any(c.isdigit() for c in token) else [token])
        if not readings:
            return []
        options = [head + reading.split() for head in options
                   for reading in readings][:MAX_VARIANTS]
    return [" ".join(words).upper() for words in options if words]


def _strings(node: Any, key: str | None = None
             ) -> Iterator[tuple[str | None, str]]:
    """Mọi cặp (khoá, chuỗi) trong cây JSON, kể cả trong danh sách."""
    if isinstance(node, dict):
        for child_key, value in node.items():
            yield from _strings(value, child_key)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value, key)
    elif isinstance(node, str):
        yield key, node


def _is_code(word: str) -> bool:
    """Mã vừa chữ vừa số ("SE7", "VTV1"), không phải số / giờ thuần."""
    return (any(c.isdigit() for c in word)
            and any(c.isalpha() for c in word))


def reply_phrases(text: str) -> set[str]:
    """Cụm viết hoa (không đầu câu) và mã chữ + số trong câu trả lời.

    Một cụm là các từ viết hoa chữ cái đầu đứng liền nhau; từ đầu câu bị bỏ
    vì viết hoa theo ngữ pháp chứ không phải vì là tên.
    """
    phrases: set[str] = set()
    current: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group()
        sentence_start = (match.start() == 0 or bool(
            _SENTENCE_END.search(text[:match.start()])))
        if _is_code(word):
            phrases.add(word)
        if word[0].isupper() and not sentence_start:
            current.append(word)
            continue
        if current:
            phrases.add(" ".join(current[:MAX_PHRASE_WORDS]))
        current = []
    if current:
        phrases.add(" ".join(current[:MAX_PHRASE_WORDS]))
    return phrases


def assistant_names(turn: dict[str, Any]) -> set[str]:
    """Mọi tên một lượt assistant để lại: ``tool_results`` + câu trả lời.

    Địa chỉ dài được tách theo dấu phẩy (hiếm khi đọc nguyên địa chỉ).
    """
    names = {part.strip()
             for key, value in _strings(turn.get("tool_results", []))
             if key in ENTITY_KEYS and len(value) <= MAX_ENTITY_CHARS
             for part in value.split(",") if part.strip()}
    return names | reply_phrases(turn.get("text") or "")


def to_hotword_string(names: set[str]) -> str:
    """Tập tên -> chuỗi hotword cho sherpa-onnx (các cụm cách nhau ``/``)."""
    return "/".join(sorted({form for name in names
                            for form in spoken_forms(name)}))


def turn_hotwords(sample: dict[str, Any]) -> list[str | None]:
    """Hotword cho từng lượt user của một mẫu, theo thứ tự giải mã.

    Returns:
        Một phần tử cho mỗi lượt ``user`` trong history (cũ nhất trước) rồi
        một phần tử cho yêu cầu cuối; ``None`` khi chưa có tên nào.
    """
    seen: set[str] = set()
    out: list[str | None] = []
    for turn in sample.get("history", []):
        if turn.get("role") == "user":
            out.append(to_hotword_string(seen) or None)
        elif turn.get("role") == "assistant":
            seen |= assistant_names(turn)
    out.append(to_hotword_string(seen) or None)
    return out
