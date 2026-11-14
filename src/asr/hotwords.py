"""Hotword theo từng mẫu: tên trong ``state.json`` -> dạng chữ của Gipformer.

Mỗi mẫu của cuộc thi có ``<split>/<split>_environment/<id>/state.json`` liệt
kê dữ liệu mà các tool sẽ trả về: danh bạ, bài hát, bến xe, mã cổ phiếu,
tiêu đề lịch... Đó chính là những tên người dùng có thể nhắc tới. Đưa danh
sách này cho bộ giải mã beam search của sherpa-onnx ("hotword") làm model ưu
tiên các cụm từ đó.

Kết quả đo trên 200 hội thoại validation (``gipformer_finetune/DECISIONS.md``
mục 12-14): tỉ lệ nghe đúng tên 85 % -> 91 %, WER 7,00 % -> 6,63 %, chỉ 2
tên bị "bịa" trên 665 câu.

Gipformer chỉ viết chữ HOA, không dấu câu, số bằng lời đọc, nên mỗi tên được
đổi sang đúng dạng đó (cùng luật với lúc chuẩn bị nhãn fine-tune).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .text_numbers import token_readings

#: Các trường của state.json được coi là "tên" (mọi loại, kể cả tiêu đề
#: lịch và nhãn báo thức).
ENTITY_KEYS = frozenset({
    "name", "title", "location", "address", "members", "artist", "channels",
    "seller", "operator", "bank", "stop", "stops", "area", "venue", "label",
    "airline", "provider", "from_name", "symbol"})
#: Chuỗi dài hơn mức này là nội dung (tin nhắn, mô tả), không phải tên.
MAX_ENTITY_CHARS = 60
#: Số cách đọc tối đa giữ cho mỗi tên có chữ số.
MAX_VARIANTS = 4

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


def build_hotwords(state: dict[str, Any]) -> str:
    """Chuỗi hotword cho sherpa-onnx: các cụm cách nhau bởi ``/``.

    Địa chỉ dài được tách theo dấu phẩy (hiếm khi đọc nguyên địa chỉ).
    """
    names = {part.strip()
             for key, value in _strings(state)
             if key in ENTITY_KEYS and len(value) <= MAX_ENTITY_CHARS
             for part in value.split(",") if part.strip()}
    phrases = {form for name in names for form in spoken_forms(name)}
    return "/".join(sorted(phrases))


def read_state(split_dir: Path, sample_id: str) -> dict[str, Any] | None:
    """``state.json`` của một mẫu, hoặc ``None`` nếu không có."""
    path = (split_dir / f"{split_dir.name}_environment" / sample_id
            / "state.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
