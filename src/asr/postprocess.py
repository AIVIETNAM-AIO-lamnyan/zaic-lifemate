"""Hậu xử lý text ASR trước khi đưa cho LLM.

Hiện chỉ có một luật: **khôi phục chữ "mươi" bị đọc tắt** trong số có hai
chữ số. Người nói hay đọc 95 là "chín lăm", 45 là "bốn lăm"; Gipformer ghi
đúng như được nói, nhưng dạng tắt dễ bị hiểu nhầm (agent có thể đọc "chín
lăm" thành 9 và 5) và lệch với cách viết chuẩn "chín mươi lăm".

Luật: ``<hai..chín> lăm`` -> ``<hai..chín> mươi lăm``. "lăm" chỉ xuất hiện
trong số (mười lăm, hai mươi lăm...), nên luật gần như không có ca sai. Đã
thử thêm "mốt" và "tư" (hai mốt, hai tư): "mốt" không đổi gì trên dev, "tư"
làm kém đi ("hôm hai tư" là ngày 24, nhưng còn "Ba Tư", "thứ hai, tư"...)
nên không dùng.

Đo trên ft06 (greedy, không hotword; gipformer_finetune/DECISIONS.md mục
22): WER dev 7,78 % -> 7,64 %, validation 7,03 % -> 6,94 %; 17 câu đổi trên
mỗi tập, không câu nào bị sai thêm do luật.
"""

from __future__ import annotations

import re

#: "<hai..chín> lăm" (không phân biệt hoa thường), chưa có "mươi" ở giữa.
_SHORT_TENS = re.compile(
    r"\b(hai|ba|bốn|năm|sáu|bảy|tám|chín) (lăm)\b", re.IGNORECASE)


def restore_tens(text: str) -> str:
    """``"xăng chín lăm"`` -> ``"xăng chín mươi lăm"`` (giữ kiểu chữ)."""
    def insert(match: re.Match) -> str:
        tens = "MƯƠI" if match.group(2).isupper() else "mươi"
        return f"{match.group(1)} {tens} {match.group(2)}"

    return _SHORT_TENS.sub(insert, text)


def postprocess(text: str) -> str:
    """Mọi luật hậu xử lý, theo thứ tự."""
    return restore_tens(text)
