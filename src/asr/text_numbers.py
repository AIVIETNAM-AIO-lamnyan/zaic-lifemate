"""Các cách đọc tiếng Việt của số và của token có chữ số.

Chép từ ``gipformer_finetune/src/gft/numbers.py`` (đã kiểm thử ở đó). Ở module
ASR, hàm này dùng để đổi tên có chữ số trong ``state.json`` (``VTV3``,
``Tàu SE7``) sang dạng lời đọc mà Gipformer viết ra ("VTV BA", "TÀU SE
BẢY"), để làm hotword. Mọi chuỗi ở đây là chữ thường.
"""


from __future__ import annotations

import re
from functools import lru_cache

#: Tên chữ số 0-9.
UNITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
         "chín")

#: Số có từ chừng này chữ số trở lên luôn đọc từng chữ số (số điện thoại).
DIGIT_BY_DIGIT_MIN_LEN = 10
#: Số có từ chừng này chữ số trở lên thì *thêm* cách đọc từng chữ số (mã,
#: năm: "2026" -> "hai không hai sáu").
ALSO_DIGITS_MIN_LEN = 4

# Đơn vị viết liền sau số và lời đọc của chúng ("50k" -> "năm mươi nghìn").
SUFFIX_READINGS = {
    "k": ("nghìn", "ngàn", "ca"),
    "đ": ("đồng",), "vnđ": ("đồng",), "vnd": ("đồng",),
    "h": ("giờ",),
}

# Tách một token thành các đoạn liên tiếp toàn chữ số hoặc toàn chữ cái.
_RUNS = re.compile(r"\d+|[^\W\d_]+")


def _unit_after_tens(unit: int, tens: int, four: str) -> str:
    """Tên hàng đơn vị đứng sau hàng chục ("mốt", "lăm", "tư"/"bốn")."""
    if unit == 1 and tens > 1:
        return "mốt"
    if unit == 5 and tens > 0:
        return "lăm"
    if unit == 4 and tens > 1:
        return four
    return UNITS[unit]


@lru_cache(maxsize=8192)
def spell(number: int, four: str = "tư", linh: str = "linh",
          thousand: str = "nghìn") -> str:
    """Cách đọc đầy đủ của ``0 <= number < 10**9``.

    Mặc định giống hệt cách đọc của thang "team" trong ``lam_lai_cuoc_doi``
    ("hai mươi tư", "một nghìn không trăm năm mươi").

    Args:
        number: Số cần đọc.
        four: Đọc số 4 sau hàng chục là "tư" hay "bốn".
        linh: "linh" hay "lẻ" ("một trăm linh năm").
        thousand: "nghìn" hay "ngàn".

    Raises:
        ValueError: Nếu số âm hoặc từ 10**9 trở lên.
    """
    if not 0 <= number < 10 ** 9:
        raise ValueError(f"Chỉ hỗ trợ 0 <= number < 10**9, nhận {number}.")
    if number == 0:
        return UNITS[0]
    words: list[str] = []
    for scale, name in ((10 ** 6, "triệu"), (1000, thousand), (1, "")):
        group, number = divmod(number, scale)
        if not group:
            continue
        hundreds, rest = divmod(group, 100)
        tens, unit = divmod(rest, 10)
        # Nhóm bên trong (đã có nhóm trước) luôn đọc đủ hàng trăm.
        if hundreds or words:
            words += [UNITS[hundreds], "trăm"]
        if tens == 0 and unit and (hundreds or words):
            words.append(linh)
        if tens == 1:
            words.append("mười")
        elif tens:
            words += [UNITS[tens], "mươi"]
        if unit:
            words.append(_unit_after_tens(unit, tens, four))
        if name:
            words.append(name)
    return " ".join(words)


def digit_by_digit(digits: str) -> str:
    """Đọc từng chữ số: ``"0909"`` -> ``"không chín không chín"``."""
    return " ".join(UNITS[int(char)] for char in digits)


@lru_cache(maxsize=8192)
def integer_readings(number: int) -> frozenset[str]:
    """Mọi cách đọc thông dụng của một số nguyên ``0 <= number < 10**9``.

    Gồm:

    * cách đọc đầy đủ, với các biến thể "tư"/"bốn", "linh"/"lẻ",
      "nghìn"/"ngàn";
    * đọc tắt hàng chục: 45 -> "bốn lăm", 21 -> "hai mốt";
    * đọc tắt hàng trăm tròn chục: 120 -> "một trăm hai", 110 -> "một trăm
      mốt";
    * đọc tắt số ba chữ số: 155 -> "một năm lăm";
    * đọc tắt nghìn / triệu tròn trăm: 2500 -> "hai nghìn năm",
      "hai nghìn rưỡi"; 1500000 -> "một triệu rưỡi".
    """
    full = {spell(number, four, linh, thousand)
            for four in ("tư", "bốn") for linh in ("linh", "lẻ")
            for thousand in ("nghìn", "ngàn")}
    short: set[str] = set()
    hundreds, rest = divmod(number, 100)
    tens, unit = divmod(rest, 10)
    if 21 <= number <= 99 and unit:
        short |= {f"{UNITS[tens]} {_unit_after_tens(unit, tens, four)}"
                  for four in ("tư", "bốn")}
    if 100 <= number <= 999 and tens and not unit:
        short.add(f"{UNITS[hundreds]} trăm "
                  f"{'mốt' if tens == 1 else UNITS[tens]}")
    if 100 <= number <= 999 and tens > 1 and unit:
        # Bỏ cả "trăm" lẫn "mươi" (hay gặp khi đọc mã chuyến bay):
        # 155 -> "một năm lăm", 151 -> "một năm mốt".
        short |= {f"{UNITS[hundreds]} {UNITS[tens]} "
                  f"{_unit_after_tens(unit, tens, four)}"
                  for four in ("tư", "bốn")}
    for scale, names in ((10 ** 6, ("triệu",)), (1000, ("nghìn", "ngàn"))):
        head, tail = divmod(number, scale)
        # Chỉ đọc tắt khi phần sau là một chữ số hàng trăm (nghìn) hoặc hàng
        # trăm nghìn (triệu): 2500, 1500000.
        if head and tail and tail % (scale // 10) == 0:
            digit = tail // (scale // 10)
            for name in names:
                short.add(f"{spell(head)} {name} {UNITS[digit]}")
                if digit == 5:
                    short.add(f"{spell(head)} {name} rưỡi")
    return frozenset(full | short)


def digit_run_readings(digits: str, after_letters: bool) -> list[str]:
    """Các cách đọc của một dãy chữ số.

    Args:
        digits: Dãy chữ số, ví dụ ``"95"``.
        after_letters: Dãy đứng ngay sau chữ cái (mã như "VTV14", "RON95").
            Mã thường được đọc cả theo số lẫn từng chữ số ("mười bốn" /
            "một bốn").
    """
    if (len(digits) >= DIGIT_BY_DIGIT_MIN_LEN
            or (len(digits) > 1 and digits.startswith("0"))):
        return [digit_by_digit(digits)]
    readings = set(integer_readings(int(digits)))
    is_code = after_letters and len(digits) > 1
    if len(digits) >= ALSO_DIGITS_MIN_LEN or is_code:
        readings.add(digit_by_digit(digits))
    return sorted(readings)


def _combine(options: list[str], continuations: list[str],
             limit: int) -> list[str]:
    """Tích Đề-các của hai danh sách cách đọc (cắt ở ``limit``)."""
    return [f"{head} {tail}".strip() for head in options
            for tail in continuations][:limit]


def token_readings(token: str, limit: int = 400) -> list[str] | None:
    """Các cách đọc của một token chữ thường có chứa chữ số.

    Token được tách thành các đoạn toàn chữ số hoặc toàn chữ cái:

    * đoạn chữ số -> :func:`digit_run_readings`;
    * đoạn chữ cái là đơn vị viết liền sau số ("k", "đ", "h") -> lời đọc
      của đơn vị (:data:`SUFFIX_READINGS`);
    * đoạn chữ cái khác (tên mã: "vtv", "ron") -> giữ nguyên.

    Giờ kiểu ``"7h30"`` được đọc "bảy giờ ba mươi" (và "... ba mươi phút",
    "... rưỡi" nếu là 30 phút).

    Returns:
        Danh sách cách đọc, hoặc ``None`` nếu token có ký tự khác chữ và số
        (khi đó câu chứa token bị bỏ, xem :mod:`gft.targets`).
    """
    runs = _RUNS.findall(token)
    if "".join(runs) != token:
        return None
    time = re.fullmatch(r"(\d{1,2})h(\d{2})", token)
    if time:
        hours = digit_run_readings(time.group(1), after_letters=False)
        minutes = int(time.group(2))
        if minutes == 0:
            return [f"{hour} giờ" for hour in hours]
        minute_words = digit_run_readings(str(minutes), after_letters=False)
        options = [f"{hour} giờ {minute}{suffix}" for hour in hours
                   for minute in minute_words for suffix in ("", " phút")]
        if minutes == 30:
            options += [f"{hour} giờ rưỡi" for hour in hours]
        return options[:limit]
    options = [""]
    previous_is_digit = False
    for index, run in enumerate(runs):
        if run.isdigit():
            choices = digit_run_readings(run, after_letters=index > 0)
        elif previous_is_digit and run in SUFFIX_READINGS:
            choices = list(SUFFIX_READINGS[run])
        else:
            choices = [run]
        options = _combine(options, choices, limit)
        previous_is_digit = run.isdigit()
    return options
