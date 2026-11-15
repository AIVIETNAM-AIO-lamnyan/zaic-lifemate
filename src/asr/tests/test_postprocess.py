"""Kiểm tra luật hậu xử lý text ASR (src/asr/postprocess.py)."""

from src.asr.postprocess import postprocess, restore_tens


def test_restore_tens_lowercase():
    assert restore_tens("xăng chín lăm bao nhiêu") == \
        "xăng chín mươi lăm bao nhiêu"


def test_restore_tens_uppercase():
    assert restore_tens("MUỘN THÊM BỐN LĂM PHÚT") == \
        "MUỘN THÊM BỐN MƯƠI LĂM PHÚT"


def test_full_form_unchanged():
    for text in ("chín mươi lăm", "mười lăm", "một trăm linh năm",
                 "ngày mốt", "hai tư giờ", "tháng năm"):
        assert postprocess(text) == text


def test_word_boundary():
    # "lăm" phải là cả một từ; "năm" đứng sau chữ khác không bị bắt.
    assert restore_tens("nắm lăm") == "nắm lăm"
    assert restore_tens("tám lắm") == "tám lắm"
