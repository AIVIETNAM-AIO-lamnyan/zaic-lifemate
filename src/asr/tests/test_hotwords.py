"""Kiểm thử hotword lấy từ lịch sử hội thoại và việc truyền vào backend."""

from __future__ import annotations

from src.asr import api
from src.asr.hotwords import (assistant_names, reply_phrases, spoken_forms,
                              turn_hotwords)

from test_api import FakeBackend, data_root  # noqa: F401  (fixture)


def test_spoken_forms():
    """Tên có chữ số đổi sang lời đọc; dấu thanh kiểu mới; viết HOA."""
    assert spoken_forms("Tàu SE7") == ["TÀU SE BẢY"]
    assert spoken_forms("VTV3") == ["VTV BA"]
    assert spoken_forms("điều hoà") == ["ĐIỀU HÒA"]


def test_reply_phrases_skip_sentence_start():
    """Lấy cụm viết hoa giữa câu và mã chữ + số; bỏ từ đầu câu, giờ."""
    phrases = reply_phrases("Đã nhắn Kevin Tran. Tàu SE7 chạy lúc 06:15.")
    assert phrases == {"Kevin Tran", "SE7"}


def test_assistant_names_from_tool_results():
    """Chỉ lấy trường tên; địa chỉ tách theo dấu phẩy; bỏ chuỗi quá dài."""
    turn = {"role": "assistant", "text": "",
            "tool_results": [{"ok": True, "items": [
                {"name": "Chị Thu", "phone": "0901234567"},
                {"address": "88 Láng Hạ, Đống Đa"},
                {"content": "không phải tên"},
                {"title": "x" * 80}]}]}
    assert assistant_names(turn) == {"Chị Thu", "88 Láng Hạ", "Đống Đa"}


def test_turn_hotwords_only_use_earlier_turns():
    """Lượt thứ k chỉ thấy các lượt assistant đứng trước nó."""
    sample = {"history": [
        {"role": "user"},
        {"role": "assistant", "text": "Kênh VTV3 đang chiếu phim.",
         "tool_results": []},
        {"role": "user"}]}
    assert turn_hotwords(sample) == [None, "VTV BA", "VTV BA"]
    assert turn_hotwords({"history": []}) == [None]


class HotwordFakeBackend(FakeBackend):
    """Backend giả có hỗ trợ hotword: ghi lại hotword nhận được."""

    supports_hotwords = True
    received: list = []

    def transcribe_batch(self, batch, hotwords=None):
        HotwordFakeBackend.received.append(hotwords)
        return super().transcribe_batch(batch)


def test_api_passes_per_turn_hotwords(data_root, monkeypatch):  # noqa: F811
    """Mỗi clip một hotword, lấy từ history; không đọc thư mục environment."""
    monkeypatch.setattr(api, "build_backend",
                        lambda spec, **kwargs: HotwordFakeBackend(spec))
    HotwordFakeBackend.received = []
    current, history = api.get_user_request_from_audio(
        "public_0002", data_root=data_root)
    api.get_user_request_from_audio("public_0005", data_root=data_root)
    assert (current, history) == ("utt9", ["utt1", "utt2"])
    # Fixture: trợ lý chỉ trả lời "Đã bật đèn." -> không có tên nào.
    assert HotwordFakeBackend.received == [[None, None, None], [None]]
    assert not (data_root / "public_test" / "public_test_environment").exists()
