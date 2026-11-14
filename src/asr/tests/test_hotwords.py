"""Kiểm thử hotword theo mẫu và việc truyền hotword vào backend."""

from __future__ import annotations

import json

from src.asr import api
from src.asr.hotwords import build_hotwords, read_state, spoken_forms

from test_api import FakeBackend, data_root  # noqa: F401  (fixture)


def test_spoken_forms():
    """Tên có chữ số đổi sang lời đọc; dấu thanh kiểu mới; viết HOA."""
    assert spoken_forms("Tàu SE7") == ["TÀU SE BẢY"]
    assert spoken_forms("VTV3") == ["VTV BA"]
    assert spoken_forms("điều hoà") == ["ĐIỀU HÒA"]


def test_build_hotwords_fields_and_addresses():
    """Chỉ lấy trường tên; địa chỉ tách theo dấu phẩy; bỏ chuỗi quá dài."""
    state = {"contacts": [{"name": "Chị Thu", "phone": "0901234567"}],
             "places": [{"address": "88 Láng Hạ, Đống Đa"}],
             "notes": [{"content": "không phải tên"}],
             "music": [{"title": "x" * 80}]}
    phrases = set(build_hotwords(state).split("/"))
    assert {"CHỊ THU", "ĐỐNG ĐA"} <= phrases
    assert "TÁM TÁM LÁNG HẠ" in phrases or "TÁM MƯƠI TÁM LÁNG HẠ" in phrases
    assert not any("KHÔNG PHẢI" in p for p in phrases)
    assert not any(p.startswith("XXX") for p in phrases)


class HotwordFakeBackend(FakeBackend):
    """Backend giả có hỗ trợ hotword: ghi lại hotword nhận được."""

    supports_hotwords = True
    received: list = []

    def transcribe_batch(self, batch, hotwords=None):
        HotwordFakeBackend.received.append(hotwords)
        return super().transcribe_batch(batch)


def test_api_passes_hotwords(data_root, monkeypatch):  # noqa: F811
    """state.json có thì truyền hotword; không có thì truyền None."""
    env = data_root / "public_test" / "public_test_environment" / \
        "public_0002"
    env.mkdir(parents=True)
    (env / "state.json").write_text(
        json.dumps({"tv": {"channels": ["VTV3"]}}), encoding="utf-8")
    monkeypatch.setattr(api, "build_backend",
                        lambda spec, **kwargs: HotwordFakeBackend(spec))
    HotwordFakeBackend.received = []
    current, history = api.get_user_request_from_audio(
        "public_0002", data_root=data_root)
    api.get_user_request_from_audio("public_0005", data_root=data_root)
    assert (current, history) == ("utt9", ["utt1", "utt2"])
    assert HotwordFakeBackend.received == ["VTV BA", None]
    assert read_state(data_root / "public_test", "public_0005") is None
