"""Kiểm thử ``get_user_request_from_audio`` (không cần GPU, model hay data).

Một backend giả trả về mã số được "giấu" trong mẫu đầu tiên của mỗi file
âm thanh. Nhờ đó test kiểm tra được hàm đọc đúng file, đúng thứ tự, và tách
đúng ``(current_req, history_req)``.

Chạy từ thư mục gốc repo::

    pytest src/asr/tests -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.asr import api
from src.asr.backends import ASRBackend
from src.asr.config import DEFAULT_MODEL, MODELS


class FakeBackend(ASRBackend):
    """Backend giả: đọc lại mã số ghi trong mẫu đầu tiên của mỗi clip."""

    load_count = 0

    def __init__(self, spec, device="cpu", dtype="float32"):
        # Bỏ qua resolve_device/resolve_dtype để test không cần torch.
        self.spec, self.device, self.dtype = spec, device, dtype
        self.model = None

    def load(self) -> None:
        FakeBackend.load_count += 1
        self.model = object()

    def transcribe_batch(self, batch):
        return [f"utt{round(clip[0] * 1000)}" for clip in batch]

    def unload(self) -> None:
        self.model = None


def _write_wav(path: Path, code: int) -> None:
    """Ghi một file WAV 0,1 s, mẫu đầu tiên mang mã ``code``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.zeros(1600, dtype="float32")
    signal[0] = code / 1000
    sf.write(path, signal, 16000, subtype="FLOAT")


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Split public_test thu nhỏ: một mẫu có 2 lượt history, một mẫu không."""
    split = tmp_path / "public_test"
    split.mkdir()
    records = [
        {"id": "public_0002", "audio": "audio/public_0002/public_0002.wav",
         "history": [
             {"role": "user",
              "audio": "audio/public_0002/public_0002_u1.wav"},
             {"role": "assistant", "text": "Đã bật đèn."},
             {"role": "user",
              "audio": "audio/public_0002/public_0002_u2.wav"},
         ]},
        {"id": "public_0005", "audio": "audio/public_0005/public_0005.wav",
         "history": []},
    ]
    (split / "public_test.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8")
    files = {"public_0002/public_0002_u1": 1, "public_0002/public_0002_u2": 2,
             "public_0002/public_0002": 9, "public_0005/public_0005": 5}
    for name, code in files.items():
        _write_wav(split / "audio" / f"{name}.wav", code)

    # Thay backend thật bằng backend giả và làm sạch mọi cache giữa các test.
    monkeypatch.setattr(api, "build_backend",
                        lambda spec, **kwargs: FakeBackend(spec))
    monkeypatch.setattr(api, "_LOADED_MODELS", {})
    api._read_split.cache_clear()
    FakeBackend.load_count = 0
    return tmp_path


def test_returns_current_and_history(data_root):
    current, history = api.get_user_request_from_audio(
        "public_0002", data_root=data_root)
    assert current == "utt9"
    assert history == ["utt1", "utt2"]      # cũ nhất trước


def test_empty_history(data_root):
    assert api.get_user_request_from_audio(
        "public_0005", data_root=data_root) == ("utt5", [])


def test_model_loaded_once(data_root):
    for sample_id in ("public_0002", "public_0005", "public_0002"):
        api.get_user_request_from_audio(sample_id, data_root=data_root)
    assert FakeBackend.load_count == 1


def test_explicit_split(data_root):
    current, _ = api.get_user_request_from_audio(
        "public_0005", split="public_test", data_root=data_root)
    assert current == "utt5"


def test_list_sample_ids(data_root):
    assert api.list_sample_ids("public_test", data_root) == [
        "public_0002", "public_0005"]


def test_unknown_sample(data_root):
    with pytest.raises(KeyError):
        api.get_user_request_from_audio("public_0001", data_root=data_root)


def test_unknown_model(data_root):
    with pytest.raises(KeyError):
        api.load_asr_model("not_a_model")


def test_infer_split():
    assert api.infer_split("train_0001") == "train"
    assert api.infer_split("public_0123") == "public_test"
    with pytest.raises(ValueError):
        api.infer_split("private_0001")


def test_default_model_registered():
    assert DEFAULT_MODEL in MODELS
