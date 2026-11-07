"""Đọc file âm thanh của cuộc thi (WAV 16 kHz, mono, PCM 16-bit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


def load_audio(path: str | Path,
               expected_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Đọc một file WAV thành mảng ``float32`` mono, giá trị trong [-1, 1].

    Args:
        path: Đường dẫn file WAV.
        expected_rate: Tần số lấy mẫu bắt buộc. Model ASR được huấn luyện ở
            16 kHz nên file ở tần số khác bị từ chối thay vì âm thầm cho ra
            kết quả sai.

    Returns:
        Mảng một chiều ``(n_samples,)``.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu tần số lấy mẫu khác ``expected_rate``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {path}")

    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != expected_rate:
        raise ValueError(
            f"{path.name}: tần số {rate} Hz, cần {expected_rate} Hz.")

    # File stereo (hiếm gặp) được trộn về mono bằng trung bình hai kênh.
    return audio if audio.ndim == 1 else audio.mean(axis=1)
