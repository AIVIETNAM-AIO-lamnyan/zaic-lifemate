"""Đặc trưng âm thanh của từng file, dùng để so sánh train và test.

Mọi phép đo chỉ dùng numpy (không cần GPU, không cần model), dựa trên năng
lượng của các khung 25 ms trượt mỗi 10 ms:

* ``duration_s``     độ dài (giây). Whisper chỉ nghe được 30 giây đầu.
* ``level_dbfs``     độ to trung bình của phần có tiếng nói (dB so với mức
                     tối đa 0 dBFS; -20 là to, -40 là nhỏ).
* ``noise_dbfs``     độ to của nền (phân vị 10 % năng lượng khung): phòng yên
                     tĩnh thường dưới -60, nhiều tạp âm thì cao hơn.
* ``snr_db``         ước lượng tỉ số tín hiệu trên nhiễu = tiếng nói - nền.
                     Trên 30 dB là sạch, dưới 15 dB là ồn.
* ``speech_ratio``   tỉ lệ khung có tiếng nói (cao hơn nền 10 dB).
* ``clip_ratio``     tỉ lệ mẫu chạm trần (|x| >= 0,999): méo do thu quá to.
* ``rolloff_hz``     tần số dưới đó chứa 99 % năng lượng phổ. Thu bằng điện
                     thoại cũ / nén mạnh thường bị cắt quanh 3,4-4 kHz.
* ``hf_ratio``       tỉ lệ năng lượng trên 4 kHz.
* ``centroid_hz``    "trọng tâm" phổ: giọng trầm/bổng, micro sáng/tối.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

FRAME, HOP = 400, 160          # 25 ms / 10 ms ở 16 kHz
EPS = 1e-10


def frame_energy_db(audio: np.ndarray) -> np.ndarray:
    """Năng lượng (dBFS) của từng khung 25 ms, bước 10 ms.

    ``sliding_window_view`` tạo ma trận (số khung x 400) mà không sao chép
    dữ liệu, nên tính được cho cả file trong một phép toán, không cần vòng
    lặp.
    """
    if len(audio) < FRAME:
        audio = np.pad(audio, (0, FRAME - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME)[::HOP]
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + EPS)


def spectrum_stats(audio: np.ndarray, rate: int) -> dict[str, float]:
    """Rolloff 99 %, tỉ lệ năng lượng trên 4 kHz và trọng tâm phổ."""
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1 / rate)
    total = power.sum() + EPS
    cumulative = np.cumsum(power) / total
    return {
        "rolloff_hz": float(freqs[np.searchsorted(cumulative, 0.99)]),
        "hf_ratio": float(power[freqs >= 4000].sum() / total),
        "centroid_hz": float((freqs * power).sum() / total),
    }


def audio_stats(path: Path) -> dict[str, float]:
    """Mọi đặc trưng của một file WAV (xem docstring của module)."""
    info = sf.info(str(path))
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                       # trộn về mono
    energy = frame_energy_db(audio)
    noise = float(np.percentile(energy, 10))
    speech = energy[energy > noise + 10]             # khung có tiếng nói
    level = float(np.percentile(speech, 50)) if speech.size else noise
    return {
        "sample_rate": rate,
        "channels": info.channels,
        "subtype": info.subtype,
        "duration_s": len(audio) / rate,
        "level_dbfs": level,
        "noise_dbfs": noise,
        "snr_db": float(np.percentile(energy, 95)) - noise,
        "speech_ratio": speech.size / energy.size,
        "clip_ratio": float(np.mean(np.abs(audio) >= 0.999)),
        "dc_offset": float(audio.mean()),
        **spectrum_stats(audio, rate),
    }
