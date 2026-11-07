"""Module ASR của LifeMate: lượt nói tiếng Việt -> text cho LLM.

Cách dùng (chạy từ thư mục gốc của repo)::

    from src.asr import get_user_request_from_audio, load_asr_model

    load_asr_model()        # tuỳ chọn: nạp model trước khi xử lý các mẫu
    current_req, history_req = get_user_request_from_audio("train_0001")

Các file:

* ``api.py``      hàm công khai (nhận dạng, nạp/giải phóng model, đọc mẫu)
* ``backends.py`` wrapper cho Whisper/PhoWhisper, Qwen3-ASR, Nemotron
* ``audio_io.py`` đọc file WAV
* ``config.py``   danh sách model, đường dẫn dữ liệu
* ``demo.py``     chạy thử trên vài mẫu: ``python -m src.asr.demo``

Xem ``README.md`` trong cùng thư mục để biết chi tiết.
"""

from .api import (get_sample, get_user_request_from_audio, list_sample_ids,
                  load_asr_model, unload_asr_model)

__all__ = [
    "get_sample",
    "get_user_request_from_audio",
    "list_sample_ids",
    "load_asr_model",
    "unload_asr_model",
]
