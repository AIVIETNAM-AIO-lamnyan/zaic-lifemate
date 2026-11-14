"""Hàm công khai của module ASR: âm thanh của một mẫu -> text cho LLM.

Phần còn lại của pipeline chỉ cần gọi một hàm::

    from src.asr import get_user_request_from_audio

    current_req, history_req = get_user_request_from_audio("train_0001")

Các lựa chọn thiết kế:

* **Model chỉ nạp một lần cho mỗi process** rồi dùng lại. Nạp PhoWhisper-large
  mất vài giây, không thể trả chi phí đó cho mỗi request khi giới hạn là
  30 giây/request. Gọi :func:`load_asr_model` lúc khởi động để nạp trước.
* **Đường dẫn âm thanh đọc từ ``<split>.jsonl``** (trường ``audio`` của mẫu và
  của các lượt ``user`` trong ``history``), không cần ``audio_text``. Vì vậy
  hàm chạy giống nhau trên train, public_test và private test.
* **Mọi lượt nói của một mẫu được nhận dạng trong một lần gọi model** (một
  batch), nhanh hơn gọi riêng từng file.
* **Text trả về là output của model** (bỏ khoảng trắng thừa). Model mặc
  định Gipformer chỉ viết chữ HOA, không dấu câu, số bằng lời đọc ("ba giờ
  chiều"); backend của nó trả text viết thường (đội Datyth thấy chữ thường
  giúp agent, còn đổi số về chữ số không giúp gì).
* **Hotword theo từng lượt** (chỉ backend hỗ trợ, như Gipformer): các tên
  trợ lý đã nói / đã tra cứu ở các lượt *trước* trong ``history`` được đưa
  cho bộ giải mã để ưu tiên (xem ``hotwords.py``). Không bao giờ đọc
  ``<split>_environment/`` (luật BTC cấm).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .audio_io import load_audio
from .backends import ASRBackend, build_backend
from .config import DATA_ROOT, DEFAULT_MODEL, MODELS
from .hotwords import turn_hotwords

#: Tiền tố của id mẫu -> tên thư mục split.
_SPLIT_BY_PREFIX = {"train": "train", "public": "public_test"}

#: Model đã nạp, theo khoá ``(model, device)``.
_LOADED_MODELS: dict[tuple[str, str], ASRBackend] = {}


# ---------------------------------------------------------------------------
# 1. QUẢN LÝ MODEL (NẠP MỘT LẦN, DÙNG LẠI)
# ---------------------------------------------------------------------------
def load_asr_model(model: str = DEFAULT_MODEL,
                   device: str = "auto") -> ASRBackend:
    """Nạp model ASR một lần và giữ trong bộ nhớ cho các lần gọi sau.

    Args:
        model: Khoá trong ``config.MODELS`` (mặc định ``phowhisper_large``).
        device: ``auto`` (GPU nếu có), ``cuda:0`` hoặc ``cpu``.

    Returns:
        Wrapper của model đã nạp.

    Raises:
        KeyError: Nếu ``model`` không có trong ``config.MODELS``.
    """
    key = (model, device)
    if key not in _LOADED_MODELS:
        if model not in MODELS:
            raise KeyError(f"Model '{model}' không tồn tại. "
                           f"Các model có sẵn: {sorted(MODELS)}")
        backend = build_backend(MODELS[model], device=device)
        backend.load()
        _LOADED_MODELS[key] = backend
    return _LOADED_MODELS[key]


def unload_asr_model() -> None:
    """Giải phóng mọi model ASR đã nạp (ví dụ để nhường GPU cho LLM)."""
    for backend in _LOADED_MODELS.values():
        backend.unload()
    _LOADED_MODELS.clear()


# ---------------------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU MẪU
# ---------------------------------------------------------------------------
def infer_split(sample_id: str) -> str:
    """Suy ra split từ id: ``train_0001`` -> ``train``.

    Raises:
        ValueError: Nếu tiền tố không quen (ví dụ private test); khi đó cần
            truyền ``split`` một cách tường minh.
    """
    prefix = sample_id.rsplit("_", 1)[0]
    if prefix not in _SPLIT_BY_PREFIX:
        raise ValueError(f"Không suy ra được split của '{sample_id}': "
                         "hãy truyền tham số split=...")
    return _SPLIT_BY_PREFIX[prefix]


@lru_cache(maxsize=None)
def _read_split(split_dir: Path) -> dict[str, dict[str, Any]]:
    """Đọc ``<split>/<split>.jsonl`` thành dict ``{id: mẫu}``.

    Kết quả được cache (``lru_cache``): mỗi split chỉ đọc từ đĩa một lần.
    """
    jsonl = split_dir / f"{split_dir.name}.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"Không tìm thấy file mẫu: {jsonl}")
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    records = (json.loads(line) for line in lines if line.strip())
    return {record["id"]: record for record in records}


def _split_dir(sample_id: str, split: str | None,
               data_root: str | Path | None) -> Path:
    """Thư mục split chứa mẫu ``sample_id``."""
    root = Path(data_root) if data_root else DATA_ROOT
    return root / (split or infer_split(sample_id))


def get_sample(sample_id: str, split: str | None = None,
               data_root: str | Path | None = None) -> dict[str, Any]:
    """Bản ghi JSON của một mẫu trong ``<split>.jsonl``.

    Raises:
        KeyError: Nếu id không có trong file (id không liên tục, ví dụ
            public_test bắt đầu từ ``public_0002``).
    """
    split_dir = _split_dir(sample_id, split, data_root)
    samples = _read_split(split_dir)
    if sample_id not in samples:
        raise KeyError(f"'{sample_id}' không có trong {split_dir.name}.jsonl")
    return samples[sample_id]


def list_sample_ids(split: str = "train",
                    data_root: str | Path | None = None) -> list[str]:
    """Danh sách id của một split, theo thứ tự trong file jsonl."""
    root = Path(data_root) if data_root else DATA_ROOT
    return list(_read_split(root / split))


def audio_paths(sample_id: str, split: str | None = None,
                data_root: str | Path | None = None
                ) -> tuple[Path, list[Path]]:
    """Các file WAV của một mẫu.

    Returns:
        ``(yêu_cầu_cuối, [lượt_u1, ..., lượt_un])``. Danh sách history chỉ
        gồm lượt ``user`` (lượt ``assistant`` là text, không có âm thanh),
        theo đúng thứ tự hội thoại.
    """
    split_dir = _split_dir(sample_id, split, data_root)
    sample = get_sample(sample_id, split_dir.name, split_dir.parent)
    history = [split_dir / turn["audio"]
               for turn in sample.get("history", [])
               if turn.get("role") == "user"]
    return split_dir / sample["audio"], history


# ---------------------------------------------------------------------------
# 3. HÀM CHÍNH CHO PIPELINE
# ---------------------------------------------------------------------------
def get_user_request_from_audio(sample_id: str, split: str | None = None,
                                data_root: str | Path | None = None,
                                model: str = DEFAULT_MODEL,
                                device: str = "auto"
                                ) -> tuple[str, list[str]]:
    """Nhận dạng toàn bộ lượt nói của người dùng trong một mẫu.

    Args:
        sample_id: Id mẫu, ví dụ ``train_0001`` hoặc ``public_0002``.
        split: Thư mục split (``train``, ``public_test``...). Bỏ trống thì
            suy ra từ tiền tố id; bắt buộc truyền với private test.
        data_root: Thư mục chứa các split. Mặc định ``config.DATA_ROOT``.
        model: Khoá model ASR (mặc định ``gipformer_zac``).
        device: ``auto``, ``cuda:0`` hoặc ``cpu``.

    Returns:
        ``(current_req, history_req)``:

        * ``current_req``: text của yêu cầu cuối (lượt được chấm điểm).
        * ``history_req``: text các lượt ``user`` trước đó, cũ nhất trước;
          ``[]`` nếu mẫu không có history.
    """
    final_path, history_paths = audio_paths(sample_id, split, data_root)

    # Ghép history và yêu cầu cuối thành một batch: một lần gọi model duy
    # nhất cho cả mẫu. Yêu cầu cuối luôn nằm ở vị trí cuối cùng.
    clips = [load_audio(path) for path in (*history_paths, final_path)]
    backend = load_asr_model(model, device)
    if backend.supports_hotwords:
        split_dir = _split_dir(sample_id, split, data_root)
        sample = get_sample(sample_id, split_dir.name, split_dir.parent)
        texts = backend.transcribe_batch(clips,
                                         hotwords=turn_hotwords(sample))
    else:
        texts = backend.transcribe_batch(clips)
    return texts[-1], texts[:-1]
