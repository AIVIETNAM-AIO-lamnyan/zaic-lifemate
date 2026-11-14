"""Dữ liệu cho fine-tune: lượt nói có nhãn, chia train/dev, batch cho model.

Mỗi mẫu của ``train.jsonl`` cho nhiều lượt nói có nhãn ``audio_text``: các
lượt ``user`` trong ``history`` và yêu cầu cuối. Toàn bộ tập train có
1.545 mẫu, 5.036 lượt nói.

**Chia train/dev theo mẫu, không theo lượt nói.** Các lượt của cùng một
hội thoại có cùng người nói và cùng chủ đề. Nếu u1 nằm ở train còn yêu cầu
cuối nằm ở dev, model đã "nghe" giọng đó lúc học và WER trên dev sẽ đẹp hơn
thực tế (rò rỉ dữ liệu).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..api import get_sample, list_sample_ids
from ..audio_io import load_audio
from ..config import DATA_ROOT, SAMPLE_RATE


@dataclass(frozen=True)
class Utterance:
    """Một lượt nói có nhãn.

    Attributes:
        utt_id: ``<sample_id>`` cho yêu cầu cuối, ``<sample_id>_u<k>`` cho
            lượt user thứ k trong history.
        sample_id: Id của mẫu (hội thoại) chứa lượt nói.
        audio_path: Đường dẫn file WAV.
        text: Transcript do người gõ (``audio_text``).
    """

    utt_id: str
    sample_id: str
    audio_path: Path
    text: str


def load_utterances(split: str = "train",
                    data_root: str | Path | None = None) -> list[Utterance]:
    """Mọi lượt nói có nhãn của một split, theo thứ tự hội thoại.

    Raises:
        ValueError: Nếu split không có ``audio_text`` (ví dụ public_test).
    """
    split_dir = Path(data_root or DATA_ROOT) / split
    samples = [get_sample(sample_id, split, split_dir.parent)
               for sample_id in list_sample_ids(split, split_dir.parent)]
    if not samples or "audio_text" not in samples[0]:
        raise ValueError(f"Split '{split}' không có audio_text: "
                         "chỉ fine-tune được trên tập train.")

    def sample_turns(sample: dict[str, Any]) -> list[Utterance]:
        """Các lượt user trong history (u1, u2...) rồi tới yêu cầu cuối."""
        users = [turn for turn in sample["history"]
                 if turn.get("role") == "user"]
        history = [Utterance(f"{sample['id']}_u{k}", sample["id"],
                             split_dir / turn["audio"], turn["audio_text"])
                   for k, turn in enumerate(users, start=1)]
        final = Utterance(sample["id"], sample["id"],
                          split_dir / sample["audio"], sample["audio_text"])
        return [*history, final]

    return [utt for sample in samples for utt in sample_turns(sample)]


def split_by_sample(utterances: list[Utterance], dev_ratio: float = 0.1,
                    seed: int = 42
                    ) -> tuple[list[Utterance], list[Utterance]]:
    """Chia train/dev theo **mẫu**: mọi lượt của một mẫu nằm cùng một phía.

    Với cùng ``seed``, kết quả chia luôn giống nhau, nên WER trước và sau
    fine-tune được đo trên đúng cùng một tập dev.
    """
    sample_ids = sorted({utt.sample_id for utt in utterances})
    n_dev = max(1, round(len(sample_ids) * dev_ratio))
    dev_ids = set(random.Random(seed).sample(sample_ids, n_dev))
    train = [utt for utt in utterances if utt.sample_id not in dev_ids]
    dev = [utt for utt in utterances if utt.sample_id in dev_ids]
    return train, dev


class SpeechDataset(Dataset):
    """Đọc âm thanh và tạo ``(input_features, labels)`` khi được truy cập.

    Âm thanh được đọc lười (lazy) từng lượt một thay vì nạp hết vào RAM:
    6,8 giờ âm thanh float32 chiếm khoảng 1,6 GB.

    Args:
        utterances: Các lượt nói.
        processor: ``WhisperProcessor`` (bộ trích log-mel và tokenizer).
    """

    def __init__(self, utterances: list[Utterance], processor: Any) -> None:
        self.utterances = utterances
        self.processor = processor

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        utt = self.utterances[index]
        audio = load_audio(utt.audio_path)
        # Whisper luôn nhận đúng 30 giây log-mel: đoạn ngắn hơn được đệm 0.
        features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt").input_features[0]
        # Token của nhãn đã có sẵn prefix <|startoftranscript|><|vi|>
        # <|transcribe|><|notimestamps|> nhờ set_prefix_tokens trong train.py.
        labels = self.processor.tokenizer(utt.text).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class SpeechCollator:
    """Gộp các phần tử thành một batch tensor.

    * ``input_features``: cùng kích thước (80 x 3000) nên chỉ cần ``stack``.
    * ``labels``: dài ngắn khác nhau, được đệm bằng ``-100``; giá trị này bị
      hàm loss bỏ qua, nên phần đệm không ảnh hưởng gradient.
    * Token đầu ``<|startoftranscript|>`` được bỏ khỏi nhãn, vì model tự
      thêm nó vào đầu chuỗi đầu vào của decoder (dịch nhãn sang phải một
      bước). Không bỏ thì token này bị lặp hai lần.
    """

    decoder_start_token_id: int

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = torch.stack([item["input_features"] for item in items])
        labels = [item["labels"][1:]
                  if item["labels"][0] == self.decoder_start_token_id
                  else item["labels"] for item in items]
        width = max(map(len, labels))
        padded = torch.tensor([ids + [-100] * (width - len(ids))
                               for ids in labels])
        return {"input_features": features, "labels": padded}
