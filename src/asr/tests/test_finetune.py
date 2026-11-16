"""Kiểm thử phần fine-tune với một model Whisper tí hon khởi tạo ngẫu nhiên.

Không tải PhoWhisper, không cần GPU: chỉ kiểm tra logic (chia dữ liệu, tạo
batch, gắn LoRA, loss giảm khi học). Bỏ qua nếu chưa cài torch/peft.

    pytest src/asr/tests/test_finetune.py -v
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from src.asr import api                                      # noqa: E402
from src.asr.finetune.data import (SpeechCollator,           # noqa: E402
                                   SpeechDataset, load_utterances,
                                   split_by_sample)
from src.asr.finetune.model import (add_lora, normalize,     # noqa: E402
                                    word_error_rate)

START = 1      # id của <|startoftranscript|> trong model tí hon


@pytest.fixture
def train_root(tmp_path):
    """Tập train thu nhỏ: 10 mẫu, mỗi mẫu 1 lượt history + yêu cầu cuối."""
    split = tmp_path / "train"
    records = []
    for k in range(10):
        sid = f"train_{k:04d}"
        records.append({
            "id": sid, "audio": f"audio/{sid}/{sid}.wav",
            "audio_text": f"yêu cầu cuối {k}",
            "history": [{"role": "user", "audio": f"audio/{sid}/{sid}_u1.wav",
                         "audio_text": f"lượt một {k}"},
                        {"role": "assistant", "text": "vâng"}]})
        for name in (sid, f"{sid}_u1"):
            path = split / "audio" / sid / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.zeros(1600, dtype="float32"), 16000)
    (split / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8")
    api._read_split.cache_clear()
    return tmp_path


def tiny_whisper():
    """Whisper 2 lớp, 32 chiều: đủ để kiểm tra luồng học trên CPU."""
    config = transformers.WhisperConfig(
        vocab_size=64, num_mel_bins=80, d_model=32, encoder_layers=2,
        decoder_layers=2, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
        max_target_positions=32, decoder_start_token_id=START,
        pad_token_id=0, eos_token_id=2)
    return transformers.WhisperForConditionalGeneration(config)


def test_load_and_split_by_sample(train_root):
    utterances = load_utterances("train", train_root)
    assert len(utterances) == 20
    assert [u.utt_id for u in utterances[:2]] == ["train_0000_u1",
                                                  "train_0000"]
    train, dev = split_by_sample(utterances, dev_ratio=0.2, seed=0)
    # Hai lượt của cùng một mẫu không bao giờ nằm ở hai phía.
    assert not {u.sample_id for u in train} & {u.sample_id for u in dev}
    assert len(dev) == 4
    assert split_by_sample(utterances, 0.2, 0)[1] == dev   # tái lập được


def test_dataset_and_collator(train_root):
    tokens = {"lượt một 0": [START, 5, 6, 2], "yêu cầu cuối 0": [START, 7, 2]}
    processor = SimpleNamespace(
        feature_extractor=transformers.WhisperFeatureExtractor(),
        tokenizer=lambda text: SimpleNamespace(input_ids=tokens[text]))
    dataset = SpeechDataset(load_utterances("train", train_root)[:2],
                            processor)
    assert dataset[0]["input_features"].shape == (80, 3000)

    batch = SpeechCollator(START)([dataset[0], dataset[1]])
    assert batch["input_features"].shape == (2, 80, 3000)
    # Bỏ token bắt đầu, đệm -100 cho bằng độ dài.
    assert batch["labels"].tolist() == [[5, 6, 2], [7, 2, -100]]


@pytest.mark.parametrize("decoder_only", [False, True])
def test_add_lora_targets(decoder_only):
    model = add_lora(tiny_whisper(), rank=4, alpha=8,
                     decoder_only=decoder_only)
    trainable = [name for name, p in model.named_parameters()
                 if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    assert any(".decoder." in name for name in trainable)
    assert any(".encoder." in name for name in trainable) != decoder_only


def test_lora_training_reduces_loss():
    torch.manual_seed(0)
    model = add_lora(tiny_whisper(), rank=4, alpha=8, dropout=0.0)
    model.train()
    batch = {"input_features": torch.randn(2, 80, 3000),
             "labels": torch.tensor([[5, 6, 2], [7, 8, 2]])}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)

    def step() -> float:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        return loss.item()

    losses = [step() for _ in range(10)]
    assert losses[-1] < losses[0]


def test_normalize_and_wer():
    assert normalize("  Bật ĐÈN, phòng khách!  ") == "bật đèn phòng khách"
    assert word_error_rate(["bật đèn"], ["Bật đèn."]) == 0
    assert word_error_rate(["bật đèn phòng khách"],
                           ["bật đèn phòng"]) == pytest.approx(25)
