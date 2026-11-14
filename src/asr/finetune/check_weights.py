"""Kiểm tra weights PhoWhisper có khớp kiến trúc Whisper và học tiếp được.

PhoWhisper được VinAI fine-tune từ Whisper, nên về lý thuyết dùng chung
kiến trúc. Script kiểm tra điều đó bằng 4 bước, từ "nhìn" tới "chạy thật":

1. **Cấu hình** (``config.json``): số lớp, số chiều, số head, kích thước bộ
   từ vựng, số kênh mel... so với Whisper chính chủ của OpenAI.
2. **Nạp weights** vào lớp ``WhisperForConditionalGeneration``: không được
   có tensor thiếu (missing), thừa (unexpected) hay lệch kích thước
   (mismatched).
3. **Đối chiếu từng tensor** với bộ khung Whisper tương ứng: cùng tên, cùng
   kích thước.
4. **Học thử**: gắn LoRA và chạy vài bước tối ưu trên cùng một batch nhỏ.
   Loss phải giảm và gradient phải hữu hạn, tức là đường lan truyền ngược
   thông suốt.

Chạy từ thư mục gốc repo::

    python -m src.asr.finetune.check_weights
    python -m src.asr.finetune.check_weights --model vinai/PhoWhisper-medium

Báo cáo được ghi vào ``results/asr/weights_check_<model>.md``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import torch

from ..config import PROJECT_ROOT
from .data import SpeechCollator, SpeechDataset, load_utterances
from .model import add_lora, load_whisper, pick_device_dtype

#: Các trường cấu hình quyết định hình dạng của tensor.
ARCH_FIELDS = ("d_model", "encoder_layers", "decoder_layers",
               "encoder_attention_heads", "decoder_attention_heads",
               "encoder_ffn_dim", "decoder_ffn_dim", "vocab_size",
               "num_mel_bins", "max_source_positions", "max_target_positions")

REPORTS = PROJECT_ROOT / "results" / "asr"           # đã có trong .gitignore


@dataclass
class Check:
    """Kết quả của một bước kiểm tra: đạt hay không và bảng chi tiết."""

    title: str
    passed: bool
    rows: list[list[str]] = field(default_factory=list)
    header: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        """Định dạng Markdown: tiêu đề, kết luận và bảng chi tiết."""
        status = "ĐẠT" if self.passed else "**KHÔNG ĐẠT**"
        table = "\n".join("| " + " | ".join(row) + " |" for row in
                          [self.header, ["---"] * len(self.header),
                           *self.rows]) if self.rows else ""
        return f"## {self.title}: {status}\n\n{table}\n"


def check_config(model_id: str, references: list[str]) -> tuple[Check, str]:
    """Bước 1: so cấu hình với các bản Whisper gốc, chọn bản giống nhất."""
    from transformers import AutoConfig

    configs = {name: AutoConfig.from_pretrained(name).to_dict()
               for name in [model_id, *references]}
    mine = configs[model_id]
    # Bản gốc giống nhất là bản có nhiều trường trùng khớp nhất.
    best = max(references, key=lambda ref: sum(
        mine.get(f) == configs[ref].get(f) for f in ARCH_FIELDS))
    rows = [[f, str(mine.get(f)), *(str(configs[r].get(f))
                                    for r in references),
             "có" if mine.get(f) == configs[best].get(f) else "**không**"]
            for f in ARCH_FIELDS]
    passed = all(row[-1] == "có" for row in rows)
    return Check(f"1. Cấu hình (gần nhất: {best})", passed, rows,
                 ["trường", model_id, *references, f"khớp {best}"]), best


def check_loading(model_id: str) -> tuple[Check, object]:
    """Bước 2: nạp weights vào lớp Whisper chuẩn, liệt kê tensor bất thường.

    Model được nạp ở float32 trên CPU: chỉ cần đọc tên và kích thước tensor,
    chưa cần GPU.
    """
    from transformers import WhisperForConditionalGeneration

    model, info = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, output_loading_info=True)
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys")
    rows = [[key, str(len(info.get(key, []))),
             ", ".join(map(str, list(info.get(key, []))[:3])) or "-"]
            for key in keys]
    passed = all(row[1] == "0" for row in rows)
    return Check("2. Nạp vào WhisperForConditionalGeneration", passed, rows,
                 ["loại", "số tensor", "ví dụ"]), model


def check_shapes(model, reference: str) -> Check:
    """Bước 3: so tên và kích thước từng tensor với bộ khung Whisper gốc.

    Bộ khung được dựng trên thiết bị ``meta``: chỉ có hình dạng, không cấp
    phát bộ nhớ, không tải weights của OpenAI.
    """
    from transformers import AutoConfig, WhisperForConditionalGeneration

    with torch.device("meta"):
        skeleton = WhisperForConditionalGeneration(
            AutoConfig.from_pretrained(reference))
    mine = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    ref = {k: tuple(v.shape) for k, v in skeleton.state_dict().items()}
    mismatched = [k for k in mine.keys() & ref.keys() if mine[k] != ref[k]]
    rows = [["số tensor (PhoWhisper)", str(len(mine))],
            [f"số tensor ({reference})", str(len(ref))],
            ["chỉ có ở PhoWhisper", str(len(mine.keys() - ref.keys()))],
            [f"chỉ có ở {reference}", str(len(ref.keys() - mine.keys()))],
            ["lệch kích thước", str(len(mismatched))],
            ["tổng tham số",
             f"{sum(p.numel() for p in model.parameters()):,}"]]
    passed = all(row[1] == "0" for row in rows[2:5])
    return Check("3. Tên và kích thước từng tensor", passed, rows,
                 ["kiểm tra", "giá trị"])


def check_training(model_id: str, n_clips: int, steps: int, lr: float,
                   device: str) -> Check:
    """Bước 4: học thử LoRA trên một batch cố định, loss phải giảm.

    Learning rate cố ý đặt cao: mục tiêu là "học thuộc" một batch nhỏ, chứng
    minh gradient chảy qua toàn bộ mạng tới các tham số LoRA.
    """
    device, dtype = pick_device_dtype(device)
    model, processor = load_whisper(model_id, device, dtype)
    mel_ok = processor.feature_extractor.feature_size == \
        model.config.num_mel_bins
    model = add_lora(model)
    model.config.use_cache = False
    model.train()

    utterances = load_utterances("train")[:n_clips]
    collate = SpeechCollator(model.config.decoder_start_token_id)
    batch = collate([SpeechDataset(utterances, processor)[i]
                     for i in range(len(utterances))])
    batch = {"input_features": batch["input_features"].to(device, dtype),
             "labels": batch["labels"].to(device)}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def step() -> tuple[float, bool]:
        """Một bước tối ưu; trả về loss và gradient có hữu hạn không."""
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        finite = all(torch.isfinite(p.grad).all() for p in params
                     if p.grad is not None)
        optimizer.step()
        return loss.item(), finite

    results = [step() for _ in range(steps)]
    losses = [loss for loss, _ in results]
    with_grad = sum(p.grad is not None for p in params)
    peak = (f"{torch.cuda.max_memory_allocated() / 2**30:.2f}"
            if device.startswith("cuda") else "n/a (CPU)")
    rows = [["số kênh mel (processor / model)",
             f"{processor.feature_extractor.feature_size} / "
             f"{model.config.num_mel_bins}"],
            ["tensor LoRA có gradient", f"{with_grad}/{len(params)}"],
            ["gradient hữu hạn ở mọi bước", str(all(f for _, f in results))],
            ["loss qua từng bước", " → ".join(f"{x:.3f}" for x in losses)],
            ["VRAM đỉnh (GB)", peak], ["thiết bị", f"{device}, {dtype}"]]
    passed = (mel_ok and with_grad == len(params) and losses[-1] < losses[0]
              and all(f for _, f in results))
    return Check(f"4. Học thử LoRA ({n_clips} lượt nói, {steps} bước)",
                 passed, rows, ["kiểm tra", "giá trị"])


def main() -> None:
    """Chạy 4 bước kiểm tra, in và lưu báo cáo."""
    parser = argparse.ArgumentParser(description="Kiểm tra weights PhoWhisper")
    parser.add_argument("--model", default="vinai/PhoWhisper-large")
    parser.add_argument("--references", nargs="+",
                        default=["openai/whisper-large-v2",
                                 "openai/whisper-large-v3"],
                        help="bản Whisper gốc để so (chỉ tải config.json)")
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config_check, reference = check_config(args.model, args.references)
    loading_check, model = check_loading(args.model)
    shape_check = check_shapes(model, reference)
    del model                                # giải phóng RAM trước bước 4
    checks = [config_check, loading_check, shape_check,
              check_training(args.model, args.clips, args.steps, args.lr,
                             args.device)]

    verdict = ("KHỚP và HỌC TIẾP ĐƯỢC" if all(c.passed for c in checks)
               else "CÓ VẤN ĐỀ, xem các bước không đạt")
    report = (f"# Kiểm tra weights: {args.model}\n\n"
              f"**Kết luận: {verdict}.**\n\n"
              + "\n".join(check.markdown() for check in checks))
    REPORTS.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w-]+", "_", args.model)       # vinai/X -> vinai_X
    path = REPORTS / f"weights_check_{slug}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Đã lưu báo cáo: {path}")


if __name__ == "__main__":
    main()
