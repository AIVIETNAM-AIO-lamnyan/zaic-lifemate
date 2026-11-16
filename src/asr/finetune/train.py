"""Fine-tune PhoWhisper-large bằng LoRA trên tập train của cuộc thi.

Quy trình:

1. Đọc mọi lượt nói có nhãn của tập train, chia train/dev 90/10 theo mẫu.
2. Đo WER của model gốc trên dev (mốc để so sánh).
3. Học LoRA; hết mỗi epoch đo WER trên dev, giữ adapter tốt nhất.
4. Gộp adapter tốt nhất vào trọng số gốc và lưu thành một model hoàn chỉnh,
   dùng được ngay bằng ``model="phowhisper_large_zac"``.

Chạy từ thư mục gốc repo::

    python -m src.asr.finetune.train                      # cấu hình mặc định
    python -m src.asr.finetune.train --epochs 1 --limit 200   # chạy thử nhanh
    python -m src.asr.finetune.train --decoder-only       # nhẹ hơn, nhanh hơn
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import FINETUNED_DIR
from .data import (SpeechCollator, SpeechDataset, load_utterances,
                   split_by_sample)
from .model import (add_lora, load_whisper, pick_device_dtype, transcribe,
                    word_error_rate)


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh; giá trị mặc định chọn cho GPU 8 GB."""
    parser = argparse.ArgumentParser(description="Fine-tune PhoWhisper (LoRA)")
    add = parser.add_argument
    add("--model", default="vinai/PhoWhisper-large", help="model gốc")
    add("--output", type=Path, default=FINETUNED_DIR,
        help="thư mục lưu model đã gộp LoRA")
    add("--epochs", type=int, default=3)
    add("--batch-size", type=int, default=4, help="số lượt nói mỗi lần chạy")
    add("--grad-accum", type=int, default=4,
        help="cộng dồn gradient: batch hiệu dụng = batch-size x grad-accum")
    add("--lr", type=float, default=1e-4, help="learning rate của LoRA")
    add("--warmup", type=float, default=0.1, help="tỉ lệ bước warmup")
    add("--rank", type=int, default=32, help="hạng LoRA")
    add("--decoder-only", action="store_true",
        help="chỉ gắn LoRA vào decoder (encoder đóng băng hoàn toàn)")
    add("--dev-ratio", type=float, default=0.1)
    add("--limit", type=int, help="chỉ dùng N lượt train đầu (chạy thử)")
    add("--eval-batch-size", type=int, default=8)
    add("--log-every", type=int, default=20, help="in loss sau mỗi N update")
    add("--seed", type=int, default=42)
    add("--device", default="auto")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, args,
                    epoch: int) -> float:
    """Một lượt qua toàn bộ dữ liệu train; trả về loss trung bình."""
    model.train()
    device, dtype = model.device, model.dtype
    total, started = 0.0, time.perf_counter()
    n_updates = math.ceil(len(loader) / args.grad_accum)

    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        batch["input_features"] = batch["input_features"].to(dtype)
        # Chia loss cho grad_accum để tổng gradient sau grad_accum batch
        # bằng gradient của một batch lớn duy nhất.
        loss = model(**batch).loss / args.grad_accum
        loss.backward()
        total += loss.item() * args.grad_accum

        # Cập nhật trọng số sau mỗi grad_accum batch (và ở batch cuối cùng).
        if step % args.grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            update = math.ceil(step / args.grad_accum)
            if update % args.log_every == 0 or update == n_updates:
                elapsed = time.perf_counter() - started
                eta = elapsed / update * (n_updates - update)
                print(f"  epoch {epoch} | update {update}/{n_updates} | "
                      f"loss {total / step:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | "
                      f"còn khoảng {eta / 60:.0f} phút", flush=True)
    return total / len(loader)


def main() -> None:
    """Chạy toàn bộ quy trình fine-tune."""
    from transformers import get_linear_schedule_with_warmup

    args = parse_args()
    torch.manual_seed(args.seed)
    device, dtype = pick_device_dtype(args.device)
    adapter_dir = args.output.with_name(args.output.name + "-lora")

    # 1. Dữ liệu ---------------------------------------------------------
    train_utts, dev_utts = split_by_sample(load_utterances("train"),
                                           args.dev_ratio, args.seed)
    train_utts = train_utts[:args.limit] if args.limit else train_utts
    print(f"Train: {len(train_utts)} lượt nói | dev: {len(dev_utts)} lượt "
          f"| thiết bị: {device} ({str(dtype).removeprefix('torch.')})")

    model, processor = load_whisper(args.model, device, dtype)
    dataset = SpeechDataset(train_utts, processor)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=SpeechCollator(model.config.decoder_start_token_id))
    references = [utt.text for utt in dev_utts]

    # 2. Mốc: WER của model gốc trên dev --------------------------------
    base_wer = word_error_rate(references, transcribe(
        model, processor, dev_utts, args.eval_batch_size))
    print(f"WER dev của model gốc: {base_wer:.2f} %")

    # 3. Gắn LoRA và học ------------------------------------------------
    model = add_lora(model, rank=args.rank, alpha=2 * args.rank,
                     decoder_only=args.decoder_only)
    model.print_trainable_parameters()
    model.config.use_cache = False          # cache chỉ dùng khi sinh text
    # Gradient checkpointing: không lưu activation của từng lớp mà tính lại
    # khi lan truyền ngược. Chậm hơn khoảng 30 % nhưng tiết kiệm nhiều VRAM.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n_updates = args.epochs * math.ceil(len(loader) / args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup * n_updates), n_updates)

    history, best_wer = [], base_wer
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, scheduler, args,
                               epoch)
        model.config.use_cache = True
        dev_wer = word_error_rate(references, transcribe(
            model, processor, dev_utts, args.eval_batch_size))
        model.config.use_cache = False
        history.append({"epoch": epoch, "loss": loss, "dev_wer": dev_wer})
        print(f"Epoch {epoch}: loss {loss:.4f} | WER dev {dev_wer:.2f} % "
              f"(gốc {base_wer:.2f} %)")
        # Chỉ giữ adapter khi WER dev tốt hơn mọi lần trước (kể cả model
        # gốc): tránh lưu một epoch đã bắt đầu overfit.
        if dev_wer < best_wer:
            best_wer = dev_wer
            model.save_pretrained(adapter_dir)
            print(f"  -> lưu adapter tốt nhất vào {adapter_dir}")

    summary = {"base_model": args.model, "base_dev_wer": base_wer,
               "best_dev_wer": best_wer, "epochs": history,
               "args": {k: str(v) for k, v in vars(args).items()},
               "dev_sample_ids": sorted({u.sample_id for u in dev_utts})}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # 4. Gộp LoRA vào trọng số gốc --------------------------------------
    if best_wer >= base_wer:
        print("Fine-tune không cải thiện WER dev: giữ model gốc, không gộp.")
        return
    merge_adapter(args.model, adapter_dir, args.output, processor)
    print(f"Xong. WER dev {base_wer:.2f} % -> {best_wer:.2f} %. "
          f"Model đã lưu ở {args.output}")


def merge_adapter(base_model: str, adapter_dir: Path, output: Path,
                  processor) -> None:
    """Gộp adapter vào model gốc và lưu thành một model Whisper bình thường.

    Việc gộp được làm ở float32 trên CPU: cộng ``B @ A`` (rất nhỏ) vào
    trọng số bfloat16 sẽ làm mất phần lớn cập nhật vì sai số làm tròn. Model
    được lưu ở float16, giữ nhiều chữ số thập phân hơn bfloat16.
    """
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    torch.cuda.empty_cache()
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.to(torch.float16).save_pretrained(output)
    processor.save_pretrained(output)


if __name__ == "__main__":
    main()
