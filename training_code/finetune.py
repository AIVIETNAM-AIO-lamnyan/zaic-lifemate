"""QLoRA fine-tuning for LifeMate tool-calling data."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from src.llm_phase.get_tool import get_available_tool
from src.llm_phase.prompt import build_messages_for_llm


class ConsoleLogCallback(TrainerCallback):
    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            print(
                f"[train] step {state.global_step}/{state.max_steps}: "
                f"{json.dumps(logs, ensure_ascii=False, default=str)}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    default_data = PROJECT_ROOT / "train+public_test" / "train" / "train.jsonl"
    parser = argparse.ArgumentParser(
        description="Fine-tune a QLoRA adapter from LifeMate train.jsonl."
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(os.environ.get("TRAIN_FILE", default_data)),
        help="JSONL training file with audio_text, context, history, and label.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Tool catalog JSON. Defaults to train_tools_catalog.json beside train-file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "finetune")),
        help="Directory for the final LoRA adapter and tokenizer.",
    )
    parser.add_argument(
        "-n",
        "-N",
        "--num-samples",
        dest="num_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to load (default: all samples).",
    )
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def load_catalog(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tool catalog: {path}")
    with path.open(encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("apps"), list):
        raise ValueError(f"Tool catalog không đúng định dạng: {path}")
    return catalog


def load_training_rows(
    train_file: Path,
    catalog: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
    max_samples: int | None = None,
) -> Tuple[List[Dict[str, str]], int]:
    """Build model-native chat-template strings for each labeled sample."""
    if not train_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy train dataset: {train_file}")

    rows: List[Dict[str, str]] = []
    tool_call_count = 0
    with train_file.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            if max_samples is not None and len(rows) >= max_samples:
                break
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON không hợp lệ tại {train_file}:{line_number}: {exc}"
                ) from exc
            if not isinstance(sample, dict):
                raise ValueError(f"Mỗi dòng trong {train_file} phải là một JSON object.")

            sample_id = sample.get("id", f"line {line_number}")
            audio_text = sample.get("audio_text")
            label = sample.get("label")
            if not isinstance(audio_text, str) or not audio_text.strip():
                raise ValueError(f"Mẫu {sample_id} thiếu audio_text hợp lệ.")
            if not isinstance(label, dict) or not isinstance(label.get("tool_calls"), list):
                raise ValueError(f"Mẫu {sample_id} thiếu label.tool_calls dạng danh sách.")

            context = sample.get("context") or {}
            if not isinstance(context, dict):
                raise ValueError(f"context của mẫu {sample_id} phải là JSON object.")
            tools = get_available_tool(sample, tool_catalog=catalog)
            available_names = {
                tool["function"]["name"]
                for tool in tools
                if isinstance(tool.get("function"), dict)
            }
            target_calls = []
            for index, call in enumerate(label["tool_calls"]):
                if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                    raise ValueError(f"Tool call {index} của mẫu {sample_id} thiếu name.")
                name = call["name"]
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"Arguments của {name} trong mẫu {sample_id} phải là object."
                    )
                if name not in available_names:
                    raise ValueError(
                        f"Tool {name} trong mẫu {sample_id} không có trong các tool "
                        "được phép theo installed_apps và catalog."
                    )
                target_calls.append({
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
            tool_call_count += len(target_calls)

            messages = build_messages_for_llm(
                audio_text=audio_text,
                context=context,
                history=sample.get("history"),
            )
            if target_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": target_calls})
            else:
                messages.append({
                    "role": "assistant",
                    "content": "Không có thao tác công cụ cần thực hiện.",
                })

            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"Không thể áp dụng chat template/tool schema cho mẫu {sample_id}. "
                    "Hãy dùng model có chat template hỗ trợ tool calling."
                ) from exc
            if len(token_ids) > max_seq_length:
                raise ValueError(
                    f"Mẫu {sample_id} có {len(token_ids)} tokens, vượt --max-seq-length "
                    f"({max_seq_length}); tăng giới hạn để không cắt mất tool call."
                )
            rows.append({"text": text})

    if not rows:
        raise ValueError(f"Dataset không có mẫu hợp lệ: {train_file}")
    return rows, tool_call_count


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if args.max_seq_length < 1:
        raise ValueError("--max-seq-length phải lớn hơn 0.")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("-N/--num-samples phải lớn hơn 0.")
    if args.logging_steps < 1:
        raise ValueError("--logging-steps phải lớn hơn 0.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs, batch-size và gradient-accumulation-steps phải lớn hơn 0.")
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA cần GPU CUDA; hãy chạy trong môi trường có NVIDIA GPU.")

    catalog_path = args.catalog or args.train_file.with_name("train_tools_catalog.json")
    catalog = load_catalog(catalog_path)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    set_seed(args.seed)

    logging.info(
        "Loading base model %s in 4-bit; max sequence length=%d.",
        args.model_name,
        args.max_seq_length,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=compute_dtype,
        load_in_4bit=True,
    )
    rows, tool_call_count = load_training_rows(
        args.train_file,
        catalog,
        tokenizer,
        args.max_seq_length,
        args.num_samples,
    )
    dataset = Dataset.from_list(rows)
    logging.info(
        "Loaded %d samples (%d tool calls); training %s in 4-bit. Output: %s",
        len(rows),
        tool_call_count,
        args.model_name,
        args.output_dir.resolve(),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    sft_config_parameters = inspect.signature(SFTConfig).parameters
    length_parameter = (
        "max_length" if "max_length" in sft_config_parameters else "max_seq_length"
    )
    steps_per_epoch = math.ceil(
        len(dataset) / (args.batch_size * args.gradient_accumulation_steps)
    )
    warmup_steps = math.ceil(steps_per_epoch * args.epochs * 0.05)
    training_config = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="no",
        optim="adamw_8bit",
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        gradient_checkpointing=True,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        report_to="none",
        **{length_parameter: args.max_seq_length},
    )
    trainer_kwargs = {"processing_class": tokenizer}
    if "processing_class" not in inspect.signature(SFTTrainer).parameters:
        trainer_kwargs = {"tokenizer": tokenizer}
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        callbacks=[ConsoleLogCallback()],
        **trainer_kwargs,
    )

    logging.info(
        "Starting training: samples=%d, epochs=%s, effective batch size=%d.",
        len(dataset),
        args.epochs,
        args.batch_size * args.gradient_accumulation_steps,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = args.output_dir / "README.md"
    original_readme = readme_path.read_bytes() if readme_path.is_file() else None
    try:
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        if original_readme is not None:
            readme_path.write_bytes(original_readme)
        elif readme_path.exists():
            readme_path.unlink()
    logging.info("QLoRA adapter and tokenizer exported to %s.", args.output_dir.resolve())


if __name__ == "__main__":
    main()
