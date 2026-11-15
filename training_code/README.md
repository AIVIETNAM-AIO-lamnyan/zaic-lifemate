# Hướng dẫn Reproduce Training Code

Tài liệu này hướng dẫn BTC tái hiện (reproduce) lại quá trình huấn luyện và chuẩn bị dữ liệu cho giải pháp LifeMate (ZAC 2026).

## Fine-tune QLoRA

`finetune.py` đọc dữ liệu JSONL có `audio_text`, `context`, `history` và `label.tool_calls`, dùng tool catalog để tạo hội thoại theo chat template của model, rồi fine-tune adapter QLoRA 4-bit. Mặc định dùng `Qwen/Qwen2.5-0.5B-Instruct`, dataset `train+public_test/train/train.jsonl`, và xuất adapter cùng tokenizer vào `saved_models/` trong thư mục gốc repo.

Chạy trong môi trường huấn luyện đã cài Unsloth, TRL, PEFT, bitsandbytes, datasets và Accelerate:

```bash
python training_code/finetune.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-file train+public_test/train/train.jsonl \
  --catalog train+public_test/train/train_tools_catalog.json \
  -N 128 \
  --output-dir saved_models
```

Có thể dùng `-n`, `-N` hoặc `--num-samples` để giới hạn số mẫu đầu tiên được đọc; mặc định dùng toàn bộ dataset. Log loss được ghi ở mỗi bước optimizer; đổi tần suất bằng `--logging-steps`. Có thể đổi model, dataset, catalog, độ dài chuỗi và nơi lưu bằng các tham số CLI tương ứng. Thư mục đầu ra chứa LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) và tokenizer; cần dùng kèm base model đã fine-tune khi suy luận.

## 1. Môi trường & Hạt giống ngẫu nhiên (Fixed Seed)
- Seed: `42`
- Cố định seed bằng hàm `seed_everything(42)` trên cả PyTorch, CUDA, NumPy và Python random.

## 2. Dữ liệu & Mô hình sử dụng
- **Dữ liệu huấn luyện**: Sử dụng bộ dữ liệu chính thức `train` do BTC cung cấp.
- **ASR Base Model**: `vinai/PhoWhisper-large` (Hugging Face Hub URL: `https://huggingface.co/vinai/PhoWhisper-large`).
- **LLM Agent Base Model**: `Qwen/Qwen2.5-3B-Instruct` (Hugging Face Hub URL: `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`).

## 3. Quy trình Reproduce
1. Chuẩn bị môi trường: `pip install -r /code/requirements.txt`.
2. Chạy kịch bản đánh giá baseline: `python3 /code/predict.py`.

