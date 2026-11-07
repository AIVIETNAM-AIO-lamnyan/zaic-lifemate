# Hướng dẫn Reproduce Training Code

Tài liệu này hướng dẫn BTC tái hiện (reproduce) lại quá trình huấn luyện và chuẩn bị dữ liệu cho giải pháp LifeMate (ZAC 2026).

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

