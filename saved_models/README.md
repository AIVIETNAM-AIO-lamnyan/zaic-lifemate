# Thư mục saved_models

Thư mục này dùng để lưu trữ weights của các mô hình phục vụ suy luận hoàn toàn ngoại tuyến (Offline Inference) bên trong Docker container theo quy định của BTC:

1. **Mô hình ASR (Nhận dạng giọng nói)**:
   - Mô hình được chọn: `vinai/PhoWhisper-large` (hoặc `PhoWhisper-medium` / `Whisper`).
   - Đặt toàn bộ checkpoint Hugging Face (hoặc safetensors/bin, tokenizer, config) tại đây.

2. **Mô hình LLM Agent (Tác nhân gọi công cụ)**:
   - Mô hình được chọn: `Qwen/Qwen2.5-3B-Instruct` hoặc `qwen3.5:4b` (dưới dạng safetensors hoặc GGUF).

> **Lưu ý**: Khi đóng gói Docker image nộp BTC, toàn bộ weights cần được copy sẵn vào trong image (không được tải online từ Hugging Face lúc container chạy).

