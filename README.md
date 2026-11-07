# Giải pháp LifeMate — Zalo AI Challenge 2026

## 1. Mô tả Ý tưởng & Kiến trúc Giải pháp (Solution Overview)
Hệ thống giải quyết trọn vẹn bài toán chuyển đổi khẩu lệnh tiếng Việt thành chuỗi hành động trên môi trường ứng dụng:

```text
Audio (.wav) -> [ASR Module] -> Văn bản -> [ReAct Agent LLM] -> [ToolEnv] -> submission.jsonl
```

## 2. Thông số Kỹ thuật Mô hình (Model Specs)
- **Mô hình ASR (Nhận dạng tiếng nói)**:
  - Tên mô hình: `vinai/PhoWhisper-large` (hoặc `PhoWhisper-medium`).
  - Số tham số: ~1.54B (PhoWhisper-large) hoặc ~0.76B (PhoWhisper-medium).
  - Ngôn ngữ: Tiếng Việt, hỗ trợ âm thanh 16kHz mono.
- **Mô hình LLM Agent (Tác nhân suy luận & gọi công cụ)**:
  - Tên mô hình: `Qwen2.5-3B-Instruct` / `Qwen3.5-4B`.
  - Số tham số: ~3.8B - 4.0B tham số (< 5.0B theo quy định BTC).
  - Context Length: 16,384 tokens (< 16K tokens).
- **Tổng số tham số hệ thống**: ~5.3B tham số (< 6.0B theo quy định BTC).

## 3. Cơ chế Vòng lặp Agent (Agentic ReAct Loop)
1. **Dynamic Tool Filtering**: Chỉ nạp schema của các tool thuộc `installed_apps` của người dùng + meta-tool `sys.ask_missing_info`.
2. **Read-Before-Write**: Ép buộc tra cứu thực thể thật (ID danh bạ, ID sự kiện, thiết bị) trước khi thực hiện hành động ghi dữ liệu.
3. **Schema Pre-Validation Guardrail**: Sử dụng `env.validate()` để tự kiểm tra đối số trước khi gọi `env.call()`. Nếu sai schema, đưa lỗi ngược lại cho LLM tự sửa mà không làm bẩn nhật ký gọi tool.
4. **Xử lý thiếu thông tin & Hủy yêu cầu**: Tự động gọi `sys.ask_missing_info` khi thiếu tham số bắt buộc và không gọi tool ghi khi người dùng đổi ý.

## 4. Cách chạy suy luận (Inference Execution)
- Trong môi trường Docker:
  ```bash
  bash /code/predict.sh
  ```
- Kết quả được ghi tự động vào: `/result/submission.jsonl`.

