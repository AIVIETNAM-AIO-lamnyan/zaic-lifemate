# Hướng dẫn chạy Test Pipeline (`tests/test_pipeline.py`)

Script dùng để kiểm thử luồng suy luận toàn trình của LLM Agent: từ tiếp nhận yêu cầu (`audio_text`), nạp bối cảnh (`context`), lọc schema công cụ (`tools`), tương tác đa bước với môi trường `zaic_tools.ToolEnv`, đối chiếu nhãn chuẩn (Gold Label) và ghi kết quả vào `result/test_llm.json`.

---

## 1. Chuẩn bị (Prerequisites)

Khởi động server vLLM trước khi chạy kiểm thử.

> [!IMPORTANT]
> `--max-model-len` **bắt buộc** phải ≥ **16384** (= `MAX_CONTEXT_LENGTH` trong `config.py`).  
> Dùng `2048` (mặc định của vLLM) sẽ gây lỗi **HTTP 400** ngay lượt đầu tiên vì prompt thực tế chứa ~50 tool schemas + system prompt + hội thoại đa bước.

```bash
# Cách 1: Dùng script có sẵn (tự đọc tham số từ config.py)
bash serve.sh
```

```bash
# Cách 2: Chạy trực tiếp qua vLLM CLI
# Các tham số bên dưới khớp với src/llm_phase/config.py:
#   MODEL_ID            = qwen3.5:4b   (env MODEL_NAME / MODEL_ID)
#   DEFAULT_VLLM_URL    = http://localhost:8000/v1/chat/completions  (env VLLM_URL)
#   MAX_CONTEXT_LENGTH  = 16384        (2^14 tokens, quy định BTC ZAC 2026)
#   TEMPERATURE         = 0.0

vllm serve <đường_dẫn_model_local> \
  --served-model-name qwen3.5:4b \
  --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.50
```

> [!TIP]
> Nếu VRAM hạn chế (ví dụ RTX 4050 6 GB), có thể giảm xuống `--max-model-len 8192` và `--gpu-memory-utilization 0.45`.  
> Đổi endpoint/model qua biến môi trường: `VLLM_URL`, `MODEL_NAME`, `VLLM_API_KEY`.

*(Mặc định script kết nối tới `http://localhost:8000/v1/chat/completions`. Có thể đổi qua biến môi trường `VLLM_URL`).*

---

## 2. Các cú pháp chạy phổ biến

### A. Chạy 1 mẫu cụ thể theo ID
```bash
# Truyền ID trực tiếp (hỗ trợ tự chuẩn hóa train_001 -> train_0001)
python tests/test_pipeline.py train_001

# Hoặc dùng cờ --id
python tests/test_pipeline.py --id train_0005
```

### B. Chạy N mẫu đầu tiên trong dataset
```bash
# Chạy 5 mẫu đầu tiên của split train
python tests/test_pipeline.py -N 5

# Chạy 3 mẫu đầu tiên của split public_test
python tests/test_pipeline.py --split public_test -N 3
```

### C. Chạy toàn bộ mẫu trong dataset (Train hoặc Public Test)
```bash
# Chạy TOÀN BỘ mẫu của split train
python tests/test_pipeline.py --all

# Chạy TOÀN BỘ mẫu của split public_test
python tests/test_pipeline.py --split public_test --all

# Chạy toàn bộ offline bằng Mock LLM (kiểm thử toàn bộ dataset không cần vLLM/GPU)
python tests/test_pipeline.py --all --mock

# Lưu kết quả toàn bộ vào file chỉ định riêng
python tests/test_pipeline.py --split public_test --all --output result/public_test_eval.json
```
*(Mẹo: Bạn cũng có thể dùng cờ `-N -1` hoặc `-N 0` thay cho `--all`).*

### D. Chạy chế độ Mock (không cần bật vLLM, phục vụ test offline / CI)
```bash
python tests/test_pipeline.py train_001 --mock
```

---

## 3. Bảng tham số CLI

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `sample_id` | ID mẫu cần test (ví dụ `train_001` hoặc `train_0001`) | `None` |
| `--id` | Cách viết khác để truyền ID mẫu | `None` |
| `-N`, `-n`, `--num-samples` | Số lượng mẫu chạy khi không chỉ định ID (truyền `-1` hoặc `0` để chạy toàn bộ) | `2` |
| `--all` | Chạy toàn bộ mẫu trong split được chọn | `False` |
| `--split` | Split dữ liệu (`train`, `public_test`) | `train` |
| `--source` | Trường chứa transcript | `audio_text` |
| `--output` | Đường dẫn file lưu kết quả | `result/test_llm.json` |
| `--mock` | Ép buộc chạy Mock LLM giả lập | `False` |
| `--no-fallback` | Không tự động chuyển sang Mock khi vLLM offline | `False` |

---

## 4. Kết quả đầu ra

Kết quả được chèn nối tiếp vào `result/test_llm.json` theo đúng cấu trúc:
```json
{
  "id": "train_0001",
  "user_request": "...",
  "thought": [{"turn": 1, "content": "..."}],
  "tool_calls": [{"name": "...", "arguments": {...}}],
  "tool_results": [...],
  "tool_expect": [...],
  "state_hash": "...",
  "state_hash_label": "...",
  "state_hash_result": true,
  "time_infer_second": 2.5
}
```
