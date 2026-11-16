# LifeMate — Zalo AI Challenge 2026 (Track 1)

> **LifeMate**: Trợ lý AI cá nhân tại Việt Nam, tiếp nhận khẩu lệnh tiếng Việt qua âm thanh, tự động tra cứu ngữ cảnh và thực hiện chuỗi thao tác chính xác trên các ứng dụng di động (`ToolEnv`).

---

## 1. Kiến trúc Tổng quan (Architecture)

```text
Khẩu lệnh Audio (.wav) ──► [ASR Module: PhoWhisper] ──► Văn bản tiếng Việt
                                                              │
                                                              ▼
Kết quả nộp: submission.jsonl ◄── [ToolEnv Runtime] ◄── [ReAct Agent LLM: Qwen]
```

* **ASR**: `vinai/PhoWhisper-large` (nhận dạng âm thanh tiếng Việt 16kHz mono).
* **LLM**: `Qwen2.5-3B-Instruct` / `Qwen3.5-4B` (chạy qua vLLM endpoint OpenAI-compatible).
* **Môi trường công cụ**: `zaic_tools.ToolEnv` (mô phỏng ứng dụng local theo quy chuẩn ZAC 2026).
* **Định dạng nộp bài**: `result/submission.jsonl` (gồm `id`, `tool_calls`, `state_hash`).

---

## 2. Chuẩn bị (Prerequisites)

Trước khi chạy suy luận, khởi động server **vLLM** phục vụ cho LLM Agent:

```bash
# Cách 1 (khuyến nghị): dùng script tự động nạp cấu hình từ src/llm_phase/config.py
bash serve.sh
```

`serve.sh` thử model chính `Qwen/Qwen3.5-4B` trước (cần vLLM >= 0.17.0, ~10 GB VRAM khi chạy BF16). Nếu model này không khởi động được (thiếu VRAM, vLLM quá cũ, thiếu `trust-remote-code`...), script tự chuyển sang model ban đầu `Qwen/Qwen2.5-0.5B-Instruct`. Cả hai model đều được serve dưới cùng tên `qwen3.5:4b`, nên phía client không cần đổi cấu hình.

```bash
# Đổi model chính / model dự phòng
VLLM_MODEL_PATH=Qwen/Qwen3.5-4B VLLM_FALLBACK_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct bash serve.sh

# Tắt fallback để nhìn thấy lỗi thật của model chính
VLLM_NO_FALLBACK=1 bash serve.sh
```

```bash
# Cách 2: Chạy trực tiếp qua lệnh vLLM CLI với Qwen3.5-4B
# (LƯU Ý: --max-model-len bắt buộc >= 8192 hoặc 16384 để chứa đủ ~50 tool schemas)
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5:4b \
  --port 8000 \
  --max-model-len 16384 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.90
```

Với model cũ `Qwen/Qwen2.5-0.5B-Instruct`, bỏ `--trust-remote-code` và dùng `--tool-call-parser hermes`, `--gpu-memory-utilization 0.50`.

*(Mặc định agent kết nối tới `http://localhost:8000/v1/chat/completions`. Có thể tùy biến qua biến môi trường `VLLM_URL`).*

Nếu endpoint trả HTTP 404 vì không có model `qwen3.5:4b` (ví dụ chạy `vllm serve` không đặt `--served-model-name`), client tự thử model dự phòng `FALLBACK_MODEL_ID` (mặc định `Qwen/Qwen2.5-0.5B-Instruct`), in cảnh báo và ghi nhớ model chạy được cho các lượt sau. Ghi đè bằng biến môi trường `MODEL_NAME` / `FALLBACK_MODEL_NAME`.

---

## 3. Fine-tune QLoRA

Script [training_code/finetune.py](./training_code/finetune.py) huấn luyện QLoRA 4-bit từ `train.jsonl`, dùng prompt và tool schema của `src/llm_phase`. Môi trường Conda `zaic-llm01` cần có sẵn các thư viện huấn luyện (Unsloth, Transformers, TRL, PEFT, bitsandbytes, datasets và Accelerate).

Chạy từ thư mục gốc repo trên máy có GPU CUDA:

```powershell
conda activate zaic-llm01

# Smoke test với 16 mẫu đầu; dùng -u để log hiển thị trực tiếp
python -u training_code\finetune.py -N 16

# Chạy với 128 mẫu đầu
python -u training_code\finetune.py -N 128
```

Mặc định model là `Qwen/Qwen2.5-0.5B-Instruct`; dataset và tool catalog lần lượt là `train+public_test\train\train.jsonl` và `train+public_test\train\train_tools_catalog.json`. Adapter, cấu hình và tokenizer được xuất vào `finetune\` trong repo; thư mục `saved_models\` được dành riêng để sử dụng sau và không bị ghi đè bởi lần fine-tune mặc định. Cần giữ base model tương ứng để nạp adapter khi suy luận.

Các tham số thường dùng:

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `--model-name` | Hugging Face model ID hoặc đường dẫn model local | `Qwen/Qwen2.5-0.5B-Instruct` |
| `--train-file` | File train JSONL | `train+public_test\train\train.jsonl` |
| `--catalog` | File tool catalog; mặc định lấy cạnh train file | `train_tools_catalog.json` cạnh dataset |
| `-n`, `-N`, `--num-samples` | Số mẫu đầu tiên dùng để train | Toàn bộ dataset |
| `--max-seq-length` | Độ dài token tối đa; mẫu vượt giới hạn sẽ báo lỗi thay vì bị cắt | `16384` |
| `--epochs` | Số epoch | `1` |
| `--batch-size` | Batch size trên mỗi GPU | `1` |
| `--gradient-accumulation-steps` | Số bước tích lũy gradient | `8` |
| `--logging-steps` | Ghi log loss sau mỗi số bước optimizer này | `1` |
| `--output-dir` | Thư mục xuất adapter và tokenizer | `finetune\` |

Ví dụ chỉ định rõ dataset, model, log mỗi bước và output:

```powershell
python -u training_code\finetune.py `
  --model-name Qwen/Qwen2.5-0.5B-Instruct `
  --train-file train+public_test\train\train.jsonl `
  --catalog train+public_test\train\train_tools_catalog.json `
  --num-samples 128 `
  --logging-steps 1 `
  --output-dir finetune
```

---

## 4. Hướng dẫn chạy `predict.py` cho các trường hợp

File `predict.py` là script suy luận chính thức (official prediction script), tự động xử lý trọn vẹn từ dữ liệu thô đến file nộp bài `submission.jsonl`.

### Trường hợp 1: Chạy chuẩn thi đấu BTC (Mặc định / Docker submission)
Tự động phát hiện dữ liệu tại `/data` (hoặc thư mục local), tự động chọn split ưu tiên (`private_test` ➔ `public_test` ➔ `train`) và chạy toàn bộ dataset:
```bash
python predict.py
```
*Kết quả nộp bài được ghi tự động vào `result/submission.jsonl` (hoặc `/result/submission.jsonl` trong container).*

---

### Trường hợp 2: Chạy trên toàn bộ một tập dữ liệu cụ thể (`train` hoặc `public_test`)
```bash
# Chạy toàn bộ tập public_test (chấm điểm thử nghiệm leaderboard)
python predict.py --split public_test

# Chạy toàn bộ tập train (đối chiếu nhãn chuẩn)
python predict.py --split train
```

---

### Trường hợp 3: Chạy thử nghiệm N mẫu đầu tiên
Hữu ích khi cần đánh giá nhanh chất lượng mô hình mà không cần chờ chạy hết hàng trăm mẫu:
```bash
# Chạy 5 mẫu đầu tiên của split public_test
python predict.py --split public_test -N 5

# Chạy 2 mẫu đầu tiên của split train
python predict.py --split train -N 2
```

---

### Trường hợp 4: Chạy đúng 1 mẫu cụ thể theo ID
Dùng để gỡ lỗi (debug) cho một mẫu bị sai hoặc kiểm tra hành vi cụ thể của Agent:
```bash
# Truyền ID trực tiếp (hỗ trợ cả train_001 hoặc train_0001)
python predict.py train_0001 --split train

# Hoặc dùng cờ --id
python predict.py --id public_0002 --split public_test
```

---

### Trường hợp 5: Chạy chế độ Mock offline (Không cần bật GPU / vLLM / ASR)
Phục vụ kiểm thử luồng code, CI/CD hoặc kiểm tra schema môi trường khi vLLM chưa khởi động:
```bash
# Chạy thử 2 mẫu với Mock LLM
python predict.py -N 2 --mock

# Chạy 1 mẫu cụ thể với Mock LLM
python predict.py train_0001 --split train --mock
```

---

### Trường hợp 6: Xem log chi tiết từng bước (Verbose Mode)
Hiển thị toàn bộ suy luận (Thought), các tool call do LLM sinh ra, và kết quả thực thi trong `ToolEnv`:
```bash
python predict.py train_0001 --split train -v
```

---

### Trường hợp 7: Tùy biến đường dẫn đầu ra và thư mục dữ liệu
```bash
# Lưu file nộp bài ra đường dẫn riêng biệt
python predict.py --split public_test --output result/public_test_submission.jsonl

# Chỉ định thư mục data root khác
python predict.py --root /path/to/custom_data --split public_test
```

---

## 5. Bảng tham số dòng lệnh của `predict.py`

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `sample_id` | ID mẫu cụ thể cần chạy (ví dụ `train_0001`, `public_0002`) | `None` (chạy batch) |
| `--id` | Cách viết khác để truyền ID mẫu | `None` |
| `-N`, `-n`, `--num-samples` | Số lượng mẫu chạy khi không chỉ định ID | Toàn bộ split |
| `--split` | Split dữ liệu (`private_test`, `public_test`, `train`) | Tự động phát hiện |
| `--root` | Đường dẫn thư mục chứa các split dữ liệu | `/data` hoặc local `train+public_test` |
| `--output` | Đường dẫn file kết quả xuất ra | `result/submission.jsonl` |
| `--mock` | Ép buộc chạy Mock LLM giả lập | `False` |
| `--no-fallback` | Vô hiệu hóa tự động fallback sang Mock khi vLLM offline | `False` |
| `-v`, `--verbose` | In log chi tiết từng bước suy luận và tool call | `False` |

---

## 6. Định dạng kết quả và Kiểm tra tính hợp lệ (Validation)

### Định dạng `submission.jsonl`
Mỗi dòng là một JSON object độc lập UTF-8, tuân thủ nghiêm ngặt 3 trường quy định của Ban tổ chức ZAC 2026:
```json
{"id": "public_0039", "tool_calls": [{"name": "alarm.list", "arguments": {}}, {"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], "state_hash": "461fcd40625c9968434b56ee113c0953f332f0c1189a3d385575e1dabec263f3"}
```

### Lệnh kiểm tra tính hợp lệ schema của bài nộp
Sử dụng chính validator chuẩn của `zaic_tools` để xác thực:
```bash
# Kiểm tra file submission.jsonl cho split public_test
python -c "import sys; sys.path.insert(0, 'train+public_test/public_test'); from zaic_tools import catalog_path, schema_validate, ToolEnv; import json; cat = json.load(open(catalog_path('public_test', 'train+public_test'), encoding='utf-8')); lines = [json.loads(l) for l in open('result/submission.jsonl', encoding='utf-8') if l.strip()]; errors = [r for line in lines for r in schema_validate(line['tool_calls'], cat, ToolEnv('public_test', line['id'], 'train+public_test').allowed) if not r['ok']]; print(f'{len(lines)} mẫu kiểm tra, {len(errors)} lỗi schema')"
```
Nếu màn hình báo `0 lỗi schema` nghĩa là file nộp bài hoàn toàn hợp lệ.

---

## 7. Chấm điểm bài nộp bằng `result/validate.py`

Script chấm điểm theo đúng công thức trong `docs/ZaloAI.md`:

- `state_hash` của bài nộp phải trùng hash vàng, nếu không mẫu đó 0 điểm.
- Ghép tool call: cùng tên tool, tham số non-fuzzy giống hệt, tham số fuzzy (`x-match: "fuzzy"` trong catalog) có token-F1 > 0.6 sau lowercase + Unicode NFC. Thứ tự call không quan trọng; mỗi call chỉ khớp tối đa một call vàng.
- Gold rỗng (out-of-scope): chỉ được 1 điểm khi submission cũng không gọi tool nào.
- Điểm từng mẫu: `2PR/(P+R)` với `P = M/(M+U)`, `R = M/|G|`; điểm cuối `100/N × Σ Score`.

Script luôn chấm trên nhãn vàng của `train` — split duy nhất có `label.tool_calls`. Không thể chấm `public_test`/`private_test` khi chưa có nhãn.

### Cách chạy

```bash
# Mặc định: chấm result/submission.jsonl (cùng thư mục với validate.py)
python result/validate.py

# Chấm một file khác trong cùng thư mục result/
python result/validate.py my_submission.jsonl
```

Kết quả:

```text
Score: 72.35/100
```

Điểm là trung bình cộng điểm các mẫu có trong file, thang 100. Các mẫu train không có trong file không được tính, nên muốn ước lượng điểm leaderboard thì file phải chứa đủ mẫu.

### Gọi từ Python

```python
from result.validate import score_submission

score = score_submission()                     # result/submission.jsonl
score = score_submission("my_submission.jsonl")
print(score)                                   # ví dụ 72.35
```


