# ASR mặc định: Gipformer fine-tune — hướng dẫn nhanh cho team

Interface **không đổi**, phần LLM không phải sửa gì:

```python
from src.asr import get_user_request_from_audio, load_asr_model

load_asr_model()          # nạp model một lần lúc khởi động (~1-2 giây)
current_req, history_req = get_user_request_from_audio("public_0002")
```

## 1. Có gì mới

| | Trước | Bây giờ (`gipformer_zac`, mặc định) |
|---|---|---|
| Model | PhoWhisper-large (1,54B) | Gipformer 1.5 fine-tune (**68M**) |
| WER (200 hội thoại validation, không dùng để train) | ~25 % | **6,6 %** |
| Nghe đúng tên riêng (danh bạ, bến xe, mã CK...) | – | **91 %** |
| Chạy trên | GPU | **CPU** (~0,1-0,4 giây / câu), không chiếm GPU của LLM |
| Ngân sách tham số còn cho LLM (tổng < 6B) | ~4,4B | gần như trọn **5B** (giới hạn LLM < 5B) |

- **Hotword theo mẫu:** các tên trong
  `<split>/<split>_environment/<id>/state.json` được đưa cho bộ giải mã để
  ưu tiên. Hàm tự đọc file này; không có file thì vẫn chạy bình thường.
- **Dạng text trả về:** chữ thường, không dấu câu, **số viết bằng chữ**, ví
  dụ `"đặt lịch họp lúc ba giờ chiều"`, `"chuyển kênh vtv ba"`. Prompt của
  LLM nên lưu ý điều này khi trích tham số (giờ, số tiền, mã kênh).

## 2. Cài đặt (môi trường riêng, không cần torch / GPU)

```bash
python -m venv .venv-sherpa
source .venv-sherpa/bin/activate        # Windows: .venv-sherpa\Scripts\activate
pip install -r src/asr/requirements-sherpa.txt
```

## 3. Lấy model (KHÔNG có trong repo vì repo public)

Model là một thư mục ONNX (~280 MB): `encoder.onnx`, `decoder.onnx`,
`joiner.onnx`, `tokens.txt`, `bpe.model`. Chọn một trong hai cách:

- **Từ Hugging Face (repo riêng tư):** xin quyền truy cập, đăng nhập một lần
  rồi trỏ biến môi trường tới repo; module tự tải về lần đầu:

  ```bash
  hf auth login                                          # dán token HF
  export ZAC_GIPFORMER_MODEL=daycuachuoithucan/gipformer1.5-zac-ft06
  ```

- **Thư mục trên máy:** đặt model vào
  `checkpoints/asr/gipformer1.5-zac-ft06/` (đường dẫn mặc định, đã nằm trong
  `.gitignore`), hoặc trỏ `ZAC_GIPFORMER_MODEL` tới thư mục khác.

Dữ liệu cuộc thi đặt như cũ (`train+public_test/train+public_test/` hoặc
`ZAC_DATA_ROOT`).

## 4. Dùng

```python
from src.asr import get_user_request_from_audio, list_sample_ids, load_asr_model

load_asr_model()
for sample_id in list_sample_ids("public_test"):
    current_req, history_req = get_user_request_from_audio(sample_id)

# Private test: tiền tố id chưa biết trước -> truyền split
get_user_request_from_audio(sample_id, split="private_test")

# Dùng lại model cũ nếu cần so sánh
get_user_request_from_audio("public_0002", model="phowhisper_large")
```

Giải mã sẵn cả một split ra JSON (để thử prompt LLM mà không phải chạy ASR):

```bash
python -m src.asr.transcribe_split --split public_test
# -> results/asr/public_test_gipformer_zac.json
#    {"public_0002": {"current_req": "...", "history_req": [...]}, ...}
```

File JSON này chứa transcript của dữ liệu cuộc thi: **không commit lên repo**.

## 5. Kiểm thử

```bash
pytest src/asr/tests -v          # unit test, không cần model / GPU / data
```

## 6. Nguồn gốc model

Fine-tune toàn bộ Gipformer 1.5 (`g-group-ai-lab/gipformer1.5-68M-rnnt`, MIT)
trên tập train (bỏ 200 hội thoại validation), 3 epoch, AdamW 2e-5 warmup +
cosine, nhiễu trắng (SNR 10-30 dB, 30 %) + SpecAugment; xuất ONNX bằng icefall.
Giải mã: modified beam search 4, hotword điểm 1,5. Code train và nhật ký mọi
thí nghiệm nằm ở repo cá nhân `gipformer_finetune` (`DECISIONS.md`).
