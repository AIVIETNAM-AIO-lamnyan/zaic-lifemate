# `src/asr`: Nhận dạng giọng nói (ASR)

Module chuyển **lượt nói tiếng Việt của người dùng** thành **text** để đưa vào
LLM agent. Đây là bước đầu tiên của pipeline LifeMate:

```text
audio (WAV 16 kHz) ──► src/asr ──► (current_req, history_req) ──► LLM agent ──► tool calls
```

Toàn bộ phần ASR nằm trong thư mục này và không phụ thuộc vào module nào khác
của repo.

---

## Mục lục

1. [Interface](#1-interface)
2. [Cài đặt](#2-cài-đặt)
3. [Dữ liệu](#3-dữ-liệu)
4. [Cách dùng](#4-cách-dùng)
5. [Chạy thử và kiểm thử](#5-chạy-thử-và-kiểm-thử)
6. [Lựa chọn model](#6-lựa-chọn-model)
7. [Thiết kế](#7-thiết-kế)
8. [Cấu trúc thư mục](#8-cấu-trúc-thư-mục)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)

---

## 1. Interface

Phần LLM chỉ cần **một hàm**:

```python
from src.asr import get_user_request_from_audio

current_req, history_req = get_user_request_from_audio("train_0001")
```

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `current_req` | `str` | Text của **yêu cầu cuối** (`<id>.wav`), là lượt được chấm điểm |
| `history_req` | `list[str]` | Text các **lượt user trước đó** (`<id>_u1.wav` … `<id>_un.wav`), cũ nhất trước; `[]` nếu không có history |

`history_req` chỉ chứa lượt **user**. Lượt **assistant** trong `history` vốn
đã là text trong file jsonl, phần LLM tự lấy từ `sample["history"]`.

Ví dụ một mẫu có history (text chỉ để minh hoạ):

```python
>>> get_user_request_from_audio("public_0006")
('tắt nó đi', ['bật đèn phòng khách', 'chỉnh độ sáng năm mươi phần trăm'])
```

### Tham số

```python
get_user_request_from_audio(
    sample_id: str,                 # "train_0001", "public_0002", ...
    split: str | None = None,       # suy ra từ tiền tố id; BẮT BUỘC với private test
    data_root: str | Path | None = None,   # mặc định: config.DATA_ROOT
    model: str = "phowhisper_large",
    device: str = "auto",           # "auto" | "cuda:0" | "cpu"
) -> tuple[str, list[str]]
```

### Các hàm phụ

| Hàm | Mục đích |
|---|---|
| `load_asr_model(model, device)` | Nạp model trước (gọi lúc khởi động để mẫu đầu tiên không bị chậm) |
| `unload_asr_model()` | Giải phóng GPU, ví dụ để nhường bộ nhớ cho LLM |
| `get_sample(sample_id, split)` | Bản ghi JSON của một mẫu |
| `list_sample_ids(split)` | Danh sách id có thật của một split |

---

## 2. Cài đặt

Yêu cầu Python ≥ 3.10.

```bash
# 1. Tạo môi trường ảo (Windows: .venv-asr\Scripts\activate)
python -m venv .venv-asr
source .venv-asr/bin/activate

# 2. Cài PyTorch đúng phiên bản CUDA (xem https://pytorch.org).
#    GPU RTX 50xx (Blackwell) cần CUDA >= 12.8:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Cài thư viện của module
pip install -r src/asr/requirements.txt
```

**Qwen3-ASR** (không bắt buộc): `qwen-asr` ghim phiên bản `transformers` riêng
nên cần một môi trường ảo khác và cài `src/asr/requirements-qwen.txt`.

**Nơi lưu model.** Trọng số PhoWhisper-large (≈ 3 GB) được tải từ Hugging Face
về `~/.cache/huggingface` ở lần chạy đầu tiên. Muốn để ở ổ khác thì đặt biến
`HF_HOME`:

```powershell
$env:HF_HOME = "D:\hf_cache"      # PowerShell
```

---

## 3. Dữ liệu

Giải nén bộ dữ liệu của BTC vào thư mục gốc repo (trùng quy ước của
`src/llm_phase`):

```text
zaic-lifemate/
└── train+public_test/
    └── train+public_test/
        ├── train/
        │   ├── train.jsonl
        │   └── audio/<id>/<id>.wav, <id>_u1.wav, ...
        └── public_test/
            ├── public_test.jsonl
            └── audio/<id>/...
```

Nếu dữ liệu nằm ở nơi khác (ví dụ khi chạy trong Docker), đặt biến môi trường
`ZAC_DATA_ROOT` trỏ tới thư mục chứa `train/` và `public_test/`, hoặc truyền
tham số `data_root=...`.

Lưu ý: **id không liên tục**. Ví dụ public_test có 634 mẫu, id chạy từ
`public_0002` đến `public_0699`. Dùng `list_sample_ids("public_test")` để lấy id
có thật.

---

## 4. Cách dùng

Mọi lệnh chạy từ **thư mục gốc repo**.

### Một mẫu

```python
from src.asr import get_user_request_from_audio

current_req, history_req = get_user_request_from_audio("train_0001")
```

### Cả một split (dùng khi tạo file nộp bài)

```python
from src.asr import get_user_request_from_audio, list_sample_ids, load_asr_model

load_asr_model()                                  # nạp một lần, khoảng 10 s

for sample_id in list_sample_ids("public_test"):
    current_req, history_req = get_user_request_from_audio(sample_id)
    # ... đưa vào LLM agent ...
```

### Private test

Tiền tố id của private test chưa biết trước, nên cần truyền `split`:

```python
get_user_request_from_audio(sample_id, split="private_test")
```

### Đổi model

```python
get_user_request_from_audio("train_0001", model="qwen3_asr_1_7b")
```

---

## 5. Chạy thử và kiểm thử

### Chạy thử với model thật

```bash
python -m src.asr.demo                              # 5 mẫu train đầu tiên
python -m src.asr.demo --split public_test -n 3
python -m src.asr.demo --ids public_0006 public_0015 public_0023   # các mẫu có history
python -m src.asr.demo --model qwen3_asr_1_7b
```

Dạng kết quả (text chỉ để minh hoạ; dòng `REF` là transcript gốc, chỉ có ở tập train):

```text
Đã nạp phowhisper_large trong 8.7 s

=== train_0001  (0.85 s, 1 lượt history)
  u1  ASR: bật đèn phòng khách
      REF: Bật đèn phòng khách
  cur ASR: tắt nó đi
      REF: Tắt nó đi
```

### Unit test (không cần GPU, model hay dữ liệu)

```bash
pytest src/asr/tests -v
```

Test dùng một backend giả để kiểm tra hàm đọc đúng file, đúng thứ tự, tách
đúng `(current_req, history_req)` và chỉ nạp model một lần.

---

## 6. Lựa chọn model

Model mặc định là **PhoWhisper-large**. Bảng dưới là kết quả benchmark trên toàn
bộ **5.036 lượt nói có nhãn** của tập train (6,8 giờ âm thanh):

| Model (`model=`) | Tham số | WER % [95% CI] | CER % | Recall tham số tool % |
|---|---|---|---|---|
| **`phowhisper_large`** | 1.54B | **24.69** [24.19, 25.24] | **16.52** | **56.7** |
| `qwen3_asr_1_7b` | 2.04B | 25.97 [25.42, 26.58] | 17.51 | 53.7 |
| `whisper_large_v3` | 1.54B | 27.08 [26.20, 28.27] | 18.61 | 52.1 |
| `phowhisper_medium` | 0.76B | 27.31 [26.51, 28.37] | 18.48 | 54.0 |
| `whisper_large_v3_turbo` | 0.81B | 31.39 [29.70, 33.38] | 22.43 | 47.4 |
| `qwen3_asr_0_6b` | 0.78B | 32.67 [32.07, 33.28] | 21.92 | 45.5 |
| `nemotron_asr_0_6b` | 0.64B | 40.77 [40.18, 41.41] | 28.29 | 36.6 |

*Recall tham số tool*: tỉ lệ giá trị tham số tool được nói ra trong câu (tên
người, số tiền, giờ…) xuất hiện đúng trong text ASR. Chỉ số này gắn trực tiếp
với điểm của agent hơn WER.

**Ngân sách tham số**: cuộc thi giới hạn tổng < 6B, trong đó LLM < 5B.
PhoWhisper-large chiếm 1,54B, nên phần còn lại cho LLM vào khoảng 4,4B.

---

## 7. Thiết kế

| Quyết định | Lý do |
|---|---|
| Model **chỉ nạp một lần** cho mỗi process | Nạp model mất khoảng 10 s, trong khi giới hạn là 30 s/request |
| Đường dẫn âm thanh đọc từ `<split>.jsonl` | Không cần `audio_text`, nên chạy được trên public test và private test |
| Mọi lượt nói của một mẫu xử lý **trong một batch** | Một lần gọi model cho cả mẫu, thay vì một lần cho mỗi file |
| Trả về **text thô** của model | Không viết thường, không đổi số thành chữ: LLM trích tham số từ "500 nghìn" tốt hơn từ "năm trăm nghìn" |
| Giải mã **greedy** (1 beam, không sampling) | Kết quả tất định: chạy lại cho cùng output |
| Thư viện nặng chỉ import khi nạp model | Môi trường chỉ cần cài thư viện của backend thực sự dùng |
| Đường dẫn dữ liệu ghi đè được bằng `ZAC_DATA_ROOT` | Đóng gói Docker không phải sửa code |

**Thêm một họ model mới**: viết một lớp con của `ASRBackend` trong
`backends.py` (cài đặt `load()` và `transcribe_batch()`), đăng ký vào
`BACKENDS`, rồi thêm một `ModelSpec` vào `config.py`.

---

## 8. Cấu trúc thư mục

```text
src/asr/
├── __init__.py             # export các hàm công khai
├── api.py                  # get_user_request_from_audio, load/unload model, đọc mẫu
├── backends.py             # wrapper Whisper/PhoWhisper, Qwen3-ASR, Nemotron
├── audio_io.py             # đọc file WAV 16 kHz
├── config.py               # danh sách model, đường dẫn dữ liệu
├── demo.py                 # python -m src.asr.demo
├── requirements.txt        # môi trường mặc định (PhoWhisper/Whisper/Nemotron)
├── requirements-qwen.txt   # môi trường riêng cho Qwen3-ASR
├── tests/
│   ├── conftest.py
│   └── test_api.py         # pytest src/asr/tests
└── README.md
```

---

## 9. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân và cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chưa đứng ở thư mục gốc repo. `cd` về gốc repo rồi chạy lại |
| `FileNotFoundError: Không tìm thấy file mẫu ... .jsonl` | Dữ liệu không ở `train+public_test/train+public_test/`. Đặt `ZAC_DATA_ROOT` hoặc truyền `data_root=` |
| `KeyError: 'public_0001' không có trong public_test.jsonl` | Id không tồn tại vì id không liên tục. Dùng `list_sample_ids(...)` |
| `ValueError: Không suy ra được split ...` | Id có tiền tố lạ (private test). Truyền `split=...` |
| Lần đầu chạy tải khoảng 3 GB | Bình thường. Lần sau dùng cache; đặt `HF_HOME` nếu muốn lưu ở ổ khác |
| Mỗi mẫu mất vài chục giây | Đang chạy trên CPU. Kiểm tra `python -c "import torch; print(torch.cuda.is_available())"` |
| Cảnh báo `Redirects are currently not supported in Windows` | Cảnh báo vô hại của PyTorch trên Windows, bỏ qua được |
