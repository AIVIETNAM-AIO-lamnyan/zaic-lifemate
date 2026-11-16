"""Fine-tune PhoWhisper-large bằng LoRA trên tập train của cuộc thi.

* ``check_weights.py`` kiểm tra weights PhoWhisper khớp kiến trúc Whisper
* ``train.py``         fine-tune LoRA, chọn epoch tốt nhất, gộp và lưu model
* ``evaluate.py``      so WER trên dev qua đúng đường chạy của pipeline
* ``data.py``          đọc lượt nói có nhãn, chia train/dev, tạo batch
* ``model.py``         nạp model, gắn LoRA, nhận dạng, tính WER

Xem phần "Fine-tune" trong ``src/asr/README.md``.
"""
