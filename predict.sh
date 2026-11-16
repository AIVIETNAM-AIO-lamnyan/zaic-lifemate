#!/bin/bash
set -e

# Đảm bảo thư mục kết quả tồn tại
mkdir -p /result

echo "=== [LifeMate] BẮT ĐẦU QUÁ TRÌNH INFERENCE ==="

# Chạy tiền xử lý nếu có
if [ -f "/code/preprocessing.py" ]; then
    echo "-> Đang chạy tiền xử lý..."
    python3 /code/preprocessing.py
fi

# Chạy mã nguồn dự đoán chính
echo "-> Đang thực thi predict.py..."
python3 /code/predict.py

echo "=== [LifeMate] HOÀN TẤT! KẾT QUẢ ĐÃ LƯU TẠI /result/submission.jsonl ==="

