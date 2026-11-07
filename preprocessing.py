"""
Module tiền xử lý (Preprocessing) trước khi chạy dự đoán.
Kiểm tra cấu trúc dữ liệu /data và chuẩn bị môi trường thực thi.
"""
import os
import sys
from pathlib import Path

def main():
    data_dir = Path("/data") if Path("/data").exists() else Path("train+public_test")
    try:
        result_dir = Path("/result")
        result_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        result_dir = Path("result")
        result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Preprocessing] Dữ liệu nguồn: {data_dir}")
    print(f"[Preprocessing] Thư mục kết quả: {result_dir}")
    
    # Tìm split trong thư mục data
    splits = [p.name for p in data_dir.iterdir() if p.is_dir() and (p / f"{p.name}_tools_catalog.json").exists()]
    if splits:
        print(f"[Preprocessing] Phát hiện split hợp lệ: {splits[0]}")
    else:
        print("[Preprocessing] Cảnh báo: Chưa phát hiện split có catalog trong thư mục data.")

if __name__ == "__main__":
    main()
