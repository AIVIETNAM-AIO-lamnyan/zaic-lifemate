"""Cấu hình pytest: thêm thư mục gốc repo vào ``sys.path``.

Nhờ vậy test import được ``src.asr`` giống hệt cách pipeline sử dụng, dù
pytest được gọi từ thư mục nào.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
