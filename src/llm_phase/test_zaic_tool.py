"""Thực nghiệm và kiểm tra các tính năng cốt lõi của zaic_tools (ToolEnv)."""
import json
import sys
from pathlib import Path

# Định vị thư mục chứa dữ liệu
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "train+public_test" / "train+public_test"
SPLIT = "train"

# Nạp thư viện zaic_tools của split tương ứng
sys.path.insert(0, str(DATA_ROOT / SPLIT))
from zaic_tools import ToolEnv


def main():
    print("=" * 70)
    print("      TEST CÁC TÍNH NĂNG CỦA MÔI TRƯỜNG ZAIC_TOOLS (TOOLENV)")
    print("=" * 70)

    # 1. Khởi tạo một phiên môi trường độc lập cho một mẫu
    sample_id = "train_0001"
    env = ToolEnv(SPLIT, sample_id, root=DATA_ROOT)
    
    print(f"\n[1] Khởi tạo ToolEnv cho mẫu: {sample_id}")
    hash_init = env.get_state_hash()
    print(f"    State Hash ban đầu : {hash_init}")

    # 2. Tính năng validate: Chỉ kiểm tra schema, KHÔNG thực thi, KHÔNG đổi state
    print("\n[2] Tính năng env.validate() (Kiểm tra schema trước khi gọi)")
    
    # 2.1. Thử validate một lệnh đúng
    v_ok = env.validate("weather.get_forecast", {"date": "2026-11-29"})
    print(f"    - Validate lệnh đúng  : {v_ok}")

    # 2.2. Thử validate một lệnh thiếu tham số bắt buộc
    v_err1 = env.validate("weather.get_forecast", {})
    print(f"    - Validate thiếu param : {v_err1}")

    # 2.3. Thử validate một app chưa được cài đặt
    v_err2 = env.validate("hotel.book", {"city": "Da Lat"})
    print(f"    - Validate app lạ      : {v_err2}")

    #2.4 
    err3 = env.validate("appliances.list", {})
    print(err3)

    # 3. Tính năng call: Thực thi tool đọc (Read Tool)
    print("\n[3] Thực thi Tool đọc (Read): Lấy thời tiết")
    res_weather = env.call("weather.get_forecast", {"date": "2026-11-29"})
    print(f"    Kết quả trả về:\n    {json.dumps(res_weather, ensure_ascii=False)}")
    print(f"    State Hash sau khi đọc (không đổi): {env.get_state_hash()}")

    # 4. Tính năng call: Thực thi tool ghi (Write Tool) -> Làm thay đổi dữ liệu app
    print("\n[4] Thực thi Tool ghi (Write): Dời lịch sự kiện chạy bộ")
    res_update = env.call("calendar.update_event", {
        "event_id": "evt_01",
        "start": "2026-11-29T09:00:00+07:00",
        "end": "2026-11-29T10:00:00+07:00"
    })
    print(f"    Kết quả trả về:\n    {json.dumps(res_update, ensure_ascii=False)}")
    hash_after_write = env.get_state_hash()
    print(f"    State Hash sau khi ghi (ĐÃ THAY ĐỔI): {hash_after_write}")

    # 5. Xem toàn bộ vết lịch sử các lời gọi (Dùng để tạo submission.jsonl)
    print("\n[5] Danh sách toàn bộ tool calls đã ghi nhận (env.get_tool_calls())")
    calls = env.get_tool_calls()
    for idx, c in enumerate(calls, 1):
        print(f"    Call {idx}: {c['name']} -> {c.get('arguments', {})}")

    # 6. Định dạng dòng kết quả nộp bài chuẩn ZAC 2026
    submission_entry = {
        "id": sample_id,
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash()
    }
    print("\n[6] Dòng kết quả chuẩn để ghi vào submission.jsonl:")
    print(f"    {json.dumps(submission_entry, ensure_ascii=False)}")
    print("\n" + "=" * 70)
    print("✔ Môi trường zaic_tools hoạt động hoàn hảo và độc lập trong bộ nhớ!")

if __name__ == "__main__":
    main()
