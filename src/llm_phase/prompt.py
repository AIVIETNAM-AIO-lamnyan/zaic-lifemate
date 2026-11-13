"""Xây dựng System Prompt, User Context và Tool Schema cho LLM Agent.
Kết nối danh sách app người dùng đã cài với train_tools_catalog.json.
Luôn mặc định bổ sung tool 'sys.ask_missing_info' cho mọi sample.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Đảm bảo đường dẫn import cho cả chạy trực tiếp lẫn gọi từ module ngoài
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import module xử lý công cụ riêng biệt
try:
    from src.llm_phase.get_tool import (
        get_catalog,
        get_available_tool,
        DATA_ROOT
    )
except ImportError:
    from get_tool import (
        get_catalog,
        get_available_tool,
        DATA_ROOT
    )


SYSTEM_PROMPT = """Bạn là trợ lý LifeMate. Chuyển lời nói mới nhất thành đúng các thao tác trên các ứng dụng đã cài.
Chỉ dùng tool được cung cấp. Đọc dữ liệu qua tool trước khi cần ID hoặc trạng thái. Không bịa ID, số tiền, người nhận,
địa chỉ hay phần lời nói bị cắt. Nếu thiếu thông tin bắt buộc sau khi tra cứu được phép, gọi sys.ask_missing_info.
Chỉ truyền tham số tùy chọn khi người dùng nói rõ hoặc khi công cụ thật sự cần; bỏ qua giá trị mặc định đã có trong bối cảnh.
Kết quả đọc không hoàn thành hành động. Nếu điều kiện đúng, tiếp tục gọi các tool ghi cho mọi hành động được yêu cầu.
"Dời lịch" cần tìm sự kiện bằng calendar.list_events rồi calendar.update_event với ID và thời gian thực tế.
"Nhắn [người]" cần tìm người qua contacts.search rồi gửi bằng sms.send nếu SMS đã cài; không thay bằng reminders.create.
Không bao giờ nói đã dời lịch, gửi tin, thanh toán hay tạo việc nếu tool ghi tương ứng chưa trả về thành công.
Trước mỗi tool ghi, xác định động từ và đối tượng tương ứng trong lời nói. Không điều khiển nhà thông minh chỉ vì có điều kiện thời tiết;
phải có yêu cầu nói rõ thiết bị, phòng, hoặc hành động với nhà. Nếu lời nhận dạng mơ hồ, không tự suy ra hành động khác.
Nếu yêu cầu ngoài khả năng các app đã cài, không gọi tool. Dừng ngay khi yêu cầu đã hoàn tất.
Các thao tác trong lịch sử đã được thực hiện; không chạy lại. Một lượt tiếng nói hiện tại là yêu cầu độc lập trừ khi
lịch sử thật sự giải quyết được tham chiếu. Phản hồi văn bản không được chấm điểm.
"""


def format_user_context(context: Optional[Dict[str, Any]] = None) -> str:
    """Định dạng thông tin bối cảnh người dùng (User Context) từ context dict.
    CHÚ Ý: Không chứa thông tin tools/ứng dụng để tránh dư thừa và tuân thủ nguyên tắc.
    """
    ctx = context or {}
    user_profile = ctx.get("user_profile", {})
    location = ctx.get("location", {})
    current_time = ctx.get("current_time", "N/A")
    timezone = ctx.get("timezone", "Asia/Ho_Chi_Minh")

    context_lines = [
        "### THỜI GIAN VÀ BỐI CẢNH HIỆN TẠI (User Context):",
        f"- Thời điểm hiện tại: {current_time} (Múi giờ: {timezone})",
    ]
    if location:
        context_lines.append(f"- Vị trí hiện tại: Vĩ độ {location.get('lat')}, Kinh độ {location.get('lng')}")

    if user_profile:
        context_lines.extend([
            "- Thông tin người dùng:",
            f"  + Tên: {user_profile.get('name', 'N/A')}",
            f"  + Nhà riêng: {user_profile.get('home_address', 'N/A')}",
            f"  + Nơi làm việc: {user_profile.get('work_address', 'N/A')}",
            f"  + Thành phố: {user_profile.get('city', 'N/A')}"
        ])
    return "\n".join(context_lines)


def create_user_message(
    audio_text: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, str]:
    """Tạo 1 message người dùng (role: user) từ text âm thanh của ASR và context người dùng.
    
    Args:
        audio_text: Bản dịch transcript từ audio do ASR cung cấp.
        context: Dict context của mẫu (thời gian, vị trí, profile... KHÔNG chứa tool).
        history: Lịch sử hội thoại trước đó (nếu có).
        
    Returns:
        Dict message theo chuẩn: {"role": "user", "content": "..."}
    """
    context_str = format_user_context(context)

    if history:
        history_text = json.dumps(history, ensure_ascii=False, indent=2)
    else:
        history_text = "Không có (Đây là yêu cầu đầu tiên)."

    content = f"""{context_str}

    ### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
    {history_text}

    ### YÊU CẦU MỚI NHẤT CỦA NGƯỜI DÙNG:
    "{audio_text}"

    Dựa trên bối cảnh thời gian thực, thông tin cá nhân và yêu cầu của người dùng ở trên, hãy phân tích và đưa ra công cụ (tool call) tiếp theo cần thực thi."""

    return {"role": "user", "content": content}


def build_messages_for_llm(
    audio_text: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Tạo danh sách messages đầy đủ cho LLM suy luận (system prompt + user message).
    Đầu vào là text của audio + context người dùng (không bao gồm tool).
    
    Args:
        audio_text: Transcript từ ASR.
        context: Context người dùng (không bao gồm tool).
        history: Lịch sử hội thoại (nếu có).
        system_prompt: System prompt (mặc định SYSTEM_PROMPT).
        
    Returns:
        List[Dict[str, Any]] gồm các message sẵn sàng gửi tới LLM.
    """
    sys_prompt = system_prompt or SYSTEM_PROMPT
    user_msg = create_user_message(audio_text=audio_text, context=context, history=history)
    return [
        {"role": "system", "content": sys_prompt},
        user_msg
    ]


