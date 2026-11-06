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


SYSTEM_PROMPT = """Bạn là LifeMate - Trợ lý AI Voice-to-Action tiếng Việt thông minh trong hệ sinh thái ứng dụng số.
Nhiệm vụ của bạn là tiếp nhận yêu cầu bằng lời nói của người dùng, phân tích ngữ cảnh, lựa chọn và thực thi các công cụ (tools) phù hợp theo đúng thứ tự để đưa trạng thái ứng dụng về đúng yêu cầu.

### CÁC NGUYÊN TẮC BẮT BUỘC:
1. **LỰA CHỌN CÔNG CỤ (TOOL SELECTION)**:
   - Chỉ được phép sử dụng các công cụ có trong danh sách công cụ khả dụng (`available_tools`) thuộc các ứng dụng mà người dùng đã cài đặt (`installed_apps`).
   - **Ngoài phạm vi (Out-of-scope)**: Nếu yêu cầu của người dùng KHÔNG CÓ ứng dụng hoặc công cụ nào phù hợp để xử lý, bạn KHÔNG ĐƯỢC gọi bất kỳ công cụ nào (kết thúc ngay với danh sách rỗng []).

2. **XỬ LÝ THIẾU THÔNG TIN (MISSING INFORMATION)**:
   - Nếu xác định được công cụ cần gọi nhưng người dùng chưa cung cấp đủ tham số bắt buộc hoặc các bước tìm kiếm trước không tìm thấy thực thể cần thiết, bạn PHẢI gọi công cụ: `sys.ask_missing_info`.
   - Tham số của `sys.ask_missing_info`:
     + `tool_name`: Tên công cụ dự định dùng (ví dụ: "phone.call", "sms.send").
     + `params`: Mảng chứa tên các tham số bắt buộc còn thiếu (ví dụ: ["contact_id"]).

3. **BẮT BUỘC SUY LUẬN TRƯỚC KHI HÀNH ĐỘNG (THOUGHT / REASONING)**:
   - Trước mỗi lần phát sinh lời gọi công cụ (tool call), bạn BẮT BUỘC phải viết ra phần suy luận ngắn gọn (Thought) trong nội dung phản hồi:
     + Tình trạng hiện tại: Đã có dữ liệu gì từ câu nói của người dùng và các bước quan sát (Observation) trước đó?
     + Kế hoạch tiếp theo: Tại sao lại chọn công cụ này? Các tham số được tính toán/trích xuất từ đâu?
   - Sau khi nêu suy luận rõ ràng, mới đưa ra tool call tương ứng.

4. **QUY TRÌNH ĐA BƯỚC (MULTI-STEP REASONING)**:
   - Đối với yêu cầu có điều kiện ("Nếu mai mưa thì...") hoặc cần dữ liệu tham chiếu (tìm tên trong danh bạ, lấy ID sự kiện lịch), hãy gọi công cụ đọc (Read) trước.
   - Sử dụng kết quả quan sát (Observation) từ công cụ đọc để điền chính xác tham số cho công cụ ghi (Write) tiếp theo.

5. **ĐỊNH DẠNG ĐẦU RA**:
   - Khi cần hành động: Trả về phần suy luận (Thought) kèm lời gọi công cụ (tool call) với arguments hợp lệ theo đúng JSON Schema.
   - Khi đã hoàn tất toàn bộ yêu cầu: Không gọi thêm công cụ nào nữa và thông báo hoàn tất trong phần suy luận.
"""


# ---------------------------------------------------------------------------
# 2. HÀM TẢI DỮ LIỆU SAMPLE TỪ DISK
# ---------------------------------------------------------------------------
def load_sample_by_id(sample_id: str, split: str = "train") -> Optional[Dict[str, Any]]:
    """Tìm và trả về dữ liệu của sample_id từ file <split>.jsonl."""
    jsonl_path = DATA_ROOT / split / f"{split}.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("id") == sample_id:
                return data

    return None


# ---------------------------------------------------------------------------
# 3. HÀM DỰNG USER PROMPT TỪ NGỮ CẢNH CỦA SAMPLE
# ---------------------------------------------------------------------------
def build_user_prompt(
    sample: Dict[str, Any],
    *args,
    **kwargs
) -> str:
    """Ghép nối ngữ cảnh người dùng (User Context) và lời nói thành Prompt hoàn chỉnh.
    LƯU Ý: Tuyệt đối không nhồi nhét danh sách tools vào đây vì đã được truyền
    qua tham số 'tools' của Ollama payload để tránh thừa thông tin và token.
    """
    context = sample.get("context", {})

    user_profile = context.get("user_profile", {})
    location = context.get("location", {})
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "")
    history = sample.get("history", [])
    audio_text = sample.get("audio_text", "")

    # Lịch sử hội thoại trước đó (nếu có)
    if history:
        history_text = json.dumps(history, ensure_ascii=False, indent=2)
    else:
        history_text = "Không có (Đây là yêu cầu đầu tiên)."

    prompt = f"""### THỜI GIAN VÀ BỐI CẢNH HIỆN TẠI (User Context):
- Thời điểm hiện tại: {current_time} (Múi giờ: {timezone})
- Vị trí hiện tại: Vĩ độ {location.get('lat')}, Kinh độ {location.get('lng')}
- Thông tin người dùng:
  + Tên: {user_profile.get('name', 'N/A')}
  + Nhà riêng: {user_profile.get('home_address', 'N/A')}
  + Nơi làm việc: {user_profile.get('work_address', 'N/A')}
  + Thành phố: {user_profile.get('city', 'N/A')}

### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
{history_text}

### YÊU CẦU MỚI NHẤT CỦA NGƯỜI DÙNG:
"{audio_text}"

Dựa trên bối cảnh thời gian thực, thông tin cá nhân và yêu cầu của người dùng ở trên, hãy phân tích và đưa ra công cụ (tool call) tiếp theo cần thực thi."""
    return prompt


# ---------------------------------------------------------------------------
# 4. HÀM TẠO GÓI PAYLOAD HOÀN CHỈNH CHO OLLAMA (MESSAGES + TOOLS)
# ---------------------------------------------------------------------------
def build_ollama_payload(sample_id: str, split: str = "train", model: str = "qwen2.5:3b") -> Dict[str, Any]:
    """Tạo toàn bộ payload JSON gồm messages và tools theo chuẩn Native Ollama API (/api/chat)."""
    sample = load_sample_by_id(sample_id, split=split)
    if not sample:
        raise ValueError(f"Không tìm thấy mẫu {sample_id} trong {split}.jsonl")

    catalog = get_catalog(split=split)
    tools_schema = get_available_tool(sample.get("context", {}), tool_catalog=catalog)
    user_prompt = build_user_prompt(sample)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    return {
        "model": model,
        "messages": messages,
        "tools": tools_schema,
        "options": {
            "temperature": 0.0
        },
        "stream": False
    }
