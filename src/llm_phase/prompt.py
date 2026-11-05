"""Xây dựng System Prompt, User Context và Tool Schema cho LLM Agent.
Kết nối danh sách app người dùng đã cài với train_tools_catalog.json.
Luôn mặc định bổ sung tool 'sys.ask_missing_info' cho mọi sample.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Định vị thư mục chứa dữ liệu
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "train+public_test" / "train+public_test"


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
# 2. HÀM TẢI DỮ LIỆU TỪ DISK (CATALOG & SAMPLE)
# ---------------------------------------------------------------------------
def load_tools_catalog(split: str = "train") -> Dict[str, Any]:
    """Tải catalog chứa schema chi tiết của toàn bộ tools trong split."""
    catalog_path = DATA_ROOT / split / f"{split}_tools_catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy catalog tại: {catalog_path}")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
# 3. KẾT NỐI INSTALLED_APPS VỚI TOOLS_CATALOG (LUÔN CÓ SYS.ASK_MISSING_INFO)
# ---------------------------------------------------------------------------
def get_allowed_tool_names(context: Dict[str, Any]) -> Set[str]:
    """Lấy danh sách các tool được phép gọi từ installed_apps.
    LƯU Ý: Luôn mặc định bổ sung 'sys.ask_missing_info' cho mọi sample.
    """
    installed_apps = context.get("installed_apps", {})
    allowed_tools = set()

    # 1. Quét toàn bộ tool của các app đã cài
    for app_name, app_info in installed_apps.items():
        for tool_name in app_info.get("tools", []):
            allowed_tools.add(tool_name)

    # 2. Mặc định luôn luôn thêm sys.ask_missing_info
    allowed_tools.add("sys.ask_missing_info")
    return allowed_tools


def clean_parameters_for_ollama(params: Dict[str, Any]) -> Dict[str, Any]:
    """Làm sạch schema tham số để tương thích 100% với JSON Schema chuẩn của Ollama.
    Loại bỏ các trường tùy biến riêng của cuộc thi như x-one-of, x-ref, x-match.
    """
    cleaned: Dict[str, Any] = {
        "type": params.get("type", "object"),
        "properties": {}
    }
    if "required" in params and isinstance(params["required"], list):
        cleaned["required"] = params["required"]

    for p_name, p_val in params.get("properties", {}).items():
        if isinstance(p_val, dict):
            prop = {}
            # Chỉ giữ lại các trường chuẩn theo chuẩn JSON Schema
            for key in ["type", "description", "enum", "items", "minimum", "exclusiveMinimum", "format"]:
                if key in p_val:
                    prop[key] = p_val[key]
            cleaned["properties"][p_name] = prop
        else:
            cleaned["properties"][p_name] = p_val

    return cleaned


def get_sample_tool_schemas(
    sample: Dict[str, Any],
    catalog: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Trích xuất JSON Schema chi tiết từ catalog cho các tool mà sample được phép dùng.
    Trả về định dạng Function Calling chuẩn hóa cho Ollama.
    """
    allowed_tool_names = get_allowed_tool_names(sample.get("context", {}))
    openai_tools = []

    # Quét qua catalog của BTC để lấy mô tả và parameters chuẩn
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            name = tool.get("name")
            if name in allowed_tool_names:
                raw_params = tool.get("parameters", {"type": "object", "properties": {}})
                clean_params = clean_parameters_for_ollama(raw_params)
                
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": clean_params
                    }
                }
                openai_tools.append(openai_tool)

    return openai_tools


# ---------------------------------------------------------------------------
# 4. HÀM DỰNG USER PROMPT TỪ NGỮ CẢNH CỦA SAMPLE
# ---------------------------------------------------------------------------
def build_user_prompt(
    sample: Dict[str, Any],
    include_tool_descriptions: bool = True,
    catalog: Optional[Dict[str, Any]] = None
) -> str:
    """Ghép nối ngữ cảnh người dùng (User Context) và lời nói thành Prompt hoàn chỉnh.
    Nếu include_tool_descriptions=True: Nhúng bản tóm tắt chức năng tool vào Prompt.
    """
    context = sample.get("context", {})

    user_profile = context.get("user_profile", {})
    location = context.get("location", {})
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "")
    history = sample.get("history", [])
    audio_text = sample.get("audio_text", "")

    # Tóm tắt các công cụ khả dụng kèm mô tả từ catalog
    allowed_tools = get_allowed_tool_names(context)
    tool_desc_lines = []

    if include_tool_descriptions and catalog:
        # Tra cứu mô tả từ catalog
        for app in catalog.get("apps", []):
            for tool in app.get("tools", []):
                t_name = tool.get("name")
                if t_name in allowed_tools:
                    desc = tool.get("description", "Không có mô tả")
                    req = tool.get("parameters", {}).get("required", [])
                    req_str = f" [Bắt buộc: {', '.join(req)}]" if req else ""
                    tool_desc_lines.append(f"  - `{t_name}`: {desc}{req_str}")
        tools_summary = "\n".join(tool_desc_lines)
    else:
        # Chỉ liệt kê tên công cụ
        installed_apps = context.get("installed_apps", {})
        for app_name, app_info in installed_apps.items():
            tool_desc_lines.append(f"  + Ứng dụng '{app_name}': {', '.join(app_info.get('tools', []))}")
        tool_desc_lines.append("  + Hệ thống: sys.ask_missing_info")
        tools_summary = "\n".join(tool_desc_lines)

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

### CÁC CÔNG CỤ ĐƯỢC PHÉP SỬ DỤNG (Available Tools):
{tools_summary}

### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
{history_text}

### YÊU CẦU MỚI NHẤT CỦA NGƯỜI DÙNG:
"{audio_text}"

Dựa trên bối cảnh thời gian thực, thông tin cá nhân và chức năng của các công cụ khả dụng ở trên, hãy phân tích và đưa ra công cụ (tool call) tiếp theo cần thực thi."""
    return prompt


# ---------------------------------------------------------------------------
# 5. HÀM TẠO GÓI PAYLOAD HOÀN CHỈNH CHO OLLAMA (MESSAGES + TOOLS)
# ---------------------------------------------------------------------------
def build_ollama_payload(sample_id: str, split: str = "train", model: str = "qwen2.5:3b") -> Dict[str, Any]:
    """Tạo toàn bộ payload JSON gồm messages và tools theo chuẩn Native Ollama API (/api/chat)."""
    sample = load_sample_by_id(sample_id, split=split)
    if not sample:
        raise ValueError(f"Không tìm thấy mẫu {sample_id} trong {split}.jsonl")

    catalog = load_tools_catalog(split=split)
    tools_schema = get_sample_tool_schemas(sample, catalog)
    user_prompt = build_user_prompt(sample, include_tool_descriptions=True, catalog=catalog)

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


# ---------------------------------------------------------------------------
# 6. CHẠY THỬ VÀ IN RA ĐỂ KIỂM TRA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target_id = sys.argv[1] if len(sys.argv) > 1 else "train_0001"

    print("=" * 80)
    print(f"   KIỂM TRA DỰNG PROMPT & TOOL SCHEMAS CHO SAMPLE ID: {target_id}")
    print("=" * 80)

    sample_data = load_sample_by_id(target_id)
    if not sample_data:
        print(f"❌ Không tìm thấy mẫu có id = {target_id}")
        sys.exit(1)

    catalog_data = load_tools_catalog("train")
    allowed = get_allowed_tool_names(sample_data.get("context", {}))
    tools_schema = get_sample_tool_schemas(sample_data, catalog_data)

    print(f"\n[1] DANH SÁCH TOOLS KHẢ DỤNG: Tổng cộng {len(allowed)} tools")
    print(f"    -> Đã bao gồm 'sys.ask_missing_info': {'sys.ask_missing_info' in allowed}")

    print("\n" + "#" * 35 + " 2. USER PROMPT (ĐÃ KÈM MÔ TẢ TOOL TỪ CATALOG) " + "#" * 35)
    user_prompt_text = build_user_prompt(sample_data, include_tool_descriptions=True, catalog=catalog_data)
    print(user_prompt_text)

    print("\n" + "#" * 30 + " 3. MẪU SCHEMA OPENAI TOOL (GỬI QUA API) " + "#" * 30)
    # In ví dụ 2 tool đầu tiên và sys.ask_missing_info để kiểm tra
    sample_preview = [t for t in tools_schema if t["function"]["name"] in ["weather.get_forecast", "sys.ask_missing_info"]]
    print(json.dumps(sample_preview, ensure_ascii=False, indent=2))

    print("\n" + "#" * 30 + " 4. ĐÁP ÁN CHUẨN CỦA MẪU (LABEL) " + "#" * 30)
    label = sample_data.get("label", {})
    print(f"- Tool calls mong đợi: {json.dumps(label.get('tool_calls', []), ensure_ascii=False, indent=2)}")
