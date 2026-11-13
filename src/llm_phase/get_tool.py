"""Module xử lý, nạp Tool Catalog và trích xuất Tool Schema cho LLM Agent.
Tinh gọn gồm 2 hàm chính:
  1. get_catalog: Nạp và lưu cache Tool Catalog (chỉ đọc disk 1 lần).
  2. get_available_tool: Lọc và tạo danh sách Tool Schema chuẩn cho tham số 'tools' của Ollama.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Định vị thư mục chứa dữ liệu
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_candidate_root = PROJECT_ROOT / "train+public_test"
if (_candidate_root / "train+public_test").is_dir():
    DATA_ROOT = _candidate_root / "train+public_test"
elif (_candidate_root / "train").is_dir():
    DATA_ROOT = _candidate_root
elif Path("/data").is_dir():
    DATA_ROOT = Path("/data")
else:
    DATA_ROOT = _candidate_root

# Cache lưu trữ Tool Catalog trong bộ nhớ (Singleton pattern theo split)
_CATALOG_CACHE: Dict[str, Dict[str, Any]] = {}

def get_catalog(
    split: str = "train",
    data_root: Optional[Path] = None,
    force_reload: bool = False
) -> Dict[str, Any]:
    """Tải và lưu cache danh mục công cụ (tools_catalog) theo split.
    Khởi tạo 1 lần duy nhất trong toàn bộ quá trình chạy qua các mẫu.

    Args:
        split: Tên tập dữ liệu ('train' hoặc 'public_test').
        data_root: Đường dẫn thư mục gốc dữ liệu (tuỳ chọn).
        force_reload: Bỏ qua cache để nạp lại từ disk nếu cần.

    Returns:
        Dict chứa catalog gồm các app và tool definitions từ BTC.
    """
    global _CATALOG_CACHE

    if not force_reload and split in _CATALOG_CACHE:
        return _CATALOG_CACHE[split]

    root = data_root or DATA_ROOT
    catalog_path = root / split / f"{split}_tools_catalog.json"

    if not catalog_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Tool Catalog tại: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    _CATALOG_CACHE[split] = catalog
    return catalog


# Alias tiện ích nếu cần gọi get_tool_catalog
get_tool_catalog = get_catalog


def _clean_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
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
            for key in ["type", "description", "enum", "items", "minimum", "exclusiveMinimum", "format"]:
                if key in p_val:
                    prop[key] = p_val[key]
            cleaned["properties"][p_name] = prop
        else:
            cleaned["properties"][p_name] = p_val

    return cleaned


def get_available_tool(
    installed_apps: Union[Dict[str, Any], Any],
    tool_catalog: Optional[Dict[str, Any]] = None,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """Từ danh sách installed_apps + tool_catalog, tạo bộ lọc để chỉ lấy các tool
    thuộc các app người dùng đã cài đặt, trả về danh sách Function Schemas chuẩn
    để truyền trực tiếp vào tham số 'tools' của Ollama payload.

    Args:
        installed_apps: Dict installed_apps, hoặc context dict, hoặc sample dict.
        tool_catalog: Catalog đã nạp (tuỳ chọn, nếu None sẽ tự gọi get_catalog).
        split: Tên split ('train' hoặc 'public_test') nếu cần tự nạp catalog.

    Returns:
        List các function tool schemas theo chuẩn Ollama/OpenAI:
        [
            {
                "type": "function",
                "function": {
                    "name": ...,
                    "description": ...,
                    "parameters": { ... }
                }
            },
            ...
        ]
    """
    # 1. Nạp catalog nếu chưa truyền vào
    catalog = tool_catalog if tool_catalog is not None else get_catalog(split=split)

    # 2. Trích xuất danh sách tên tool được phép từ installed_apps
    if isinstance(installed_apps, dict):
        if "context" in installed_apps and isinstance(installed_apps["context"], dict):
            apps_dict = installed_apps["context"].get("installed_apps", {})
        elif "installed_apps" in installed_apps and isinstance(installed_apps["installed_apps"], dict):
            apps_dict = installed_apps["installed_apps"]
        else:
            apps_dict = installed_apps
    else:
        apps_dict = {}

    allowed_names: Set[str] = set()
    for app_name, app_info in apps_dict.items():
        if isinstance(app_info, dict):
            for t_name in app_info.get("tools", []):
                allowed_names.add(t_name)
        elif isinstance(app_info, list):
            for t_name in app_info:
                allowed_names.add(t_name)

    # Luôn luôn bổ sung tool hệ thống sys.ask_missing_info
    allowed_names.add("sys.ask_missing_info")

    # 3. Lọc từ catalog và tạo chuẩn schema function calling
    tools_schema: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            name = tool.get("name")
            if name in allowed_names and name not in seen:
                seen.add(name)
                raw_params = tool.get("parameters", {"type": "object", "properties": {}})
                clean_params = _clean_parameters(raw_params)

                tools_schema.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": clean_params
                    }
                })

    return tools_schema
