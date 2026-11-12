import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

# 1. Thêm đường dẫn để nạp prompt và zaic_tools
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_phase.prompt import (
    SYSTEM_PROMPT,
    load_sample_by_id,
    build_user_prompt,
    DATA_ROOT
)
from src.llm_phase.get_tool import (
    get_catalog,
    get_available_tool
)

# Nạp ToolEnv từ zaic_tools
SPLIT = "train"
sys.path.insert(0, str(DATA_ROOT / SPLIT))
from zaic_tools import ToolEnv

    
# ---------------------------------------------------------------------------
# HÀM KIỂM TRA MODEL TRONG OLLAMA (NATIVE API: GET /api/tags)
# ---------------------------------------------------------------------------
def check_ollama_models(host: str) -> List[str]:
    """Lấy danh sách các model đang có sẵn trong Ollama qua GET /api/tags."""
    try:
        url = f"{host.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# HÀM BÓC TÁCH TOOL CALL TỪ PHẢN HỒI NATIVE CỦA OLLAMA
# ---------------------------------------------------------------------------
def extract_single_tool_call(message: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Trích xuất duy nhất 1 tool call từ phản hồi Native của Ollama.
    Trong Native Ollama API:
    message = {
        "role": "assistant",
        "content": "...",
        "tool_calls": [
            {
                "function": {
                    "name": "...",
                    "arguments": { ... }
                }
            }
        ]
    }
    Trả về: (tool_name, tool_arguments)
    """
    tool_calls = message.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        first = tool_calls[0]
        func = first.get("function", {})
        name = func.get("name")
        args_raw = func.get("arguments", {})
        
        # Native Ollama thường trả về dict trực tiếp, nhưng nếu là string thì parse JSON
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
        else:
            args = args_raw or {}
        return name, args

    # Dự phòng nếu mô hình sinh text dạng <tool_call>...</tool_call>
    content = message.get("content") or ""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("name"), data.get("arguments", {})
        except Exception:
            pass

    return None, None


# ---------------------------------------------------------------------------
# HÀM GỌI NATIVE OLLAMA API (POST /api/chat)
# ---------------------------------------------------------------------------
def call_native_ollama_chat(
    host: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    num_ctx: int = 16384,
    timeout: int = 90
) -> Dict[str, Any]:
    """Gửi HTTP POST tới endpoint Native /api/chat của Ollama.
    Không dùng /v1/chat/completions.
    """
    url = f"{host.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    
    # Payload chuẩn theo tài liệu chính thức của Ollama Native API
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "options": {
            "temperature": 0.0,  # Cố định nhiệt độ theo quy chế thi ZAC 2026
            "num_ctx": num_ctx   # Mặc định 16384 (16k tokens) theo đúng quy định thi
        },
        "stream": False  # Nhận 1 lần phản hồi hoàn chỉnh
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if not resp.ok:
            try:
                err_data = resp.json().get("error", resp.text)
            except Exception:
                err_data = resp.text
            if resp.status_code == 404:
                raise ValueError(
                    f"Ollama trả về 404 Not Found từ endpoint /api/chat.\n"
                    f"Chi tiết từ Ollama: {err_data}\n"
                    f"👉 Hãy tải model bằng lệnh: ollama run {model}"
                )
            raise ValueError(f"Ollama trả về lỗi HTTP {resp.status_code} (Bad Request):\n{err_data}")
            
        data = resp.json()
        # Native API trả về trường 'message' ở tầng gốc (root)
        return data.get("message", {})

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Không thể kết nối tới Ollama tại {host}.\n"
            f"👉 Vui lòng mở terminal và chạy: ollama run {model}"
        )


def save_result(result: Dict[str, Any], result_path: Optional[Path] = None) -> None:
    """Append one run to the JSON array at result/test_llm.json."""
    path = result_path or PROJECT_ROOT / "result" / "test_llm.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            results = json.load(file)
        if not isinstance(results, list):
            raise ValueError(f"File kết quả phải chứa một JSON array: {path}")
    else:
        results = []

    results.append(result)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")


# ---------------------------------------------------------------------------
# VÒNG LẶP CHÍNH: THOUGHT -> ACTION -> OBSERVE (NATIVE OLLAMA)
# ---------------------------------------------------------------------------
def run_agent_loop(
    sample_id: str,
    model_name: str,
    ollama_host: str = "http://localhost:11434",
    num_ctx: int = 16384,
    max_turns: int = 8
) -> Dict[str, Any]:
    run_start_time = time.perf_counter()

    # 0. Kiểm tra danh sách model đang có trong Ollama
    installed_models = check_ollama_models(ollama_host)
    if installed_models:
        found = any(model_name == m or model_name in m or m in model_name for m in installed_models)

    # 1. Tải sample và catalog (catalog được cache trong get_tool)
    sample = load_sample_by_id(sample_id, split=SPLIT)
    if not sample:
        raise ValueError(f"Không tìm thấy mẫu {sample_id} trong tập {SPLIT}!")

    catalog = get_catalog(split=SPLIT)
    tools_schema = get_available_tool(sample.get("context", {}), tool_catalog=catalog)
    
    # 2. Khởi tạo môi trường ToolEnv (In-memory Sandbox)
    env = ToolEnv(SPLIT, sample_id, root=DATA_ROOT)

    # 3. Khởi tạo hội thoại Messages (user_prompt chỉ chứa bối cảnh người dùng, không lặp lại tool)
    user_prompt = build_user_prompt(sample)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    turn = 0
    thoughts: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    # 4. Vòng lặp suy luận từng bước (Thought - Action - Observe)
    while turn < max_turns:
        turn += 1

        # Gửi request sang Native Ollama API
        try:
            bot_msg = call_native_ollama_chat(ollama_host, model_name, messages, tools_schema, num_ctx=num_ctx)
        except Exception as e:
            print(f"\n❌ [LỖI GỌI MODEL]: {e}")
            sys.exit(1)

        thought = bot_msg.get("content") or bot_msg.get("thinking") or ""
        tool_name, tool_args = extract_single_tool_call(bot_msg)

        # BƯỚC 1: THOUGHT (Suy luận của model)
        if thought.strip():
            # Xử lý nếu mô hình dùng thẻ <think>...</think>
            clean_thought = re.sub(r"</?think>", "", thought).strip()
        else:
            clean_thought = ""
        thoughts.append({"turn": turn, "content": clean_thought})

        # BƯỚC 2: ACTION (Hành động gọi tool)
        # Nếu model không gọi tool nào -> Tác vụ đã xong hoặc Ngoài phạm vi
        if not tool_name:
            break

        tool_call = {"name": tool_name, "arguments": tool_args}
        tool_calls.append(tool_call)

        # Lưu ý: Model KHÔNG ĐƯỢC PHÉP tự bịa ra kết quả quan sát! Dừng suy luận tại đây và để Env thực thi.
        
        # BƯỚC 3: OBSERVE (Môi trường ToolEnv thực thi và trả kết quả thật)
        # Trường hợp gọi sys.ask_missing_info: thực thi và kết thúc lượt
        if tool_name == "sys.ask_missing_info":
            obs_result = env.call(tool_name, tool_args)
            tool_results.append(obs_result)
            break

        # Kiểm tra tính hợp lệ trước khi thực thi
        val = env.validate(tool_name, tool_args)
        if not val.get("ok"):
            print(f"   ⚠ [Cảnh báo Schema Error]: {val.get('error')}")

        # Thực thi thật trên ToolEnv
        obs_result = env.call(tool_name, tool_args)
        tool_results.append(obs_result)

        # BƯỚC 4: APPEND KẾT QUẢ VÀO MESSAGES CHO LƯỢT TIẾP THEO (CHUẨN NATIVE OLLAMA)
        # Thêm lượt Assistant (chứa tool_calls chuẩn Ollama)
        messages.append({
            "role": "assistant",
            "content": thought if thought else "",
            "tool_calls": [{
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }]
        })

        # Thêm lượt Tool (chứa kết quả observation)
        messages.append({
            "role": "tool",
            "content": json.dumps(obs_result, ensure_ascii=False)
        })

        # Nếu tool trả về lỗi không thể tiếp tục
        if not obs_result.get("ok", True):
            break

    # ---------------------------------------------------------------------------
    # TỔNG KẾT VÀ ĐẦU RA ĐỊNH DẠNG NỘP BÀI CHUẨN ZAC 2026
    # ---------------------------------------------------------------------------
    final_output = {
        "id": sample_id,
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash()
    }

    # Đối chiếu với Gold Label (nếu có trong train.jsonl)
    gold_label = sample.get("label")
    state_hash_label = None
    state_hash_result = None
    if gold_label:
        label_env = ToolEnv(SPLIT, sample_id, root=DATA_ROOT)
        for label_call in gold_label.get("tool_calls", []):
            label_env.call(label_call.get("name", ""), label_call.get("arguments") or {})
        state_hash_label = label_env.get_state_hash()
        state_hash_result = final_output["state_hash"] == state_hash_label

    save_result({
        "id": sample_id,
        "user_request": sample.get("audio_text", ""),
        "thought": thoughts,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "tool_expect": (gold_label or {}).get("tool_calls", []),
        "state_hash": final_output["state_hash"],
        "state_hash_label": state_hash_label,
        "state_hash_result": state_hash_result,
        "time_infer_second": round(time.perf_counter() - run_start_time, 1)
    })

    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LifeMate Phase 3: Test Native Ollama Agent")
    parser.add_argument("--id", default="train_0001", help="Sample ID cần test (ví dụ: train_0001, train_0019, train_0025)")
    parser.add_argument("--model", default="qwen2.5:3b", help="Tên model trên Ollama (mặc định: qwen2.5:3b)")
    parser.add_argument("--host", default="http://localhost:11434", help="Địa chỉ Ollama host (mặc định: http://localhost:11434)")
    parser.add_argument("--num-ctx", type=int, default=16384, help="Kích thước Context window (mặc định: 16384 theo luật thi)")
    parser.add_argument("--max-turns", type=int, default=8, help="Số lượt suy luận tối đa")
    args = parser.parse_args()

    run_agent_loop(
        sample_id=args.id,
        model_name=args.model,
        ollama_host=args.host,
        num_ctx=args.num_ctx,
        max_turns=args.max_turns
    )
