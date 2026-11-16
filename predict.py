#!/usr/bin/env python3
"""
LifeMate Official Prediction Script (predict.py)
Tuân thủ định dạng trang 6-7 trong Docker_Guide_ZAC2026_LifeMate.pdf
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# 1. Xác định đường dẫn dữ liệu (mặc định BTC mount vào /data)
ROOT = Path("/data") if Path("/data").is_dir() else Path("train+public_test")

# 2. Tự động phát hiện split (private_test, public_test, hoặc train)
candidates = [p.name for p in ROOT.iterdir() if p.is_dir() and (p / f"{p.name}_tools_catalog.json").exists()] if ROOT.is_dir() else []
SPLIT = candidates[0] if candidates else "private_test"

# 3. Nạp zaic_tools của chính split dữ liệu (BẮT BUỘC theo quy định BTC)
sys.path.insert(0, str(ROOT / SPLIT))
try:
    from zaic_tools import ToolEnv
except ImportError:
    pass  # Sẽ được nạp lúc chạy thực tế khi /data được mount

# Nạp catalog của split nếu có
catalog_tools_map = {}
catalog_path = ROOT / SPLIT / f"{SPLIT}_tools_catalog.json"
if catalog_path.is_file():
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_tools_map = {t["name"]: t for a in catalog["apps"] for t in a.get("tools", [])}

# Cấu hình mô hình ASR và LLM
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.5:4b")
ASR_MODEL_DIR = os.environ.get("ASR_MODEL_DIR", "/code/saved_models/asr")
ASK_TOOL = "sys.ask_missing_info"


# ==============================================================================
# KHỐI ASR (NHẬN DẠNG GIỌNG NÓI OFFLINE)
# ==============================================================================
class ASRModel:
    def __init__(self, model_dir_or_name: str):
        self.model_dir = model_dir_or_name
        self.pipeline = None

    def load(self):
        if self.pipeline is not None:
            return
        try:
            import torch
            from transformers import pipeline
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
                torch.float16 if device.startswith("cuda") else torch.float32
            )
            # Ưu tiên load từ thư mục offline saved_models
            model_target = self.model_dir if Path(self.model_dir).is_dir() else "vinai/PhoWhisper-large"
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_target,
                device=device,
                torch_dtype=dtype,
            )
        except Exception as e:
            print(f"[ASR Warning] Không thể khởi tạo ASR pipeline: {e}")

    def transcribe(self, audio_path: str) -> str:
        if not audio_path:
            return ""
        self.load()
        if self.pipeline is None:
            return ""
        full_audio_path = ROOT / audio_path if not Path(audio_path).is_absolute() else Path(audio_path)
        if not full_audio_path.is_file():
            # Thử tìm trong split directory
            alt = ROOT / SPLIT / audio_path
            if alt.is_file():
                full_audio_path = alt
        try:
            res = self.pipeline(str(full_audio_path), generate_kwargs={"num_beams": 1, "do_sample": False})
            return res.get("text", "").strip()
        except Exception:
            return ""


asr_engine = ASRModel(ASR_MODEL_DIR)


# ==============================================================================
# KHỐI AGENT & TOOL SPEC
# ==============================================================================
def build_tools_spec(installed_apps):
    allowed_tools = []
    seen = set()
    for app_name, app_info in installed_apps.items():
        for tool_name in app_info.get("tools", []):
            if tool_name in catalog_tools_map and tool_name not in seen:
                seen.add(tool_name)
                t = catalog_tools_map[tool_name]
                allowed_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
    if ASK_TOOL in catalog_tools_map and ASK_TOOL not in seen:
        t = catalog_tools_map[ASK_TOOL]
        allowed_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return allowed_tools


def clean_arguments(args, schema_properties):
    cleaned = dict(args) if isinstance(args, dict) else {}
    for k, v in cleaned.items():
        prop = schema_properties.get(k, {})
        if prop.get("type") == "integer" and isinstance(v, str) and v.isdigit():
            cleaned[k] = int(v)
        if prop.get("format") == "date-time" and isinstance(v, str):
            v = v.replace(" ", "T")
            if not re.search(r"(Z|[+-]\d{2}:?\d{2})$", v):
                v = v + "+07:00"
            cleaned[k] = v
    return cleaned


def query_llm(messages, tools):
    import urllib.request
    payload = {
        "model": MODEL_NAME,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    if tools:
        payload["tools"] = list(tools)
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def run_agent(item, env):
    context = item.get("context", {})
    installed_apps = context.get("installed_apps", {})
    tools = build_tools_spec(installed_apps)

    system_prompt = f"""Bạn là LifeMate - trợ lý AI cá nhân tại Việt Nam có nhiệm vụ thực hiện hành động dựa trên yêu cầu người dùng.
Thông tin bối cảnh hiện tại:
- Thời gian hiện tại: {context.get('current_time')}
- Múi giờ: {context.get('timezone', 'Asia/Ho_Chi_Minh')}
- Vị trí hiện tại: {context.get('location', {})}
- Hồ sơ người dùng: {json.dumps(context.get('user_profile', {}), ensure_ascii=False)}

Nguyên tắc thực thi bắt buộc:
1. TUYỆT ĐỐI KHÔNG TỰ BỊA ID (contact_id, event_id, device_id). BẮT BUỘC gọi tool tra cứu (read) trước rồi mới gọi tool ghi.
2. Nếu câu lệnh thiếu tham số bắt buộc không thể tự tra cứu, BẮT BUỘC gọi tool 'sys.ask_missing_info' với 'tool_name' và danh sách 'params' còn thiếu.
3. Nếu người dùng đổi ý ('à thôi', 'không cần nữa') hoặc yêu cầu ngoài phạm vi, KHÔNG gọi bất kỳ tool thay đổi dữ liệu nào.
4. Mọi tham số thời gian (start, end, time) phải tính toán theo chuẩn ISO-8601 dựa trên thời gian hiện tại."""

    messages = [{"role": "system", "content": system_prompt}]

    # Nạp lịch sử hội thoại
    for h in item.get("history", []):
        role = h.get("role")
        if role == "user":
            txt = h.get("audio_text")
            if not txt and "audio" in h:
                txt = asr_engine.transcribe(h["audio"])
            messages.append({"role": "user", "content": txt or ""})
        elif role == "assistant":
            msg = {"role": "assistant", "content": h.get("text", "")}
            if h.get("tool_calls"):
                msg["tool_calls"] = [{"type": "function", "function": c} for c in h["tool_calls"]]
            messages.append(msg)
            for tc, tr in zip(h.get("tool_calls", []), h.get("tool_results", [])):
                res_str = json.dumps(tr, ensure_ascii=False)
                if len(res_str) > 2000:
                    res_str = res_str[:2000] + "...(truncated)"
                messages.append({"role": "tool", "content": res_str})

    # Lượt hiện tại
    latest_text = item.get("audio_text")
    if not latest_text and "audio" in item:
        latest_text = asr_engine.transcribe(item["audio"])
    messages.append({"role": "user", "content": latest_text or ""})

    previous_calls = None
    for step in range(6):
        resp = query_llm(messages, tools)
        if not resp or "message" not in resp:
            break
        assistant_msg = resp["message"]
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls or tool_calls == previous_calls:
            break
        previous_calls = tool_calls

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}

            tool_spec = catalog_tools_map.get(name, {})
            schema_props = (tool_spec.get("parameters") or {}).get("properties") or {}
            args = clean_arguments(raw_args, schema_props)

            check = env.validate(name, args)
            if not check.get("ok"):
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"ok": False, "error": check.get("error")}, ensure_ascii=False),
                })
                continue

            result = env.call(name, args)
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

            if name == ASK_TOOL:
                return


# ==============================================================================
# HÀM SUY LUẬN ĐƠN MẪU (INFER)
# ==============================================================================
def infer(item):
    t1 = time.perf_counter()  # bắt đầu mẫu: trước khi đọc audio
    env = ToolEnv(SPLIT, item["id"], root=ROOT)  # đúng một ToolEnv cho mỗi mẫu
    run_agent(item, env)
    tool_calls = env.get_tool_calls()
    state_hash = env.get_state_hash()
    ms = int((time.perf_counter() - t1) * 1000)
    return tool_calls, state_hash, ms


# ==============================================================================
# ENTRYPOINT CHÍNH
# ==============================================================================
def main():
    try:
        result_dir = Path("/result")
        result_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        result_dir = Path("result")
        result_dir.mkdir(parents=True, exist_ok=True)
        
    out_path = result_dir / "submission.jsonl"

    data_file = ROOT / SPLIT / f"{SPLIT}.jsonl"
    if not data_file.is_file():
        print(f"[predict.py] Không tìm thấy file dữ liệu tại {data_file}")
        sys.exit(1)

    print(f"[predict.py] Bắt đầu suy luận tập '{SPLIT}' từ {data_file}...")

    count = 0
    with open(data_file, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            tool_calls, state_hash, ms = infer(item)
            fout.write(json.dumps({
                "id": item["id"],
                "tool_calls": tool_calls,
                "state_hash": state_hash,
                "time": ms,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count % 20 == 0:
                print(f"  -> Đã xử lý {count} mẫu...")

    print(f"[predict.py] Hoàn thành {count} mẫu -> {out_path}")


if __name__ == "__main__":
    main()
