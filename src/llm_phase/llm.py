"""Module gửi request tới LLM Endpoint (vLLM OpenAI-compatible).
Nhận messages, tools (từ get_tool.py) và model (từ model.py), tạo request gửi tới endpoint vLLM
và xử lý bóc tách phản hồi của LLM.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import requests

from .model import ModelConfig, get_default_model

# Model đang hoạt động thật sự trên từng endpoint, để không phải thử lại mỗi lượt.
_WORKING_MODELS: Dict[str, str] = {}


class ModelNotFoundError(ValueError):
    """Endpoint không có model được yêu cầu (HTTP 404 / model không tồn tại)."""


def build_request_payload(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[ModelConfig] = None
) -> Dict[str, Any]:
    """Tạo payload JSON tương thích với endpoint vLLM (OpenAI Chat Completions API).
    
    Args:
        messages: Danh sách messages hội thoại (system, user, assistant, tool...).
        tools: Danh sách tool schema từ get_tool.py (hoặc None nếu không dùng tool).
        model: Đối tượng ModelConfig (nếu None sẽ dùng cấu hình mặc định).
        
    Returns:
        Dict payload sẵn sàng serialize gửi qua HTTP POST.
    """
    cfg = model or get_default_model()

    payload: Dict[str, Any] = {
        "model": cfg.name,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "top_p": cfg.top_p,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    # Gộp các tùy chọn bổ sung nếu có
    for k, v in cfg.extra_options.items():
        if k not in payload:
            payload[k] = v

    return payload


def _post_chat(
    cfg: ModelConfig,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Gửi đúng một request chat completions và trả về JSON phản hồi."""
    payload = build_request_payload(messages=messages, tools=tools, model=cfg)

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    try:
        resp = requests.post(
            cfg.endpoint_url,
            headers=headers,
            json=payload,
            timeout=cfg.timeout
        )
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Không thể kết nối tới vLLM Endpoint tại: {cfg.endpoint_url}.\n"
            f"Vui lòng đảm bảo server vLLM đã khởi chạy và đúng cổng.\n"
            f"Chi tiết: {e}"
        ) from e

    if resp.status_code == 404 or (
        resp.status_code == 400
        and re.search(r"model.*(does not exist|not found|unknown)", resp.text or "", re.I)
    ):
        # vLLM trả 404 (đôi khi 400) khi tên model trong payload không có trên server.
        raise ModelNotFoundError(
            f"Endpoint '{cfg.endpoint_url}' trả HTTP {resp.status_code} cho model "
            f"'{cfg.name}' (sai URL hoặc model không tồn tại trên server)."
        )

    if not resp.ok:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        raise ValueError(
            f"Lỗi HTTP {resp.status_code} từ endpoint '{cfg.endpoint_url}':\n{err_body}"
        )

    return resp.json()


def send_llm_request(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Tạo payload và gửi HTTP POST request tới vLLM endpoint.

    Nếu endpoint báo không có model chính (HTTP 404), tự động thử model dự phòng
    ``ModelConfig.fallback_name`` (mặc định ``Qwen/Qwen2.5-0.5B-Instruct``) và
    ghi nhớ model chạy được cho các lượt sau.

    Args:
        messages: Danh sách messages.
        tools: Danh sách tool schema.
        model: ModelConfig chứa endpoint_url và cấu hình.

    Returns:
        Phản hồi dạng dict từ server vLLM.

    Raises:
        ConnectionError: Nếu không kết nối được tới server vLLM.
        ModelNotFoundError: Nếu cả model chính lẫn dự phòng đều không có.
        ValueError: Nếu server trả về mã lỗi HTTP khác.
    """
    cfg = model or get_default_model()

    # Thử model đã chạy được trước, rồi tới model chính, cuối cùng là dự phòng.
    candidates: List[str] = []
    for name in (_WORKING_MODELS.get(cfg.endpoint_url), cfg.name, cfg.fallback_name):
        if name and name not in candidates:
            candidates.append(name)

    last_error: Optional[Exception] = None
    for name in candidates:
        attempt = cfg if name == cfg.name else replace(cfg, name=name)
        try:
            response = _post_chat(attempt, messages, tools)
        except ModelNotFoundError as exc:
            last_error = exc
            continue

        if _WORKING_MODELS.get(cfg.endpoint_url) != name:
            if name != cfg.name:
                print(
                    f"[llm] Cảnh báo: endpoint '{cfg.endpoint_url}' không có model "
                    f"'{cfg.name}', chuyển sang model dự phòng '{name}'.",
                    file=sys.stderr,
                )
            _WORKING_MODELS[cfg.endpoint_url] = name
        return response

    assert last_error is not None
    raise last_error


def parse_llm_response(
    response_data: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Bóc tách phản hồi từ vLLM thành (thought, tool_calls, assistant_message).
    
    Args:
        response_data: Dữ liệu JSON trả về từ vLLM Chat Completions endpoint.
        
    Returns:
        Tuple gồm:
        - thought (str): Nội dung suy luận text hoặc reasoning của model.
        - tool_calls (List[Dict[str, Any]]): Danh sách các tool call chuẩn [{'name': ..., 'arguments': ..., 'id': ...}].
        - assistant_message (Dict[str, Any]): Object message chuẩn của assistant để append vào lịch sử messages.
    """
    raw_message: Dict[str, Any] = {}
    choices = response_data.get("choices", [])
    if choices:
        raw_message = choices[0].get("message", {})

    content = raw_message.get("content") or ""
    thinking = raw_message.get("thinking") or ""
    thought = (thinking or content).strip()

    raw_tool_calls = raw_message.get("tool_calls") or []
    parsed_tool_calls: List[Dict[str, Any]] = []

    if raw_tool_calls:
        for idx, tc in enumerate(raw_tool_calls):
            func = tc.get("function", {})
            name = func.get("name")
            args_raw = func.get("arguments", {})
            call_id = tc.get("id") or f"call_{idx}"

            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {}
            else:
                args = args_raw or {}

            if name:
                parsed_tool_calls.append({
                    "id": call_id,
                    "name": name,
                    "arguments": args
                })

    # Fallback trích xuất bằng regex nếu model sinh dạng text: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    if not parsed_tool_calls and content:
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
        for idx, match_str in enumerate(matches):
            try:
                data = json.loads(match_str)
                name = data.get("name")
                args = data.get("arguments", {})
                if name:
                    parsed_tool_calls.append({
                        "id": f"call_regex_{idx}",
                        "name": name,
                        "arguments": args
                    })
            except Exception:
                pass

    # Chuẩn hóa assistant_message theo chuẩn OpenAI Function Calling để append vào messages
    assistant_message = {
        "role": "assistant",
        "content": content,
    }
    if parsed_tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False) if isinstance(tc["arguments"], dict) else str(tc["arguments"])
                }
            }
            for i, tc in enumerate(parsed_tool_calls)
        ]

    return thought, parsed_tool_calls, assistant_message


def call_llm(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[ModelConfig] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Gửi request tới vLLM và bóc tách kết quả.
    
    Args:
        messages: Lịch sử hội thoại.
        tools: Danh sách tool schemas từ get_tool.py.
        model: Cấu hình ModelConfig từ model.py.
        
    Returns:
        Tuple: (thought, tool_calls, assistant_message)
    """
    cfg = model or get_default_model()
    resp_data = send_llm_request(messages=messages, tools=tools, model=cfg)
    return parse_llm_response(resp_data)
