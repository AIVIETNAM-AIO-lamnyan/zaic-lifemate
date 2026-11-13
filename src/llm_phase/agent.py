"""Module điều phối luồng chạy của Agent (agent_loop) cho vLLM.
Kết nối message từ prompt.py, tools từ get_tool.py, model từ model.py và gửi qua llm.py,
tương tác đa bước với ToolEnv và lưu kết quả qua submission.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import DEFAULT_SUBMISSION_PATH, MAX_TURNS
from .get_tool import get_available_tool, get_catalog
from .llm import call_llm
from .model import ModelConfig, get_default_model
from .prompt import build_messages_for_llm
from .submission import save_submission_sample


def _build_tool_message(
    content: str,
    tool_call_id: Optional[str] = None,
    tool_name: Optional[str] = None
) -> Dict[str, Any]:
    """Tạo dict message với role='tool' theo chuẩn OpenAI Chat Completions cho vLLM."""
    tool_msg: Dict[str, Any] = {
        "role": "tool",
        "content": content
    }
    if tool_call_id:
        tool_msg["tool_call_id"] = tool_call_id
    if tool_name:
        tool_msg["name"] = tool_name
    return tool_msg


def agent_loop(
    sample_id: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: Optional[ModelConfig] = None,
    env: Any = None,
    max_turns: int = MAX_TURNS,
    submission_path: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Điều phối vòng lặp suy luận của Agent cho một mẫu:
    Thought -> Tool Calls -> Execute Tool in ToolEnv -> Observation -> Repeat.

    Args:
        sample_id: ID của mẫu đang xử lý.
        messages: Danh sách messages đầu vào (được sinh từ prompt.py).
        tools: Danh sách tool schemas (được lấy từ get_tool.py).
        model: ModelConfig từ model.py (mặc định lấy từ get_default_model).
        env: Môi trường ToolEnv (Session của zaic_tools).
        max_turns: Số lượt tương tác tối đa (mặc định 10 từ config.MAX_TURNS).
        submission_path: Đường dẫn file jsonl để lưu submission (nếu có).
        verbose: In log các bước suy luận.

    Returns:
        Dict kết quả chứa sample_id, tool_calls, state_hash, messages, turns, time.
    """
    cfg = model or get_default_model()
    start_time = time.perf_counter()
    turn = 0
    previous_calls = None
    executed_history: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n[{sample_id}] Bắt đầu agent_loop với {len(tools)} tools khả dụng...")

    while turn < max_turns:
        turn += 1
        if verbose:
            print(f"  -> Turn {turn}/{max_turns}: Đang gửi request tới vLLM ({cfg.endpoint_url})...")

        # 1. Gửi request qua llm.py
        thought, tool_calls, assistant_msg = call_llm(
            messages=messages,
            tools=tools,
            model=cfg
        )

        # 2. Append phản hồi của assistant vào danh sách messages
        messages.append(assistant_msg)

        if verbose and thought:
            print(f"     [Thought]: {thought[:120]}..." if len(thought) > 120 else f"     [Thought]: {thought}")

        # 3. Kiểm tra xem LLM có gọi tool nào không
        if not tool_calls:
            if verbose:
                print("     [Action]: Không có tool call nào. Hoàn tất yêu cầu.")
            break

        # Tránh lặp vô tận nếu model gọi lại chính xác các tool call của lượt trước
        current_calls_signature = [(tc.get("name"), tc.get("arguments")) for tc in tool_calls]
        if current_calls_signature == previous_calls:
            if verbose:
                print("     [Cảnh báo]: Model gọi lặp lại chính xác tool calls của lượt trước. Dừng.")
            break
        previous_calls = current_calls_signature

        # 4. Thực thi từng tool call trên môi trường ToolEnv
        stop_agent = False
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            call_id = tc.get("id")

            if verbose:
                print(f"     [Tool Call]: {name}({args})")

            # Validate schema với ToolEnv nếu có method validate
            if env is not None and hasattr(env, "validate"):
                val = env.validate(name, args)
                if not val.get("ok"):
                    if verbose:
                        print(f"       ⚠ Schema Error: {val.get('error')}")

            # Thực thi tool
            if env is not None and hasattr(env, "call"):
                obs_result = env.call(name, args)
            else:
                obs_result = {"ok": True, "simulated": True}

            executed_history.append({"name": name, "arguments": args, "result": obs_result})

            # Append kết quả quan sát vào messages cho lượt tiếp theo
            obs_str = json.dumps(obs_result, ensure_ascii=False)
            tool_msg = _build_tool_message(
                content=obs_str,
                tool_call_id=call_id,
                tool_name=name
            )
            messages.append(tool_msg)

            # Quy tắc ZAC: nếu gọi sys.ask_missing_info thì kết thúc phiên ngay lập tức
            if name == "sys.ask_missing_info":
                if verbose:
                    print("     [Action]: Đã gọi 'sys.ask_missing_info'. Dừng theo quy định.")
                stop_agent = True
                break

        if stop_agent:
            break

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # 5. Xử lý đầu ra từ env (hỗ trợ cả get_tool_calls / get_tool_call và get_state_hash / get_status)
    if env is not None:
        if hasattr(env, "get_tool_calls"):
            final_tool_calls = env.get_tool_calls()
        elif hasattr(env, "get_tool_call"):
            final_tool_calls = env.get_tool_call()
        else:
            final_tool_calls = [
                {"name": h["name"], "arguments": h["arguments"]} for h in executed_history
            ]

        if hasattr(env, "get_state_hash"):
            final_state_hash = env.get_state_hash()
        elif hasattr(env, "get_status"):
            final_state_hash = env.get_status()
        elif hasattr(env, "get_state"):
            final_state_hash = str(env.get_state())
        else:
            final_state_hash = ""
    else:
        final_tool_calls = [
            {"name": h["name"], "arguments": h["arguments"]} for h in executed_history
        ]
        final_state_hash = ""

    # 6. Tự động lưu kết quả vào submission.jsonl nếu được yêu cầu
    if submission_path:
        save_submission_sample(
            output_file=submission_path,
            sample_id=sample_id,
            tool_calls=final_tool_calls,
            state_hash=final_state_hash,
            infer_time_ms=elapsed_ms
        )
        if verbose:
            print(f"  -> Đã lưu kết quả của mẫu '{sample_id}' vào: {submission_path}")

    return {
        "id": sample_id,
        "tool_calls": final_tool_calls,
        "state_hash": final_state_hash,
        "messages": messages,
        "turns": turn,
        "infer_time_ms": elapsed_ms
    }


# ---------------------------------------------------------------------------
# HÀM PIPELINE TOÀN TRÌNH CHO 1 MẪU (HIGH-LEVEL INFERENCE PIPELINE)
# ---------------------------------------------------------------------------
def run_sample_pipeline(
    sample_id: str,
    audio_text: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    installed_apps: Optional[Union[Dict[str, Any], Any]] = None,
    env: Any = None,
    model: Optional[ModelConfig] = None,
    split: str = "train",
    submission_path: Optional[Union[str, Path]] = None,
    max_turns: int = MAX_TURNS,
    verbose: bool = False
) -> Dict[str, Any]:
    """Kết nối toàn bộ pipeline LLM cho 1 mẫu:
    ASR Transcript Text + Context -> prompt.py -> get_tool.py -> agent_loop -> submission.py.

    Args:
        sample_id: Mã ID mẫu.
        audio_text: Text transcript từ ASR cung cấp.
        context: Context dict của người dùng (thời gian, profile... KHÔNG chứa tool).
        history: Lịch sử hội thoại trước đó (nếu có).
        installed_apps: Danh sách apps đã cài (hoặc context chứa installed_apps).
        env: ToolEnv instance.
        model: ModelConfig từ model.py.
        split: Split tập dữ liệu ('train', 'public_test', 'private_test').
        submission_path: Đường dẫn file jsonl để lưu kết quả.
        max_turns: Số lượt tối đa (mặc định 10 từ config.MAX_TURNS).
        verbose: Bật log.

    Returns:
        Dict kết quả trả về từ agent_loop.
    """
    ctx = context or {}

    # 1. Trích xuất tool schemas từ get_tool.py
    app_source = installed_apps if installed_apps is not None else ctx.get("installed_apps", ctx)
    catalog = get_catalog(split=split)
    tools = get_available_tool(installed_apps=app_source, tool_catalog=catalog, split=split)

    # 2. Xây dựng messages từ prompt.py (chỉ gồm audio_text + context, không chứa tool)
    messages = build_messages_for_llm(
        audio_text=audio_text,
        context=ctx,
        history=history
    )

    # 3. Chạy agent loop điều phối
    return agent_loop(
        sample_id=sample_id,
        messages=messages,
        tools=tools,
        model=model,
        env=env,
        max_turns=max_turns,
        submission_path=submission_path,
        verbose=verbose
    )
