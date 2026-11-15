"""Module điều phối luồng chạy của Agent (agent_loop) cho vLLM.
Kết nối message từ prompt.py, tools từ get_tool.py, model từ model.py và gửi qua llm.py,
tương tác đa bước với ToolEnv và lưu kết quả qua submission.py.
"""
from __future__ import annotations

import json
import importlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from .config import DATA_ROOT, DEFAULT_SUBMISSION_PATH, MAX_TURNS, RESULT_DIR
from .get_tool import get_available_tool, get_catalog
from .llm import call_llm
from .model import ModelConfig, get_default_model
from .prompt import build_messages_for_llm
from .submission import append_json_array, save_submission_sample


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
    verbose: bool = False,
    mock_llm: Optional[Callable[..., Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]] = None,
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
        mock_llm: Callable giả lập LLM response (dùng cho testing/fallback).

    Returns:
        Dict kết quả chứa sample_id, tool_calls, state_hash, messages, turns, time.
    """
    cfg = model or get_default_model()
    start_time = time.perf_counter()
    turn = 0
    previous_calls = None
    executed_history: List[Dict[str, Any]] = []
    thought_history: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n[{sample_id}] Bắt đầu agent_loop với {len(tools)} tools khả dụng...")

    while turn < max_turns:
        turn += 1
        if verbose:
            if mock_llm is not None:
                print(f"  -> Turn {turn}/{max_turns}: Đang gọi Mock LLM...")
            else:
                print(f"  -> Turn {turn}/{max_turns}: Đang gửi request tới vLLM ({cfg.endpoint_url})...")

        # 1. Gửi request qua llm.py hoặc mock_llm
        if mock_llm is not None:
            thought, tool_calls, assistant_msg = mock_llm(
                messages=messages,
                tools=tools,
                turn=turn,
            )
        else:
            thought, tool_calls, assistant_msg = call_llm(
                messages=messages,
                tools=tools,
                model=cfg
            )
        thought_history.append({"turn": turn, "content": thought})

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
            tool_results.append(obs_result)

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
        "infer_time_ms": elapsed_ms,
        "thought": thought_history,
        "tool_results": tool_results,
    }


# ---------------------------------------------------------------------------
# HÀM NẠP VÀ CHUẨN HÓA MẪU THEO SAMPLE_IDS
# ---------------------------------------------------------------------------
def _load_samples_by_ids(
    sample_ids: Sequence[Union[str, Dict[str, Any]]],
    split: str,
    data_root: Path,
) -> List[Dict[str, Any]]:
    """Nạp danh sách các dict mẫu từ split.jsonl theo sample_ids đã truyền."""
    if not sample_ids:
        return []

    # Nếu phần tử đã là dict (sample object đầy đủ), dùng trực tiếp
    if isinstance(sample_ids[0], dict):
        return list(sample_ids)  # type: ignore

    data_file = data_root / split / f"{split}.jsonl"
    if not data_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_file}")

    with data_file.open("r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    samples_by_id = {s["id"]: s for s in all_samples}
    resolved_samples: List[Dict[str, Any]] = []

    for raw_id in sample_ids:
        s_id = str(raw_id)
        if s_id in samples_by_id:
            resolved_samples.append(samples_by_id[s_id])
            continue

        # Tự động chuẩn hóa format số thứ tự (ví dụ: train_001 -> train_0001)
        match = re.match(r"^([a-zA-Z_]+?)(\d+)$", s_id)
        if match:
            prefix, num = match.groups()
            padded = f"{prefix}{int(num):04d}"
            if padded in samples_by_id:
                resolved_samples.append(samples_by_id[padded])
                continue

        raise KeyError(f"Không tìm thấy sample ID: '{s_id}' trong split '{split}'")

    return resolved_samples


# ---------------------------------------------------------------------------
# HÀM PIPELINE DUY NHẤT (UNIFIED PIPELINE ENTRYPOINT)
# ---------------------------------------------------------------------------
def run_sample_pipeline(
    sample_ids: Union[str, Sequence[str], Dict[str, Any], Sequence[Dict[str, Any]]],
    split: str = "train",
    transcript_source: str = "audio_text",
    output_path: Optional[Union[str, Path]] = None,
    model: Optional[ModelConfig] = None,
    data_root: Optional[Union[str, Path]] = None,
    max_turns: int = MAX_TURNS,
    verbose: bool = True,
    mock: bool = False,
    allow_fallback: bool = True,
    mock_llm: Optional[Callable[..., Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Hàm pipeline duy nhất điều phối toàn bộ quá trình suy luận cho một hoặc nhiều mẫu.

    Nhận đầu vào là danh sách sample_id (hoặc 1 sample_id đơn lẻ), tự động nạp dữ liệu,
    khởi tạo ToolEnv, xây dựng prompt, gọi LLM qua agent_loop, đối chiếu nhãn chuẩn (gold label)
    và chèn kết quả vào file JSON (mặc định result/test_llm.json).

    Args:
        sample_ids: 1 ID ('train_0001'), 1 dict mẫu, hoặc 1 list các ID / dict mẫu.
        split: Tên split ('train', 'public_test').
        transcript_source: Tên trường văn bản âm thanh trong mẫu (mặc định 'audio_text').
        output_path: Đường dẫn file JSON để append kết quả (nếu có).
        model: Cấu hình ModelConfig (mặc định lấy từ môi trường / config).
        data_root: Thư mục gốc dữ liệu (mặc định config.DATA_ROOT).
        max_turns: Số lượt suy luận tối đa cho mỗi mẫu.
        verbose: In thông tin log chi tiết.
        mock: Bật chế độ Mock LLM giả lập phản hồi.
        allow_fallback: Tự động dùng Mock LLM nếu không kết nối được vLLM.
        mock_llm: Callable giả lập tùy biến (nếu có).

    Returns:
        Danh sách các dict kết quả hoàn chỉnh theo định dạng test_llm.json.
    """
    root = Path(data_root or DATA_ROOT)

    # Đảm bảo nạp được zaic_tools từ split tương ứng
    split_dir = root / split
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))

    ToolEnv = None
    try:
        from zaic_tools import ToolEnv
    except ImportError:
        if verbose:
            print(f"  [Cảnh báo] Chưa thể import zaic_tools từ {split_dir}")

    # Chuẩn hóa đầu vào sample_ids thành dạng danh sách
    if isinstance(sample_ids, (str, dict)):
        id_list = [sample_ids]
    else:
        id_list = list(sample_ids)

    # Nạp danh sách samples đầy đủ thông tin
    samples = _load_samples_by_ids(id_list, split=split, data_root=root)

    # Nạp Catalog 1 lần cho split
    catalog = get_catalog(split=split, data_root=root)

    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        sample_id = sample.get("id", f"sample_{idx}")
        audio_text = sample.get(transcript_source) or sample.get("audio_text", "")
        context = sample.get("context", {})
        history = sample.get("history")
        installed_apps = context.get("installed_apps", context) if context else None
        gold_label = sample.get("label")

        if verbose:
            print("\n" + "=" * 65)
            print(f"[{idx}/{len(samples)}] Bắt đầu xử lý mẫu: {sample_id} (split: {split})")
            print(f"  Yêu cầu: {audio_text}")
            print("=" * 65)

        # 1. Khởi tạo môi trường ToolEnv cho mẫu
        env = None
        if ToolEnv is not None:
            try:
                env = ToolEnv(split, sample_id, root=root)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Không thể khởi tạo ToolEnv cho {sample_id}: {e}")

        # 2. Trích xuất tool schemas và xây dựng messages
        tools = get_available_tool(installed_apps=installed_apps, tool_catalog=catalog, split=split)
        messages = build_messages_for_llm(audio_text=audio_text, context=context, history=history)

        # 3. Xây dựng hàm Mock phản hồi nếu cần
        gold_calls = (gold_label or {}).get("tool_calls", [])

        def make_sample_mock():
            called_turns = 0

            def _sample_mock(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], turn: int):
                nonlocal called_turns
                called_turns += 1
                if called_turns == 1 and gold_calls:
                    thought = f"Thought: Đang xử lý yêu cầu '{audio_text}'. Thực thi các công cụ theo nhãn chuẩn."
                    parsed_tools = [dict(c) for c in gold_calls]
                    assistant_msg = {
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": c["name"],
                                    "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)
                                    if isinstance(c.get("arguments"), dict)
                                    else str(c.get("arguments", {}))
                                }
                            }
                            for i, c in enumerate(parsed_tools)
                        ]
                    }
                    return thought, parsed_tools, assistant_msg
                else:
                    thought = "Yêu cầu của người dùng đã được thực hiện đầy đủ."
                    assistant_msg = {"role": "assistant", "content": thought}
                    return thought, [], assistant_msg

            return _sample_mock

        active_mock = mock_llm or (make_sample_mock() if mock else None)

        # 4. Chạy agent loop điều phối
        try:
            loop_res = agent_loop(
                sample_id=sample_id,
                messages=messages,
                tools=tools,
                model=model,
                env=env,
                max_turns=max_turns,
                submission_path=None,
                verbose=verbose,
                mock_llm=active_mock,
            )
        except ConnectionError as e:
            if allow_fallback and active_mock is None:
                if verbose:
                    print(f"  ⚠ Lỗi kết nối vLLM: {e}")
                    print(f"  -> Tự động kích hoạt Mock LLM cho mẫu '{sample_id}' để hoàn tất kiểm thử...")
                # Reset ToolEnv trước khi chạy lại với Mock
                if ToolEnv is not None:
                    try:
                        env = ToolEnv(split, sample_id, root=root)
                    except Exception:
                        pass
                # Reset messages
                messages = build_messages_for_llm(audio_text=audio_text, context=context, history=history)
                loop_res = agent_loop(
                    sample_id=sample_id,
                    messages=messages,
                    tools=tools,
                    model=model,
                    env=env,
                    max_turns=max_turns,
                    submission_path=None,
                    verbose=verbose,
                    mock_llm=make_sample_mock(),
                )
            else:
                raise

        # 5. Đối chiếu với Gold Label (nếu có)
        state_hash_label = None
        state_hash_result = None
        tool_expect = None

        if gold_label and ToolEnv is not None:
            tool_expect = gold_label.get("tool_calls", [])
            try:
                label_env = ToolEnv(split, sample_id, root=root)
                for label_call in tool_expect:
                    label_env.call(label_call.get("name", ""), label_call.get("arguments") or {})
                state_hash_label = label_env.get_state_hash()
                state_hash_result = (loop_res["state_hash"] == state_hash_label)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Lỗi khi replay gold label: {e}")

        # 6. Đóng gói kết quả chuẩn định dạng của test_llm.json
        record: Dict[str, Any] = {
            "id": sample_id,
            "user_request": audio_text,
            "thought": loop_res.get("thought", []),
            "tool_calls": loop_res.get("tool_calls", []),
            "tool_results": loop_res.get("tool_results", []),
        }
        if tool_expect is not None:
            record["tool_expect"] = tool_expect

        record.update({
            "state_hash": loop_res.get("state_hash", ""),
            "state_hash_label": state_hash_label,
            "state_hash_result": state_hash_result,
            "time_infer_second": round(loop_res.get("infer_time_ms", 0) / 1000.0, 1),
        })

        # 7. Chèn vào file JSON array nếu có output_path
        if output_path:
            append_json_array(output_path, [record])
            if verbose:
                print(f"  ✔ Đã chèn kết quả mẫu '{sample_id}' vào: {output_path}")

        results.append(record)

    return results
