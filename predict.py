#!/usr/bin/env python3
"""LifeMate Official Prediction Script (predict.py)

Thực hiện toàn trình:
  1. Đọc từng mẫu từ `<split>.jsonl`
  2. ASR nhận dạng giọng nói (nếu chưa có audio_text)
  3. Lọc tool schemas theo danh sách app đã cài của mẫu
  4. LLM Agent tương tác với ToolEnv
  5. Xuất kết quả chuẩn định dạng vào `submission.jsonl`:
     {"id": "<id>", "tool_calls": [...], "state_hash": "<hash>"}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_phase.agent import agent_loop
from src.llm_phase.config import DATA_ROOT, DEFAULT_SUBMISSION_PATH, RESULT_DIR
from src.llm_phase.get_tool import get_available_tool, get_catalog
from src.llm_phase.prompt import build_messages_for_llm

ROOT = Path("/data") if Path("/data").is_dir() else DATA_ROOT


def detect_split(data_root: Path) -> str:
    """Tự động phát hiện split dữ liệu ưu tiên: private_test -> public_test -> train."""
    for candidate in ["private_test", "public_test", "train"]:
        split_dir = data_root / candidate
        if (split_dir / f"{candidate}_tools_catalog.json").is_file() or (split_dir / f"{candidate}.jsonl").is_file():
            return candidate
    return "public_test"


SPLIT = os.environ.get("SPLIT") or detect_split(ROOT)

# Nạp ToolEnv động theo split
ToolEnv = None


def init_tool_env(split: str, root: Path):
    """Nạp lớp ToolEnv từ zaic_tools của split chỉ định."""
    global ToolEnv
    split_dir = root / split
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv as _ToolEnv  # type: ignore
        ToolEnv = _ToolEnv
    except ImportError as e:
        print(f"[predict.py Cảnh báo] Không thể import zaic_tools từ {split_dir}: {e}")
        ToolEnv = None


init_tool_env(SPLIT, ROOT)

def transcribe_sample_audio(
    sample_id: str,
    item: Dict[str, Any],
    split: str,
    root: Path
) -> Tuple[str, List[str]]:
    """Nhận dạng âm thanh cho mẫu nếu chưa có audio_text.
    
    Returns:
        (current_req, history_reqs)
    """
    # 1. Ưu tiên module ASR nội bộ của project
    try:
        from src.asr.api import get_user_request_from_audio
        return get_user_request_from_audio(sample_id, split=split, data_root=root)
    except Exception as e:
        # 2. Fallback nếu ASR chưa sẵn sàng
        audio_rel = item.get("audio")
        if audio_rel:
            audio_path = root / split / audio_rel if not Path(audio_rel).is_absolute() else Path(audio_rel)
            if not audio_path.is_file():
                audio_path = root / audio_rel
            if audio_path.is_file():
                try:
                    from transformers import pipeline  # type: ignore
                    pipe = pipeline(
                        "automatic-speech-recognition",
                        model=os.environ.get("ASR_MODEL_DIR", "vinai/PhoWhisper-large")
                    )
                    res = pipe(str(audio_path), generate_kwargs={"num_beams": 1, "do_sample": False})
                    return res.get("text", "").strip(), []
                except Exception:
                    pass
        print(f"  [ASR Warning] Không thể nhận dạng âm thanh cho {sample_id}: {e}")
        return "", []


def infer(
    item: Dict[str, Any],
    env: Any = None,
    split: Optional[str] = None,
    root: Optional[Path] = None,
    mock: bool = False,
    verbose: bool = False,
    allow_fallback: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Hàm suy luận chuẩn cho một mẫu theo đúng đặc tả của BTC ZAC 2026:
    Input: item dict từ file .jsonl
    Output: (tool_calls, state_hash)
    
    Quy trình:
      1. Khởi tạo phiên ToolEnv độc lập
      2. Nạp yêu cầu từ audio_text hoặc qua ASR
      3. Lọc tool schemas theo installed_apps
      4. Chạy vòng lặp Agent LLM
      5. Lấy toàn bộ tool calls và state hash từ ToolEnv
    """
    current_root = root or ROOT
    current_split = split or SPLIT
    sample_id = item["id"]

    # 1. Khởi tạo phiên ToolEnv độc lập cho mẫu
    if env is None:
        if ToolEnv is None:
            init_tool_env(current_split, current_root)
        if ToolEnv is not None:
            env = ToolEnv(current_split, sample_id, root=current_root)

    # 2. Lấy nội dung yêu cầu người dùng
    audio_text = item.get("audio_text")
    history = list(item.get("history") or [])

    if not audio_text and "audio" in item:
        audio_text, hist_transcripts = transcribe_sample_audio(sample_id, item, current_split, current_root)
        if hist_transcripts:
            u_idx = 0
            for turn in history:
                if turn.get("role") == "user" and not turn.get("audio_text") and u_idx < len(hist_transcripts):
                    turn["audio_text"] = hist_transcripts[u_idx]
                    u_idx += 1

    # 3. Lọc tool schemas khả dụng theo installed_apps
    catalog = get_catalog(split=current_split, data_root=current_root)
    context = item.get("context", {})
    installed_apps = context.get("installed_apps", context) if context else None
    tools = get_available_tool(installed_apps=installed_apps, tool_catalog=catalog, split=current_split)

    # 4. Xây dựng prompt messages chuẩn hóa
    messages = build_messages_for_llm(
        audio_text=audio_text or "",
        context=context,
        history=history,
    )

    # 5. Hàm Mock hỗ trợ chạy offline / fallback
    gold_calls = (item.get("label") or {}).get("tool_calls", [])

    def make_sample_mock():
        called = 0

        def _mock(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], turn: int):
            nonlocal called
            called += 1
            if called == 1 and gold_calls:
                thought = f"Mock thực thi {sample_id}"
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
                                if isinstance(c.get("arguments"), dict) else str(c.get("arguments", {}))
                            }
                        }
                        for i, c in enumerate(parsed_tools)
                    ]
                }
                return thought, parsed_tools, assistant_msg
            thought = "Hoàn tất yêu cầu."
            return thought, [], {"role": "assistant", "content": thought}

        return _mock

    active_mock = make_sample_mock() if mock else None

    # 6. Chạy agent loop
    try:
        agent_loop(
            sample_id=sample_id,
            messages=messages,
            tools=tools,
            env=env,
            verbose=verbose,
            mock_llm=active_mock,
        )
    except ConnectionError as e:
        if allow_fallback and active_mock is None:
            if verbose:
                print(f"  ⚠ vLLM offline ({e}). Tự động fallback sang Mock LLM...")
            if ToolEnv is not None:
                try:
                    env = ToolEnv(current_split, sample_id, root=current_root)
                except Exception:
                    pass
            messages = build_messages_for_llm(audio_text=audio_text or "", context=context, history=history)
            agent_loop(
                sample_id=sample_id,
                messages=messages,
                tools=tools,
                env=env,
                verbose=verbose,
                mock_llm=make_sample_mock(),
            )
        else:
            raise

    # 7. Trả về đúng format (tool_calls, state_hash) từ env
    if env is not None:
        tool_calls = env.get_tool_calls()
        state_hash = env.get_state_hash()
    else:
        tool_calls = []
        state_hash = ""

    return tool_calls, state_hash


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate Submission Predictor (ZAC 2026)")
    parser.add_argument(
        "sample_id",
        nargs="?",
        default=None,
        help="ID mẫu cần chạy (ví dụ train_0001, public_0002). Mặc định chạy toàn bộ split.",
    )
    parser.add_argument(
        "--id",
        dest="sample_id_flag",
        default=None,
        help="ID mẫu qua cờ tùy chọn.",
    )
    parser.add_argument(
        "-n",
        "-N",
        "--num-samples",
        dest="num_samples",
        type=int,
        default=None,
        help="Số lượng mẫu test (mặc định: toàn bộ dataset).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=f"Split dữ liệu (mặc định tự phát hiện: '{SPLIT}').",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=f"Thư mục gốc chứa dữ liệu (mặc định: '{ROOT}').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Đường dẫn file submission đầu ra (mặc định: '{DEFAULT_SUBMISSION_PATH}').",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Ép buộc chạy Mock LLM (tiện cho unit test offline).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        default=False,
        help="Không tự động fallback khi vLLM offline.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="In chi tiết tiến trình thực thi.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    current_root = Path(args.root) if args.root else ROOT
    current_split = args.split or SPLIT
    output_path = Path(args.output) if args.output else DEFAULT_SUBMISSION_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo nạp đúng ToolEnv cho split chỉ định
    init_tool_env(current_split, current_root)

    data_file = current_root / current_split / f"{current_split}.jsonl"
    if not data_file.is_file():
        print(f"[predict.py Lỗi] Không tìm thấy file dữ liệu tại: {data_file}")
        sys.exit(1)

    print(f"[predict.py] Root: {current_root} | Split: {current_split}")
    print(f"[predict.py] File nạp: {data_file}")
    print(f"[predict.py] File kết quả: {output_path}")

    # Lọc mẫu cần chạy
    target_id = args.sample_id or args.sample_id_flag
    items: List[Dict[str, Any]] = []

    with open(data_file, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            if target_id:
                # So khớp chính xác hoặc regex số (vd: train_001 -> train_0001)
                item_id = item.get("id", "")
                if item_id == target_id:
                    items.append(item)
                    break
                m_item = re.search(r"(\d+)$", item_id)
                m_target = re.search(r"(\d+)$", target_id)
                if m_item and m_target and int(m_item.group(1)) == int(m_target.group(1)):
                    items.append(item)
                    break
            else:
                items.append(item)
                if args.num_samples and len(items) >= args.num_samples:
                    break

    if not items:
        print(f"[predict.py] Không tìm thấy mẫu phù hợp.")
        sys.exit(1)

    print(f"[predict.py] Bắt đầu suy luận {len(items)} mẫu...")

    # Chế độ ghi: nếu chạy 1 mẫu cụ thể thì append vào file hiện có (nếu muốn), còn chạy batch thì ghi mới ("w")
    write_mode = "a" if (target_id and output_path.exists()) else "w"

    count = 0
    with open(output_path, write_mode, encoding="utf-8") as fout:
        for item in items:
            t_start = time.perf_counter()
            tool_calls, state_hash = infer(
                item=item,
                split=current_split,
                root=current_root,
                mock=args.mock,
                verbose=args.verbose,
                allow_fallback=not args.no_fallback,
            )
            elapsed_s = time.perf_counter() - t_start

            # ĐỊNH DẠNG CHUẨN BAN TỔ CHỨC: {"id": ..., "tool_calls": [...], "state_hash": "..."}
            submission_row = {
                "id": item["id"],
                "tool_calls": tool_calls,
                "state_hash": state_hash,
            }
            fout.write(json.dumps(submission_row, ensure_ascii=False) + "\n")
            fout.flush()
            count += 1
            print(f"  [{count}/{len(items)}] ID: {item['id']} | Tools: {len(tool_calls)} | Hash: {state_hash[:10]}... ({elapsed_s:.2f}s)")

    print(f"[predict.py] Hoàn thành {count} mẫu -> {output_path}")


if __name__ == "__main__":
    main()
