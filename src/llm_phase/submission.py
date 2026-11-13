"""Module quản lý lưu kết quả suy luận (submission.jsonl) và đánh giá score theo chuẩn ZAC 2026.
Format mỗi dòng trong file .jsonl:
{"id": "<sample_id>", "tool_calls": [...], "state_hash": "<sha256>"}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def format_submission_entry(
    sample_id: str,
    tool_calls: List[Dict[str, Any]],
    state_hash: str,
    infer_time_ms: Optional[int] = None
) -> Dict[str, Any]:
    """Tạo dict bản ghi kết quả chuẩn quy định của Zalo AI Challenge.
    
    Args:
        sample_id: ID của mẫu (ví dụ: 'train_0001', 'public_0002').
        tool_calls: Danh sách các tool call do ToolEnv ghi nhận.
        state_hash: Mã băm sha256 trạng thái cuối cùng của ToolEnv.
        infer_time_ms: Thời gian suy luận tính bằng mili-giây (tuỳ chọn).
        
    Returns:
        Dict chuẩn cho submission.
    """
    entry: Dict[str, Any] = {
        "id": sample_id,
        "tool_calls": tool_calls,
        "state_hash": state_hash,
    }
    if infer_time_ms is not None:
        entry["time"] = infer_time_ms
    return entry


def save_submission_sample(
    output_file: Union[str, Path],
    sample_id: str,
    tool_calls: List[Dict[str, Any]],
    state_hash: str,
    infer_time_ms: Optional[int] = None
) -> Dict[str, Any]:
    """Ghi append kết quả của 1 mẫu vào file JSONL.
    Tự động tạo thư mục cha nếu chưa tồn tại.
    
    Args:
        output_file: Đường dẫn file jsonl (ví dụ: 'result/submission.jsonl').
        sample_id: ID mẫu.
        tool_calls: Danh sách tool calls từ env.
        state_hash: State hash từ env.
        infer_time_ms: Thời gian suy luận.
        
    Returns:
        Dict bản ghi đã được ghi vào file.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = format_submission_entry(
        sample_id=sample_id,
        tool_calls=tool_calls,
        state_hash=state_hash,
        infer_time_ms=infer_time_ms
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


class SubmissionWriter:
    """Class hỗ trợ ghi stream các mẫu vào file submission.jsonl."""

    def __init__(self, output_path: Union[str, Path], overwrite: bool = False):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self._file = None
        self.count = 0

    def __enter__(self) -> "SubmissionWriter":
        mode = "w" if self.overwrite else "a"
        self._file = open(self.path, mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None

    def write_sample(
        self,
        sample_id: str,
        tool_calls: List[Dict[str, Any]],
        state_hash: str,
        infer_time_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ghi 1 mẫu vào file hiện tại."""
        entry = format_submission_entry(
            sample_id=sample_id,
            tool_calls=tool_calls,
            state_hash=state_hash,
            infer_time_ms=infer_time_ms
        )
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        if self._file:
            self._file.write(line)
            self._file.flush()
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

        self.count += 1
        return entry


def load_submission(file_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Đọc file submission.jsonl thành dictionary dạng {sample_id: entry}."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file submission tại: {path}")

    records: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item.get("id")
            if sample_id:
                records[sample_id] = item
    return records


def evaluate_submission(
    submission_path: Union[str, Path],
    ground_truth_path: Union[str, Path]
) -> Dict[str, Any]:
    """So sánh kết quả submission với ground truth (ví dụ train.jsonl) để tính điểm.
    
    Args:
        submission_path: Đường dẫn file submission.jsonl.
        ground_truth_path: Đường dẫn file nhãn chuẩn (ví dụ train.jsonl).
        
    Returns:
        Dict tổng hợp tỷ lệ chính xác (state_hash_accuracy, tool_calls_accuracy).
    """
    sub_data = load_submission(submission_path)
    gt_path = Path(ground_truth_path)

    total = 0
    hash_matches = 0
    tool_matches = 0
    evaluated_samples = []

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gt_item = json.loads(line)
            s_id = gt_item.get("id")
            if s_id not in sub_data:
                continue

            sub_item = sub_data[s_id]
            total += 1

            # Lấy thông tin nhãn chuẩn
            gt_label = gt_item.get("label", {})
            gt_tools = gt_label.get("tool_calls", [])
            # So sánh tool calls
            sub_tools = sub_item.get("tool_calls", [])
            tools_match = (sub_tools == gt_tools)
            if tools_match:
                tool_matches += 1

            # So sánh hash nếu có nhãn state hash hoặc qua replay
            gt_hash = gt_label.get("state_hash")
            hash_match = False
            if gt_hash:
                hash_match = (sub_item.get("state_hash") == gt_hash)
                if hash_match:
                    hash_matches += 1

            evaluated_samples.append({
                "id": s_id,
                "tool_calls_match": tools_match,
                "state_hash_match": hash_match
            })

    hash_acc = (hash_matches / total) if total > 0 else 0.0
    tool_acc = (tool_matches / total) if total > 0 else 0.0

    return {
        "total_evaluated": total,
        "state_hash_accuracy": round(hash_acc * 100, 2),
        "tool_calls_accuracy": round(tool_acc * 100, 2),
        "samples": evaluated_samples
    }
