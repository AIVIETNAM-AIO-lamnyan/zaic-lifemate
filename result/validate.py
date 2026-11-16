#!/usr/bin/env python3
"""Chấm điểm một file submission.jsonl theo công thức ZAC 2026.

Cách dùng (file mặc định nằm cùng thư mục với script)::

    python result/validate.py                  # chấm result/submission.jsonl
    python result/validate.py my_submission.jsonl

Công thức mỗi mẫu (docs/ZaloAI.md):

* hash sai hoặc M = 0  -> 0 điểm
* gold rỗng            -> 1 điểm nếu submission cũng rỗng, ngược lại 0
* còn lại              -> 2*P*R / (P + R), P = M/(M+U), R = M/|G|

Điểm cuối = 100/N * tổng điểm các mẫu. Chỉ split train có nhãn vàng.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "train+public_test"
TRAIN_DIR = DATA_ROOT / "train"
GOLD_PATH = TRAIN_DIR / "train.jsonl"
CATALOG_PATH = TRAIN_DIR / "train_tools_catalog.json"
ENVIRONMENT_DIR = TRAIN_DIR / "train_environment"

FUZZY_THRESHOLD = 0.6

__all__ = ["score_submission"]


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_submission(path: Path) -> Dict[str, Dict[str, Any]]:
    return {
        row["id"]: {
            "tool_calls": row.get("tool_calls") or [],
            "state_hash": row.get("state_hash") or "",
        }
        for row in _read_jsonl(path)
    }


def _load_gold() -> Dict[str, List[Dict[str, Any]]]:
    return {
        row["id"]: row["label"].get("tool_calls") or []
        for row in _read_jsonl(GOLD_PATH)
        if isinstance(row.get("label"), dict)
    }


def _load_fuzzy_arguments() -> Dict[str, set[str]]:
    """{tool_name: {tham số fuzzy}} từ x-match: fuzzy trong catalog."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    fuzzy: Dict[str, set[str]] = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            properties = (tool.get("parameters") or {}).get("properties") or {}
            keys = {
                key
                for key, spec in properties.items()
                if isinstance(spec, dict) and spec.get("x-match") == "fuzzy"
            }
            if keys:
                fuzzy[tool["name"]] = keys
    return fuzzy


# ---------------------------------------------------------------------------
# So khớp tool call
# ---------------------------------------------------------------------------
def _normalize(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value).lower())


def _token_f1(left: Any, right: Any) -> float:
    left_tokens = _normalize(left).split()
    right_tokens = _normalize(right).split()
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _calls_match(
    submitted: Dict[str, Any],
    gold: Dict[str, Any],
    fuzzy_keys: set[str],
) -> bool:
    if submitted.get("name") != gold.get("name"):
        return False

    submitted_args = submitted.get("arguments") or {}
    gold_args = gold.get("arguments") or {}
    if set(submitted_args) != set(gold_args):
        return False

    for key, gold_value in gold_args.items():
        submitted_value = submitted_args[key]
        if key in fuzzy_keys:
            if isinstance(submitted_value, str) and isinstance(gold_value, str):
                if _token_f1(submitted_value, gold_value) <= FUZZY_THRESHOLD:
                    return False
            elif not _values_equal(submitted_value, gold_value):
                return False
        elif not _values_equal(submitted_value, gold_value):
            return False

    return True


def _maximum_matching(
    submitted: Sequence[Dict[str, Any]],
    gold: Sequence[Dict[str, Any]],
    fuzzy_by_tool: Dict[str, set[str]],
) -> int:
    """Số call khớp nhiều nhất; mỗi call chỉ ghép với một call vàng."""
    if not submitted or not gold:
        return 0

    compatible = [
        [
            _calls_match(call, reference, fuzzy_by_tool.get(call["name"], set()))
            for reference in gold
        ]
        for call in submitted
    ]
    matched_gold = [-1] * len(gold)

    def augment(submitted_index: int, seen: List[bool]) -> bool:
        for gold_index, is_compatible in enumerate(compatible[submitted_index]):
            if not is_compatible or seen[gold_index]:
                continue
            seen[gold_index] = True
            if matched_gold[gold_index] == -1 or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = submitted_index
                return True
        return False

    return sum(augment(index, [False] * len(gold)) for index in range(len(submitted)))


# ---------------------------------------------------------------------------
# Replay state_hash bằng zaic_tools của train
# ---------------------------------------------------------------------------
def _load_executor():
    sys.path.insert(0, str(TRAIN_DIR))
    try:
        from zaic_tools import Executor  # type: ignore
    finally:
        if str(TRAIN_DIR) in sys.path:
            sys.path.remove(str(TRAIN_DIR))
    return Executor(str(ENVIRONMENT_DIR), str(CATALOG_PATH))


def _replay_hash(executor: Any, sample_id: str, calls: Sequence[Dict[str, Any]]) -> str:
    session = executor.open(sample_id)
    for call in calls:
        session.call(call.get("name", ""), call.get("arguments") or {})
    return session.get_state_hash()


# ---------------------------------------------------------------------------
# Chấm điểm
# ---------------------------------------------------------------------------
def score_submission(file_name: str | Path | None = None) -> float:
    """Chấm file .jsonl và trả về điểm thang 100.

    Args:
        file_name: Tên file cần chấm. Mặc định ``submission.jsonl`` nằm cùng
            thư mục với ``validate.py``. Tên tương đối được hiểu là nằm trong
            thư mục của script; đường dẫn tuyệt đối được dùng nguyên vẹn.
    """
    if file_name is None:
        path = SCRIPT_DIR / "submission.jsonl"
    else:
        path = Path(file_name)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file submission: {path}")

    submissions = _load_submission(path)
    if not submissions:
        raise ValueError(f"{path} không có dòng submission nào")

    gold = _load_gold()
    fuzzy_by_tool = _load_fuzzy_arguments()
    executor = _load_executor()

    total_score = 0.0
    for sample_id, submission in submissions.items():
        if sample_id not in gold:
            raise KeyError(f"'{sample_id}' không có trong {GOLD_PATH}")

        submitted_calls = submission["tool_calls"]
        gold_calls = gold[sample_id]
        gold_hash = _replay_hash(executor, sample_id, gold_calls)
        matched = _maximum_matching(submitted_calls, gold_calls, fuzzy_by_tool)

        if not gold_calls:
            # Out-of-scope: chỉ đúng khi không gọi tool nào.
            total_score += 1.0 if not submitted_calls else 0.0
        elif submission["state_hash"] != gold_hash or matched == 0:
            total_score += 0.0
        else:
            precision = matched / len(submitted_calls)
            recall = matched / len(gold_calls)
            total_score += 2 * precision * recall / (precision + recall)

    return round(100 * total_score / len(submissions), 2)


def main() -> None:
    file_name = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Score: {score_submission(file_name):.2f}/100")


if __name__ == "__main__":
    main()
