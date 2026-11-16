"""ASR có nghe đúng các tham số tool trên tập test (không có transcript)?

Tập test không có ``audio_text``, nhưng ``history`` có cấu trúc::

    user      (audio u_k)       "tìm giúp tôi gạo ST25"
    assistant (tool_calls)      grocery.search(query="gạo ST25")

Tham số chuỗi của lượt assistant (``query``, ``symbol``, ``text``...) phần lớn
là lời người dùng vừa nói ở lượt u_k. Script chạy ASR trên u_k rồi kiểm tra
từng tham số có xuất hiện trong text hay không:

* ``recall từ``: tỉ lệ từ của tham số có trong text ASR;
* ``khớp đủ``: tỉ lệ tham số có **mọi** từ xuất hiện.

Có những tham số không được nói nguyên văn (ví dụ "nhà" -> địa chỉ đầy đủ),
nên con số tuyệt đối thấp hơn thực tế. Ý nghĩa nằm ở phép **so sánh**: app
cũ và app mới, model gốc và model fine-tune, trên cùng các lượt nói.

Chạy từ thư mục gốc repo (cần GPU, khoảng 5-10 phút mỗi model)::

    python -m src.asr.audit.slot_probe
    python -m src.asr.audit.slot_probe --models phowhisper_large
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ..api import get_sample, get_user_request_from_audio, list_sample_ids
from ..config import DATA_ROOT, MODELS, PROJECT_ROOT
from ..finetune.model import normalize
from .compare_splits import catalog_apps, table

OUTPUT = PROJECT_ROOT / "results" / "asr" / "audit"

#: Giá trị là ID, ngày giờ hoặc mã hệ thống: không phải lời người dùng nói.
NOT_SPOKEN = re.compile(r"^[a-z]+_[\w]+$|^\d{4}-\d{2}-\d{2}|^[a-z]{2}$")


def enum_params(split: str, root: Path) -> set[str]:
    """``"tool.tham_số"`` có danh sách giá trị cố định (enum) trong catalog.

    Các giá trị này là mã (``buy``, ``cool``...), người dùng nói "mua",
    "chế độ mát"..., nên không so được với text ASR.
    """
    path = root / split / f"{split}_tools_catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return {f"{tool['name']}.{key}"
            for app in catalog["apps"] for tool in app["tools"]
            for key, spec in tool.get("parameters", {})
            .get("properties", {}).items() if "enum" in spec}


def spoken_slots(sample: dict,
                 skip: set[str]) -> list[tuple[int, str, str, str]]:
    """Tham số có thể đã được nói ra, gắn với lượt user ngay trước nó.

    Returns:
        Danh sách ``(chỉ số lượt user, app, "tool.tham_số", giá trị)``;
        chỉ số tính trong các lượt user của history (0 = u1).
    """
    slots, user_index = [], -1
    for turn in sample["history"]:
        if turn.get("role") == "user":
            user_index += 1
            continue
        for call in turn.get("tool_calls") or []:
            app = call["name"].split(".")[0]
            slots += [(user_index, app, f"{call['name']}.{key}", value)
                      for key, value in call["arguments"].items()
                      if isinstance(value, str) and user_index >= 0
                      and f"{call['name']}.{key}" not in skip
                      and not NOT_SPOKEN.match(value)]
    return slots


def token_recall(value: str, text: str) -> tuple[int, int]:
    """(số từ của ``value`` có trong ``text``, tổng số từ của ``value``)."""
    words, heard = normalize(value).split(), set(normalize(text).split())
    return sum(word in heard for word in words), len(words)


def main() -> None:
    """Chạy ASR trên các lượt có tham số, tổng hợp theo app cũ/mới."""
    parser = argparse.ArgumentParser(description="Đo ASR trên tham số tool")
    parser.add_argument("--split", default="public_test")
    parser.add_argument("--train-split", default="train",
                        help="split dùng để xác định app nào là 'cũ'")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS),
                        default=["phowhisper_large", "phowhisper_large_zac"])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    old_apps = set(catalog_apps(args.train_split, args.data_root))
    samples = [get_sample(sid, args.split, args.data_root)
               for sid in list_sample_ids(args.split, args.data_root)]
    skip = enum_params(args.split, args.data_root)
    probes = [(sample, spoken_slots(sample, skip)) for sample in samples]
    probes = [(sample, slots) for sample, slots in probes if slots]
    n_slots = sum(len(slots) for _, slots in probes)
    print(f"{len(probes)} mẫu, {n_slots} tham số cần kiểm tra")

    rows, examples = [], defaultdict(list)
    for key in args.models:
        model_dir = Path(MODELS[key].model_id)
        if model_dir.is_absolute() and not model_dir.exists():
            print(f"Bỏ qua {key}: chưa có {model_dir}")
            continue
        # stats[nhóm] = [số từ nghe đúng, tổng số từ, số tham số khớp đủ,
        #                tổng số tham số]
        stats = defaultdict(lambda: [0, 0, 0, 0])
        for sample, slots in probes:
            _, history = get_user_request_from_audio(
                sample["id"], args.split, args.data_root, model=key)
            for index, app, slot, value in slots:
                hit, total = token_recall(value, history[index])
                group = "app mới" if app not in old_apps else "app cũ"
                for name in (group, f"{group}: {app}"):
                    stats[name][0] += hit
                    stats[name][1] += total
                    stats[name][2] += hit == total
                    stats[name][3] += 1
                if hit < total and app not in old_apps:
                    examples[key].append(f"`{slot}` = \"{value}\" -> ASR: "
                                         f"\"{history[index]}\"")
        rows += [[key, name, s[3], f"{100 * s[0] / s[1]:.1f}",
                  f"{100 * s[2] / s[3]:.1f}"]
                 for name, s in sorted(stats.items())]

    report = ("# ASR trên tham số tool của tập test\n\n"
              + table(["model", "nhóm", "số tham số", "recall từ %",
                       "khớp đủ %"], rows)
              + "".join(f"\n\n## Lỗi ở app mới: {key} (20 ví dụ đầu)\n\n"
                        + "\n".join(f"- {line}" for line in lines[:20])
                        for key, lines in examples.items()) + "\n")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"slot_probe_{args.split}.md"
    path.write_text(report, encoding="utf-8")
    (OUTPUT / f"slot_probe_{args.split}_errors.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"Đã lưu: {path}")


if __name__ == "__main__":
    main()
