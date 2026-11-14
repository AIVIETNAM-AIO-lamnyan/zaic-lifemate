"""So sánh tập train và tập test: nội dung, phương ngữ và chất lượng âm thanh.

Ba nhóm câu hỏi:

1. **Nội dung**: test có app/tool mới không? Lịch sử hội thoại dài ngắn ra sao?
2. **Phương ngữ**: trong train, phương ngữ gắn chặt với thành phố trong
   ``user_profile`` (Huế/Đà Nẵng: "rứa, ni, mô"; Cần Thơ/TP.HCM: "nghen,
   hông"). Vì vậy tỉ lệ thành phố là thước đo phương ngữ cho tập test (vốn
   không có transcript).
3. **Âm thanh**: độ dài, độ to, nhiễu nền, SNR, méo tiếng, dải tần
   (xem ``audio_stats.py``). Mỗi đặc trưng được so bằng thống kê KS: khoảng
   cách lớn nhất giữa hai hàm phân phối tích luỹ (0 = giống hệt, 1 = tách
   rời). Chỉ đánh dấu "lệch" khi KS vừa lớn hơn dao động ngẫu nhiên do cỡ
   mẫu (mức ý nghĩa 5 %), vừa từ 0,1 trở lên (đủ lớn để đáng bận tâm).

Chạy từ thư mục gốc repo (chỉ dùng CPU, vài phút)::

    python -m src.asr.audit.compare_splits
    python -m src.asr.audit.compare_splits --test public_test \\
        --predictions path/to/phowhisper_large.csv   # thêm WER theo SNR

Kết quả: ``results/asr/audit/compare_<train>_vs_<test>.md`` và một file CSV
đặc trưng của từng file âm thanh.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..api import get_sample, list_sample_ids
from ..config import DATA_ROOT, PROJECT_ROOT
from .audio_stats import audio_stats

OUTPUT = PROJECT_ROOT / "results" / "asr" / "audit"

#: Đặc trưng âm thanh được so sánh, kèm đơn vị hiển thị.
METRICS = {"duration_s": "s", "level_dbfs": "dBFS", "noise_dbfs": "dBFS",
           "snr_db": "dB", "speech_ratio": "", "clip_ratio": "",
           "rolloff_hz": "Hz", "hf_ratio": "", "centroid_hz": "Hz"}

#: Ngưỡng "có vấn đề" cho từng file: (tên, hàm kiểm tra).
FLAGS = {
    "dài hơn 30 s (Whisper cắt mất)": lambda s: s["duration_s"] > 30,
    "SNR < 15 dB (ồn)": lambda s: s["snr_db"] < 15,
    "SNR < 10 dB (rất ồn)": lambda s: s["snr_db"] < 10,
    "méo (> 0,1 % mẫu chạm trần)": lambda s: s["clip_ratio"] > 1e-3,
    "nhỏ tiếng (< -40 dBFS)": lambda s: s["level_dbfs"] < -40,
    "dải hẹp (rolloff < 4 kHz)": lambda s: s["rolloff_hz"] < 4000,
    "ít tiếng nói (< 30 % khung)": lambda s: s["speech_ratio"] < 0.3,
}


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def load_split(split: str, root: Path) -> list[dict]:
    """Mọi mẫu của một split."""
    return [get_sample(sid, split, root)
            for sid in list_sample_ids(split, root)]


def user_turns(sample: dict) -> list[dict]:
    """Các lượt nói của người dùng: history (u1, u2...) rồi yêu cầu cuối."""
    history = [t for t in sample["history"] if t.get("role") == "user"]
    return [*history, sample]


def catalog_apps(split: str, root: Path) -> dict[str, str]:
    """``{app_id: tên tiếng Việt}`` trong ``<split>_tools_catalog.json``."""
    path = root / split / f"{split}_tools_catalog.json"
    apps = json.loads(path.read_text(encoding="utf-8"))["apps"]
    return {app["id"]: app.get("name_vi", app["id"]) for app in apps}


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Khoảng cách Kolmogorov-Smirnov giữa hai mẫu (không cần scipy)."""
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_threshold(n_a: int, n_b: int) -> float:
    """Ngưỡng "lệch thật": max(0,1; giá trị tới hạn KS ở mức 5 %).

    Với mẫu nhỏ, hai tập giống nhau vẫn có KS lớn do ngẫu nhiên; giá trị tới
    hạn 1,36 * sqrt((n_a + n_b) / (n_a * n_b)) loại bỏ trường hợp đó.
    """
    return max(0.1, 1.36 * np.sqrt((n_a + n_b) / (n_a * n_b)))


def table(header: list[str], rows: list[list]) -> str:
    """Bảng Markdown."""
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in lines)


def pct(part: float, whole: float) -> str:
    """Phần trăm làm tròn 1 chữ số."""
    return f"{100 * part / whole:.1f} %" if whole else "-"


# ---------------------------------------------------------------------------
# Ba phần của báo cáo
# ---------------------------------------------------------------------------
def content_section(names: tuple[str, str], data: dict[str, list[dict]],
                    root: Path) -> str:
    """Số mẫu, độ dài history, app mới, phân bố thành phố."""
    a, b = names
    apps = {s: catalog_apps(s, root) for s in names}
    new_apps = sorted(set(apps[b]) - set(apps[a]))
    exposed = sum(bool(set(x["context"]["installed_apps"]) & set(new_apps))
                  for x in data[b])
    turns = {s: Counter(len(user_turns(x)) - 1 for x in data[s])
             for s in names}
    cities = {s: Counter(x["context"]["user_profile"].get("city")
                         for x in data[s]) for s in names}

    basic = table(["", a, b], [
        ["số mẫu", len(data[a]), len(data[b])],
        ["số lượt nói (file âm thanh)",
         *(sum(len(user_turns(x)) for x in data[s]) for s in names)],
        ["mẫu có history", *(pct(len(data[s]) - turns[s][0], len(data[s]))
                             for s in names)],
        ["số app trong catalog", len(apps[a]), len(apps[b])],
    ])
    city_rows = [[c, pct(cities[a][c], len(data[a])),
                  pct(cities[b][c], len(data[b]))]
                 for c in sorted(cities[a] | cities[b],
                                 key=lambda c: -cities[a][c])]
    new_list = ", ".join(f"{apps[b][k]} (`{k}`)"
                         for k in new_apps) or "không"
    return (
        f"## 1. Nội dung\n\n{basic}\n\n"
        f"**App chỉ có ở {b}: {len(new_apps)}/{len(apps[b])}**: "
        f"{new_list}.\n\n"
        f"{pct(exposed, len(data[b]))} mẫu {b} có cài ít nhất một app mới.\n\n"
        "## 2. Phương ngữ (đo qua thành phố của người dùng)\n\n"
        + table(["thành phố", a, b], city_rows) + "\n")


def audio_section(names: tuple[str, str],
                  stats: dict[str, list[dict]]) -> str:
    """Phân vị, KS và tỉ lệ file có vấn đề cho từng đặc trưng."""
    a, b = names
    threshold = ks_threshold(len(stats[a]), len(stats[b]))
    rows = []
    for metric, unit in METRICS.items():
        va, vb = (np.array([s[metric] for s in stats[x]]) for x in names)
        qa, qb = (np.percentile(v, [10, 50, 90]) for v in (va, vb))
        ks = ks_statistic(va, vb)
        rows.append([f"{metric} ({unit})" if unit else metric,
                     " / ".join(f"{q:.3g}" for q in qa),
                     " / ".join(f"{q:.3g}" for q in qb),
                     f"{ks:.3f}" + (" **lệch**" if ks >= threshold else "")])
    flag_rows = [[name, *(pct(sum(map(check, stats[x])), len(stats[x]))
                          for x in names)]
                 for name, check in FLAGS.items()]
    formats = {x: Counter(f"{s['sample_rate']} Hz, {s['channels']} kênh, "
                          f"{s['subtype']}" for s in stats[x]) for x in names}
    return (
        "## 3. Âm thanh\n\n"
        f"Định dạng: {a}: {dict(formats[a])}; {b}: {dict(formats[b])}.\n\n"
        + table(["đặc trưng", f"{a} p10 / p50 / p90",
                 f"{b} p10 / p50 / p90", "KS"], rows)
        + f"\n\nNgưỡng đánh dấu lệch: KS >= {threshold:.3f}."
        + "\n\n### Tỉ lệ file có vấn đề\n\n"
        + table(["vấn đề", a, b], flag_rows) + "\n")


def wer_by_snr(stats: list[dict], predictions: Path) -> str:
    """WER của model trên train theo nhóm SNR: nhiễu có làm sai nhiều hơn?"""
    import jiwer

    with predictions.open(encoding="utf-8-sig") as handle:
        rows = {r["utt_id"]: r for r in csv.DictReader(handle)}
    bins = [(-np.inf, 15, "< 15 dB"), (15, 25, "15-25 dB"),
            (25, 40, "25-40 dB"), (40, np.inf, ">= 40 dB")]
    out = []
    for low, high, name in bins:
        group = [rows[s["utt_id"]] for s in stats
                 if low <= s["snr_db"] < high and s["utt_id"] in rows]
        if group:
            wer = 100 * jiwer.wer([g["reference"] for g in group],
                                  [g["hypothesis"] for g in group])
            out.append([name, len(group), f"{wer:.1f}"])
    return ("### WER theo SNR (tập train, từ file dự đoán)\n\n"
            + table(["SNR", "số lượt nói", "WER %"], out) + "\n")


# ---------------------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------------------
def main() -> None:
    """Đo, so sánh và ghi báo cáo."""
    parser = argparse.ArgumentParser(description="So sánh train và test")
    parser.add_argument("--train", default="train")
    parser.add_argument("--test", default="public_test")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--predictions", type=Path,
                        help="CSV dự đoán ASR trên train (utt_id, "
                             "reference, hypothesis) để tính WER theo SNR")
    args = parser.parse_args()
    names = (args.train, args.test)

    data = {s: load_split(s, args.data_root) for s in names}
    stats: dict[str, list[dict]] = {}
    for split in names:
        turns = [(f"{x['id']}_u{k}" if k else x["id"], t["audio"])
                 for x in data[split]
                 for k, t in enumerate(user_turns(x)[:-1], start=1)]
        turns += [(x["id"], x["audio"]) for x in data[split]]
        print(f"Đang đo {len(turns)} file âm thanh của {split}...",
              flush=True)
        stats[split] = [{"utt_id": uid, "split": split, **audio_stats(
            args.data_root / split / rel)} for uid, rel in turns]

    report = (f"# So sánh {args.train} và {args.test}\n\n"
              + content_section(names, data, args.data_root) + "\n"
              + audio_section(names, stats))
    if args.predictions:
        report += "\n" + wer_by_snr(stats[args.train], args.predictions)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"compare_{args.train}_vs_{args.test}"
    (OUTPUT / f"{stem}.md").write_text(report, encoding="utf-8")
    rows = [*stats[args.train], *stats[args.test]]
    with (OUTPUT / f"{stem}.csv").open("w", newline="",
                                       encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"\nĐã lưu: {OUTPUT / stem}.md (+ .csv)")


if __name__ == "__main__":
    main()
