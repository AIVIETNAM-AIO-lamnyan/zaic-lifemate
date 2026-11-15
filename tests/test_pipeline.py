import argparse
import json
import sys
from pathlib import Path

# Đảm bảo import được module src từ mọi thư mục làm việc
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_phase.agent import run_sample_pipeline
from src.llm_phase.config import DATA_ROOT, RESULT_DIR

SPLIT = "train"
NUM_SAMPLES = 2
TRANSCRIPT_SOURCE = "audio_text"


def get_sample_ids(num_samples: int | None = None, split: str = SPLIT) -> list[str]:
    """Lấy danh sách sample ID từ dataset.
    
    Nếu num_samples là None hoặc <= 0, lấy toàn bộ sample ID có trong file.
    """
    data_file = DATA_ROOT / split / f"{split}.jsonl"
    if not data_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu mẫu: {data_file}")
    sample_ids = []
    with data_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_ids.append(json.loads(line)["id"])
                if num_samples is not None and num_samples > 0 and len(sample_ids) >= num_samples:
                    break
    return sample_ids


def parse_args():
    parser = argparse.ArgumentParser(description="LifeMate Pipeline Test Runner")
    parser.add_argument(
        "sample_id",
        nargs="?",
        default=None,
        help="Sample ID cần test (ví dụ: train_001 hoặc train_0001). Nếu để trống, sẽ lấy num_samples.",
    )
    parser.add_argument(
        "--id",
        dest="sample_id_flag",
        default=None,
        help="Sample ID cần test qua cờ tùy chọn (ví dụ: --id train_0001).",
    )
    parser.add_argument(
        "--split",
        default=SPLIT,
        help=f"Split dữ liệu ('train' hoặc 'public_test'). Mặc định: '{SPLIT}'.",
    )
    parser.add_argument(
        "-n",
        "-N",
        "--num-samples",
        dest="num_samples",
        type=int,
        default=None,
        help="Số lượng mẫu test khi không truyền sample_id (mặc định: 2). Truyền -1 hoặc 0 để lấy toàn bộ.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Chạy toàn bộ mẫu có trong split được chỉ định (bỏ qua -N).",
    )
    parser.add_argument(
        "--source",
        default=TRANSCRIPT_SOURCE,
        help=f"Trường transcript nguồn (mặc định: '{TRANSCRIPT_SOURCE}').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Đường dẫn file kết quả (mặc định: result/test_llm.json).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Ép buộc chạy chế độ Mock LLM (tiện cho unit test khi vLLM chưa khởi chạy).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        default=False,
        help="Vô hiệu hóa tự động fallback sang Mock khi vLLM offline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_id = args.sample_id or args.sample_id_flag
    split = args.split or SPLIT
    output_path = Path(args.output) if args.output else RESULT_DIR / "test_llm.json"
    transcript_source = args.source or TRANSCRIPT_SOURCE

    # Xây dựng danh sách sample_ids đưa vào agent
    if target_id is not None:
        sample_ids = [target_id]
    elif args.all or (args.num_samples is not None and args.num_samples <= 0):
        sample_ids = get_sample_ids(num_samples=None, split=split)
    else:
        num = args.num_samples if args.num_samples is not None else NUM_SAMPLES
        sample_ids = get_sample_ids(num_samples=num, split=split)

    # Chạy duy nhất hàm run_sample_pipeline cho danh sách sample_ids
    run_sample_pipeline(
        sample_ids=sample_ids,
        split=split,
        transcript_source=transcript_source,
        output_path=output_path,
        data_root=DATA_ROOT,
        mock=args.mock,
        allow_fallback=not args.no_fallback,
    )


if __name__ == "__main__":
    main()
