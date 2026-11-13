"""Kiểm thử tự động toàn diện (Unit & End-to-End Test) cho pipeline mới của llm_phase (vLLM).
Bao gồm:
  1. config.py: kiểm tra các hằng số cố định (context length, temperature, max_turns, model_id...)
  2. prompt.py: input text transcript + user context -> messages
  3. model.py: khởi tạo cấu hình model vLLM
  4. get_tool.py: lấy tools khả dụng
  5. llm.py: build request payload & parse responses cho vLLM
  6. agent.py & submission.py: chạy agent_loop tương tác với ToolEnv và lưu submission.jsonl
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Thêm đường dẫn root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_phase import (
    DATA_ROOT,
    MAX_CONTEXT_LENGTH,
    MAX_TURNS,
    MODEL_ID,
    TEMPERATURE,
    ModelConfig,
    agent_loop,
    build_messages_for_llm,
    build_request_payload,
    create_user_message,
    format_submission_entry,
    get_available_tool,
    get_catalog,
    load_submission,
    parse_llm_response,
    run_sample_pipeline,
    save_submission_sample,
)


def test_config():
    print("[1] Kiểm tra config.py...")
    assert MAX_CONTEXT_LENGTH == 16384, f"Context length phải là 2^14 (16384), nhận được: {MAX_CONTEXT_LENGTH}"
    assert TEMPERATURE == 0.0, f"Temperature phải là 0.0, nhận được: {TEMPERATURE}"
    assert MAX_TURNS == 10, f"Max turns phải là 10, nhận được: {MAX_TURNS}"
    assert "qwen3.5" in MODEL_ID or MODEL_ID == "qwen3.5:4b", f"Model ID không đúng: {MODEL_ID}"
    assert DATA_ROOT.exists(), f"DATA_ROOT không tồn tại: {DATA_ROOT}"
    print(f"    ✔ Cấu hình hợp lệ: Context={MAX_CONTEXT_LENGTH}, Temp={TEMPERATURE}, Turns={MAX_TURNS}, Model={MODEL_ID}")


def test_prompt_generation():
    print("\n[2] Kiểm tra prompt.py...")
    # Đọc thử 1 sample từ train.jsonl
    train_file = DATA_ROOT / "train" / "train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    audio_text = sample.get("audio_text", "xem thời tiết hôm nay")
    context = sample.get("context", {})

    # 2.1. Tạo single user message
    user_msg = create_user_message(audio_text=audio_text, context=context)
    assert user_msg["role"] == "user"
    assert audio_text in user_msg["content"]
    assert "installed_apps" not in user_msg["content"], "User message không được chứa tools/apps!"
    print("    ✔ create_user_message hoạt động đúng (chỉ chứa context người dùng + audio text).")

    # 2.2. Tạo list messages đầy đủ cho LLM
    messages = build_messages_for_llm(audio_text=audio_text, context=context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    print("    ✔ build_messages_for_llm tạo messages hợp lệ (System + User).")


def test_model_and_llm():
    print("\n[3] Kiểm tra model.py và llm.py (chuẩn vLLM)...")
    vllm_cfg = ModelConfig.from_env()

    catalog = get_catalog(split="train")
    # Lấy sample đầu tiên
    train_file = DATA_ROOT / "train" / "train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    tools = get_available_tool(sample.get("context", {}), tool_catalog=catalog)

    messages = [{"role": "user", "content": "Kiểm tra thời tiết"}]

    # 3.1. Kiểm tra build payload cho vLLM
    vllm_payload = build_request_payload(messages=messages, tools=tools, model=vllm_cfg)
    assert vllm_payload["model"] == vllm_cfg.name
    assert "tools" in vllm_payload
    assert vllm_payload["temperature"] == 0.0
    print("    ✔ build_request_payload cho vLLM hợp lệ.")

    # 3.2. Kiểm tra parse response (OpenAI / vLLM format)
    mock_vllm_resp = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Tôi sẽ tra thời tiết.",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "weather.get_forecast",
                        "arguments": json.dumps({"date": "2026-11-29"})
                    }
                }]
            }
        }]
    }
    thought, parsed_tools, assistant_msg = parse_llm_response(mock_vllm_resp)
    assert thought == "Tôi sẽ tra thời tiết."
    assert len(parsed_tools) == 1
    assert parsed_tools[0]["name"] == "weather.get_forecast"
    assert parsed_tools[0]["arguments"] == {"date": "2026-11-29"}
    assert assistant_msg["role"] == "assistant"
    print("    ✔ parse_llm_response cho vLLM format hoạt động chuẩn xác.")


def test_agent_and_submission():
    print("\n[4] Kiểm tra agent.py và submission.py kết nối ToolEnv...")
    split = "train"
    sys.path.insert(0, str(DATA_ROOT / split))
    from zaic_tools import ToolEnv

    sample_id = "train_0001"
    train_file = DATA_ROOT / split / f"{split}.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        
    env = ToolEnv(split, sample_id, root=DATA_ROOT)

    mock_resp_1 = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Bước 1: Tra cứu thời tiết ngày 29-11.",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "weather.get_forecast",
                        "arguments": json.dumps({"date": "2026-11-29"})
                    }
                }]
            }
        }]
    }

    mock_resp_2 = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Đã xem xong dự báo thời tiết, không cần làm gì thêm.",
                "tool_calls": []
            }
        }]
    }

    calls = 0
    def mock_send(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return mock_resp_1
        return mock_resp_2

    test_sub_file = Path("result/test_agent_vllm_submission.jsonl")
    if test_sub_file.exists():
        test_sub_file.unlink()

    with patch("src.llm_phase.llm.send_llm_request", side_effect=mock_send):
        model = ModelConfig(name="qwen3.5:4b", endpoint_url="http://mock-vllm:8000/v1/chat/completions")
        res = run_sample_pipeline(
            sample_id=sample_id,
            audio_text=sample["audio_text"],
            context=sample["context"],
            env=env,
            model=model,
            split=split,
            submission_path=test_sub_file,
            verbose=False
        )

    assert res["id"] == sample_id
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["name"] == "weather.get_forecast"
    assert res["state_hash"] is not None
    assert res["turns"] == 2
    print("    ✔ agent_loop tương tác hoàn chỉnh qua vLLM endpoint với ToolEnv.")

    # Kiểm tra submission.py lưu file jsonl
    records = load_submission(test_sub_file)
    assert sample_id in records
    saved = records[sample_id]
    assert saved["id"] == sample_id
    assert saved["tool_calls"] == res["tool_calls"]
    assert saved["state_hash"] == res["state_hash"]
    print("    ✔ submission.py ghi và đọc file jsonl chuẩn định dạng ZAC.")

    if test_sub_file.exists():
        test_sub_file.unlink()


if __name__ == "__main__":
    print("=" * 65)
    print("   CHẠY KIỂM THỬ TOÀN DIỆN PIPELINE MỚI LLM_PHASE (VLLM)")
    print("=" * 65)
    test_config()
    test_prompt_generation()
    test_model_and_llm()
    test_agent_and_submission()
    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ CÁC BÀI TEST ĐỀU THÀNH CÔNG RỰC RỠ 100%!")
    print("=" * 65)
