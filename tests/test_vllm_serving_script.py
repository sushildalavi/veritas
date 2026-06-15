from scripts.serve_vllm_explanations import build_vllm_command


def test_build_vllm_command_includes_model_and_port() -> None:
    command = build_vllm_command(
        model="Qwen/Test",
        host="0.0.0.0",
        port=9000,
        api_key="token",
        max_model_len=8192,
        tensor_parallel_size=2,
    )

    assert "--model" in command
    assert "Qwen/Test" in command
    assert "--port" in command
    assert "9000" in command
    assert "--api-key" in command
