import pytest

from serving.onnx_verifier_backend import ONNXVerifierBackend, ort


@pytest.mark.skipif(ort is not None, reason="onnxruntime is installed; skip-path test not applicable")
def test_onnx_backend_raises_clear_error_without_onnxruntime(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
        ONNXVerifierBackend(tmp_path / "model.onnx", "checkpoints/transformer_verifier_clean")


@pytest.mark.skipif(ort is None, reason="onnxruntime not installed")
def test_onnx_backend_raises_for_missing_model(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ONNXVerifierBackend(tmp_path / "missing.onnx", "checkpoints/transformer_verifier_clean")
