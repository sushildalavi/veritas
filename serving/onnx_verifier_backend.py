"""Optional ONNX Runtime backend for the transformer verifier.

This module imports cleanly even when ``onnxruntime`` is not installed --
only constructing ``ONNXVerifierBackend`` requires it. This keeps
``onnxruntime`` an optional dependency: it is never required for tests or for
the default (PyTorch) verifier path.
"""

from __future__ import annotations

from pathlib import Path

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ort = None

import numpy as np
from transformers import AutoTokenizer


class ONNXVerifierBackend:
    """Runs the exported verifier ONNX model via ``onnxruntime.InferenceSession``."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with `pip install onnxruntime` "
                "to use ONNXVerifierBackend."
            )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. Run scripts/export_verifier_onnx.py first."
            )
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def predict_logits(self, texts: list[str], max_length: int = 512) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        outputs = self.session.run(
            ["logits"],
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )
        return outputs[0]
