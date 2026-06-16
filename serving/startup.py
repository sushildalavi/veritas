"""Startup checks for the Veritas API service."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def check_artifacts() -> dict[str, object]:
    """Return a status dict for all expected artifacts."""
    checks: dict[str, object] = {}

    verifier_path = ROOT / "checkpoints" / "transformer_verifier_clean"
    checks["verifier_checkpoint"] = verifier_path.exists()
    if not verifier_path.exists():
        LOGGER.warning(
            "Verifier checkpoint missing at %s. "
            "Run: python3 scripts/train_transformer_verifier_clean.py",
            verifier_path,
        )

    onnx_path = ROOT / "checkpoints" / "transformer_verifier_clean_onnx"
    checks["onnx_checkpoint"] = onnx_path.exists()

    mlx_verifier_path = ROOT / "checkpoints" / "mlx_lora_verifier"
    checks["mlx_lora_verifier"] = mlx_verifier_path.exists()

    mlx_explanation_path = ROOT / "adapters" / "mlx_qwen_veritas_lora"
    checks["mlx_explanation_adapter"] = mlx_explanation_path.exists()

    phi3_qlora_path = ROOT / "checkpoints" / "phi3_qlora_adapter"
    checks["phi3_qlora_adapter"] = phi3_qlora_path.exists()
    if not phi3_qlora_path.exists():
        checks["phi3_qlora_note"] = "CUDA required; train on Colab via notebooks/phi3_colab_training.ipynb"

    final_results = ROOT / "reports" / "final_veritas_metrics_summary.json"
    checks["final_results_report"] = final_results.exists()

    demo_corpus = ROOT / "data" / "demo_corpus.jsonl"
    checks["demo_corpus"] = demo_corpus.exists()

    return checks


def log_startup_report(checks: dict[str, object]) -> None:
    ok = [k for k, v in checks.items() if v is True]
    missing = [k for k, v in checks.items() if v is False]
    notes = {k: v for k, v in checks.items() if isinstance(v, str)}

    if ok:
        LOGGER.info("Startup OK: %s", ", ".join(ok))
    if missing:
        LOGGER.warning("Startup missing artifacts: %s", ", ".join(missing))
    for k, v in notes.items():
        LOGGER.info("Note [%s]: %s", k, v)
