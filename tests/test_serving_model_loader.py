from core.config import ProjectSettings
from serving.model_loader import _load_reranker_runtime


def test_load_reranker_runtime_none_backend() -> None:
    settings = ProjectSettings(reranker_backend="none", use_cross_encoder=False)

    runtime = _load_reranker_runtime(settings)

    assert runtime.reranker is None
    assert runtime.reranker_backend == "none"
    assert runtime.fallback_used is False


def test_load_reranker_runtime_heuristic_backend() -> None:
    settings = ProjectSettings(reranker_backend="heuristic", use_cross_encoder=False)

    runtime = _load_reranker_runtime(settings)

    assert runtime.reranker is not None
    assert runtime.reranker_backend == "heuristic"
    assert runtime.fallback_used is False


def test_load_reranker_runtime_cross_encoder_backend_uses_mock(monkeypatch) -> None:
    class FakeCrossEncoderReranker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.backend_name = "cross-encoder"

        def rank(self, claim, evidence_list):  # noqa: ANN001
            return list(reversed(evidence_list))

    monkeypatch.setattr("serving.model_loader.CrossEncoderReranker", FakeCrossEncoderReranker)

    settings = ProjectSettings(
        reranker_backend="cross_encoder",
        use_cross_encoder=True,
        cross_encoder_model="fake-model",
    )

    runtime = _load_reranker_runtime(settings)

    assert runtime.reranker is not None
    assert getattr(runtime.reranker, "model_name", None) == "fake-model"
    assert runtime.reranker_backend == "cross_encoder"
    assert runtime.cross_encoder_model == "fake-model"
    assert runtime.fallback_used is False


def test_load_reranker_runtime_cross_encoder_falls_back(monkeypatch) -> None:
    class ExplodingCrossEncoderReranker:
        def __init__(self, model_name: str) -> None:  # noqa: ARG002
            raise RuntimeError("download failed")

    monkeypatch.setattr("serving.model_loader.CrossEncoderReranker", ExplodingCrossEncoderReranker)

    settings = ProjectSettings(
        reranker_backend="cross_encoder",
        use_cross_encoder=True,
        cross_encoder_model="fake-model",
    )

    runtime = _load_reranker_runtime(settings)

    assert runtime.reranker is not None
    assert runtime.reranker_backend == "heuristic"
    assert runtime.cross_encoder_model == "fake-model"
    assert runtime.fallback_used is True
