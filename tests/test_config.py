from pathlib import Path

from core.config import ProjectSettings, load_project_settings, load_yaml_config


def test_load_yaml_config_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_yaml_config(tmp_path / "missing.yaml") == {}


def test_load_project_settings_honors_env_overrides(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        """
api:
  host: 127.0.0.1
  port: 9000
cache:
  ttl_seconds: 30
demo:
  top_k: 3
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VERITAS_CONFIG", str(config_path))
    monkeypatch.setenv("VERITAS_VERIFIER_BACKEND", "sklearn")
    monkeypatch.setenv("VERITAS_RETRIEVAL_BACKEND", "bm25_hashing_hybrid")
    monkeypatch.setenv("VERITAS_USE_NEURAL_RETRIEVAL", "true")
    monkeypatch.setenv("VERITAS_RERANKER_BACKEND", "cross_encoder")
    monkeypatch.setenv("VERITAS_USE_CROSS_ENCODER", "true")
    monkeypatch.setenv("VERITAS_CROSS_ENCODER_MODEL", "fake-reranker")
    monkeypatch.setenv("VERITAS_BM25_TOP_K", "100")
    monkeypatch.setenv("VERITAS_DENSE_TOP_K", "80")
    monkeypatch.setenv("VERITAS_TITLE_TOP_K", "25")
    monkeypatch.setenv("VERITAS_QUERY_EXPANSION_TOP_K", "15")
    monkeypatch.setenv("VERITAS_INCLUDE_TITLE_IN_INDEX", "true")
    monkeypatch.setenv("VERITAS_CACHE_TTL_SECONDS", "45")
    monkeypatch.setenv("VERITAS_EVIDENCE_CORPUS", "custom.jsonl")

    settings = load_project_settings()

    assert isinstance(settings, ProjectSettings)
    assert settings.verifier_backend == "sklearn"
    assert settings.retrieval_backend == "bm25_hashing_hybrid"
    assert settings.use_neural_retrieval is True
    assert settings.reranker_backend == "cross_encoder"
    assert settings.use_cross_encoder is True
    assert settings.cross_encoder_model == "fake-reranker"
    assert settings.bm25_top_k == 100
    assert settings.dense_top_k == 80
    assert settings.title_top_k == 25
    assert settings.query_expansion_top_k == 15
    assert settings.include_title_in_index is True
    assert settings.cache_ttl_seconds == 45
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
    assert settings.demo_top_k == 3
    assert settings.evidence_corpus_path == "custom.jsonl"


def test_load_project_settings_includes_transformer_checkpoint(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        """
transformer_checkpoint: checkpoints/transformer_verifier
retrieval:
  backend: bm25_only
  use_neural_retrieval: false
  bm25_top_k: 40
  dense_top_k: 30
  title_top_k: 10
  query_expansion_top_k: 12
ranking:
  reranker_backend: heuristic
  use_cross_encoder: false
  cross_encoder_model: cross-encoder/ms-marco-MiniLM-L-6-v2
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VERITAS_CONFIG", str(config_path))

    settings = load_project_settings()

    assert settings.transformer_checkpoint == "checkpoints/transformer_verifier"
    assert settings.retrieval_backend == "bm25_only"
    assert settings.use_neural_retrieval is False
    assert settings.bm25_top_k == 40
    assert settings.dense_top_k == 30
    assert settings.title_top_k == 10
    assert settings.query_expansion_top_k == 12
    assert settings.reranker_backend == "heuristic"
    assert settings.use_cross_encoder is False


def test_load_project_settings_reads_verifier_thresholds_from_yaml(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        """
verifier:
  aggregation: per_passage_max
  support_threshold: 0.61
  refute_threshold: 0.42
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VERITAS_CONFIG", str(config_path))

    settings = load_project_settings()

    assert settings.verifier_aggregation == "per_passage_max"
    assert settings.support_threshold == 0.61
    assert settings.refute_threshold == 0.42
