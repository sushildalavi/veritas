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
    monkeypatch.setenv("VERITAS_CACHE_TTL_SECONDS", "45")
    monkeypatch.setenv("VERITAS_EVIDENCE_CORPUS", "custom.jsonl")

    settings = load_project_settings()

    assert isinstance(settings, ProjectSettings)
    assert settings.verifier_backend == "sklearn"
    assert settings.cache_ttl_seconds == 45
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
    assert settings.demo_top_k == 3
    assert settings.evidence_corpus_path == "custom.jsonl"
