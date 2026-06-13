"""Central project settings loaded from YAML and environment variables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import os

import yaml


@dataclass(frozen=True)
class ProjectSettings:
    verifier_backend: str = "auto"
    sklearn_checkpoint: str = "checkpoints/verifier"
    transformer_checkpoint: str = "checkpoints/transformer_verifier"
    transformer_clean_checkpoint: str = "checkpoints/transformer_verifier_clean"
    deberta_checkpoint: str = "checkpoints/deberta_verifier_clean"
    mlx_lora_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    mlx_lora_adapter: str = "checkpoints/mlx_lora_verifier"
    legacy_verifier_checkpoint: str | None = None
    retrieval_backend: str = "bm25_only"
    use_neural_retrieval: bool = False
    reranker_backend: str = "none"
    use_cross_encoder: bool = False
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    optional_research_embedding_model: str = "BAAI/bge-m3"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    explanation_mode: str = "template"
    max_candidates_for_reranking: int = 3
    strict_json_output: bool = True
    max_evidence: int = 5
    cache_ttl_seconds: int = 120
    log_level: str = "INFO"
    evidence_corpus_path: str = "data/processed/evidence_corpus.jsonl"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    demo_top_k: int = 5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}


def load_project_settings(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProjectSettings:
    env = dict(environ or os.environ)
    resolved_config_path = Path(config_path or env.get("VERITAS_CONFIG", "configs/serving.yaml"))
    yaml_config = load_yaml_config(resolved_config_path)

    def env_or_yaml(env_key: str, yaml_path: tuple[str, ...], default: Any) -> Any:
        if env_key in env and env[env_key] != "":
            return env[env_key]
        value = _nested_get(yaml_config, yaml_path)
        return default if value is None else value

    return ProjectSettings(
        verifier_backend=str(env_or_yaml("VERITAS_VERIFIER_BACKEND", ("verifier_backend",), "auto")),
        sklearn_checkpoint=str(env_or_yaml("VERITAS_SKLEARN_CHECKPOINT", ("checkpoint", "output_dir"), "checkpoints/verifier")),
        transformer_checkpoint=str(env_or_yaml("VERITAS_TRANSFORMER_CHECKPOINT", ("transformer_checkpoint",), "checkpoints/transformer_verifier")),
        transformer_clean_checkpoint=str(env_or_yaml("VERITAS_TRANSFORMER_CLEAN_CHECKPOINT", ("transformer_clean_checkpoint",), "checkpoints/transformer_verifier_clean")),
        deberta_checkpoint=str(env_or_yaml("VERITAS_DEBERTA_CHECKPOINT", ("deberta_checkpoint",), "checkpoints/deberta_verifier_clean")),
        mlx_lora_model=str(env_or_yaml("VERITAS_MLX_LORA_MODEL", ("mlx_lora_model",), "mlx-community/Qwen2.5-1.5B-Instruct-4bit")),
        mlx_lora_adapter=str(env_or_yaml("VERITAS_MLX_LORA_ADAPTER", ("mlx_lora_adapter",), "checkpoints/mlx_lora_verifier")),
        legacy_verifier_checkpoint=_optional_str(env.get("VERITAS_VERIFIER_CHECKPOINT")),
        retrieval_backend=str(env_or_yaml("VERITAS_RETRIEVAL_BACKEND", ("retrieval", "backend"), "bm25_only")),
        use_neural_retrieval=_coerce_bool(env_or_yaml("VERITAS_USE_NEURAL_RETRIEVAL", ("retrieval", "use_neural_retrieval"), False)),
        reranker_backend=str(env_or_yaml("VERITAS_RERANKER_BACKEND", ("ranking", "reranker_backend"), "none")),
        use_cross_encoder=_coerce_bool(env_or_yaml("VERITAS_USE_CROSS_ENCODER", ("ranking", "use_cross_encoder"), False)),
        embedding_backend=str(env_or_yaml("VERITAS_EMBEDDING_BACKEND", ("dense", "backend"), "hashing")),
        embedding_model=str(
            env_or_yaml(
                "VERITAS_EMBEDDING_MODEL",
                ("retrieval", "embedding_model"),
                env_or_yaml("VERITAS_EMBEDDING_MODEL", ("dense", "model_name"), "sentence-transformers/all-MiniLM-L6-v2"),
            )
        ),
        optional_research_embedding_model=str(
            env_or_yaml(
                "VERITAS_RESEARCH_EMBEDDING_MODEL",
                ("retrieval", "research_embedding_model"),
                "BAAI/bge-m3",
            )
        ),
        cross_encoder_model=str(env_or_yaml("VERITAS_CROSS_ENCODER_MODEL", ("ranking", "cross_encoder_model"), "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        explanation_mode=str(env_or_yaml("VERITAS_EXPLANATION_MODE", ("explanation_mode",), "template")),
        max_candidates_for_reranking=int(env_or_yaml("VERITAS_MAX_CANDIDATES_FOR_RERANKING", ("max_candidates_for_reranking",), 3)),
        strict_json_output=_coerce_bool(env_or_yaml("VERITAS_STRICT_JSON_OUTPUT", ("strict_json_output",), True)),
        max_evidence=int(env_or_yaml("VERITAS_MAX_EVIDENCE", ("demo", "top_k"), 5)),
        cache_ttl_seconds=int(env_or_yaml("VERITAS_CACHE_TTL_SECONDS", ("cache", "ttl_seconds"), 120)),
        log_level=str(env_or_yaml("VERITAS_LOG_LEVEL", ("log_level",), "INFO")),
        evidence_corpus_path=str(env_or_yaml("VERITAS_EVIDENCE_CORPUS", ("evidence_corpus_path",), "data/processed/evidence_corpus.jsonl")),
        api_host=str(env_or_yaml("VERITAS_API_HOST", ("api", "host"), "0.0.0.0")),
        api_port=int(env_or_yaml("VERITAS_API_PORT", ("api", "port"), 8000)),
        demo_top_k=int(env_or_yaml("VERITAS_DEMO_TOP_K", ("demo", "top_k"), 5)),
    )


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}
