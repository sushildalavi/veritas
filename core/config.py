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
    verifier_checkpoint: str | None = None
    sklearn_checkpoint: str = "checkpoints/verifier"
    transformer_checkpoint: str = "checkpoints/transformer_verifier"
    transformer_clean_checkpoint: str = "checkpoints/transformer_verifier_clean"
    challenger_verifier_checkpoint: str = "checkpoints/deberta_verifier_clean"
    deberta_checkpoint: str = "checkpoints/deberta_verifier_clean"
    mlx_lora_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    mlx_lora_adapter: str = "checkpoints/mlx_lora_verifier"
    vllm_base_url: str = "http://127.0.0.1:8001"
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vllm_api_key: str | None = None
    vllm_timeout_seconds: float = 30.0
    vllm_max_retries: int = 2
    vllm_max_new_tokens: int = 256
    explanation_backend: str = "vllm"
    legacy_verifier_checkpoint: str | None = None
    retrieval_backend: str = "bm25_only"
    use_neural_retrieval: bool = False
    reranker_backend: str = "none"
    use_cross_encoder: bool = False
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    research_embedding_model: str = "BAAI/bge-m3"
    optional_research_embedding_model: str = "BAAI/bge-m3"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_top_k: int = 20
    dense_top_k: int = 20
    title_top_k: int = 0
    query_expansion_top_k: int = 0
    rrf_top_k: int = 60
    rerank_top_k: int = 20
    final_top_k: int = 5
    include_title_in_index: bool = False
    include_metadata_window: bool = False
    verifier_aggregation: str = "per_passage_max"
    support_threshold: float = 0.55
    refute_threshold: float = 0.5
    explanation_mode: str = "template"
    max_candidates_for_reranking: int = 3
    num_explanation_candidates: int = 3
    strict_json_output: bool = True
    max_evidence: int = 5
    max_claim_length: int = 1000
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

    def env_or_yaml(env_key: str, yaml_paths: tuple[tuple[str, ...], ...], default: Any) -> Any:
        if env_key in env and env[env_key] != "":
            return env[env_key]
        for yaml_path in yaml_paths:
            value = _nested_get(yaml_config, yaml_path)
            if value is not None:
                return value
        return default

    return ProjectSettings(
        verifier_backend=str(env_or_yaml("VERITAS_VERIFIER_BACKEND", (("verifier_backend",),), "auto")),
        verifier_checkpoint=_optional_str(env_or_yaml("VERITAS_VERIFIER_CHECKPOINT", (("verifier_checkpoint",),), None)),
        sklearn_checkpoint=str(env_or_yaml("VERITAS_SKLEARN_CHECKPOINT", (("sklearn_checkpoint",), ("checkpoint", "output_dir")), "checkpoints/verifier")),
        transformer_checkpoint=str(env_or_yaml("VERITAS_TRANSFORMER_CHECKPOINT", (("transformer_checkpoint",),), "checkpoints/transformer_verifier")),
        transformer_clean_checkpoint=str(env_or_yaml("VERITAS_TRANSFORMER_CLEAN_CHECKPOINT", (("transformer_clean_checkpoint",),), "checkpoints/transformer_verifier_clean")),
        challenger_verifier_checkpoint=str(env_or_yaml("VERITAS_CHALLENGER_VERIFIER_CHECKPOINT", (("challenger_verifier_checkpoint",),), "checkpoints/deberta_verifier_clean")),
        deberta_checkpoint=str(env_or_yaml("VERITAS_DEBERTA_CHECKPOINT", (("deberta_checkpoint",),), "checkpoints/deberta_verifier_clean")),
        mlx_lora_model=str(env_or_yaml("VERITAS_MLX_LORA_MODEL", (("mlx_lora_model",),), "mlx-community/Qwen2.5-1.5B-Instruct-4bit")),
        mlx_lora_adapter=str(env_or_yaml("VERITAS_MLX_LORA_ADAPTER", (("mlx_lora_adapter",),), "checkpoints/mlx_lora_verifier")),
        vllm_base_url=str(env_or_yaml("VERITAS_VLLM_BASE_URL", (("vllm_base_url",), ("vllm", "base_url")), "http://127.0.0.1:8001")),
        vllm_model=str(env_or_yaml("VERITAS_VLLM_MODEL", (("vllm_model",), ("vllm", "model")), "Qwen/Qwen2.5-1.5B-Instruct")),
        vllm_api_key=_optional_str(env_or_yaml("VERITAS_VLLM_API_KEY", (("vllm_api_key",), ("vllm", "api_key")), None)),
        vllm_timeout_seconds=float(env_or_yaml("VERITAS_VLLM_TIMEOUT_SECONDS", (("vllm_timeout_seconds",), ("vllm", "timeout_seconds")), 30.0)),
        vllm_max_retries=int(env_or_yaml("VERITAS_VLLM_MAX_RETRIES", (("vllm_max_retries",), ("vllm", "max_retries")), 2)),
        vllm_max_new_tokens=int(env_or_yaml("VERITAS_VLLM_MAX_NEW_TOKENS", (("vllm_max_new_tokens",), ("vllm", "max_new_tokens")), 256)),
        explanation_backend=str(env_or_yaml("VERITAS_EXPLANATION_BACKEND", (("explanation_backend",),), "vllm")),
        legacy_verifier_checkpoint=_optional_str(env.get("VERITAS_VERIFIER_CHECKPOINT")),
        retrieval_backend=str(env_or_yaml("VERITAS_RETRIEVAL_BACKEND", (("retrieval_backend",), ("retrieval", "backend")), "bm25_only")),
        use_neural_retrieval=_coerce_bool(env_or_yaml("VERITAS_USE_NEURAL_RETRIEVAL", (("use_neural_retrieval",), ("retrieval", "use_neural_retrieval")), False)),
        reranker_backend=str(env_or_yaml("VERITAS_RERANKER_BACKEND", (("reranker_backend",), ("ranking", "reranker_backend")), "none")),
        use_cross_encoder=_coerce_bool(env_or_yaml("VERITAS_USE_CROSS_ENCODER", (("use_cross_encoder",), ("ranking", "use_cross_encoder")), False)),
        embedding_backend=str(env_or_yaml("VERITAS_EMBEDDING_BACKEND", (("embedding_backend",), ("dense", "backend")), "hashing")),
        embedding_model=str(
            env_or_yaml(
                "VERITAS_EMBEDDING_MODEL",
                (("embedding_model",), ("retrieval", "embedding_model"), ("dense", "model_name")),
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        ),
        optional_research_embedding_model=str(
            env_or_yaml(
                "VERITAS_RESEARCH_EMBEDDING_MODEL",
                (("research_embedding_model",), ("retrieval", "research_embedding_model")),
                "BAAI/bge-m3",
            )
        ),
        cross_encoder_model=str(env_or_yaml("VERITAS_CROSS_ENCODER_MODEL", (("cross_encoder_model",), ("ranking", "cross_encoder_model")), "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        bm25_top_k=int(env_or_yaml("VERITAS_BM25_TOP_K", (("bm25_top_k",), ("retrieval", "bm25_top_k")), 20)),
        dense_top_k=int(env_or_yaml("VERITAS_DENSE_TOP_K", (("dense_top_k",), ("retrieval", "dense_top_k")), 20)),
        title_top_k=int(env_or_yaml("VERITAS_TITLE_TOP_K", (("title_top_k",), ("retrieval", "title_top_k")), 0)),
        query_expansion_top_k=int(env_or_yaml("VERITAS_QUERY_EXPANSION_TOP_K", (("query_expansion_top_k",), ("retrieval", "query_expansion_top_k")), 0)),
        rrf_top_k=int(env_or_yaml("VERITAS_RRF_TOP_K", (("rrf_top_k",), ("retrieval", "rrf_top_k"), ("hybrid", "rrf_k")), 60)),
        rerank_top_k=int(env_or_yaml("VERITAS_RERANK_TOP_K", (("rerank_top_k",), ("retrieval", "rerank_top_k")), 20)),
        final_top_k=int(env_or_yaml("VERITAS_FINAL_TOP_K", (("final_top_k",), ("retrieval", "final_top_k")), 5)),
        include_title_in_index=_coerce_bool(env_or_yaml("VERITAS_INCLUDE_TITLE_IN_INDEX", (("include_title_in_index",), ("retrieval", "include_title_in_index")), False)),
        include_metadata_window=_coerce_bool(env_or_yaml("VERITAS_INCLUDE_METADATA_WINDOW", (("include_metadata_window",), ("retrieval", "include_metadata_window")), False)),
        verifier_aggregation=str(env_or_yaml("VERITAS_VERIFIER_AGGREGATION", (("verifier_aggregation",), ("verifier", "aggregation")), "per_passage_max")),
        support_threshold=float(env_or_yaml("VERITAS_SUPPORT_THRESHOLD", (("support_threshold",), ("verifier", "support_threshold")), 0.55)),
        refute_threshold=float(env_or_yaml("VERITAS_REFUTE_THRESHOLD", (("refute_threshold",), ("verifier", "refute_threshold")), 0.5)),
        explanation_mode=str(env_or_yaml("VERITAS_EXPLANATION_MODE", (("explanation_mode",),), "template")),
        max_candidates_for_reranking=int(env_or_yaml("VERITAS_MAX_CANDIDATES_FOR_RERANKING", (("max_candidates_for_reranking",),), 3)),
        num_explanation_candidates=int(env_or_yaml("VERITAS_NUM_EXPLANATION_CANDIDATES", (("num_explanation_candidates",),), 3)),
        strict_json_output=_coerce_bool(env_or_yaml("VERITAS_STRICT_JSON_OUTPUT", (("strict_json_output",),), True)),
        max_evidence=int(env_or_yaml("VERITAS_MAX_EVIDENCE", (("max_evidence",), ("demo", "top_k")), 5)),
        max_claim_length=int(env_or_yaml("VERITAS_MAX_CLAIM_LENGTH", (("max_claim_length",),), 1000)),
        cache_ttl_seconds=int(env_or_yaml("VERITAS_CACHE_TTL_SECONDS", (("cache_ttl_seconds",), ("cache", "ttl_seconds")), 120)),
        log_level=str(env_or_yaml("VERITAS_LOG_LEVEL", (("log_level",),), "INFO")),
        evidence_corpus_path=str(env_or_yaml("VERITAS_EVIDENCE_CORPUS", (("evidence_corpus_path",),), "data/processed/evidence_corpus.jsonl")),
        api_host=str(env_or_yaml("VERITAS_API_HOST", (("api_host",), ("api", "host")), "0.0.0.0")),
        api_port=int(env_or_yaml("VERITAS_API_PORT", (("api_port",), ("api", "port")), 8000)),
        demo_top_k=int(env_or_yaml("VERITAS_DEMO_TOP_K", (("demo_top_k",), ("demo", "top_k")), 5)),
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
