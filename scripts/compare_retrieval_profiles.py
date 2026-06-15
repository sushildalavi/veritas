"""Compare retrieval profiles with honest skip reporting."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings
from scripts.eval_oracle_vs_retrieved_v2 import evaluate_oracle_vs_retrieved_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrieval profiles for Veritas.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--split-prefixes", default="fever_test,scifact_test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--report-json", default="reports/retrieval_profile_comparison.json")
    parser.add_argument("--report-md", default="reports/retrieval_profile_comparison.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    base_settings = load_project_settings(args.config)
    split_prefixes = tuple(part.strip() for part in args.split_prefixes.split(",") if part.strip())
    profiles = [
        (
            "bm25_only",
            replace(
                base_settings,
                retrieval_backend="bm25_only",
                reranker_backend="none",
                use_cross_encoder=False,
            ),
        ),
        (
            "dense_only",
            replace(
                base_settings,
                retrieval_backend="dense_only",
                embedding_backend="hashing",
                use_neural_retrieval=False,
                reranker_backend="none",
                use_cross_encoder=False,
            ),
        ),
        (
            "hybrid_bm25_dense",
            replace(
                base_settings,
                retrieval_backend="bm25_hashing_hybrid",
                reranker_backend="none",
                use_cross_encoder=False,
                query_expansion_top_k=0,
            ),
        ),
        (
            "hybrid_with_query_expansion",
            replace(
                base_settings,
                retrieval_backend="bm25_hashing_hybrid",
                reranker_backend="none",
                use_cross_encoder=False,
                query_expansion_top_k=max(base_settings.query_expansion_top_k, 10),
            ),
        ),
        (
            "hybrid_with_reranker",
            replace(
                base_settings,
                retrieval_backend="bm25_hashing_hybrid",
                reranker_backend="cross_encoder",
                use_cross_encoder=True,
                query_expansion_top_k=max(base_settings.query_expansion_top_k, 10),
            ),
        ),
    ]

    report = {
        "config_path": args.config,
        "suffix": args.suffix,
        "split_prefixes": list(split_prefixes),
        "max_examples": args.max_examples,
        "profiles": {},
    }
    for profile_name, settings in profiles:
        started = time.perf_counter()
        if profile_name == "hybrid_with_reranker" and _cpu_only_environment():
            report["profiles"][profile_name] = {
                "status": "skipped",
                "reason": "CPU-only environment; cross-encoder reranker profile skipped for runtime budget",
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "settings": {
                    "retrieval_backend": settings.retrieval_backend,
                    "embedding_backend": settings.embedding_backend,
                    "reranker_backend": settings.reranker_backend,
                    "use_cross_encoder": settings.use_cross_encoder,
                    "query_expansion_top_k": settings.query_expansion_top_k,
                },
            }
            continue
        try:
            profile_report = evaluate_oracle_vs_retrieved_v2(
                settings=settings,
                checkpoint=settings.transformer_clean_checkpoint,
                data_dir=Path(args.data_dir),
                suffix=args.suffix,
                split_prefixes=split_prefixes,
                top_k=args.top_k,
                max_examples=args.max_examples,
                config_path=args.config,
            )
            report["profiles"][profile_name] = {
                "status": "measured",
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "retrieval_backend": profile_report["retrieval_backend"],
                "reranker_backend": profile_report["reranker_backend"],
                "retrieval_metrics": profile_report["retrieval_metrics"],
                "retrieved_bundle_macro_f1": profile_report["retrieved"]["bundle"]["macro_f1"],
                "retrieved_per_passage_macro_f1": profile_report["retrieved"]["per_passage_max"]["macro_f1"],
                "retrieved_bundle_nei_false_positive_rate": profile_report["retrieved"]["bundle"]["nei_false_positive_rate"],
                "retrieved_per_passage_nei_false_positive_rate": profile_report["retrieved"]["per_passage_max"]["nei_false_positive_rate"],
                "latency_ms": {
                    "bundle_p50": profile_report["retrieved"]["bundle"]["latency_ms"]["p50"],
                    "per_passage_p50": profile_report["retrieved"]["per_passage_max"]["latency_ms"]["p50"],
                },
                "settings": profile_report["settings"],
                "git_commit": profile_report["git_commit"],
            }
        except Exception as exc:
            report["profiles"][profile_name] = {
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=2),
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "settings": {
                    "retrieval_backend": settings.retrieval_backend,
                    "embedding_backend": settings.embedding_backend,
                    "reranker_backend": settings.reranker_backend,
                    "use_cross_encoder": settings.use_cross_encoder,
                    "query_expansion_top_k": settings.query_expansion_top_k,
                },
            }

    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print("retrieval profile comparison complete")


def _cpu_only_environment() -> bool:
    try:
        import torch

        return not bool(torch.cuda.is_available())
    except Exception:
        return True


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieval Profile Comparison",
        "",
        f"- Config path: `{report['config_path']}`",
        f"- Split prefixes: `{', '.join(report['split_prefixes'])}`",
        f"- Max examples: {report['max_examples']}",
        "",
        "| profile | status | retrieval_backend | reranker_backend | recall@10 | ndcg@10 | per_passage_macro_f1 | runtime_seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for profile_name, payload in report["profiles"].items():
        if payload["status"] == "measured":
            lines.append(
                f"| {profile_name} | measured | {payload['retrieval_backend']} | {payload['reranker_backend']} | "
                f"{payload['retrieval_metrics']['recall@10']} | {payload['retrieval_metrics']['ndcg@10']} | "
                f"{payload['retrieved_per_passage_macro_f1']} | {payload['runtime_seconds']} |"
            )
        else:
            lines.append(
                f"| {profile_name} | skipped | {payload['settings'].get('retrieval_backend')} | "
                f"{payload['settings'].get('reranker_backend')} | - | - | - | {payload['runtime_seconds']} |"
            )
    lines.extend(["", "## Skip notes", ""])
    for profile_name, payload in report["profiles"].items():
        if payload["status"] == "skipped":
            lines.append(f"- `{profile_name}`: {payload['reason']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
