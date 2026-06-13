"""Aggregate the final Veritas research artifacts into one report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.reporting import write_report


@dataclass(frozen=True)
class ReportRef:
    key: str
    path: Path


REPORTS = [
    ReportRef("data_quality", Path("reports/data_quality_large.json")),
    ReportRef("verifier_data", Path("reports/verifier_data_audit.json")),
    ReportRef("retrieval", Path("reports/retrieval_eval_neural_large.json")),
    ReportRef("ranking", Path("reports/ranking_eval_cross_encoder_large.json")),
    ReportRef("verifier", Path("reports/transformer_verifier_clean_eval.json")),
    ReportRef("oracle_verifier", Path("reports/oracle_verifier_eval.json")),
    ReportRef("end_to_end_verifier", Path("reports/end_to_end_verifier_eval.json")),
    ReportRef("topk_verifier", Path("reports/topk_verifier_eval.json")),
    ReportRef("mlx_lora", Path("reports/mlx_lora_eval_200.json")),
    ReportRef("mlx_lora_comparison", Path("reports/mlx_lora_comparison.json")),
    ReportRef("preference_pair_stats", Path("reports/preference_pair_stats.json")),
    ReportRef("faithfulness", Path("reports/faithfulness_comparison.json")),
    ReportRef("pareto", Path("reports/final_pareto_analysis.json")),
    ReportRef("retrieval_ablation", Path("reports/retrieval_ablation_research.json")),
    ReportRef("deberta_challenger", Path("reports/deberta_verifier_clean_eval.json")),
    ReportRef("explanation_sft", Path("reports/explanation_sft_data_stats.json")),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final Veritas evaluation suite.")
    parser.add_argument("--output-json", default="reports/final_results.json")
    parser.add_argument("--output-md", default="reports/final_results.md")
    parser.add_argument("--audit-md", default="reports/final_research_audit.md")
    return parser


def main() -> None:  # pragma: no cover - script entrypoint
    args = build_parser().parse_args()
    report = build_final_suite()
    write_report(report, Path(args.output_json))
    Path(args.output_md).write_text(_to_markdown(report), encoding="utf-8")
    Path(args.audit_md).write_text(_to_audit_markdown(report), encoding="utf-8")
    print(f"Wrote final results to {args.output_json}")


def build_final_suite() -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    missing: list[str] = []
    for ref in REPORTS:
        payload = _load_json(ref.path)
        if payload is None:
            missing.append(ref.key)
        else:
            loaded[ref.key] = payload

    return {
        "highlights": _highlights(loaded),
        "dataset_sizes": _dataset_sizes(loaded),
        "retrieval": _retrieval_summary(loaded),
        "ranking": _ranking_summary(loaded),
        "verifier": _verifier_summary(loaded),
        "oracle_vs_retrieved": _oracle_vs_retrieved_summary(loaded),
        "topk_verifier": _topk_summary(loaded),
        "explanation": _explanation_summary(loaded),
        "preference_reranking": _preference_summary(loaded),
        "faithfulness": _faithfulness_summary(loaded),
        "pareto": _pareto_summary(loaded),
        "deberta": _deberta_summary(loaded),
        "missing_reports": missing,
        "resume_safe_bullets": _resume_safe_bullets(loaded),
        "what_is_not_claimed": _what_is_not_claimed(),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".md":
        return {"markdown": path.read_text(encoding="utf-8")}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _highlights(loaded: dict[str, Any]) -> dict[str, Any]:
    oracle = loaded.get("oracle_verifier", {})
    end_to_end = loaded.get("end_to_end_verifier", {})
    topk = loaded.get("topk_verifier", {})
    verifier = loaded.get("verifier", {})
    retrieval = loaded.get("retrieval", {})
    ranking = loaded.get("ranking", {})
    mlx = loaded.get("mlx_lora", {})
    faithfulness = loaded.get("faithfulness", {})
    deberta = loaded.get("deberta_challenger", {})
    deberta_test = deberta.get("test", {})
    template = faithfulness.get("generators", {}).get("template", {})
    best_top_k = topk.get("best_top_k")
    top_k_metrics = topk.get("top_k", {})
    best_top_k_metrics = top_k_metrics.get(f"top_{best_top_k}", {}) if best_top_k is not None else {}
    return {
        "oracle_verifier_accuracy": oracle.get("accuracy"),
        "oracle_verifier_macro_f1": oracle.get("macro_f1"),
        "end_to_end_accuracy": end_to_end.get("accuracy"),
        "end_to_end_macro_f1": end_to_end.get("macro_f1"),
        "topk_best_top_k": best_top_k,
        "topk_best_accuracy": best_top_k_metrics.get("accuracy"),
        "topk_best_macro_f1": best_top_k_metrics.get("macro_f1"),
        "verifier_test_accuracy": _nested_get(verifier, ("test", "accuracy")),
        "verifier_test_macro_f1": _nested_get(verifier, ("test", "macro_f1")),
        "dense_recall_at_10": _nested_get(retrieval, ("metrics", "dense", "recall@10")),
        "hybrid_recall_at_10": _nested_get(retrieval, ("metrics", "hybrid", "recall@10")),
        "cross_encoder_map": _nested_get(ranking, ("strategies", "cross_encoder", "map")),
        "template_citation_valid_rate": template.get("citation_valid_rate"),
        "mlx_lora_verdict_accuracy": mlx.get("verdict_accuracy"),
        "mlx_lora_macro_f1": mlx.get("macro_f1"),
        "deberta_challenger_accuracy": deberta_test.get("accuracy"),
        "deberta_challenger_macro_f1": deberta_test.get("macro_f1"),
    }


def _dataset_sizes(loaded: dict[str, Any]) -> dict[str, Any]:
    verifier_data = loaded.get("verifier_data", {})
    data_quality = loaded.get("data_quality", {})
    explanation_sft = loaded.get("explanation_sft", {})
    return {
        "verifier_splits": {
            split: details.get("example_count")
            for split, details in verifier_data.get("splits", {}).items()
            if isinstance(details, dict)
        },
        "verifier_label_distribution": {
            split: details.get("label_distribution")
            for split, details in verifier_data.get("splits", {}).items()
            if isinstance(details, dict)
        },
        "evidence_corpus_size": data_quality.get("evidence_corpus_size"),
        "data_quality_label_distribution": data_quality.get("label_distribution"),
        "fever_sizes": data_quality.get("fever_sizes"),
        "scifact_sizes": data_quality.get("scifact_sizes"),
        "explanation_sft_total": explanation_sft.get("total_examples"),
    }


def _retrieval_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    retrieval = loaded.get("retrieval", {})
    ablation = loaded.get("retrieval_ablation", {})
    metrics = retrieval.get("metrics", {})
    best_metric_key = None
    best_metric_value = None
    if isinstance(metrics, dict):
        for strategy_name, strategy_metrics in metrics.items():
            if not isinstance(strategy_metrics, dict):
                continue
            candidate = strategy_metrics.get("recall@10")
            if candidate is None:
                continue
            if best_metric_value is None or candidate > best_metric_value:
                best_metric_key = strategy_name
                best_metric_value = candidate
    ablation_strategies = ablation.get("strategies", {})
    ablation_best = None
    if isinstance(ablation_strategies, dict) and ablation_strategies:
        ablation_best = max(
            ablation_strategies.items(),
            key=lambda item: item[1].get("recall@10", item[1].get("recall_at_10", 0.0)) if isinstance(item[1], dict) else 0.0,
        )[0]
    return {
        "sample_size": retrieval.get("num_queries"),
        "split": retrieval.get("split"),
        "dense_backend": retrieval.get("dense_backend"),
        "embedding_model": retrieval.get("embedding_model"),
        "metrics": metrics,
        "best_recall_at_10_strategy": best_metric_key,
        "best_recall_at_10": best_metric_value,
        "ablation_available": bool(ablation),
        "ablation_best_strategy": ablation_best,
        "ablation_notes": ablation.get("notes", []),
        "ablation_strategies": ablation_strategies,
    }


def _ranking_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    ranking = loaded.get("ranking", {})
    strategies = ranking.get("strategies", {})
    best_map_strategy = None
    best_map_value = None
    if isinstance(strategies, dict):
        for name, metrics in strategies.items():
            if not isinstance(metrics, dict):
                continue
            candidate = metrics.get("map")
            if candidate is None:
                continue
            if best_map_value is None or candidate > best_map_value:
                best_map_strategy = name
                best_map_value = candidate
    return {
        "sample_size": ranking.get("num_queries"),
        "candidate_k": ranking.get("candidate_k"),
        "cross_encoder_model": ranking.get("cross_encoder_model"),
        "learned_ranker_backend": ranking.get("learned_ranker_backend"),
        "best_map_strategy": best_map_strategy,
        "best_map": best_map_value,
        "strategies": strategies,
        "limitations": ranking.get("limitations", []),
    }


def _verifier_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    verifier = loaded.get("verifier", {})
    return {
        "model_name": verifier.get("model_name"),
        "checkpoint_path": verifier.get("checkpoint_path"),
        "label_order": verifier.get("label_order"),
        "train": verifier.get("train", {}),
        "validation": verifier.get("validation", {}),
        "test": verifier.get("test", {}),
        "training_runtime_seconds": verifier.get("training_runtime_seconds"),
        "thresholds": verifier.get("thresholds", {}),
    }


def _oracle_vs_retrieved_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    oracle = loaded.get("oracle_verifier", {})
    end_to_end = loaded.get("end_to_end_verifier", {})
    gap = end_to_end.get("oracle_vs_retrieved_gap", {})
    oracle_accuracy = oracle.get("accuracy")
    end_accuracy = end_to_end.get("accuracy")
    oracle_macro_f1 = oracle.get("macro_f1")
    end_macro_f1 = end_to_end.get("macro_f1")
    return {
        "oracle": {
            "accuracy": oracle_accuracy,
            "macro_f1": oracle_macro_f1,
            "refuted_recall": _nested_get(oracle, ("per_class", "REFUTED", "recall")),
            "example_count": oracle.get("example_count"),
            "evidence_source": oracle.get("evidence_source"),
        },
        "retrieved": {
            "accuracy": end_accuracy,
            "macro_f1": end_macro_f1,
            "refuted_recall": _nested_get(end_to_end, ("per_class", "REFUTED", "recall")),
            "example_count": end_to_end.get("example_count"),
            "evidence_source": end_to_end.get("evidence_source"),
        },
        "gap": gap,
        "delta_from_oracle": {
            "accuracy": round((oracle_accuracy or 0.0) - (end_accuracy or 0.0), 4) if oracle_accuracy is not None and end_accuracy is not None else None,
            "macro_f1": round((oracle_macro_f1 or 0.0) - (end_macro_f1 or 0.0), 4) if oracle_macro_f1 is not None and end_macro_f1 is not None else None,
        },
    }


def _topk_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    topk = loaded.get("topk_verifier", {})
    top_k = topk.get("top_k", {})
    best_key = f"top_{topk.get('best_top_k')}" if topk.get("best_top_k") is not None else None
    best = top_k.get(best_key, {}) if best_key else {}
    return {
        "headline": topk.get("headline"),
        "retrieval_mode": topk.get("retrieval_mode"),
        "retrieval_backend": topk.get("retrieval_backend"),
        "sample_size": topk.get("sample_size"),
        "oracle": topk.get("oracle", {}),
        "best_top_k": topk.get("best_top_k"),
        "best_result": best,
        "top_k": top_k,
        "oracle_vs_retrieved_gap": topk.get("oracle_vs_retrieved_gap", {}),
        "top_errors": topk.get("top_errors", []),
    }


def _explanation_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    explanation = loaded.get("mlx_lora", {})
    return {
        "sample_size": explanation.get("sample_size"),
        "verdict_accuracy": explanation.get("verdict_accuracy"),
        "macro_f1": explanation.get("macro_f1"),
        "parseable_rate": explanation.get("parseable_rate"),
        "citation_valid_rate": explanation.get("citation_valid_rate"),
        "unsupported_sentence_rate": explanation.get("unsupported_sentence_rate"),
        "mean_latency_seconds": explanation.get("mean_latency_seconds"),
        "top_error_examples": explanation.get("top_error_examples", []),
    }


def _preference_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    comparison = loaded.get("mlx_lora_comparison", {})
    pair_stats = loaded.get("preference_pair_stats", {})
    adapters = comparison.get("adapters", [])
    best_adapter = comparison.get("best_adapter")
    return {
        "best_adapter": best_adapter,
        "adapters": adapters,
        "pair_stats": pair_stats,
        "available": bool(adapters or pair_stats),
    }


def _faithfulness_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    faithfulness = loaded.get("faithfulness", {})
    generators = faithfulness.get("generators", {})
    return {
        "checkpoint": faithfulness.get("checkpoint"),
        "data_file": faithfulness.get("data_file"),
        "template": generators.get("template", {}),
        "qlora": generators.get("qlora", {}),
        "dpo": generators.get("dpo", {}),
    }


def _pareto_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    pareto = loaded.get("pareto", {})
    return {
        "frontier": pareto.get("frontier", []),
        "summaries": pareto.get("summaries", []),
        "not_trained": pareto.get("not_trained", []),
    }


def _deberta_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    challenger = loaded.get("deberta_challenger", {})
    return {
        "measured": bool(challenger),
        "challenge_report": challenger,
    }


def _resume_safe_bullets(loaded: dict[str, Any]) -> list[str]:
    bullets = [
        "Built a reproducible Mac-first verification stack with retrieval, reranking, transformer verification, and citation-grounded explanations.",
        "Measured the oracle-vs-retrieved gap explicitly and added a top-k verifier evaluation to test how much retrieval quality still matters.",
        "Packaged the project with a FastAPI backend, Gradio frontend, and a final results/audit generator for research review.",
    ]
    if loaded.get("retrieval_ablation"):
        bullets.append("Ran a retrieval ablation to compare sparse, dense, and hybrid evidence retrieval strategies.")
    if loaded.get("deberta_challenger"):
        bullets.append("Measured a DeBERTa challenger against the DistilRoBERTa verifier baseline.")
    if loaded.get("faithfulness"):
        bullets.append("Audited explanation faithfulness with citation validity and unsupported-sentence measurements.")
    return bullets


def _what_is_not_claimed() -> list[str]:
    return [
        "SOTA performance",
        "production-scale benchmark coverage",
        "perfect citation faithfulness",
        "CUDA QLoRA",
        "CUDA DPO",
        "verified DeBERTa challenger training if the checkpoint is absent",
    ]


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _to_markdown(report: dict[str, Any]) -> str:
    sections = [
        ("Highlights", report["highlights"]),
        ("Dataset Sizes", report["dataset_sizes"]),
        ("Retrieval", report["retrieval"]),
        ("Ranking", report["ranking"]),
        ("Verifier", report["verifier"]),
        ("Oracle vs Retrieved", report["oracle_vs_retrieved"]),
        ("Top-k Verifier", report["topk_verifier"]),
        ("Explanation", report["explanation"]),
        ("Preference Reranking", report["preference_reranking"]),
        ("Faithfulness", report["faithfulness"]),
        ("Pareto", report["pareto"]),
        ("DeBERTa", report["deberta"]),
    ]
    lines = ["# Final Veritas Results", ""]
    for title, payload in sections:
        lines.extend([f"## {title}", "", json.dumps(payload, indent=2, sort_keys=True), ""])
    lines.extend(
        [
            "## Missing Reports",
            "",
            json.dumps(report["missing_reports"], indent=2, sort_keys=True),
            "",
            "## Resume-Safe Bullets",
            "",
            "\n".join(f"- {bullet}" for bullet in report["resume_safe_bullets"]),
            "",
            "## What Is Not Claimed",
            "",
            "\n".join(f"- {item}" for item in report["what_is_not_claimed"]),
            "",
        ]
    )
    return "\n".join(lines)


def _to_audit_markdown(report: dict[str, Any]) -> str:
    rows = [
        ("Dataset sizes", bool(report["dataset_sizes"])),
        ("Retrieval", bool(report["retrieval"].get("metrics"))),
        ("Ranking", bool(report["ranking"].get("strategies"))),
        ("Verifier", bool(report["verifier"].get("test"))),
        ("Oracle vs retrieved", bool(report["oracle_vs_retrieved"].get("gap"))),
        ("Top-k verifier", bool(report["topk_verifier"].get("top_k"))),
        ("Explanation", bool(report["explanation"].get("sample_size"))),
        ("Preference reranking", bool(report["preference_reranking"].get("available"))),
        ("Faithfulness", bool(report["faithfulness"].get("template"))),
        ("Pareto", bool(report["pareto"].get("summaries"))),
        ("DeBERTa", report["deberta"].get("measured")),
    ]
    lines = [
        "# Final Research Audit",
        "",
        "| Area | Status |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'measured' if status else 'missing'} |" for name, status in rows)
    if report["missing_reports"]:
        lines.extend(["", "## Missing Reports", "", "\n".join(f"- {item}" for item in report["missing_reports"])])
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
