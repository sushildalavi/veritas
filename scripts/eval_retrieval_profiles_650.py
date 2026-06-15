"""Full 650-example comparison of bm25_only vs. the two strongest hybrid retrieval profiles.

Reuses `evaluate_oracle_vs_retrieved_v2` (same evaluator as the headline v2 reports) on the
full fever_test_large + scifact_test_large set (650 examples) for each profile, and reports
recall@1/3/5/10/50, nDCG@5/10, retrieved per-passage macro-F1, label-wise F1, REFUTED
prediction rate, the oracle-vs-retrieved gap, and per-profile runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings
from evaluation.reporting import write_report
from scripts.eval_oracle_vs_retrieved_v2 import LABEL_ORDER, evaluate_oracle_vs_retrieved_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare bm25_only vs. hybrid profiles on the full 650-example set.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--split-prefixes", default="fever_test,scifact_test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=650)
    parser.add_argument("--report-json", default="reports/retrieval_profile_comparison_650.json")
    parser.add_argument("--report-md", default="reports/retrieval_profile_comparison_650.md")
    return parser


def _profiles(base_settings):
    return [
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
            "hybrid_bm25_sentence_transformer",
            replace(
                base_settings,
                retrieval_backend="bm25_sentence_transformer_hybrid",
                embedding_backend="sentence-transformers",
                use_neural_retrieval=True,
                reranker_backend="none",
                use_cross_encoder=False,
                query_expansion_top_k=0,
            ),
        ),
    ]


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    base_settings = load_project_settings(args.config)
    split_prefixes = tuple(part.strip() for part in args.split_prefixes.split(",") if part.strip())

    report = {
        "config_path": args.config,
        "suffix": args.suffix,
        "split_prefixes": list(split_prefixes),
        "max_examples": args.max_examples,
        "profiles": {},
    }

    for profile_name, settings in _profiles(base_settings):
        started = time.perf_counter()
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
        total_runtime = round(time.perf_counter() - started, 3)

        retrieved_per_passage = profile_report["retrieved"]["per_passage_max"]
        confusion = retrieved_per_passage["confusion_matrix"]
        sample_size = profile_report["sample_size"]
        refuted_predicted = sum(row[LABEL_ORDER.index("REFUTED")] for row in confusion)

        report["profiles"][profile_name] = {
            "retrieval_backend": profile_report["retrieval_backend"],
            "reranker_backend": profile_report["reranker_backend"],
            "sample_size": sample_size,
            "retrieval_metrics": profile_report["retrieval_metrics"],
            "retrieved_per_passage_max": {
                "accuracy": retrieved_per_passage["accuracy"],
                "macro_f1": retrieved_per_passage["macro_f1"],
                "per_class": retrieved_per_passage["per_class"],
                "confusion_matrix": confusion,
                "nei_false_positive_rate": retrieved_per_passage["nei_false_positive_rate"],
            },
            "refuted_prediction_rate": round(refuted_predicted / sample_size, 4) if sample_size else 0.0,
            "oracle_per_passage_max": {
                "accuracy": profile_report["oracle"]["per_passage_max"]["accuracy"],
                "macro_f1": profile_report["oracle"]["per_passage_max"]["macro_f1"],
            },
            "oracle_gap_per_passage_max": profile_report["delta_from_oracle"]["per_passage_max"],
            "retrieval_runtime_seconds": profile_report["retrieval_runtime_seconds"],
            "total_runtime_seconds": total_runtime,
            "git_commit": profile_report["git_commit"],
            "settings": profile_report["settings"],
        }
        print(
            f"{profile_name}: recall@10={profile_report['retrieval_metrics']['recall@10']:.4f}, "
            f"retrieved_macro_f1={retrieved_per_passage['macro_f1']:.4f}, "
            f"refuted_rate={report['profiles'][profile_name]['refuted_prediction_rate']:.4f}, "
            f"runtime={total_runtime:.1f}s"
        )

    write_report(report, Path(args.report_json))
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieval Profile Comparison (Full 650-Example Set)",
        "",
        f"- Config path: `{report['config_path']}`",
        f"- Split prefixes: `{', '.join(report['split_prefixes'])}`",
        f"- Suffix: `{report['suffix']}`",
        f"- Sample size: {report['max_examples']}",
        "",
        "## Retrieval metrics",
        "",
        "| profile | recall@1 | recall@3 | recall@5 | recall@10 | recall@50 | nDCG@5 | nDCG@10 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, profile in report["profiles"].items():
        rm = profile["retrieval_metrics"]
        lines.append(
            f"| {name} | {rm['recall@1']} | {rm['recall@3']} | {rm['recall@5']} | {rm['recall@10']} | "
            f"{rm['recall@50']} | {rm['ndcg@5']} | {rm['ndcg@10']} |"
        )

    lines.extend(["", "## Retrieved per-passage verifier metrics", ""])
    lines.append("| profile | accuracy | macro_f1 | SUPPORTED F1 | REFUTED F1 | NEI F1 | refuted_pred_rate | nei_fp_rate |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, profile in report["profiles"].items():
        rpm = profile["retrieved_per_passage_max"]
        pc = rpm["per_class"]
        lines.append(
            f"| {name} | {rpm['accuracy']} | {rpm['macro_f1']} | {pc['SUPPORTED']['f1']} | {pc['REFUTED']['f1']} | "
            f"{pc['NOT ENOUGH INFO']['f1']} | {profile['refuted_prediction_rate']} | {rpm['nei_false_positive_rate']} |"
        )

    lines.extend(["", "## Oracle gap and runtime", ""])
    lines.append("| profile | oracle macro_f1 | retrieved macro_f1 | macro_f1 gap | total runtime (s) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, profile in report["profiles"].items():
        lines.append(
            f"| {name} | {profile['oracle_per_passage_max']['macro_f1']} | {profile['retrieved_per_passage_max']['macro_f1']} | "
            f"{profile['oracle_gap_per_passage_max']['macro_f1']} | {profile['total_runtime_seconds']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
