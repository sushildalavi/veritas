"""Compare default vs calibrated verifier thresholds on final evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_project_settings
from scripts.eval_oracle_vs_retrieved_v2 import evaluate_oracle_vs_retrieved_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare default and calibrated verifier thresholds.")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--threshold-config", default="configs/verifier_thresholds.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--suffix", default="_large")
    parser.add_argument("--split-prefixes", default="fever_test,scifact_test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--report-json", default="reports/threshold_comparison.json")
    parser.add_argument("--report-md", default="reports/threshold_comparison.md")
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = build_parser().parse_args()
    settings = load_project_settings(args.config)
    threshold_payload = json.loads(Path(args.threshold_config).read_text(encoding="utf-8"))
    split_prefixes = tuple(part.strip() for part in args.split_prefixes.split(",") if part.strip())
    default_report = evaluate_oracle_vs_retrieved_v2(
        settings=settings,
        checkpoint=settings.transformer_clean_checkpoint,
        data_dir=Path(args.data_dir),
        suffix=args.suffix,
        split_prefixes=split_prefixes,
        top_k=args.top_k,
        max_examples=args.max_examples,
        config_path=args.config,
        support_threshold=0.5,
        refute_threshold=0.5,
    )
    calibrated_report = evaluate_oracle_vs_retrieved_v2(
        settings=settings,
        checkpoint=settings.transformer_clean_checkpoint,
        data_dir=Path(args.data_dir),
        suffix=args.suffix,
        split_prefixes=split_prefixes,
        top_k=args.top_k,
        max_examples=args.max_examples,
        config_path=args.config,
        support_threshold=float(threshold_payload["support_threshold"]),
        refute_threshold=float(threshold_payload["refute_threshold"]),
    )
    report = {
        "threshold_config_path": args.threshold_config,
        "threshold_config": threshold_payload,
        "default_thresholds": {
            "support_threshold": 0.5,
            "refute_threshold": 0.5,
            "retrieved": calibrated_subset(default_report),
        },
        "calibrated_thresholds": {
            "support_threshold": threshold_payload["support_threshold"],
            "refute_threshold": threshold_payload["refute_threshold"],
            "retrieved": calibrated_subset(calibrated_report),
        },
        "delta": {
            "bundle_macro_f1": round(
                calibrated_report["retrieved"]["bundle"]["macro_f1"] - default_report["retrieved"]["bundle"]["macro_f1"], 4
            ),
            "per_passage_macro_f1": round(
                calibrated_report["retrieved"]["per_passage_max"]["macro_f1"]
                - default_report["retrieved"]["per_passage_max"]["macro_f1"],
                4,
            ),
            "per_passage_nei_false_positive_rate": round(
                calibrated_report["retrieved"]["per_passage_max"]["nei_false_positive_rate"]
                - default_report["retrieved"]["per_passage_max"]["nei_false_positive_rate"],
                4,
            ),
        },
        "metadata": {
            "sample_size": calibrated_report["sample_size"],
            "dataset_source": calibrated_report["dataset_source"],
            "config_path": args.config,
            "git_commit": calibrated_report["git_commit"],
        },
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_md).write_text(_to_markdown(report), encoding="utf-8")
    print(
        "threshold comparison complete: "
        f"default={default_report['retrieved']['per_passage_max']['macro_f1']:.3f}, "
        f"calibrated={calibrated_report['retrieved']['per_passage_max']['macro_f1']:.3f}"
    )


def calibrated_subset(report: dict[str, object]) -> dict[str, object]:
    return {
        "bundle": report["retrieved"]["bundle"],
        "per_passage_max": report["retrieved"]["per_passage_max"],
    }


def _to_markdown(report: dict[str, object]) -> str:
    default = report["default_thresholds"]
    calibrated = report["calibrated_thresholds"]
    lines = [
        "# Threshold Comparison",
        "",
        f"- Git commit: `{report['metadata']['git_commit']}`",
        f"- Config path: `{report['metadata']['config_path']}`",
        f"- Dataset source: `{report['metadata']['dataset_source']}`",
        f"- Sample size: {report['metadata']['sample_size']}",
        "",
        "| setting | support | refute | bundle_macro_f1 | per_passage_macro_f1 | per_passage_nei_fp_rate |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| default | {default['support_threshold']} | {default['refute_threshold']} | "
        f"{default['retrieved']['bundle']['macro_f1']} | {default['retrieved']['per_passage_max']['macro_f1']} | "
        f"{default['retrieved']['per_passage_max']['nei_false_positive_rate']} |",
        f"| calibrated | {calibrated['support_threshold']} | {calibrated['refute_threshold']} | "
        f"{calibrated['retrieved']['bundle']['macro_f1']} | {calibrated['retrieved']['per_passage_max']['macro_f1']} | "
        f"{calibrated['retrieved']['per_passage_max']['nei_false_positive_rate']} |",
        "",
        "## Delta",
        "",
        f"- bundle_macro_f1: {report['delta']['bundle_macro_f1']}",
        f"- per_passage_macro_f1: {report['delta']['per_passage_macro_f1']}",
        f"- per_passage_nei_false_positive_rate: {report['delta']['per_passage_nei_false_positive_rate']}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
