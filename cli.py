"""Command line entrypoint for a single Veritas verification."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from serving.model_loader import load_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a claim with the Veritas demo pipeline.")
    parser.add_argument("claim")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-corpus", default=os.getenv("VERITAS_EVIDENCE_CORPUS"))
    parser.add_argument("--verifier-checkpoint", default=os.getenv("VERITAS_VERIFIER_CHECKPOINT"))
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    pipeline = load_pipeline(
        evidence_corpus_path=args.evidence_corpus,
        verifier_checkpoint=args.verifier_checkpoint,
    )
    outcome = pipeline.reflection_loop.run(args.claim, top_k=args.top_k)
    payload = {
        "claim": args.claim,
        "verdict": outcome.verification.verdict if outcome.verification else "NOT ENOUGH INFO",
        "confidence": outcome.verification.confidence if outcome.verification else 0.0,
        "explanation": outcome.explanation,
        "citation_valid": outcome.citation_valid,
        "fallback_used": pipeline.fallback_used,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:  # pragma: no cover - script entrypoint
    run()


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
