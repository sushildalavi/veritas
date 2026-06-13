from pathlib import Path

from data.schemas import EvidenceSpan
from scripts.eval_topk_verifier import EvaluationResult, evaluate_topk_verifier


def test_topk_verifier_report_structure(monkeypatch) -> None:
    rows = [
        {"claim_id": "1", "claim": "Paris is in France", "label": "SUPPORTED", "evidence": "Paris is in France."},
    ]
    corpus = [EvidenceSpan(doc_id="0", text="Paris is in France.")]

    monkeypatch.setattr("scripts.eval_topk_verifier.read_jsonl", lambda path: rows)
    monkeypatch.setattr("scripts.eval_topk_verifier.load_evidence_corpus", lambda data_dir, suffix="": corpus)
    monkeypatch.setattr("scripts.eval_topk_verifier._load_transformer", lambda checkpoint: (object(), object()))

    responses = [
        EvaluationResult(1, 0.7, 0.7, {"SUPPORTED": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}, [[1]], 1.0),
        EvaluationResult(1, 0.5, 0.5, {"SUPPORTED": {"precision": 0.5, "recall": 0.5, "f1": 0.5}}, [[1]], 1.0),
        EvaluationResult(1, 0.6, 0.6, {"SUPPORTED": {"precision": 0.6, "recall": 0.6, "f1": 0.6}}, [[1]], 1.0),
        EvaluationResult(1, 0.8, 0.8, {"SUPPORTED": {"precision": 0.8, "recall": 0.8, "f1": 0.8}}, [[1]], 1.0),
    ]
    monkeypatch.setattr("scripts.eval_topk_verifier._score_examples", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("scripts.eval_topk_verifier._predict_labels", lambda *args, **kwargs: [0])

    report = evaluate_topk_verifier(
        checkpoint="checkpoint",
        data_dir=Path("data/processed"),
        test_file="verifier_test.jsonl",
        evidence_corpus_suffix="_large",
        top_ks=[1, 3, 5],
        retrieval_mode="bm25",
        max_length=384,
        batch_size=1,
        max_errors=5,
    )

    assert report["best_top_k"] == 5
    assert "top_1" in report["top_k"]
    assert "top_3" in report["top_k"]
    assert "top_5" in report["top_k"]
    assert "oracle_vs_retrieved_gap" in report
