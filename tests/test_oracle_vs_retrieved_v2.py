from pathlib import Path

from core.config import ProjectSettings
from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from scripts.eval_oracle_vs_retrieved_v2 import evaluate_oracle_vs_retrieved_v2


class _StubRetriever:
    def __init__(self, rankings: list[list[EvidenceSpan]]) -> None:
        self._rankings = rankings
        self._index = 0

    def retrieve(self, claim: str, top_k: int):  # noqa: ANN001, ARG002
        ranking = self._rankings[self._index]
        self._index += 1
        return ranking[:top_k]


class _StubRuntime:
    def __init__(self, retriever) -> None:  # noqa: ANN001
        self.retriever = retriever
        self.retrieval_backend = "stub"


class _StubRerankerRuntime:
    def __init__(self) -> None:
        self.reranker = None
        self.reranker_backend = "none"


class _StubRouter:
    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = verdicts
        self._index = 0

    def predict(self, claim: str, evidence: list[EvidenceSpan]):  # noqa: ANN001, ARG002
        verdict = self._verdicts[self._index]
        self._index += 1
        return type(
            "Result",
            (),
            {
                "verdict": verdict,
                "confidence": 0.8,
            },
        )()


def test_oracle_vs_retrieved_v2_uses_structured_doc_ids(monkeypatch) -> None:
    records = [
        ClaimEvidenceRecord(
            claim_id="c1",
            claim="Paris is in France.",
            label="SUPPORTED",
            evidence=(EvidenceSpan(doc_id="doc-1", text="Paris is in France."),),
        ),
        ClaimEvidenceRecord(
            claim_id="c2",
            claim="Ottawa is in France.",
            label="REFUTED",
            evidence=(EvidenceSpan(doc_id="doc-2", text="Ottawa is in Canada."),),
        ),
    ]
    corpus = [
        EvidenceSpan(doc_id="doc-1", text="Paris is in France."),
        EvidenceSpan(doc_id="doc-2", text="Ottawa is in Canada."),
    ]
    rankings = [
        [corpus[0], corpus[1]],
        [corpus[0], corpus[1]],
    ]
    routers = [
        _StubRouter(["SUPPORTED", "REFUTED", "SUPPORTED", "NOT ENOUGH INFO"]),
        _StubRouter(["SUPPORTED", "REFUTED", "SUPPORTED", "REFUTED"]),
    ]

    monkeypatch.setattr(
        "scripts.eval_oracle_vs_retrieved_v2.load_records",
        lambda data_dir, split_names: {name: list(records) for name in split_names},
    )
    monkeypatch.setattr("scripts.eval_oracle_vs_retrieved_v2.load_evidence_corpus", lambda data_dir, suffix="": list(corpus))
    monkeypatch.setattr(
        "scripts.eval_oracle_vs_retrieved_v2._load_retrieval_runtime",
        lambda corpus_arg, settings: _StubRuntime(_StubRetriever(rankings)),
    )
    monkeypatch.setattr("scripts.eval_oracle_vs_retrieved_v2._load_reranker_runtime", lambda settings: _StubRerankerRuntime())
    monkeypatch.setattr("scripts.eval_oracle_vs_retrieved_v2.ModelRouter", lambda **kwargs: routers.pop(0))

    report = evaluate_oracle_vs_retrieved_v2(
        settings=ProjectSettings(verifier_backend="mock", final_top_k=5, rrf_top_k=10, rerank_top_k=5),
        checkpoint="checkpoint",
        data_dir=Path("data/processed"),
        suffix="_large",
        split_prefixes=("fever_test",),
        top_k=1,
    )

    assert report["retrieval_metrics"]["recall@1"] == 0.5
    assert report["retrieved"]["bundle"]["macro_f1"] < report["oracle"]["bundle"]["macro_f1"]
    assert report["split_prefixes"] == ["fever_test"]
