from pathlib import Path

from core.config import ProjectSettings
from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from scripts.calibrate_verifier_thresholds import PassageScoreSummary, _aggregate_label, calibrate_verifier_thresholds


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
    def __init__(self, scores) -> None:  # noqa: ANN001
        self._scores = scores
        self._index = 0

    def score_evidence_passages(self, claim: str, evidence: list[EvidenceSpan]):  # noqa: ANN001, ARG002
        score = self._scores[self._index]
        self._index += 1
        return score


def test_aggregate_label_uses_thresholds() -> None:
    summary = PassageScoreSummary(support=0.7, refute=0.2, nei=0.1)

    assert _aggregate_label(summary, 0.5, 0.5) == "SUPPORTED"
    assert _aggregate_label(summary, 0.8, 0.5) == "NOT ENOUGH INFO"


def test_calibration_report_includes_best_candidate(monkeypatch) -> None:
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
    router_scores = [
        [type("Score", (), {"logits": {"SUPPORTED": 0.8, "REFUTED": 0.1, "NOT ENOUGH INFO": 0.1}})()],
        [type("Score", (), {"logits": {"SUPPORTED": 0.2, "REFUTED": 0.75, "NOT ENOUGH INFO": 0.05}})()],
    ]

    monkeypatch.setattr(
        "scripts.calibrate_verifier_thresholds.load_records",
        lambda data_dir, split_names: {name: list(records) for name in split_names},
    )
    monkeypatch.setattr("scripts.calibrate_verifier_thresholds.load_evidence_corpus", lambda data_dir, suffix="": list(corpus))
    monkeypatch.setattr(
        "scripts.calibrate_verifier_thresholds._load_retrieval_runtime",
        lambda corpus_arg, settings: _StubRuntime(_StubRetriever([corpus, corpus])),
    )
    monkeypatch.setattr("scripts.calibrate_verifier_thresholds._load_reranker_runtime", lambda settings: _StubRerankerRuntime())
    monkeypatch.setattr("scripts.calibrate_verifier_thresholds.ModelRouter", lambda **kwargs: _StubRouter(router_scores))

    report = calibrate_verifier_thresholds(
        settings=ProjectSettings(verifier_backend="mock"),
        checkpoint="checkpoint",
        data_dir=Path("data/processed"),
        suffix="_large",
        split_prefixes=("fever_val",),
        top_k=1,
    )

    assert report["baseline"]["macro_f1"] == 0.6667
    assert report["best"]["macro_f1"] == 0.6667
    assert report["top_candidates"]
