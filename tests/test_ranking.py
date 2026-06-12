from data.schemas import ClaimEvidenceRecord, EvidenceSpan
from pathlib import Path

from ranking.ab_simulation import bootstrap_ab_simulation
from ranking.features import extract_features, feature_names, to_matrix
from ranking.learned_ranker import LearnedRanker
from ranking.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k
from ranking.reranker import CrossEncoderReranker, HeuristicReranker
from scripts.run_ranking_eval import evaluate_ranking


def test_extract_features_includes_expected_keys() -> None:
    evidence = EvidenceSpan(doc_id="1", text="Paris is the capital of France.", score=0.7)

    features = extract_features(
        "France capital city",
        evidence,
        bm25_score=1.2,
        dense_score=0.8,
        bm25_rank=1,
        dense_rank=2,
    )

    assert set(feature_names()).issubset(features.keys())
    assert features["bm25_score"] == 1.2
    assert features["bm25_rank_position"] == 1.0
    assert features["dense_rank_position"] == 2.0
    assert features["cross_encoder_score"] == 0.7


def test_learned_ranker_heuristic_fallback_scores_positive_examples_higher() -> None:
    claim = "Paris is in France"
    positive = extract_features(
        claim,
        EvidenceSpan(doc_id="p", text="Paris is the capital of France."),
        bm25_score=2.0,
        dense_score=0.9,
        bm25_rank=1,
        dense_rank=1,
    )
    negative = extract_features(
        claim,
        EvidenceSpan(doc_id="n", text="Ottawa is the capital of Canada."),
        bm25_score=0.1,
        dense_score=0.1,
        bm25_rank=5,
        dense_rank=5,
    )

    ranker = LearnedRanker().fit([positive, negative], [1, 0])
    scores = ranker.predict_scores([positive, negative])

    assert scores[0] >= scores[1]
    assert ranker.backend_name in {"lightgbm", "sklearn-logistic", "sklearn-gradient-boosting", "heuristic"}


def test_ranking_metrics_and_ab_simulation() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "c"}, 3) > 0.0
    assert mean_average_precision([["a", "b"], ["x", "y"]], [{"b"}, {"z"}]) == 0.25
    assert mean_reciprocal_rank([["b", "a"], ["x", "y"]], [{"a"}, {"z"}]) == 0.25

    summary = bootstrap_ab_simulation([0.2, 0.4, 0.6], [0.3, 0.5, 0.7], iterations=100)
    assert summary.delta > 0.0
    assert summary.ci_low <= summary.delta <= summary.ci_high


def test_cross_encoder_reranker_uses_mock_model() -> None:
    class FakeModel:
        def predict(self, pairs, batch_size=16):  # noqa: ANN001
            return [0.9 if "France" in pair[1] else 0.1 for pair in pairs]

    reranker = CrossEncoderReranker(model_name="fake-model", model=FakeModel())
    passages = [
        EvidenceSpan(doc_id="1", text="Paris is the capital of France."),
        EvidenceSpan(doc_id="2", text="Ottawa is the capital of Canada."),
    ]

    scores = reranker.score_pairs("Paris is in France", passages)
    reranked = reranker.rerank("Paris is in France", passages)

    assert scores[0] > scores[1]
    assert reranked[0].doc_id == "1"
    assert reranker.rank("Paris is in France", passages)[0].doc_id == "1"


def test_heuristic_reranker_prefers_overlap() -> None:
    reranker = HeuristicReranker()
    passages = [
        EvidenceSpan(doc_id="1", text="Paris is the capital of France."),
        EvidenceSpan(doc_id="2", text="Ottawa is the capital of Canada."),
    ]

    scores = reranker.score_pairs("Paris is in France", passages)
    reranked = reranker.rank("Paris is in France", passages)

    assert scores[0] > scores[1]
    assert reranked[0].doc_id == "1"


def test_ranking_eval_report_schema_with_cross_encoder(monkeypatch) -> None:
    corpus = [
        EvidenceSpan(doc_id="0", text="Paris is the capital of France."),
        EvidenceSpan(doc_id="1", text="Ottawa is the capital of Canada."),
    ]
    record = ClaimEvidenceRecord(
        claim_id="c1",
        claim="Paris is the capital of France.",
        label="SUPPORTED",
        evidence=(corpus[0],),
        split="val",
        metadata={},
    )

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def score_pairs(self, claim, evidence_list):  # noqa: ANN001
            return [0.9 if span.doc_id == "0" else 0.1 for span in evidence_list]

    monkeypatch.setattr("scripts.run_ranking_eval.load_records", lambda data_dir: {"fever_train": [record], "fever_val": [record], "scifact_train": [], "scifact_val": []})
    monkeypatch.setattr("scripts.run_ranking_eval.load_evidence_corpus", lambda data_dir: corpus)
    monkeypatch.setattr("scripts.run_ranking_eval.CrossEncoderReranker", lambda model_name: FakeCrossEncoder(model_name))

    report = evaluate_ranking(
        data_dir=Path("data/processed"),
        split="val",
        max_queries=1,
        candidate_k=2,
        use_cross_encoder=True,
        cross_encoder_model="fake-model",
    )

    assert report["cross_encoder_enabled"] is True
    assert report["cross_encoder_model"] == "fake-model"
    assert "cross_encoder" in report["strategies"]
    assert "cross_encoder_plus_learned" in report["strategies"]
