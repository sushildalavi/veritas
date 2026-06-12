from data.schemas import EvidenceSpan
from ranking.ab_simulation import bootstrap_ab_simulation
from ranking.features import extract_features, feature_names, to_matrix
from ranking.learned_ranker import LearnedRanker
from ranking.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k


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
