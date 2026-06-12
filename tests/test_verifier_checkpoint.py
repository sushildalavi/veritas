from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.schemas import EvidenceSpan
from models.deberta_verifier import DebertaVerifier
from core.config import ProjectSettings
from serving.model_loader import _resolve_checkpoint_path
from scripts.train_transformer_verifier import _to_markdown


def test_deberta_verifier_loads_joblib_checkpoint(tmp_path: Path) -> None:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    texts = [
        "claim: Paris is in France\nevidence: Paris is in France.",
        "claim: Paris is in France\nevidence: Paris is in Germany.",
        "claim: Paris is in France\nevidence: ",
    ]
    labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
    pipeline.fit(texts, labels)
    joblib.dump({"pipeline": pipeline, "label_order": list(pipeline.classes_)}, tmp_path / "model.joblib")

    verifier = DebertaVerifier(checkpoint_path=tmp_path)
    result = verifier.predict("Paris is in France", [EvidenceSpan(doc_id="1", text="Paris is in France.")])

    assert verifier._backend == "sklearn"
    assert result.verdict in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}
    assert result.confidence >= 0.0


def test_transformer_checkpoint_takes_priority_over_sklearn(tmp_path: Path) -> None:
    transformer_dir = tmp_path / "transformer_verifier"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    sklearn_dir = tmp_path / "verifier"
    sklearn_dir.mkdir()
    (sklearn_dir / "model.joblib").write_text("placeholder", encoding="utf-8")

    settings = ProjectSettings(
        sklearn_checkpoint=str(sklearn_dir),
        transformer_checkpoint=str(transformer_dir),
    )

    resolved = _resolve_checkpoint_path(settings)

    assert resolved == transformer_dir


def test_transformer_verifier_report_markdown_schema() -> None:
    report = {
        "model_name": "distilroberta-base",
        "checkpoint_path": "checkpoints/transformer_verifier",
        "train_example_count": 2,
        "validation_example_count": 1,
        "test_example_count": 1,
        "test_latency_ms_per_example": 12.5,
        "train": {"example_count": 2, "accuracy": 0.5, "macro_f1": 0.33},
        "validation": {"example_count": 1, "accuracy": 0.0, "macro_f1": 0.0},
        "test": {"example_count": 1, "accuracy": 1.0, "macro_f1": 1.0},
        "test_per_class": {
            "SUPPORTED": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "REFUTED": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "NOT_ENOUGH_INFO": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        },
        "test_confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    }

    markdown = _to_markdown(report)

    assert "Transformer Verifier Evaluation" in markdown
    assert "distilroberta-base" in markdown
    assert "SUPPORTED" in markdown
