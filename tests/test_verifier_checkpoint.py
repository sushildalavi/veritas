from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.schemas import EvidenceSpan
from models.deberta_verifier import DebertaVerifier


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
