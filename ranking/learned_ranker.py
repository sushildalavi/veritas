"""Learned ranker with optional LightGBM and sklearn fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

from .features import feature_names, to_matrix


@dataclass
class LearnedRanker:
    """Train a ranker using the best available local dependency."""

    backend: object | None = None
    backend_name: str = "heuristic"

    def fit(self, feature_rows: list[dict[str, float]], labels: list[int], groups: list[int] | None = None) -> "LearnedRanker":
        X = to_matrix(feature_rows)
        y = labels
        self.backend, self.backend_name = _build_backend()

        if self.backend_name == "lightgbm":  # pragma: no cover - optional dependency
            self.backend.fit(X, y, group=groups)
        elif self.backend_name in {"sklearn-logistic", "sklearn-gradient-boosting"}:  # pragma: no cover - optional dependency
            self.backend.fit(X, y)
        else:
            self.backend = _HeuristicRanker().fit(X, y)
            self.backend_name = "heuristic"
        return self

    def predict_scores(self, feature_rows: list[dict[str, float]]) -> list[float]:
        X = to_matrix(feature_rows)
        if self.backend_name == "lightgbm":  # pragma: no cover - optional dependency
            return [float(score) for score in self.backend.predict(X)]
        if self.backend_name == "sklearn-logistic":  # pragma: no cover - optional dependency
            return [float(row[1]) for row in self.backend.predict_proba(X)]
        if self.backend_name == "sklearn-gradient-boosting":  # pragma: no cover - optional dependency
            return [float(score) for score in self.backend.predict_proba(X)[:, 1]]
        return self.backend.predict_scores(X) if self.backend else [0.0 for _ in X]


def _build_backend() -> tuple[object, str]:
    try:  # pragma: no cover - optional dependency
        from lightgbm import LGBMRanker  # type: ignore

        return LGBMRanker(n_estimators=32, learning_rate=0.1, random_state=42), "lightgbm"
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        from sklearn.linear_model import LogisticRegression  # type: ignore

        return LogisticRegression(max_iter=200), "sklearn-logistic"
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

        return GradientBoostingClassifier(random_state=42), "sklearn-gradient-boosting"
    except Exception:
        pass
    return _HeuristicRanker(), "heuristic"


class _HeuristicRanker:
    def fit(self, X: list[list[float]], y: list[int]) -> "_HeuristicRanker":
        if not X:
            self.weights = [0.0] * len(feature_names())
            return self
        pos = [row for row, label in zip(X, y) if label]
        neg = [row for row, label in zip(X, y) if not label]
        if not pos:
            self.weights = [0.0] * len(X[0])
            return self
        if not neg:
            self.weights = [1.0] * len(X[0])
            return self
        self.weights = [
            (sum(row[index] for row in pos) / len(pos)) - (sum(row[index] for row in neg) / len(neg))
            for index in range(len(X[0]))
        ]
        return self

    def predict_scores(self, X: list[list[float]]) -> list[float]:
        return [sum(weight * value for weight, value in zip(self.weights, row)) for row in X]
