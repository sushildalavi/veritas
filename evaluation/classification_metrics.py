"""Classification metrics for verifier outputs."""

from __future__ import annotations


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    return correct / len(y_true)


def per_class_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    return {label: _f1_for_label(y_true, y_pred, label) for label in labels}


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = list(per_class_f1(y_true, y_pred).values())
    return sum(scores) / len(scores) if scores else 0.0


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {label: {pred: 0 for pred in labels} for label in labels}
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    return matrix


def _f1_for_label(y_true: list[str], y_pred: list[str], label: str) -> float:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
    if not tp and not fp and not fn:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)
