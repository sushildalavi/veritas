"""Confidence calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float


def expected_calibration_error(y_true: list[bool], confidences: list[float], *, bins: int = 10) -> float:
    if not y_true:
        return 0.0
    bin_bins = _make_bins(bins)
    ece = 0.0
    for lower, upper in bin_bins:
        indices = [index for index, confidence in enumerate(confidences) if lower <= confidence < upper or (upper == 1.0 and confidence == 1.0)]
        if not indices:
            continue
        bin_accuracy = sum(1 for index in indices if y_true[index]) / len(indices)
        bin_confidence = sum(confidences[index] for index in indices) / len(indices)
        ece += abs(bin_accuracy - bin_confidence) * (len(indices) / len(y_true))
    return ece


def calibration_bins(y_true: list[bool], confidences: list[float], *, bins: int = 10) -> list[CalibrationBin]:
    results: list[CalibrationBin] = []
    for lower, upper in _make_bins(bins):
        indices = [index for index, confidence in enumerate(confidences) if lower <= confidence < upper or (upper == 1.0 and confidence == 1.0)]
        if not indices:
            results.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        results.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(indices),
                accuracy=sum(1 for index in indices if y_true[index]) / len(indices),
                confidence=sum(confidences[index] for index in indices) / len(indices),
            )
        )
    return results


def _make_bins(bins: int) -> list[tuple[float, float]]:
    step = 1.0 / bins
    intervals = []
    start = 0.0
    for index in range(bins):
        end = 1.0 if index == bins - 1 else round(start + step, 10)
        intervals.append((start, end))
        start = end
    return intervals
