"""Evaluation package for Veritas."""

from .beir_benchmark import BeirBenchmarkResult, run_beir_benchmark
from .classification_metrics import accuracy, confusion_matrix, macro_f1, per_class_f1
from .confidence_analysis import CalibrationBin, calibration_bins, expected_calibration_error
from .error_analysis import ErrorExample, build_error_analysis, categorize_claim, write_error_report
from .faithfulness_metrics import citation_precision, unsupported_sentence_rate, verdict_consistency
from .hallucination_eval import hallucination_rate
from .pareto_analysis import ParetoPoint, pareto_report

__all__ = [
    "BeirBenchmarkResult",
    "CalibrationBin",
    "ErrorExample",
    "ParetoPoint",
    "accuracy",
    "build_error_analysis",
    "calibration_bins",
    "categorize_claim",
    "citation_precision",
    "confusion_matrix",
    "expected_calibration_error",
    "hallucination_rate",
    "macro_f1",
    "pareto_report",
    "per_class_f1",
    "run_beir_benchmark",
    "unsupported_sentence_rate",
    "verdict_consistency",
    "write_error_report",
]
