from evaluation import accuracy, categorize_claim, expected_calibration_error, macro_f1


def test_classification_metrics_compute_expected_values() -> None:
    y_true = ["SUPPORTED", "REFUTED", "SUPPORTED"]
    y_pred = ["SUPPORTED", "SUPPORTED", "SUPPORTED"]

    assert accuracy(y_true, y_pred) == 2 / 3
    assert 0.0 < macro_f1(y_true, y_pred) <= 1.0


def test_expected_calibration_error_and_categorization() -> None:
    ece = expected_calibration_error([True, False, True], [0.9, 0.2, 0.8], bins=3)

    assert ece >= 0.0
    assert categorize_claim("Who won in 2020?") in {"entity", "temporal", "numerical"}
