"""Evaluation and benchmarking scripts."""

__all__ = ["expected_calibration_error"]


def expected_calibration_error(*args, **kwargs):
    from src.eval.research_metrics import expected_calibration_error as _expected_calibration_error

    return _expected_calibration_error(*args, **kwargs)
