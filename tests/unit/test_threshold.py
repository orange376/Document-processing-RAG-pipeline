from __future__ import annotations

import pytest

from src.confidence.threshold import ThresholdStrategy


class TestThresholdStrategy:
    """Suite of tests for ThresholdStrategy classification."""

    # ------------------------------------------------------------------
    # Initialisation and configuration
    # ------------------------------------------------------------------

    def test_default_thresholds(self):
        strategy = ThresholdStrategy()
        assert strategy.accept_threshold == 0.75
        assert strategy.reject_threshold == 0.40

    def test_custom_thresholds(self):
        strategy = ThresholdStrategy(accept=0.8, reject=0.3)
        assert strategy.accept_threshold == 0.8
        assert strategy.reject_threshold == 0.3

    def test_accept_must_be_greater_than_reject(self):
        with pytest.raises(ValueError, match="must be greater than"):
            ThresholdStrategy(accept=0.3, reject=0.5)

    def test_equal_thresholds_raises_error(self):
        with pytest.raises(ValueError, match="must be greater than"):
            ThresholdStrategy(accept=0.5, reject=0.5)

    # ------------------------------------------------------------------
    # classify — accept
    # ------------------------------------------------------------------

    def test_classify_accept_default(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(0.85) == "accept"

    def test_classify_accept_at_threshold(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(0.75) == "accept"

    def test_classify_accept_custom_threshold(self):
        strategy = ThresholdStrategy(accept=0.9, reject=0.3)
        assert strategy.classify(0.95) == "accept"
        assert strategy.classify(0.9) == "accept"
        assert strategy.classify(0.89) == "review"

    def test_classify_accept_perfect_score(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(1.0) == "accept"

    # ------------------------------------------------------------------
    # classify — review
    # ------------------------------------------------------------------

    def test_classify_review_default(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(0.60) == "review"

    def test_classify_review_just_above_reject(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(0.40) == "review"

    def test_classify_review_range(self):
        strategy = ThresholdStrategy()
        for score in (0.41, 0.50, 0.60, 0.74):
            assert strategy.classify(score) == "review", f"failed at {score}"

    def test_classify_review_custom_threshold(self):
        strategy = ThresholdStrategy(accept=0.8, reject=0.3)
        assert strategy.classify(0.79) == "review"
        assert strategy.classify(0.30) == "review"

    # ------------------------------------------------------------------
    # classify — reject
    # ------------------------------------------------------------------

    def test_classify_reject_default(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(0.30) == "reject"

    def test_classify_reject_below_threshold(self):
        strategy = ThresholdStrategy()
        for score in (0.0, 0.10, 0.20, 0.39):
            assert strategy.classify(score) == "reject", f"failed at {score}"

    def test_classify_reject_custom_threshold(self):
        strategy = ThresholdStrategy(accept=0.8, reject=0.3)
        assert strategy.classify(0.29) == "reject"
        assert strategy.classify(0.0) == "reject"

    def test_classify_reject_zero(self):
        strategy = ThresholdStrategy()
        assert strategy.classify(0.0) == "reject"
