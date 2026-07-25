"""Test suite for ConfidenceScorer and ThresholdStrategy."""

import pytest
from src.confidence.scorer import ConfidenceScorer
from src.confidence.threshold import ThresholdStrategy


def test_scorer_query_signals():
    scorer = ConfidenceScorer()
    result = scorer.score(
        reranker_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        num_results=5,
    )
    assert "overall" in result
    assert "details" in result
    assert 0.0 <= result["overall"] <= 1.0


def test_scorer_no_results():
    scorer = ConfidenceScorer()
    result = scorer.score(reranker_scores=[], num_results=0)
    assert result["overall"] == 0.0


def test_threshold_accept():
    t = ThresholdStrategy(accept=0.75, reject=0.40)
    assert t.classify(0.85) == "accept"
    assert t.classify(0.80) == "accept"


def test_threshold_review():
    t = ThresholdStrategy(accept=0.75, reject=0.40)
    assert t.classify(0.60) == "review"
    assert t.classify(0.50) == "review"


def test_threshold_reject():
    t = ThresholdStrategy(accept=0.75, reject=0.40)
    assert t.classify(0.30) == "reject"
    assert t.classify(0.10) == "reject"


def test_threshold_boundaries():
    t = ThresholdStrategy(accept=0.75, reject=0.40)
    assert t.classify(0.75) == "accept"  # ≥ accept threshold
    assert t.classify(0.40) == "review"  # ≥ reject but < accept


def test_scorer_with_layout_elements():
    scorer = ConfidenceScorer()
    from src.domain import LayoutElement, BBox
    elements = [
        LayoutElement(bbox=BBox(0, 0, 100, 100, 1), category="text", confidence=0.9, reading_order=0),
        LayoutElement(bbox=BBox(0, 100, 100, 200, 1), category="table", confidence=0.7, reading_order=1),
        LayoutElement(bbox=BBox(0, 200, 100, 300, 1), category="formula", confidence=0.8, reading_order=2),
    ]
    result = scorer.score(layout_elements=elements)
    assert result["overall"] > 0.0
