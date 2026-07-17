from __future__ import annotations

import pytest

from src.confidence.fallback import (
    RouteResult,
    route,
    route_accept,
    route_reject,
    route_review,
)


class TestRouteResult:
    """Tests for RouteResult dataclass."""

    def test_default_construction(self):
        result = RouteResult(decision="accept")
        assert result.decision == "accept"
        assert result.data == {}
        assert result.needs_review is False
        assert result.metadata == {}

    def test_full_construction(self):
        result = RouteResult(
            decision="review",
            data={"key": "val"},
            needs_review=True,
            metadata={"source": "test"},
        )
        assert result.decision == "review"
        assert result.data == {"key": "val"}
        assert result.needs_review is True
        assert result.metadata == {"source": "test"}


class TestRouteAccept:
    """Tests for route_accept."""

    def test_returns_accept_decision(self):
        result = route_accept({"answer": "hello"})
        assert result.decision == "accept"
        assert result.needs_review is False

    def test_preserves_data(self):
        data = {"answer": "hello", "score": 0.95}
        result = route_accept(data)
        assert result.data == data

    def test_preserves_metadata(self):
        data = {"answer": "hello", "metadata": {"source": "doc1"}}
        result = route_accept(data)
        assert result.metadata == {"source": "doc1"}

    def test_empty_result(self):
        result = route_accept({})
        assert result.data == {}
        assert result.metadata == {}


class TestRouteReview:
    """Tests for route_review."""

    def test_returns_review_decision(self):
        result = route_review({"answer": "maybe"})
        assert result.decision == "review"
        assert result.needs_review is True

    def test_preserves_data(self):
        data = {"answer": "maybe", "score": 0.55}
        result = route_review(data)
        assert result.data == data

    def test_default_reason(self):
        result = route_review({"answer": "maybe"})
        assert result.needs_review is True

    def test_custom_reason(self):
        result = route_review(
            {"answer": "maybe"}, reason="Score just below threshold"
        )
        assert result.needs_review is True

    def test_preserves_metadata(self):
        data = {"answer": "maybe", "metadata": {"page": 3}}
        result = route_review(data)
        assert result.metadata == {"page": 3}


class TestRouteReject:
    """Tests for route_reject."""

    def test_returns_reject_decision(self):
        result = route_reject()
        assert result.decision == "reject"
        assert result.needs_review is False

    def test_empty_data(self):
        result = route_reject()
        assert result.data == {}

    def test_empty_metadata(self):
        result = route_reject()
        assert result.metadata == {}

    def test_logs_reason(self):
        result = route_reject(reason="Score too low")
        assert result.decision == "reject"
        assert result.data == {}

    def test_accepts_original_result(self):
        result = route_reject({"answer": "bad"})
        assert result.decision == "reject"
        assert result.data == {}  # data is always empty on reject


class TestRouteDispatch:
    """Tests for the top-level ``route()`` dispatcher."""

    def test_route_accept(self):
        result = route("accept", {"answer": "yes"})
        assert result.decision == "accept"
        assert result.data == {"answer": "yes"}
        assert result.needs_review is False

    def test_route_review(self):
        result = route("review", {"answer": "maybe"})
        assert result.decision == "review"
        assert result.needs_review is True

    def test_route_reject(self):
        result = route("reject", {"answer": "no"})
        assert result.decision == "reject"
        assert result.data == {}

    def test_unknown_decision(self):
        with pytest.raises(ValueError, match="Unknown decision"):
            route("unknown", {})

    def test_route_empty_result(self):
        result = route("accept", {})
        assert result.data == {}
        assert result.metadata == {}
