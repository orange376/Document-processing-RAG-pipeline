from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """The result of routing a confidence-scored pipeline output.

    Attributes:
        decision: One of ``"accept"``, ``"review"``, ``"reject"``.
        data: The original result data. For ``"accept"`` this is the
              full result; for ``"review"`` it is the original data
              with ``needs_review=True`` semantics; for ``"reject"``
              it is an empty/default result.
        needs_review: Flag indicating the result requires human review
                      (``True`` only for ``"review"`` decisions).
        metadata: Optional arbitrary metadata preserved from the
                  original pipeline stage.
    """

    decision: str
    data: dict[str, Any] = field(default_factory=dict)
    needs_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def route_accept(result: dict[str, Any]) -> RouteResult:
    """Return the result as-is with an ``"accept"`` decision.

    Args:
        result: The full pipeline result dict.

    Returns:
        A ``RouteResult`` with ``decision="accept"`` and the original data.
    """
    return RouteResult(
        decision="accept",
        data=result,
        needs_review=False,
        metadata=result.get("metadata", {}),
    )


def route_review(
    result: dict[str, Any],
    reason: str = "Moderate confidence — human review requested",
) -> RouteResult:
    """Flag the result for human review.

    A warning is logged with the provided *reason*.

    Args:
        result: The pipeline result dict (may be partial / lower confidence).
        reason: Human-readable explanation for the review.

    Returns:
        A ``RouteResult`` with ``decision="review"`` and
        ``needs_review=True``.
    """
    logger.warning("RouteResult: review — %s", reason)
    return RouteResult(
        decision="review",
        data=result,
        needs_review=True,
        metadata=result.get("metadata", {}),
    )


def route_reject(
    result: dict[str, Any] | None = None,
    reason: str = "Insufficient confidence — result rejected",
) -> RouteResult:
    """Return an empty/default result indicating rejection.

    An info message is logged with the provided *reason*.

    Args:
        result: Optional original result (primarily for auditing / logging).
        reason: Human-readable explanation for the rejection.

    Returns:
        A ``RouteResult`` with ``decision="reject"`` and empty data.
    """
    logger.info("RouteResult: reject — %s", reason)
    if result is not None:
        logger.info("Rejected result: %s", result)
    return RouteResult(
        decision="reject",
        data={},
        needs_review=False,
        metadata={},
    )


def route(decision: str, result: dict[str, Any]) -> RouteResult:
    """Dispatch *result* to the appropriate routing function based on *decision*.

    Args:
        decision: One of ``"accept"``, ``"review"``, ``"reject"``.
        result: The pipeline result dict.

    Returns:
        A ``RouteResult`` produced by the matching route function.

    Raises:
        ValueError: If *decision* is not one of the recognised values.
    """
    if decision == "accept":
        return route_accept(result)
    elif decision == "review":
        return route_review(result)
    elif decision == "reject":
        return route_reject(result)
    else:
        raise ValueError(
            f"Unknown decision: {decision!r}. "
            f"Expected one of 'accept', 'review', 'reject'."
        )
