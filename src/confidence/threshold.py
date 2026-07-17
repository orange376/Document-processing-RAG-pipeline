from __future__ import annotations


class ThresholdStrategy:
    """Classify a confidence score into an action category.

    Categories:
        "accept" — score is high enough to use the result as-is.
        "review" — score is moderate; the result needs human review.
        "reject" — score is too low; the result should be discarded.

    Thresholds are configurable at initialisation.
    """

    def __init__(self, accept: float = 0.75, reject: float = 0.40) -> None:
        """Store acceptance and rejection thresholds.

        Args:
            accept: Scores >= this value are classified as "accept".
                    Must be greater than ``reject``.
            reject: Scores >= this value are classified as "review";
                    anything below is "reject".

        Raises:
            ValueError: If ``accept <= reject``.
        """
        if accept <= reject:
            raise ValueError(
                f"accept threshold ({accept}) must be greater than "
                f"reject threshold ({reject})"
            )
        self._accept = accept
        self._reject = reject

    @property
    def accept_threshold(self) -> float:
        """Return the configured accept threshold."""
        return self._accept

    @property
    def reject_threshold(self) -> float:
        """Return the configured reject threshold."""
        return self._reject

    def classify(self, score: float) -> str:
        """Classify *score* into ``"accept"``, ``"review"``, or ``"reject"``.

        Args:
            score: A confidence score (expected in [0, 1]).

        Returns:
            One of ``"accept"``, ``"review"``, ``"reject"``.
        """
        if score >= self._accept:
            return "accept"
        if score >= self._reject:
            return "review"
        return "reject"
