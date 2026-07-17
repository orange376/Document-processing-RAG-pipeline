from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.confidence import ConfidenceScorer
from src.domain import Chunk, ChunkMetadata, LayoutElement, Table
from src.domain.layout import BBox


class TestConfidenceScorer:
    """Suite of tests for the 5-dimension ConfidenceScorer."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_layout(confidence: float, category: str = "text") -> LayoutElement:
        return LayoutElement(
            bbox=BBox(x0=0, y0=0, x1=10, y1=10, page_num=1),
            category=category,
            confidence=confidence,
            reading_order=0,
        )

    @staticmethod
    def _make_ocr_item(confidence: float) -> object:
        """Return a duck-typed object with a ``confidence`` attribute."""
        return type("OcrResult", (), {"confidence": confidence})()

    @staticmethod
    def _make_table(num_rows: int, num_cols: int) -> Table:
        return Table(
            bbox=BBox(x0=0, y0=0, x1=10, y1=10, page_num=1),
            num_rows=num_rows,
            num_cols=num_cols,
        )

    @staticmethod
    def _make_chunk(content: str, has_metadata: bool = True) -> Chunk:
        meta = (
            ChunkMetadata(
                source_file="test.pdf",
                page_num=1,
                section="s1",
                chunk_type="text",
            )
            if has_metadata
            else None
        )
        return Chunk(content=content, metadata=meta)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization(self):
        scorer = ConfidenceScorer()
        assert scorer is not None
        assert scorer.WEIGHTS == {
            "layout_quality": 0.25,
            "ocr_confidence": 0.20,
            "table_integrity": 0.15,
            "chunk_coherence": 0.20,
            "reranker_score": 0.20,
        }

    def test_import_from_package(self):
        """ConfidenceScorer should be importable from src.confidence."""
        from src.confidence import ConfidenceScorer as CS

        assert CS is ConfidenceScorer

    # ------------------------------------------------------------------
    # score() — general contract
    # ------------------------------------------------------------------

    def test_score_defaults(self):
        """All empty inputs should produce 0.0 overall."""
        scorer = ConfidenceScorer()
        result = scorer.score(
            layout_elements=[],
            ocr_results=[],
            tables=[],
            chunks=[],
            reranker_scores=[],
        )
        assert result["overall"] == 0.0
        for v in result["details"].values():
            assert v == 0.0

    def test_score_returns_correct_keys(self):
        scorer = ConfidenceScorer()
        result = scorer.score(
            layout_elements=[],
            ocr_results=[],
            tables=[],
            chunks=[],
            reranker_scores=[],
        )
        assert set(result.keys()) == {"overall", "details"}
        assert set(result["details"].keys()) == {
            "layout_quality",
            "ocr_confidence",
            "table_integrity",
            "chunk_coherence",
            "reranker_score",
        }

    def test_score_range(self):
        """Overall and all details should be in [0, 1]."""
        scorer = ConfidenceScorer()
        result = scorer.score(
            layout_elements=[self._make_layout(0.8)],
            ocr_results=[self._make_ocr_item(0.9)],
            tables=[self._make_table(3, 4)],
            chunks=[self._make_chunk("hello")],
            reranker_scores=[0.85, 0.72],
        )
        assert 0 <= result["overall"] <= 1.0
        for v in result["details"].values():
            assert 0 <= v <= 1.0

    def test_score_accepts_none(self):
        """None inputs should be handled as empty lists."""
        scorer = ConfidenceScorer()
        result = scorer.score(
            layout_elements=None,
            ocr_results=None,
            tables=None,
            chunks=None,
            reranker_scores=None,
        )
        assert result["overall"] == 0.0

    # ------------------------------------------------------------------
    # _score_layout
    # ------------------------------------------------------------------

    def test_layout_empty(self):
        scorer = ConfidenceScorer()
        assert scorer._score_layout([]) == 0.0

    def test_layout_mean_confidence(self):
        scorer = ConfidenceScorer()
        elements = [
            self._make_layout(0.7),
            self._make_layout(0.9),
            self._make_layout(0.5),
        ]
        assert scorer._score_layout(elements) == pytest.approx(0.7)

    def test_layout_single_element(self):
        scorer = ConfidenceScorer()
        elements = [self._make_layout(0.42)]
        assert scorer._score_layout(elements) == 0.42

    def test_layout_ignores_category(self):
        scorer = ConfidenceScorer()
        elements = [
            self._make_layout(0.8, category="table"),
            self._make_layout(0.6, category="figure"),
        ]
        assert scorer._score_layout(elements) == 0.7

    # ------------------------------------------------------------------
    # _score_ocr
    # ------------------------------------------------------------------

    def test_ocr_empty(self):
        scorer = ConfidenceScorer()
        assert scorer._score_ocr([]) == 0.0

    def test_ocr_mean_confidence(self):
        scorer = ConfidenceScorer()
        results = [
            self._make_ocr_item(0.95),
            self._make_ocr_item(0.85),
        ]
        assert scorer._score_ocr(results) == pytest.approx(0.9)

    def test_ocr_single_result(self):
        scorer = ConfidenceScorer()
        results = [self._make_ocr_item(0.5)]
        assert scorer._score_ocr(results) == 0.5

    def test_ocr_handles_tuples(self):
        """Support (bbox, text, confidence) tuples from easyocr."""
        scorer = ConfidenceScorer()
        results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "hello", 0.88),
            ([[5, 5], [15, 5], [15, 15], [5, 15]], "world", 0.92),
        ]
        assert scorer._score_ocr(results) == pytest.approx(0.9)

    def test_ocr_mixed_objects_and_tuples(self):
        """Gracefully handle heterogeneous lists."""
        scorer = ConfidenceScorer()
        results = [
            self._make_ocr_item(0.8),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "text", 0.9),
        ]
        assert scorer._score_ocr(results) == pytest.approx(0.85)

    def test_ocr_items_without_confidence(self):
        """Items without a confidence attribute / non-tuple should be skipped."""
        scorer = ConfidenceScorer()
        results = ["invalid", 42, None]
        assert scorer._score_ocr(results) == 0.0

    # ------------------------------------------------------------------
    # _score_tables
    # ------------------------------------------------------------------

    def test_tables_empty(self):
        scorer = ConfidenceScorer()
        assert scorer._score_tables([]) == 0.0

    def test_tables_all_well_formed(self):
        scorer = ConfidenceScorer()
        tables = [
            self._make_table(3, 4),
            self._make_table(1, 1),
            self._make_table(5, 2),
        ]
        assert scorer._score_tables(tables) == 1.0

    def test_tables_some_issues(self):
        scorer = ConfidenceScorer()
        tables = [
            self._make_table(3, 4),
            self._make_table(0, 0),  # zero rows and cols
        ]
        assert scorer._score_tables(tables) == 0.5

    def test_tables_none_well_formed(self):
        scorer = ConfidenceScorer()
        tables = [
            self._make_table(0, 0),
            self._make_table(0, 1),
        ]
        assert scorer._score_tables(tables) == 0.0

    def test_tables_single_well_formed(self):
        scorer = ConfidenceScorer()
        tables = [self._make_table(2, 3)]
        assert scorer._score_tables(tables) == 1.0

    # ------------------------------------------------------------------
    # _score_chunks
    # ------------------------------------------------------------------

    def test_chunks_empty(self):
        scorer = ConfidenceScorer()
        assert scorer._score_chunks([]) == 0.0

    def test_chunks_all_good(self):
        scorer = ConfidenceScorer()
        chunks = [
            self._make_chunk("content a"),
            self._make_chunk("content b"),
        ]
        assert scorer._score_chunks(chunks) == 1.0

    def test_chunks_partial_good(self):
        scorer = ConfidenceScorer()
        chunks = [
            self._make_chunk("good"),
            self._make_chunk("", has_metadata=True),  # empty content
            self._make_chunk("good", has_metadata=False),  # no metadata
        ]
        assert scorer._score_chunks(chunks) == pytest.approx(1 / 3)

    def test_chunks_no_metadata(self):
        scorer = ConfidenceScorer()
        chunks = [
            self._make_chunk("text", has_metadata=False),
            self._make_chunk("text2", has_metadata=False),
        ]
        assert scorer._score_chunks(chunks) == 0.0

    def test_chunks_empty_content(self):
        scorer = ConfidenceScorer()
        chunks = [
            self._make_chunk(""),
            self._make_chunk(""),
        ]
        assert scorer._score_chunks(chunks) == 0.0

    # ------------------------------------------------------------------
    # _score_reranker
    # ------------------------------------------------------------------

    def test_reranker_empty(self):
        scorer = ConfidenceScorer()
        assert scorer._score_reranker([]) == 0.0

    def test_reranker_single_score(self):
        scorer = ConfidenceScorer()
        assert scorer._score_reranker([0.75]) == 0.75

    def test_reranker_max_score(self):
        scorer = ConfidenceScorer()
        scores = [0.5, 0.95, 0.8, 0.3]
        assert scorer._score_reranker(scores) == 0.95

    def test_reranker_with_zeros(self):
        scorer = ConfidenceScorer()
        scores = [0.0, 0.0, 0.6]
        assert scorer._score_reranker(scores) == 0.6

    # ------------------------------------------------------------------
    # Integration: overall weighted score
    # ------------------------------------------------------------------

    def test_weighted_overall_calculation(self):
        """Verify the overall score is a weighted sum of details."""
        scorer = ConfidenceScorer()
        result = scorer.score(
            layout_elements=[self._make_layout(0.8)],
            ocr_results=[self._make_ocr_item(0.9)],
            tables=[self._make_table(3, 4)],
            chunks=[self._make_chunk("hello")],
            reranker_scores=[0.85],
        )
        expected = (
            0.25 * 0.8  # layout_quality
            + 0.20 * 0.9  # ocr_confidence
            + 0.15 * 1.0  # table_integrity
            + 0.20 * 1.0  # chunk_coherence
            + 0.20 * 0.85  # reranker_score
        )
        assert result["overall"] == pytest.approx(expected)

    def test_weighted_overall_with_zeros(self):
        """Zeroing one dimension should correctly reduce overall."""
        scorer = ConfidenceScorer()
        result = scorer.score(
            layout_elements=[self._make_layout(0.8)],
            ocr_results=[self._make_ocr_item(0.9)],
            tables=[],  # 0 for table_integrity
            chunks=[self._make_chunk("hello")],
            reranker_scores=[0.85],
        )
        expected = (
            0.25 * 0.8  # layout_quality
            + 0.20 * 0.9  # ocr_confidence
            + 0.15 * 0.0  # table_integrity (empty)
            + 0.20 * 1.0  # chunk_coherence
            + 0.20 * 0.85  # reranker_score
        )
        assert result["overall"] == pytest.approx(expected)
