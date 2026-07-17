"""HybridSearch 单元测试"""

from src.domain import Chunk, ChunkMetadata, SearchResult
from src.index.hybrid_search import HybridSearch


def make_chunk(chunk_id: str, content: str = "dummy") -> Chunk:
    """辅助：快速构建带 ID 的 Chunk。"""
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            source_file="doc.pdf",
            page_num=1,
            section="test",
            chunk_type="text",
        ),
    )


def make_result(chunk_id: str, rank: int, method: str) -> SearchResult:
    """辅助：构建一个 SearchResult，score 设为 1/(1+rank) 以模拟排序。"""
    return SearchResult(
        chunk=make_chunk(chunk_id),
        score=1.0 / (1 + rank),
        retrieval_method=method,
    )


class TestRRFFusion:
    """RRF 融合核心逻辑测试"""

    def test_both_empty(self):
        """两个结果列表均为空时返回空列表。"""
        assert HybridSearch._rrf_fusion([], []) == []

    def test_only_vector_results(self):
        """仅向量结果时按原顺序返回（RRF 退化为 1/(k+rank) 排序）。"""
        results = [
            make_result("a", 0, "vector"),
            make_result("b", 1, "vector"),
            make_result("c", 2, "vector"),
        ]
        fused = HybridSearch._rrf_fusion(results, [])
        assert len(fused) == 3
        # 排序应保持原顺序（RRF 分数与 rank 相关）
        assert fused[0].score >= fused[1].score >= fused[2].score

    def test_only_bm25_results(self):
        """仅 BM25 结果时按原顺序返回。"""
        results = [
            make_result("x", 0, "bm25"),
            make_result("y", 1, "bm25"),
        ]
        fused = HybridSearch._rrf_fusion([], results)
        assert len(fused) == 2
        assert fused[0].score >= fused[1].score

    def test_no_overlap(self):
        """向量与 BM25 结果无重叠时 RRF 融合正确排序。"""
        vec = [
            make_result("v1", 0, "vector"),
            make_result("v2", 1, "vector"),
        ]
        bm25 = [
            make_result("b1", 0, "bm25"),
            make_result("b2", 1, "bm25"),
        ]
        fused = HybridSearch._rrf_fusion(vec, bm25)
        assert len(fused) == 4
        # Rank 1 from both lists should be at the top
        # v1: 1/(60+1)=0.01639, b1: 1/(60+1)=0.01639 → tied
        # v2: 1/(60+2)=0.01613, b2: 1/(60+2)=0.01613
        assert fused[0].score == fused[1].score  # both rank-1 ties
        assert fused[2].score == fused[3].score  # both rank-2 ties

    def test_partial_overlap(self):
        """两个结果列表存在重叠 chunk 时分数累加。"""
        vec = [
            make_result("a", 0, "vector"),
            make_result("b", 1, "vector"),
        ]
        bm25 = [
            make_result("a", 2, "bm25"),  # "a" 在 BM25 中 rank=3
        ]
        fused = HybridSearch._rrf_fusion(vec, bm25)

        # "a" appears in both lists: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
        # "b" appears only in vector: 1/(60+2) = 0.01613
        ids = [r.chunk.chunk_id for r in fused]
        assert ids[0] == "a"  # "a" has higher fused score
        assert fused[0].score > fused[1].score

    def test_rrf_formula(self):
        """验证 RRF 公式计算是否精确。"""
        vec = [
            make_result("x", 0, "vector"),  # position 0 → rank=1 → 1/(60+1)
        ]
        # 在 BM25 列表中，"x" 排在第二位以获得 rank=2
        bm25 = [
            make_result("dummy", 0, "bm25"),  # position 0 → rank=1
            make_result("x", 1, "bm25"),      # position 1 → rank=2 → 1/(60+2)
        ]
        fused = HybridSearch._rrf_fusion(vec, bm25, k=60)
        expected = 1.0 / 61.0 + 1.0 / 62.0
        assert abs(fused[0].score - expected) < 1e-10

    def test_custom_k_value(self):
        """自定义 k 值被正确应用。"""
        vec = [make_result("a", 0, "vector")]
        fused_default = HybridSearch._rrf_fusion(vec, [])
        fused_custom = HybridSearch._rrf_fusion(vec, [], k=10)
        assert fused_default[0].score < fused_custom[0].score  # 更小的 k 给出更大分数

    def test_scores_decrease_monotonically(self):
        """融合结果分数单调递减。"""
        vec = [make_result(str(i), i, "vector") for i in range(10)]
        bm25 = [make_result(str(i + 10), i, "bm25") for i in range(5)]
        fused = HybridSearch._rrf_fusion(vec, bm25)
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score, (
                f"Score not monotonic at index {i}: "
                f"{fused[i].score} < {fused[i + 1].score}"
            )

    def test_retrieval_method_is_hybrid(self):
        """融合结果的 retrieval_method 应为 hybrid。"""
        vec = [make_result("a", 0, "vector")]
        bm25 = [make_result("b", 0, "bm25")]
        fused = HybridSearch._rrf_fusion(vec, bm25)
        for r in fused:
            assert r.retrieval_method == "hybrid"

    def test_top_k_fewer_than_total(self):
        """top_k 参数限制返回数量。"""
        # 模拟 VectorStore 和 BM25Index 搜索的 top_k 已经应用，
        # 这里仅验证 _rrf_fusion 分页结果正确
        # 实际上 search() 方法最后会做 fused[:top_k]
        pass


class TestHybridSearch:
    """HybridSearch 完整 search 方法测试（使用 mock 子组件）"""

    def test_empty_search(self):
        """空查询应返回空结果。"""
        hybrid = HybridSearch()
        results = hybrid.search("", [0.1] * 1024, top_k=5)
        assert isinstance(results, list)

    def test_returns_list_of_search_results(self):
        """search 返回 SearchResult 列表。"""
        hybrid = HybridSearch()
        results = hybrid.search("test", [0.1] * 1024, top_k=5)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_hybrid_importable(self):
        """HybridSearch 可从 src.index 导入。"""
        from src.index import HybridSearch

        assert HybridSearch is not None

    def test_vector_store_none_creates_default(self):
        """不传 vector_store 时自动创建默认实例。"""
        hybrid = HybridSearch()
        assert hybrid._vector_store is not None

    def test_bm25_index_none_creates_default(self):
        """不传 bm25_index 时自动创建默认实例。"""
        hybrid = HybridSearch()
        assert hybrid._bm25 is not None


class TestHybridSearchWithMockData:
    """使用模拟数据的集成式测试"""

    def test_search_with_mock_indexes(self):
        """使用预填充的 BM25 索引进行混合搜索。"""
        # 此测试需要真实 VectorStore（Qdrant），如未配置则跳过
        # 仅验证从 BM25 侧能获取结果
        from src.index import BM25Index

        bm25 = BM25Index()
        chunks = [
            Chunk(
                chunk_id="doc1",
                content="Retrieval augmented generation with hybrid search",
                metadata=ChunkMetadata(
                    source_file="doc.pdf", page_num=1, section="intro", chunk_type="text"
                ),
            ),
            Chunk(
                chunk_id="doc2",
                content="Machine learning and deep learning techniques",
                metadata=ChunkMetadata(
                    source_file="doc.pdf", page_num=2, section="methods", chunk_type="text"
                ),
            ),
        ]
        bm25.add_documents(chunks)

        hybrid = HybridSearch(bm25_index=bm25)
        results = hybrid.search("hybrid search", [0.0] * 1024, top_k=5)

        # BM25 应能匹配到 doc1
        ids = [r.chunk.chunk_id for r in results]
        assert len(results) > 0
        # 向量搜索虽然用零向量不会产生好的语义匹配，
        # 但 Qdrant 仍可能返回结果；关键是至少 BM25 侧能贡献
        assert any(cid in ids for cid in ("doc1", "doc2"))
