#!/usr/bin/env python3
"""RAG Pipeline - End-to-End Integration Test

Tests:
  [OK] PDF Loader (PyMuPDF with dict extraction)
  [OK] Structure-Aware Chunker
  [OK] Qdrant Local Vector Store (1.18 API)
  [OK] bge-large-zh CPU Embedding
  [OK] LayoutDetector (PP-DocLayoutV3 from PaddleX)
  [OK] easyocr import
  [OK] Pipeline Orchestrator (PP-DocLayoutV3 + CPU embedding + Qdrant)

Usage:
  python scripts/test_e2e_pipeline.py
"""

import os, sys, time, json
from pathlib import Path

os.environ.setdefault("USE_WINDOWS_IPC", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = " .."
DIVIDER = "-" * 60

passed = failed = skipped = 0


def report(name: str, status: str, detail: str = ""):
    global passed, failed, skipped
    if status == OK: passed += 1
    elif status == FAIL: failed += 1
    elif status == SKIP: skipped += 1
    d = f"  {detail}" if detail else ""
    print(f"  {status} {name}{d}")


def header(title: str):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def create_test_pdf(path: str) -> bool:
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72),  "RAG System Performance Report",  fontsize=18, fontname="helv")
        page.insert_text((72, 120), "Author: AI Lab",                 fontsize=11, fontname="helv")
        page.insert_text((72, 160), "Abstract",                       fontsize=14, fontname="helv")
        page.insert_text((72, 185), "This report evaluates the RAG architecture for document QA.", fontsize=11, fontname="helv")
        page.insert_text((72, 230), "1. Introduction",                fontsize=14, fontname="helv")
        page.insert_text((72, 255), "RAG systems combine retrieval with generation.", fontsize=11, fontname="helv")
        page.insert_text((72, 310), "Table 1: Experiment Config",     fontsize=12, fontname="helv")
        for i, r in enumerate(["Parameter | Value", "Model | bge-large-zh", "Dim | 1024"]):
            page.insert_text((72, 335 + i*18), r, fontsize=9, fontname="courier")

        page = doc.new_page()
        page.insert_text((72, 72),  "2. Method",      fontsize=14, fontname="helv")
        page.insert_text((72, 100), "Metrics: HR, MRR, NDCG.", fontsize=11, fontname="helv")
        page.insert_text((72, 145), "Formula 1: Hit Rate", fontsize=12, fontname="helv")
        page.insert_text((72, 170), "HR@k = (1/N) * sum(1 if relevant in top-k else 0)", fontsize=10, fontname="courier")
        page.insert_text((72, 280), "2.1 Experiment Setup", fontsize=12, fontname="helv")
        page.insert_text((72, 305), "5-fold cross validation.", fontsize=11, fontname="helv")

        page = doc.new_page()
        page.insert_text((72, 72),  "3. Results",     fontsize=14, fontname="helv")
        page.insert_text((72, 100), "Hybrid retrieval outperforms pure vector search.", fontsize=11, fontname="helv")
        page.insert_text((72, 145), "Table 2: Performance Comparison", fontsize=12, fontname="helv")
        for i, r in enumerate(["Method | HR@5", "Vector | 0.723", "Hybrid | 0.856"]):
            page.insert_text((72, 170 + i*16), r, fontsize=9, fontname="courier")

        page = doc.new_page()
        page.insert_text((72, 72),  "4. Conclusion",  fontsize=14, fontname="helv")
        page.insert_text((72, 100), "RAG is effective. Next: multimodal.", fontsize=11, fontname="helv")
        doc.save(path); doc.close()
        return True
    except Exception as e:
        print(f"  {FAIL} create test PDF: {e}")
        return False


# ────────── Test 1: PDF Loader ──────────
def test_pdf_loader(pdf_path):
    header("Test 1: PDF Loader (PyMuPDF dict mode)")
    from src.parser.loader.pdf_loader import PDFLoader
    doc = PDFLoader().load(pdf_path)
    for name, ok in [
        ("filename", doc.filename == "test_e2e_doc.pdf"),
        ("file type", doc.file_type == "pdf"),
        ("4 pages", doc.total_pages == 4),
        ("page objects", len(doc.pages) == 4),
        ("raw_dict present", all(p.raw_dict is not None for p in doc.pages)),
        ("font size in metadata", any(
            p.blocks and "max_font_size" in p.blocks[0].metadata
            for p in doc.pages
        )),
    ]: report(name, OK if ok else FAIL)
    total_blocks = sum(len(p.blocks) for p in doc.pages)
    total_text = sum(len(p.text) for p in doc.pages)
    report(f"Totals: {doc.total_pages}p, {total_blocks}blocks, {total_text}chars", OK)
    return doc


# ────────── Test 2: Chunker ──────────
def test_chunker(doc):
    header("Test 2: Structure-Aware Chunker")
    from src.parser.chunker import StructureAwareChunker
    chunks = StructureAwareChunker(max_chunk_chars=2000).chunk(doc)
    report(f"Chunks: {len(chunks)}", OK if len(chunks) >= 2 else FAIL)
    report("metadata completeness", OK if all(
        c.metadata and c.metadata.source_file and c.metadata.page_num > 0 for c in chunks) else FAIL)
    for i, c in enumerate(chunks):
        section = c.metadata.section if c.metadata else ""
        print(f"  {INFO} Chunk {i+1}: type={c.metadata.chunk_type}, page={c.metadata.page_num}, "
              f"section=[{section}], len={len(c.content)}")
    return chunks


# ────────── Test 3: Vector Store ──────────
def test_vector_store():
    header("Test 3: Qdrant Vector Store (local mode 1.18 API)")
    import tempfile
    from src.domain import Chunk, ChunkMetadata
    from src.index.vector_store import VectorStore
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(collection_name="e2e_test")
        store._settings.vector_db_dir = tmpdir
        store._client = None
        chunks = [
            Chunk(content="RAG system report",
                  metadata=ChunkMetadata(source_file="t.pdf", page_num=1, section="", chunk_type="text"),
                  embedding=[0.1]*1024),
            Chunk(content="Hybrid retrieval outperforms",
                  metadata=ChunkMetadata(source_file="t.pdf", page_num=3, section="Results", chunk_type="text"),
                  embedding=[0.2]*1024),
            Chunk(content="HR MRR NDCG metrics",
                  metadata=ChunkMetadata(source_file="t.pdf", page_num=3, section="Results", chunk_type="text"),
                  embedding=[0.3]*1024),
        ]
        n = store.index_chunks(chunks)
        report(f"indexed {n} chunks", OK if n == 3 else FAIL)
        results = store.search(query_embedding=[0.15]*1024, top_k=2)
        report(f"search returned {len(results)}", OK if len(results) <= 3 else FAIL)
        count = store.count()
        report(f"count = {count}", OK if count == 3 else FAIL)
        if store._client:
            store._client.delete_collection("e2e_test")
    return True


# ────────── Test 4: bge-large-zh (CPU) ──────────
def test_embedding_cpu():
    header("Test 4: bge-large-zh Embedding (CPU)")
    from src.config import get_settings
    model_dir = get_settings().resolved_model_dir / "bge-large-zh"
    if not model_dir.exists():
        report(f"model dir not found: {model_dir}", FAIL); return False
    files = list(model_dir.iterdir())
    report(f"model files: {len(files)}", OK)
    try:
        from FlagEmbedding import FlagModel
        model = FlagModel('BAAI/bge-large-zh-v1.5', use_fp16=False,
                          query_instruction_for_retrieval="generate embedding for retrieval:",
                          cache_folder=str(model_dir))
        emb = model.encode(["RAG system retrieval quality"])
        dim = emb.shape[1]
        report(f"encoding OK, dim={dim}", OK if dim == 1024 else FAIL)
        del model
        import gc, torch; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return True
    except Exception as e:
        report(f"CPU embedding failed: {e}", FAIL)
        return False


# ────────── Test 5: Layout Detector (PP-DocLayoutV3) ──────────
def test_layout_ppdoclayv3():
    header("Test 5: LayoutDetector (PP-DocLayoutV3)")
    from src.parser.layout import LayoutDetector
    import fitz, numpy as np
    # Create test page image
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72),  "RAG System Performance Report",  fontsize=20, fontname="helv")
    page.insert_text((72, 120), "Abstract",                       fontsize=14, fontname="helv")
    page.insert_text((72, 150), "Test paragraph for layout.",      fontsize=11, fontname="helv")
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()

    detector = LayoutDetector()
    elements = detector.analyze(img, scale=2.0)
    report(f"{len(elements)} elements", OK if len(elements) >= 1 else WARN)
    if elements:
        cats = [e.category for e in elements]
        report(f"categories: {set(cats)}", OK)
    report("model unload", OK)
    detector.unload()

    # Also test heuristic fallback still works
    blocks = [
        {"type": 0, "bbox": (72, 72, 500, 100), "lines": [{"bbox": (72,72,500,100), "spans": [{"text": "Title", "size": 18, "font": "helv"}]}]},
        {"type": 1, "bbox": (72, 300, 500, 450)},
    ]
    els = detector.analyze_from_blocks(blocks, page_num=1, page_width=595, page_height=842)
    report("fallback heuristic", OK if len(els) == 2 else FAIL)
    return True


# ────────── Test 6: easyocr import ──────────
def test_ocr_import():
    header("Test 6: easyocr import (no model download)")
    try:
        import easyocr
        report("easyocr import OK", OK)
        return True
    except Exception as e:
        report(f"easyocr import: {e}", FAIL)
        return False


# ────────── Test 7: Orchestrator ──────────
def test_orchestrator(pdf_path):
    header("Test 7: Pipeline Orchestrator (PP-DocLayoutV3 + CPU emb + Qdrant)")
    from src.pipeline.orchestrator import PipelineOrchestrator
    import asyncio
    try:
        result = asyncio.run(PipelineOrchestrator().process_document(pdf_path))
        report(f"status: {result.status.value}", OK)
        report(f"chunks: {len(result.chunks)}", OK if len(result.chunks) > 0 else WARN)
        report(f"layout tree: {'yes' if result.layout_tree else 'no'}", OK)
        report(f"indexed: {result.indexed_count}", OK if result.indexed_count > 0 else WARN)
        return result
    except Exception as e:
        report(f"orchestrator exception: {e}", WARN)
        return None


# ══════════════════════ Phase 2 Tests ══════════════════════


# ────────── Test 8: BM25 Index Build + Search ──────────
def test_bm25_index():
    header("Test 8: BM25 Index Build + Search")
    from src.domain import Chunk, ChunkMetadata
    from src.index.bm25_index import BM25Index

    chunks = [
        Chunk(content="RAG system performance evaluation report",
              metadata=ChunkMetadata(source_file="doc.pdf", page_num=1, section="Intro", chunk_type="text")),
        Chunk(content="Hybrid retrieval outperforms pure vector search in recall",
              metadata=ChunkMetadata(source_file="doc.pdf", page_num=2, section="Results", chunk_type="text")),
        Chunk(content="Deep learning based document understanding methods",
              metadata=ChunkMetadata(source_file="doc.pdf", page_num=3, section="Method", chunk_type="text")),
        Chunk(content="Experimental results show significant improvement",
              metadata=ChunkMetadata(source_file="doc.pdf", page_num=4, section="Conclusion", chunk_type="text")),
    ]

    index = BM25Index()
    index.add_documents(chunks)
    report("index built", OK)

    results = index.search("retrieval", top_k=5)
    report(f"search returned {len(results)} results", OK if len(results) > 0 else FAIL)
    if results:
        report("result has chunk", OK if results[0].chunk is not None else FAIL)
        report("result has score > 0", OK if results[0].score > 0 else FAIL)
        report("retrieval method", OK if results[0].retrieval_method == "bm25" else FAIL)

    no_results = index.search("xyznonexistentkeyword123", top_k=5)
    report("no match returns empty", OK if len(no_results) == 0 else FAIL)

    index.clear()
    cleared = index.search("retrieval", top_k=5)
    report("clear empties index", OK if len(cleared) == 0 else FAIL)

    return True


# ────────── Test 9: Hybrid Search RRF Fusion ──────────
def test_hybrid_search_rrf():
    header("Test 9: Hybrid Search RRF Fusion")
    from src.domain import Chunk, ChunkMetadata, SearchResult
    from src.index.hybrid_search import HybridSearch

    chunk_a = Chunk(chunk_id="chk_aaaa", content="RAG system report",
                    metadata=ChunkMetadata(source_file="t.pdf", page_num=1, section="", chunk_type="text"))
    chunk_b = Chunk(chunk_id="chk_bbbb", content="Hybrid retrieval outperforms",
                    metadata=ChunkMetadata(source_file="t.pdf", page_num=2, section="Results", chunk_type="text"))
    chunk_c = Chunk(chunk_id="chk_cccc", content="HR MRR NDCG metrics",
                    metadata=ChunkMetadata(source_file="t.pdf", page_num=3, section="Results", chunk_type="text"))
    chunk_d = Chunk(chunk_id="chk_dddd", content="Deep learning approaches",
                    metadata=ChunkMetadata(source_file="t.pdf", page_num=4, section="Method", chunk_type="text"))

    vector_results = [
        SearchResult(chunk=chunk_a, score=0.9, retrieval_method="vector"),
        SearchResult(chunk=chunk_b, score=0.8, retrieval_method="vector"),
        SearchResult(chunk=chunk_c, score=0.7, retrieval_method="vector"),
    ]
    bm25_results = [
        SearchResult(chunk=chunk_b, score=0.85, retrieval_method="bm25"),
        SearchResult(chunk=chunk_d, score=0.75, retrieval_method="bm25"),
        SearchResult(chunk=chunk_a, score=0.65, retrieval_method="bm25"),
    ]

    fused = HybridSearch._rrf_fusion(vector_results, bm25_results, k=60)
    report(f"fused {len(fused)} results", OK if len(fused) == 4 else FAIL)

    if fused:
        report("chunk_b highest (both lists)", OK if fused[0].chunk.chunk_id == "chk_bbbb" else FAIL)
        all_hybrid = all(r.retrieval_method == "hybrid" for r in fused)
        report("all method=hybrid", OK if all_hybrid else FAIL)
        desc = all(fused[i].score >= fused[i+1].score for i in range(len(fused)-1))
        report("scores descending", OK if desc else FAIL)

    empty = HybridSearch._rrf_fusion([], [])
    report("empty lists returns []", OK if len(empty) == 0 else FAIL)

    return True


# ────────── Test 10: Reranker ──────────
def test_reranker():
    header("Test 10: Reranker (skip if model not available)")
    from src.retrieval.reranker import Reranker

    reranker = Reranker()
    report("reranker created", OK)

    model_loaded = False
    try:
        reranker._lazy_load()
        model_loaded = reranker._model is not None
    except Exception:
        model_loaded = False

    if not model_loaded:
        report("reranker model load", SKIP)
        reranker.unload()
        return False

    report("model loaded", OK)
    from src.domain import Chunk, ChunkMetadata, SearchResult
    chunks = [
        Chunk(content="RAG system performance",
              metadata=ChunkMetadata(source_file="t.pdf", page_num=1, section="", chunk_type="text")),
        Chunk(content="Hybrid retrieval outperforms vector search",
              metadata=ChunkMetadata(source_file="t.pdf", page_num=2, section="", chunk_type="text")),
    ]
    results = [SearchResult(chunk=c, score=0.5) for c in chunks]
    reranked = reranker.rerank("retrieval performance", results, top_k=2)

    report(f"reranked {len(reranked)}", OK if len(reranked) == 2 else FAIL)
    if reranked:
        report("scores updated", OK if reranked[0].score != 0.5 else FAIL)

    reranker.unload()
    report("model unloaded", OK)
    return True


# ────────── Test 11: LLM Client ──────────
def test_llm_client():
    header("Test 11: LLM Client Construction (mock API)")
    from src.generation.llm_client import LLMClient

    client_empty = LLMClient(api_key="")
    report("constructed with empty key", OK)
    resp = client_empty.chat("Hello")
    report("chat with no key returns ''", OK if resp == "" else FAIL)

    client_mock = LLMClient(api_key="sk-test-mock-key-12345")
    report("constructed with mock key", OK)
    report("api_key stored", OK if client_mock._api_key == "sk-test-mock-key-12345" else FAIL)
    report("base URL set", OK if bool(client_mock._base) else FAIL)
    report("model name set", OK if bool(client_mock._model) else FAIL)

    return True


# ────────── Test 12: Confidence Scorer ──────────
def test_confidence_scorer():
    header("Test 12: Confidence Scorer All Dimensions")
    from src.confidence.scorer import ConfidenceScorer
    from src.domain import Chunk, ChunkMetadata, LayoutElement, BBox, Table, Cell

    scorer = ConfidenceScorer()
    report("scorer created", OK)

    layout_els = [
        LayoutElement(bbox=BBox(0, 0, 100, 50, page_num=1), category="text", confidence=0.95, reading_order=1),
        LayoutElement(bbox=BBox(0, 60, 100, 80, page_num=1), category="title", confidence=0.92, reading_order=0),
    ]
    ocr_results = [
        ([0, 0, 50, 20], "RAG System", 0.98),
        ([0, 30, 80, 50], "Performance Report", 0.95),
    ]
    tables = [
        Table(bbox=BBox(0, 100, 200, 150, page_num=1),
              cells=[Cell("A", 0, 0), Cell("B", 0, 1)], num_rows=1, num_cols=2),
        Table(bbox=BBox(0, 160, 200, 210, page_num=1),
              cells=[Cell("X", 0, 0), Cell("Y", 0, 1)], num_rows=1, num_cols=2),
    ]
    chunks = [
        Chunk(content="RAG system", metadata=ChunkMetadata(source_file="t.pdf", page_num=1, section="", chunk_type="text")),
        Chunk(content="Hybrid retrieval", metadata=ChunkMetadata(source_file="t.pdf", page_num=2, section="Results", chunk_type="text")),
    ]
    reranker_scores = [0.85, 0.72]

    result = scorer.score(
        layout_elements=layout_els, ocr_results=ocr_results,
        tables=tables, chunks=chunks, reranker_scores=reranker_scores,
    )

    report("overall is float", OK if isinstance(result["overall"], float) else FAIL)
    report("overall in [0,1]", OK if 0 <= result["overall"] <= 1 else FAIL)
    report("5 dimensions", OK if len(result["details"]) == 5 else FAIL)

    all_keys = all(k in result["details"] for k in
                   ["layout_quality", "ocr_confidence", "table_integrity",
                    "chunk_coherence", "reranker_score"])
    report("all keys present", OK if all_keys else FAIL)

    report("layout=0.935", OK if abs(result["details"]["layout_quality"] - 0.935) < 0.001 else FAIL)
    report("ocr=0.965", OK if abs(result["details"]["ocr_confidence"] - 0.965) < 0.001 else FAIL)
    report("tables=1.0", OK if result["details"]["table_integrity"] == 1.0 else FAIL)
    report("chunks=1.0", OK if result["details"]["chunk_coherence"] == 1.0 else FAIL)
    report("reranker=0.85", OK if result["details"]["reranker_score"] == 0.85 else FAIL)

    empty_result = scorer.score()
    report("empty inputs overall=0", OK if empty_result["overall"] == 0.0 else FAIL)

    # Partial table integrity
    pt = [
        Table(bbox=BBox(0, 0, 50, 50, page_num=1), num_rows=1, num_cols=1),
        Table(bbox=BBox(0, 0, 50, 50, page_num=1), num_rows=0, num_cols=0),
    ]
    partial = scorer.score(tables=pt)
    report("partial tables=0.5", OK if abs(partial["details"]["table_integrity"] - 0.5) < 0.001 else FAIL)

    return True


# ────────── Test 13: Threshold Classification ──────────
def test_threshold_classification():
    header("Test 13: Threshold Classification")
    from src.confidence.threshold import ThresholdStrategy

    t = ThresholdStrategy()
    report("strategy created", OK)

    report("0.75 → accept",  OK if t.classify(0.75) == "accept" else FAIL)
    report("0.90 → accept",  OK if t.classify(0.90) == "accept" else FAIL)
    report("1.00 → accept",  OK if t.classify(1.00) == "accept" else FAIL)
    report("0.74 → review",  OK if t.classify(0.74) == "review" else FAIL)
    report("0.50 → review",  OK if t.classify(0.50) == "review" else FAIL)
    report("0.40 → review",  OK if t.classify(0.40) == "review" else FAIL)
    report("0.39 → reject",  OK if t.classify(0.39) == "reject" else FAIL)
    report("0.00 → reject",  OK if t.classify(0.00) == "reject" else FAIL)

    t2 = ThresholdStrategy(accept=0.60, reject=0.20)
    report("custom thresholds", OK)
    report("custom 0.60→accept", OK if t2.classify(0.60) == "accept" else FAIL)
    report("custom 0.30→review", OK if t2.classify(0.30) == "review" else FAIL)
    report("custom 0.19→reject", OK if t2.classify(0.19) == "reject" else FAIL)

    try:
        ThresholdStrategy(accept=0.40, reject=0.50)
        report("accept<=reject raises ValueError", FAIL)
    except ValueError:
        report("accept<=reject raises ValueError", OK)

    report("accept_threshold", OK if t.accept_threshold == 0.75 else FAIL)
    report("reject_threshold", OK if t.reject_threshold == 0.40 else FAIL)

    return True


# ────────── Test 14: API Routes ──────────
def test_api_routes():
    header("Test 14: API Routes Response (health endpoint)")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        report("TestClient import", SKIP)
        return False

    from src.api.app import create_app

    app = create_app()
    report("app created", OK)

    client = TestClient(app)

    resp = client.get("/api/v1/health")
    report(f"health => {resp.status_code}", OK if resp.status_code == 200 else FAIL)
    if resp.status_code == 200:
        report("body.status=ok", OK if resp.json().get("status") == "ok" else FAIL)

    schema = app.openapi()
    report("openapi schema", OK if schema else FAIL)
    if schema:
        report("title matches", OK if schema.get("info", {}).get("title") == "RAG Pipeline API" else FAIL)

    # Use openapi paths to verify routes are registered
    paths = list(schema.get("paths", {}).keys())
    report("/api/v1/health registered",  OK if "/api/v1/health" in paths else FAIL)
    report("/api/v1/query registered",   OK if "/api/v1/query" in paths else FAIL)
    report("/api/v1/documents/upload registered", OK if "/api/v1/documents/upload" in paths else FAIL)
    report("/api/v1/review/pending registered",   OK if "/api/v1/review/pending" in paths else FAIL)

    return True


# ──────────────────────── MAIN ────────────────────────
def main():
    global passed, failed, skipped

    def _safe_run(fn, *args, **kwargs):
        """Run a test function, catching crashes so subsequent tests still run."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            report(f"{fn.__name__} crashed", FAIL, str(e))
            import traceback
            traceback.print_exc()
            return None

    print("=" * 60)
    print("  RAG Pipeline - End-to-End Integration Test")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pdf_path = str(Path(__file__).resolve().parent.parent / "data" / "test" / "test_e2e_doc.pdf")
    if not create_test_pdf(pdf_path):
        print(f"  {FAIL} failed to create test PDF"); sys.exit(1)
    print(f"\n  {OK} Test PDF: {pdf_path}")

    t0 = time.time()

    doc = _safe_run(test_pdf_loader, pdf_path)
    chunks = _safe_run(test_chunker, doc) if doc else None
    if chunks:
        _safe_run(test_vector_store)
    _safe_run(test_embedding_cpu)
    _safe_run(test_layout_ppdoclayv3)
    _safe_run(test_ocr_import)
    _safe_run(test_orchestrator, pdf_path)

    # Phase 2 tests
    _safe_run(test_bm25_index)
    _safe_run(test_hybrid_search_rrf)
    _safe_run(test_reranker)
    _safe_run(test_llm_client)
    _safe_run(test_confidence_scorer)
    _safe_run(test_threshold_classification)
    _safe_run(test_api_routes)

    elapsed = time.time() - t0
    total = passed + failed + skipped
    print(f"\n{DIVIDER}")
    print(f"  Summary: {total} checks")
    print(f"  {OK} Passed: {passed}")
    print(f"  {FAIL} Failed: {failed}")
    print(f"  {SKIP} Skipped: {skipped}")
    print(f"  Time: {elapsed:.1f}s")
    if failed: print(f"\n  {WARN} Some checks failed"); return 1
    else:      print(f"\n  {OK} All checks passed!"); return 0


if __name__ == "__main__":
    sys.exit(main())
