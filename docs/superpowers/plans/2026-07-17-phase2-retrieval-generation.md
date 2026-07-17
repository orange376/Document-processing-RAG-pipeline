# Phase 2: 检索 + 生成 + 置信度 + API 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 RAG 流水线的检索层、生成层、置信度评估和 API 接口，实现完整的问答链路。

**架构:** 纯手撸，无 LangChain/LlamaIndex。BM25 + 向量混合检索 → BGE-Reranker → Qwen3.7-Plus API 生成（强制溯源）→ 5 维置信度评分 → FastAPI 接口。

**Tech Stack:** rank-bm25, Qdrant (已集成), FlagEmbedding (reranker), httpx (Qwen API), FastAPI, pydantic-settings

---

## 全局约束

- 纯手撸，任何模块不引入 LangChain/LlamaIndex
- bge-reranker-v2-m3 通过 FlagEmbedding `AutoModelForSequenceClassification` 加载，CPU 推理
- Qwen3.7-Plus API 通过 httpx 调用阿里云百炼兼容接口
- 置信度阈值: accept ≥ 0.75, reject < 0.40, 中间转人工
- 溯源引用格式: `[来源: 文件名 | 第N页 | §章节]`
- 每个新文件必须有对应的 pytest 单元测试
- API 路由遵循 RESTful 风格，前缀 `/api/v1/`

---

## 文件结构

### 创建的文件

```
src/domain/                    # 新增领域类型（在现有文件追加）
  __init__.py                  # 追加 SearchResult, ConfidenceResult
  chunk.py                     # 追加 SearchResult, CitationSource, QueryResult

src/index/
  bm25_index.py                # BM25 关键词索引
  hybrid_search.py             # 向量 + BM25 混合检索 (RRF)

src/retrieval/
  __init__.py
  retriever.py                 # 主检索编排
  reranker.py                  # BGE-Reranker 封装
  query_rewriter.py            # Query 改写
  fusion.py                    # RRF 排序融合

src/generation/
  __init__.py
  llm_client.py                # Qwen3.7-Plus API 客户端
  context_builder.py           # 上下文组装 + 元数据注入
  prompt_manager.py            # Prompt 模板管理
  citation.py                  # 溯源引用格式化

src/confidence/
  __init__.py
  scorer.py                    # 5 维评分引擎
  threshold.py                 # 三级阈值判定
  fallback.py                  # 降级路由

src/api/
  __init__.py
  app.py                       # FastAPI 应用工厂
  routers/
    __init__.py
    upload.py                  # POST /api/v1/documents/upload
    query.py                   # POST /api/v1/query
    review.py                  # GET/POST /api/v1/review/...
    admin.py                   # 管理接口
  schemas/
    __init__.py
    requests.py                # 请求体 Pydantic 模型
    responses.py               # 响应体 Pydantic 模型
```

### 修改的文件

```
src/domain/__init__.py         # 导出新类型
src/domain/chunk.py            # 追加 SearchResult, CitationSource, QueryResult
```

---

### Task 1: 领域模型扩展

**Files:**
- Modify: `src/domain/chunk.py` — 追加 SearchResult, CitationSource, QueryResult
- Modify: `src/domain/__init__.py` — 导出新类型
- Test: `tests/unit/test_domain.py` — 追加新测试

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class SearchResult:
      chunk: Chunk
      score: float
      retrieval_method: str  # "vector" | "bm25" | "hybrid"
  
  @dataclass
  class CitationSource:
      source_file: str
      page_num: int
      section: str
      chunk_type: str
      content: str
  
  @dataclass  
  class QueryResult:
      answer: str
      citations: list[CitationSource]
      confidence: float
      confidence_details: dict[str, float]
      needs_review: bool = False
  ```

**Steps:**

- [ ] **Step 1: 追加领域类到 chunk.py**

```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    retrieval_method: str = "hybrid"

@dataclass
class CitationSource:
    source_file: str
    page_num: int
    section: str
    chunk_type: str
    text: str

@dataclass
class QueryResult:
    answer: str
    citations: list[CitationSource]
    confidence: float = 0.0
    confidence_details: dict[str, float] = field(default_factory=dict)
    needs_review: bool = False
```

- [ ] **Step 2: 更新 __init__.py**
- [ ] **Step 3: 写测试** — 验证 dataclass 创建和默认值
- [ ] **Step 4: 运行测试验证通过**

---

### Task 2: BM25 关键词索引

**Files:**
- Create: `src/index/bm25_index.py`
- Test: `tests/unit/test_bm25_index.py`

**Interfaces:**
- Produces:
  ```python
  class BM25Index:
      def __init__(self): ...
      def add_documents(self, chunks: list[Chunk]): ...    # 构建索引
      def search(self, query: str, top_k: int = 30) -> list[SearchResult]: ...  # 检索
      def save(self, path: str): ...   # 持久化
      def load(self, path: str): ...   # 加载
      def clear(self): ...             # 清空
  ```

- [ ] **Step 1: 写测试**

```python
def test_bm25_add_and_search():
    index = BM25Index()
    chunks = [
        Chunk(content="Retrieval augmented generation", metadata=ChunkMetadata(...)),
        Chunk(content="Machine learning transformers", metadata=ChunkMetadata(...)),
    ]
    index.add_documents(chunks)
    results = index.search("retrieval", top_k=5)
    assert len(results) >= 1
    assert results[0].retrieval_method == "bm25"

def test_bm25_empty_index():
    index = BM25Index()
    assert index.search("test", top_k=5) == []

def test_bm25_clear():
    index = BM25Index()
    index.add_documents([Chunk(content="test", metadata=ChunkMetadata(...))])
    index.clear()
    assert index.search("test") == []
```

- [ ] **Step 2: 实现 BM25Index**

用 `rank_bm25.BM25Okapi` 包装。核心逻辑：
- `add_documents()`: tokenize 所有 chunk content，构建 BM25Okapi
- `search()`: tokenize query，调用 `bm25.get_scores()`，返回 top_k SearchResult
- 使用 `jieba` 分词（如果可用）或简单空格分词支持中文词组分词
- 维护 `_chunks: list[Chunk]` 列表与 BM25 内部文档顺序对应

- [ ] **Step 3: 运行测试验证通过**

---

### Task 3: RRF 混合检索融合

**Files:**
- Create: `src/index/hybrid_search.py`
- Test: `tests/unit/test_hybrid_search.py`

**Interfaces:**
- Produces:
  ```python
  class HybridSearch:
      def __init__(self, vector_store: VectorStore, bm25_index: BM25Index): ...
      def search(self, query: str, query_embedding: list[float], top_k: int = 30) -> list[SearchResult]: ...
  ```

**RRF 公式:** `score(d) = Σ(1 / (k + rank(d)))`，其中 k=60

- [ ] **Step 1: 写测试**

```python
def test_rrf_fusion():
    hybrid = HybridSearch(...)
    # mock vector results + bm25 results
    results = hybrid._rrf_fusion(vector_results, bm25_results, top_k=10)
    assert len(results) <= 10
    assert results[0].score >= results[-1].score

def test_hybrid_empty():
    hybrid = HybridSearch()
    results = hybrid.search("test", [0.1]*1024, top_k=5)
    assert results == []
```

- [ ] **Step 2: 实现 HybridSearch**

`search()` 方法：
1. 调用 `vector_store.search(query_embedding, top_k=30)` → 得到向量结果
2. 调用 `bm25_index.search(query, top_k=30)` → 得到 BM25 结果
3. 调用 `_rrf_fusion()` 按 RRF 公式合并
4. 返回 top_k 个 SearchResult

- [ ] **Step 3: 运行测试验证通过**

---

### Task 4: Query 改写

**Files:**
- Create: `src/retrieval/query_rewriter.py`
- Test: `tests/unit/test_query_rewriter.py`

通过 LLM API 将用户原始问题改写成更利于检索的形式。如果 LLM API 不可用，返回原始 query。

- [ ] **Step 1: 写测试**

```python
def test_rewrite_returns_original_when_no_llm():
    rewriter = QueryRewriter()
    # 无 API Key 或 LLM 失败时返回原 query
    result = rewriter.rewrite("test query")
    assert result == "test query"
```

- [ ] **Step 2: 实现 QueryRewriter**

```python
class QueryRewriter:
    def __init__(self, llm_client=None):
        self._llm = llm_client
    
    def rewrite(self, query: str) -> str:
        """改写 query。LLM 不可用时返回原文。"""
        if not self._llm:
            return query
        try:
            prompt = f"将以下用户问题改写成更适合检索的形式，保持核心意图：\n{query}"
            result = self._llm.chat(prompt, system="你是一个检索优化专家。请简洁改写。")
            return result.strip() or query
        except Exception:
            return query
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 5: BGE-Reranker 封装

**Files:**
- Create: `src/retrieval/reranker.py`
- Test: `tests/unit/test_reranker.py`

**注意:** bge-reranker-v2-m3 约 2.2GB，使用 FlagEmbedding 的 `AutoModelForSequenceClassification` 加载。

- [ ] **Step 1: 写测试**

```python
def test_reranker_initialization():
    reranker = Reranker()
    assert reranker is not None

def test_reranker_rerank_empty():
    reranker = Reranker()
    assert reranker.rerank("query", []) == []

@pytest.mark.skip(reason="需要下载 reranker 模型 (~2.2GB)")
def test_reranker_actual():
    reranker = Reranker()
    results = reranker.rerank("RAG", [SearchResult(...), SearchResult(...)])
    assert len(results) == 2
    assert results[0].score >= results[1].score
```

- [ ] **Step 2: 实现 Reranker**

```python
from FlagEmbedding import AutoModelForSequenceClassification

class Reranker:
    def __init__(self):
        self._model = None
    
    def _lazy_load(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-v2-m3",
                cache_folder=str(get_settings().resolved_model_dir),
            )
    
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
        if not results:
            return []
        self._lazy_load()
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.compute_score(pairs)
        for r, s in zip(results, scores):
            r.score = float(s)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def unload(self):
        self._model = None
        import gc; gc.collect()
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 6: 主检索编排器

**Files:**
- Create: `src/retrieval/retriever.py`
- Create: `src/retrieval/fusion.py`
- Test: `tests/unit/test_retriever.py`

将混合检索、重排串联为统一接口。

- [ ] **Step 1: 写测试**

```python
def test_retriever_initialization():
    retriever = Retriever()
    assert retriever is not None

def test_retriever_empty():
    retriever = Retriever()
    result = retriever.retrieve("test", embedding=[0.1]*1024)
    assert len(result) == 0
```

- [ ] **Step 2: 实现 Retriever**

```python
class Retriever:
    def __init__(self, vector_store=None, bm25_index=None, reranker=None, hybrid_search=None):
        self._vector_store = vector_store or VectorStore()
        self._bm25 = bm25_index or BM25Index()
        self._hybrid = hybrid_search or HybridSearch(self._vector_store, self._bm25)
        self._reranker = reranker or Reranker()
    
    def retrieve(self, query: str, embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        # 1. 混合检索获取 30 条候选
        candidates = self._hybrid.search(query, embedding, top_k=30)
        if not candidates:
            return []
        # 2. Reranker 精排到 top_k
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        return reranked
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 7: LLM 客户端 (Qwen3.7-Plus)

**Files:**
- Create: `src/generation/llm_client.py`
- Test: `tests/unit/test_llm_client.py`

**注意:** 这个模块是 API 调用，测试时用 httpx mock。

- [ ] **Step 1: 写测试**

```python
def test_llm_client_init():
    client = LLMClient(api_key="test-key")
    assert client is not None

def test_llm_client_chat_empty_key():
    client = LLMClient(api_key="")
    result = client.chat("hello")
    assert result == ""  # 无 API Key 返回空

def test_llm_client_chat_with_mock(httpx_mock):
    httpx_mock.add_response(json={"choices": [{"message": {"content": "test answer"}}]})
    client = LLMClient(api_key="test-key")
    result = client.chat("hello")
    assert "test answer" in result
```

- [ ] **Step 2: 实现 LLMClient**

```python
import httpx
from src.config import get_settings

class LLMClient:
    def __init__(self, api_key: str | None = None):
        s = get_settings()
        self._api_key = api_key or s.qwen_api_key
        self._base = s.qwen_api_base
        self._model = s.qwen_model
    
    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        if not self._api_key:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        resp = httpx.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 8: Prompt 模板管理

**Files:**
- Create: `src/generation/prompt_manager.py`
- Test: `tests/unit/test_prompt_manager.py`

- [ ] **Step 1: 写测试**

```python
def test_render_qa_prompt():
    pm = PromptManager()
    result = pm.render("qa", query="test question", context="test context")
    assert "test question" in result
    assert "test context" in result
```

- [ ] **Step 2: 实现 PromptManager**

```python
class PromptManager:
    TEMPLATES = {
        "qa": """你是一个专业的知识库问答助手。请基于提供的上下文回答问题。

上下文：
{context}

问题：{query}

要求：
1. 只基于上下文提供的信息回答
2. 如果上下文不足以回答问题，明确说"根据提供的资料无法回答"
3. 保持客观，不要添加自己的知识
4. 在回答中引用具体来源，使用 [来源: 文件名 | 第N页] 格式

回答：""",
        "rewrite": """将以下用户问题改写成更适合检索的形式：
原问题：{query}
改写：""",
    }
    
    def render(self, template_name: str, **kwargs) -> str:
        template = self.TEMPLATES.get(template_name)
        if template is None:
            raise ValueError(f"Unknown template: {template_name}")
        return template.format(**kwargs)
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 9: 上下文组装

**Files:**
- Create: `src/generation/context_builder.py`
- Test: `tests/unit/test_context_builder.py`

- [ ] **Step 1: 写测试**

```python
def test_build_empty():
    cb = ContextBuilder()
    ctx, sources = cb.build([])
    assert ctx == ""
    assert sources == []

def test_build_with_results():
    cb = ContextBuilder()
    results = [SearchResult(chunk=Chunk(content="test", metadata=ChunkMetadata(...)), score=0.9)]
    ctx, sources = cb.build(results)
    assert "test" in ctx
    assert len(sources) == 1
```

- [ ] **Step 2: 实现 ContextBuilder**

```python
class ContextBuilder:
    def build(self, results: list[SearchResult]) -> tuple[str, list[CitationSource]]:
        """组装上下文并提取引用源"""
        blocks = []
        sources = []
        for i, r in enumerate(results):
            c = r.chunk
            block = c.to_context_block()
            blocks.append(f"[{i+1}] {block}")
            meta = c.metadata
            if meta:
                sources.append(CitationSource(
                    source_file=meta.source_file,
                    page_num=meta.page_num,
                    section=meta.section,
                    chunk_type=meta.chunk_type,
                    text=c.content,
                ))
        return "\n\n".join(blocks), sources
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 10: 溯源引用格式化

**Files:**
- Create: `src/generation/citation.py`
- Test: `tests/unit/test_citation.py`

- [ ] **Step 1: 写测试**

```python
def test_format_citation():
    source = CitationSource(source_file="doc.pdf", page_num=3, section="Results", chunk_type="text", text="data")
    result = format_citation(source)
    assert "doc.pdf" in result
    assert "第3页" in result
```

- [ ] **Step 2: 实现**

```python
def format_citation(source: CitationSource) -> str:
    parts = [f"📄 {source.source_file}"]
    parts.append(f"📍 第{source.page_num}页")
    if source.section:
        parts.append(f"📂 §{source.section}")
    return " | ".join(parts)
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 11: 5 维置信度评分

**Files:**
- Create: `src/confidence/scorer.py`
- Test: `tests/unit/test_scorer.py`

**评分维度:**
- `layout_quality` (0.25): 版面分析元素置信度均值
- `ocr_confidence` (0.20): OCR 识别置信度均值
- `table_integrity` (0.15): 表格行列对齐完整度
- `chunk_coherence` (0.20): 切片语义连贯性
- `reranker_score` (0.20): 检索重排最高分

- [ ] **Step 1: 写测试**

```python
def test_score_defaults():
    scorer = ConfidenceScorer()
    result = scorer.score(
        layout_elements=[LayoutElement(...)],
        ocr_results=[],
        tables=[],
        chunks=[],
        reranker_scores=[0.85, 0.72],
    )
    assert 0 <= result["overall"] <= 1.0
    assert set(result["details"].keys()) == {"layout_quality", "ocr_confidence", "table_integrity", "chunk_coherence", "reranker_score"}
```

- [ ] **Step 2: 实现**

```python
class ConfidenceScorer:
    WEIGHTS = {"layout_quality": 0.25, "ocr_confidence": 0.20, "table_integrity": 0.15, "chunk_coherence": 0.20, "reranker_score": 0.20}
    
    def score(self, layout_elements, ocr_results, tables, chunks, reranker_scores) -> dict:
        details = {
            "layout_quality": self._score_layout(layout_elements),
            "ocr_confidence": self._score_ocr(ocr_results),
            "table_integrity": self._score_tables(tables),
            "chunk_coherence": self._score_chunks(chunks),
            "reranker_score": self._score_reranker(reranker_scores),
        }
        overall = sum(details[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return {"overall": round(overall, 4), "details": details}
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 12: 阈值策略 + 降级路由

**Files:**
- Create: `src/confidence/threshold.py`
- Create: `src/confidence/fallback.py`
- Test: `tests/unit/test_threshold.py`, `tests/unit/test_fallback.py`

- [ ] **Step 1: 写测试**

```python
def test_classify_accept():
    threshold = ThresholdStrategy()
    assert threshold.classify(0.85) == "accept"

def test_classify_review():
    threshold = ThresholdStrategy()
    assert threshold.classify(0.60) == "review"

def test_classify_reject():
    threshold = ThresholdStrategy()
    assert threshold.classify(0.30) == "reject"
```

- [ ] **Step 2: 实现**

```python
class ThresholdStrategy:
    def __init__(self, accept=0.75, reject=0.40):
        self._accept = accept
        self._reject = reject
    
    def classify(self, score: float) -> str:
        if score >= self._accept: return "accept"
        if score >= self._reject: return "review"
        return "reject"
```

- [ ] **Step 3: 运行测试验证通过**

---

### Task 13: API Schemas

**Files:**
- Create: `src/api/schemas/requests.py`
- Create: `src/api/schemas/responses.py`
- Test: `tests/unit/test_api_schemas.py`

- [ ] **Step 1: 定义请求/响应模型**

```python
# requests.py
class UploadRequest(BaseModel): ...
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    document_ids: list[str] | None = None

# responses.py
class UploadResponse(BaseModel): ...
class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationSource]
    confidence: float
    needs_review: bool
```

- [ ] **Step 2: 写测试**
- [ ] **Step 3: 运行测试验证通过**

---

### Task 14: API 路由

**Files:**
- Create: `src/api/routers/upload.py`
- Create: `src/api/routers/query.py`
- Create: `src/api/routers/review.py`
- Create: `src/api/routers/admin.py`
- Test: `tests/integration/test_api.py`

- [ ] **Step 1: 实现各路由**
  - `POST /api/v1/documents/upload` — 上传并异步处理
  - `GET /api/v1/documents/{task_id}/status` — 查询处理状态
  - `POST /api/v1/query` — RAG 问答
  - `GET /api/v1/review/pending` — 待审核列表
  - `POST /api/v1/review/{task_id}/approve` — 审核通过

- [ ] **Step 2: 实现 app.py 应用工厂**

```python
from fastapi import FastAPI
from .routers import upload, query, review, admin

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Pipeline API")
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(review.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    return app
```

- [ ] **Step 3: 写测试（用 TestClient）**
- [ ] **Step 4: 运行测试验证通过**

---

### Task 15: 集成测试 + 端到端验证

**Files:**
- Modify: `scripts/test_e2e_pipeline.py` — 追加 Phase 2 测试

**测试项:**
1. BM25 构建 + 检索
2. 混合检索 RRF 融合
3. Reranker 加载（跳过模型下载）
4. LLM 客户端构造（mock API）
5. 置信度评分全维度
6. 阈值分类
7. API 路由响应

- [ ] **Step 1: 追加 Phase 2 测试到集成测试脚本**
- [ ] **Step 2: 运行全量测试验证**
