# 工业级 RAG 文档处理流水线 — 设计方案

> 日期：2026-07-17
> 状态：已批准
> 硬件环境：i5-13代 + RTX 4060 (8GB) + 16GB RAM
> API: 阿里云百炼 (Qwen3.7-Plus + Qwen2.5-VL-3B)

## 一、项目概述

构建端到端的 RAG 文档处理流水线，支持 PDF 和 Word 文档的智能解析、结构感知切片、混合检索增强生成、置信度评估闭环。

核心链路：
```
用户上传 → 结构解析/多模态处理 → 版面树 → 结构感知切片
→ 向量+关键词混合检索 → BGE-Reranker → LLM生成(强制溯源)
→ 置信度评分 → 入库/拦截
```

## 二、硬件方案（方案 B：均衡流水线）

### GPU 分阶段加载策略

| 阶段 | 模型 | VRAM | 操作 |
|------|------|------|------|
| 1 | PP-DocLayoutV3 (版面分析) | ~3.5GB | 用完卸载 |
| 2 | PaddleOCR-VL-1.5 (OCR+表格) | ~4.5GB | 用完卸载 |
| 3 | bge-large-zh-v1.5 (Embedding) | ~1.3GB | 用完卸载 |
| 4 | bge-reranker-v2-m3 (重排) | ~1GB | 用完卸载 |

同一时间只加载一个模型，峰值 VRAM < 5GB，8GB 绰绰有余。

### API 依赖

| 用途 | 服务 | 计费 |
|------|------|------|
| LLM 生成 | Qwen3.7-Plus API | 免费额度 100万 tokens |
| 多模态兜底（公式/扫描件） | Qwen2.5-VL-3B API | 完全免费 |
| 复杂文档 fallback | Qwen3.7-Plus (多模态) | 同上免费额度 |

## 三、架构路线

**纯手撸 + 精选库** — 不依赖 LangChain/LlamaIndex 等 RAG 框架。
核心价值在解析层和置信度评估，框架无法覆盖，手撸可获得完全控制权。

## 四、项目目录结构

```
rag-pipeline/
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── src/
│   ├── config/                    # 全局配置 (pydantic-settings)
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── defaults.py
│   │
│   ├── domain/                    # 核心领域模型（纯 dataclass）
│   │   ├── __init__.py
│   │   ├── document.py            # Document, Page, Block
│   │   ├── layout.py              # LayoutElement, BBox
│   │   ├── table.py               # Table, Cell, Row
│   │   ├── chunk.py               # Chunk, ChunkMetadata
│   │   └── enums.py               # BlockType, ProcessingStatus
│   │
│   ├── parser/                    # 文档解析层
│   │   ├── __init__.py
│   │   ├── loader/                # 文件加载器
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # DocumentLoader 抽象基类
│   │   │   ├── pdf_loader.py      # PDF 加载 (PyMuPDF)
│   │   │   └── word_loader.py     # Word 加载 (python-docx)
│   │   ├── layout/                # 版面分析
│   │   │   ├── __init__.py
│   │   │   ├── detector.py        # PP-DocLayoutV3 封装
│   │   │   └── models.py          # 版面模型定义
│   │   ├── ocr/                   # OCR 识别
│   │   │   ├── __init__.py
│   │   │   ├── engine.py          # PaddleOCR 封装
│   │   │   └── postprocess.py     # OCR 后处理
│   │   ├── table/                 # 表格还原
│   │   │   ├── __init__.py
│   │   │   ├── detector.py        # 表格检测
│   │   │   ├── structure.py       # 表格结构还原
│   │   │   └── merger.py          # 跨页拼接
│   │   ├── formula/               # 公式识别
│   │   │   ├── __init__.py
│   │   │   ├── omml_parser.py     # Word OMML 解析
│   │   │   └── multimodal.py      # 多模态 API 兜底
│   │   ├── layout_tree.py         # 版面树构建
│   │   └── chunker.py             # 结构感知切片
│   │
│   ├── pipeline/                  # 流水线编排
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # 主编排器
│   │   ├── stages.py              # 流水线阶段定义
│   │   └── task_queue.py          # 异步任务队列 (Celery)
│   │
│   ├── index/                     # 索引层
│   │   ├── __init__.py
│   │   ├── embedding.py           # bge-large-zh 封装
│   │   ├── vector_store.py        # Qdrant 封装
│   │   ├── bm25_index.py          # BM25 索引
│   │   └── hybrid_search.py       # 混合检索融合
│   │
│   ├── retrieval/                 # 检索层（在线）
│   │   ├── __init__.py
│   │   ├── retriever.py           # 主检索器
│   │   ├── reranker.py            # BGE-Reranker 封装
│   │   ├── query_rewriter.py      # Query 改写
│   │   └── fusion.py              # RRF 排序融合
│   │
│   ├── generation/                # 生成层
│   │   ├── __init__.py
│   │   ├── context_builder.py     # 上下文拼装 + 元数据注入
│   │   ├── llm_client.py          # Qwen API 调用
│   │   ├── prompt_manager.py      # Prompt 模板管理
│   │   └── citation.py            # 溯源引用格式化
│   │
│   ├── confidence/                # 置信度评估
│   │   ├── __init__.py
│   │   ├── scorer.py              # 5 维评分引擎
│   │   ├── threshold.py           # 阈值策略
│   │   └── fallback.py            # 二次解析 / 人工审核路由
│   │
│   └── api/                       # FastAPI 接口层
│       ├── __init__.py
│       ├── app.py                 # FastAPI 应用
│       ├── routers/
│       │   ├── upload.py          # POST /api/v1/documents/upload
│       │   ├── query.py           # POST /api/v1/query
│       │   ├── review.py          # 人工审核接口
│       │   └── admin.py           # 管理接口
│       └── schemas/               # Pydantic 请求/响应
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                  # 测试文档样本
│
├── data/
│   ├── uploads/                   # 上传文件
│   ├── vector_db/                 # Qdrant 持久化
│   └── models/                    # 本地模型权重
│
├── scripts/
│   ├── download_models.py         # 模型下载脚本
│   └── benchmark.py               # 性能基准测试
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    └── superpowers/specs/         # 设计文档
```

## 五、技术选型明细

### Python 核心依赖

| 用途 | 库/框架 | 版本 |
|------|---------|------|
| API 框架 | FastAPI | ≥0.115 |
| 异步任务 | Celery + Redis | 最新 |
| 配置 | pydantic-settings | 最新 |
| PDF 加载 | PyMuPDF (fitz) | 最新 |
| Word 加载 | python-docx | ≥1.1 |

### 文档解析

| 模块 | 方案 | VRAM |
|------|------|------|
| 版面分析 | PP-DocLayoutV3 (via PaddleOCR ≥3.4.0) | <4GB |
| OCR + 端到端 | PaddleOCR-VL-1.5 (via PaddleOCR) | ~5GB |
| 表格检测 | PaddleOCR-VL-1.5 内置 + 自研行列还原 | 同上 |
| 跨页拼接 | 自研：哈希匹配 + 行结构融合 | CPU |
| Word 公式 | python-docx XML 提取 + 截图中转 Qwen-VL | CPU/API |
| 扫描件兜底 | Qwen2.5-VL-3B API (免费) | API |

### 索引与检索

| 模块 | 方案 | VRAM |
|------|------|------|
| Embedding | BAAI/bge-large-zh-v1.5 (FlagEmbedding) | ~1.3GB |
| 向量库 | Qdrant 本地持久化模式 | ~200MB |
| 关键词 | rank_bm25 自封装 | CPU |
| 重排 | BAAI/bge-reranker-v2-m3 (FlagEmbedding) | ~1GB |
| 融合 | RRF / DBSF 自研 | CPU |

### LLM

| 用途 | 服务 | 成本 |
|------|------|------|
| 生成 | Qwen3.7-Plus (阿里云百炼) | 免费 100万 tokens |
| 多模态兜底 | Qwen2.5-VL-3B API | 免费 |

## 六、核心领域模型接口

### Document / Page / Block

```python
@dataclass
class BBox:
    x0: float; y0: float; x1: float; y1: float
    page_num: int

@dataclass
class LayoutElement:
    bbox: BBox; category: str; confidence: float; reading_order: int

@dataclass
class Table:
    bbox: BBox; rows: list[list[str]]; header_rows: int; is_page_break: bool

@dataclass
class Chunk:
    id: str; content: str; metadata: ChunkMetadata; embedding: list[float] | None

@dataclass
class ChunkMetadata:
    source_file: str; page_num: int; section: str
    chunk_type: str; bbox: BBox | None; layout_tree_path: list[str]
```

### 解析器接口

```python
class DocumentLoader(ABC):
    @abstractmethod def load(self, path: str) -> list[Page]: ...

class LayoutAnalyzer(ABC):
    @abstractmethod def analyze(self, page: Page) -> list[LayoutElement]: ...

class OCRProcessor(ABC):
    @abstractmethod def recognize(self, page: Page, layout: list[LayoutElement]) -> Page: ...

class TableProcessor(ABC):
    @abstractmethod def recover(self, page: Page) -> list[Table]: ...
    @abstractmethod def merge_cross_page(self, tables: list[list[Table]]) -> list[Table]: ...

class Chunker(ABC):
    @abstractmethod def chunk(self, page: Page, metadata: DocMetadata) -> list[Chunk]: ...
```

## 七、数据流与状态机

### 文档处理状态

```
UPLOADED → QUEUED → PROCESSING → INDEXING → SCORING → ACCEPTED → INDEXED
                                                         ↘ REVIEW → FALLBACK → ACCEPTED → INDEXED
```

### RAG 问答流

```
用户 Query → Query 改写 → 向量检索(top30) + BM25(top30)
→ RRF 融合(→30) → BGE-Reranker(→10)
→ 元数据注入 → 上下文组装 → LLM 生成(强制溯源)
→ 结构化输出 {answer, citations}
```

### API 路由

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | /api/v1/documents/upload | 上传文档 |
| GET | /api/v1/documents/{task_id}/status | 查询状态 |
| POST | /api/v1/query | RAG 问答 |
| GET | /api/v1/review/pending | 待审核列表 |
| POST | /api/v1/review/{task_id}/approve | 审核通过 |

## 八、置信度评估

### 5 维评分

| 维度 | 权重 | 说明 |
|------|------|------|
| layout_quality | 0.25 | 版面分析元素置信度均值 |
| ocr_confidence | 0.20 | OCR 识别置信度均值 |
| table_integrity | 0.15 | 表格行列对齐完整度 |
| chunk_coherence | 0.20 | 切片语义连贯性 |
| reranker_score | 0.20 | 检索重排得分 |

### 阈值策略

- ≥ 0.75: accept（直接入库）
- 0.40 ~ 0.75: review（转人工审核）
- < 0.40: reject（拦截，触发二次解析或拒绝入库）

## 九、降级策略

```
本地 GPU → 本地 CPU → Qwen API → 人工审核
```

| 级别 | 场景 | 动作 |
|------|------|------|
| 🟢 可恢复 | OCR 单页失败 | 重试 2 次，跳过该页 |
| 🟡 局部降级 | GPU OOM | 切到 CPU / 切到 API |
| 🔴 全局失败 | 文件损坏 | 返回明确错误码 |
| ⚫ 置信度拦截 | 评分低 | 转人工审核池 |

## 十、第一阶段执行计划（3 周）

### 第 1 周：骨架搭建
- [ ] 初始化项目结构 (pyproject.toml, 目录树)
- [ ] 实现 domain/ 全部数据类
- [ ] 实现 config/ 配置加载 (.env)
- [ ] 编写 scripts/download_models.py 模型下载脚本
- [ ] 下载 PP-DocLayoutV3 / PaddleOCR-VL-1.5 / bge 模型
- [ ] 验证 PP-DocLayoutV3 单图推理通过

### 第 2 周：PDF 解析管线
- [ ] 实现 loader/pdf_loader.py (PyMuPDF)
- [ ] 实现 layout/detector.py (PP-DocLayoutV3 封装)
- [ ] 实现 layout_tree.py (版面树构建)
- [ ] 实现 ocr/engine.py (PaddleOCR-VL-1.5 封装)
- [ ] 单页 PDF 全流程测试通过

### 第 3 周：切片 + 入库
- [ ] 实现 table/ 基础表格还原
- [ ] 实现 chunker.py (结构感知切片)
- [ ] 实现 embedding.py + vector_store.py (Qdrant)
- [ ] 实现 pipeline/orchestrator.py (基础编排)
- [ ] 完整文档处理链路跑通

## 十一、质量门禁

| 指标 | 目标 |
|------|------|
| 单元测试覆盖率 | ≥ 85% |
| 版面分析 mAP | ≥ 0.85 |
| 表格还原准确率 | ≥ 80% |
| 检索 Recall@10 | ≥ 0.85 |
| Reranker NDCG@10 | ≥ 0.90 |
