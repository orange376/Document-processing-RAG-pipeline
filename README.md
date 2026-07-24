# RAG Pipeline — 工业级文档解析与检索增强生成系统

> 无 LangChain / LlamaIndex 依赖 · 全模块单元测试覆盖（Claude code辅助编码）

一个面向复杂文档的 RAG（检索增强生成）系统，支持 **PDF/Word** 文档的解析、版面分析、结构感知切片、混合检索（稠密 + 稀疏 + RRF 融合）、BGE 重排、LLM 生成与置信度评估闭环。所有组件均为自主实现，无大框架黑盒依赖。

---

## 功能亮点

### 📄 文档解析

| 能力 | 状态 |
|------|------|
| PDF 加载（PyMuPDF） | ✅ 文本块+字体+位置+图片流 |
| Word 文档加载（python-docx） | ✅ 就绪 |
| PP-DocLayoutV3 版面分析（25 类） | ✅ 深度学习模型 + 启发式降级 |
| 双栏自动检测 | ✅ 启发式阈值判断 |
| EasyOCR 缺文本页面补全 | ✅ 纯 Torch，支持 GPU |
| 版面树构建 | ✅ 标题→子节层次组织 |
| **纯解析 API（不索引）** | ✅ 供其他 Agent 消费 |

### 🔍 检索增强

| 能力 | 状态 |
|------|------|
| BGE-large-zh-v1.5 稠密向量化 | ✅ 1024 维，FP16 条件启用 |
| Qdrant 本地向量存储 | ✅ COSINE 距离，持久化 |
| BM25 关键词索引（rank_bm25） | ✅ jieba 分词，IDF 下限修复 |
| **RRF 混合搜索融合** | ✅ k=60，向量+关键词联合排序 |
| BGE-reranker-v2-m3 重排 | ✅ sigmoid 归一化 0-1 分 |
| Query 改写 | ✅ LLM 驱动的检索优化 |

### 🤖 LLM 生成

| 能力 | 状态 |
|------|------|
| **DeepSeek** 主语言模型 | ✅ deepseek-v4-flash |
| **Qwen VL** 多模态备用 | ✅ qwen2.5-vl-3b-instruct |
| 结构感知上下文拼装 | ✅ 来源+页码+章节+类型标注 |
| 强制溯源引用 | ✅ [来源: 文件 | 第N页 |
| 低置信度降级回答 | ✅ |

### 🎯 置信度闭环

| 能力 | 状态 |
|------|------|
| 5 维评分引擎（布局/OCR/表格/切片/重排） | ✅ 查询时动态加权 |
| 三级阈值策略（accept / review / reject） | ✅ 可配置阈值 |
| 降级路由 | ✅ 自动日志+空结果返回 |

### 🌐 API 服务

| 端点 | 用途 |
|------|------|
| `POST /api/v1/documents/upload` | 上传文档 → 全流程处理 |
| `GET /api/v1/documents/{task_id}/status` | 轮询处理状态 |
| `POST /api/v1/documents/parse` | **纯解析**（不上索引，给其他 Agent 用） |
| `POST /api/v1/query` | RAG 问答 |
| `GET /api/v1/review/pending` | 低置信度待审核列表 |
| `POST /api/v1/review/{task_id}/approve` | 审核通过/拒绝 |
| `GET /api/v1/health` | 健康检查 |

---

## 快速开始

### 1. 克隆并安装

```bash
git clone https://github.com/<your>/rag-pipeline.git
cd rag-pipeline

# 创建虚拟环境（Python >= 3.10）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖（不含 PaddlePaddle）
pip install -e ".[dev]"
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```ini
# DeepSeek（主语言模型）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-v4-flash

# Qwen（多模态备用）
QWEN_API_KEY=sk-xxx
```

### 3. 下载模型

```bash
python scripts/download_models.py
```

自动下载：
- **BAAI/bge-large-zh-v1.5**（Embedding 模型，~1.3 GB）
- **BAAI/bge-reranker-v2-m3**（重排模型，~2.2 GB）

> 若需 PP-DocLayoutV3 深度学习版面分析：
> ```bash
> pip install paddlepaddle-gpu==3.2.1 paddlex==3.7.0
> ```
> 缺省时自动降级为 PyMuPDF 启发式版面分析。

### 4. 启动服务

```bash
python main.py
# → http://localhost:8000
# → API 文档: http://localhost:8000/docs
```

---

## 使用示例

### 纯解析（供其他 Agent 消费）

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/api/v1/documents/parse",
    files={"file": ("report.pdf", open("report.pdf", "rb"), "application/pdf")},
)
data = resp.json()
print(f"共 {data['total_pages']} 页，{len(data['chunks'])} 个切片")
for chunk in data["chunks"]:
    print(f"  [{chunk['section']}] {chunk['content'][:80]}...")
```

### RAG 问答

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "文档中提到了什么技术方案？", "top_k": 10}'
```

响应包含 `answer`、`citations`（带来源页码）、`confidence` 评分。

---

## 项目结构

```
rag-pipeline/
├── main.py                  # 服务启动入口
├── .env                     # 配置（API Key、路径、阈值）
├── src/
│   ├── api/                 # FastAPI 应用 + 路由 + Schemas
│   │   ├── app.py           #   应用工厂
│   │   ├── schemas/         #   Pydantic 请求/响应模型
│   │   └── routers/         #   端点（upload, query, review, admin）
│   ├── pipeline/            # 主编排器
│   │   └── orchestrator.py  #   全流程 & 纯解析流程
│   ├── parser/              # 解析层
│   │   ├── loader/          #   PDF/Word 文档加载
│   │   ├── layout/          #   版面分析（PP-DocLayoutV3 + 启发式）
│   │   ├── ocr/             #   EasyOCR 引擎
│   │   ├── table/           #   表格检测与结构化
│   │   ├── layout_tree.py   #   版面树构建
│   │   └── chunker.py       #   结构感知切片
│   ├── index/               # 索引层
│   │   ├── embedding.py     #   BGE 向量化
│   │   ├── vector_store.py  #   Qdrant 向量存储
│   │   ├── bm25_index.py    #   BM25 关键词索引
│   │   └── hybrid_search.py #   RRF 混合检索融合
│   ├── retrieval/           # 检索层
│   │   ├── retriever.py     #   统一检索入口
│   │   ├── reranker.py      #   BGE 重排模型
│   │   └── query_rewriter.py#   LLM 查询改写
│   ├── generation/          # 生成层
│   │   ├── llm_client.py    #   DeepSeek / Qwen API 客户端
│   │   ├── context_builder.py#  上下文拼装 + 溯源引用
│   │   ├── prompt_manager.py#   提示词模板管理
│   │   └── citation.py      #   引用格式化
│   ├── confidence/          # 置信度评估
│   │   ├── scorer.py        #   5 维评分引擎
│   │   ├── threshold.py     #   三级阈值策略
│   │   └── fallback.py      #   降级路由逻辑
│   ├── domain/              # 领域模型
│   │   ├── document.py      #   Document / Page / Block
│   │   ├── chunk.py         #   Chunk / CitationSource / SearchResult
│   │   ├── layout.py        #   BBox / LayoutElement
│   │   ├── table.py         #   Table / Cell
│   │   └── enums.py         #   枚举定义
│   └── config/              # 应用配置（Pydantic Settings）
├── tests/                   #  215+ 测试，全部通过
│   ├── unit/                #   单元测试
│   └── integration/         #   集成测试（FastAPI TestClient）
├── data/                    # 数据目录（运行时生成）
│   ├── models/              #   下载的模型权重
│   ├── uploads/             #   上传文件临时存储
│   └── vector_db/           #   Qdrant 持久化数据
└── docs/                    # 设计文档
```

---

## 架构概览

```
用户/Curl/Agent
     │
     ▼
┌─────────────────┐
│   FastAPI 服务   │  ← API 路由
└────────┬────────┘
         │
         ├─── ▶ 全流程：上传 → 解析 → 切片 → 向量化 → 索引
         │
         │         PipelineOrchestrator
         │         ┌──────────────────────────────────────┐
         │         │ PDF/Word 加载  →  版面分析  →  OCR   │
         │         │ 版面树构建  →  结构感知切片            │
         │         │ Embedding  →  Qdrant 向量索引         │
         │         │ BM25 关键词索引                        │
         │         └──────────────────────────────────────┘
         │
         ├─── ▶ 查询：Embed → 改写 → 混合检索 → 重排 → LLM → 置信度
         │
         │         Retriever ─── HybridSearch ─┬─ VectorStore (Qdrant)
         │                                     └─ BM25Index
         │                │
         │                ▼
         │         Reranker (BGE-reranker-v2-m3)
         │                │
         │                ▼
         │         ContextBuilder ─── LLMClient ─── PromptManager
         │                │
         │                ▼
         │         ConfidenceScorer ─── ThresholdStrategy ─── Fallback
         │
         └─── ▶ 纯解析：不上索引，返回结构化结果（供其他 Agent）
```

### 流水线阶段

1. **加载** — PyMuPDF 提取文本块、字体、位置、图片流
2. **版面分析** — PP-DocLayoutV3（25 类，可用时）→ 启发式降级（双栏检测 + 字号分级）
3. **OCR** — EasyOCR 补全缺文本页面的文字
4. **版面树** — 标题→子节的层次结构
5. **切片** — 结构感知：标题边界、公式/表格独立切片，512 token 上限
6. **索引** — BGE 向量化 → Qdrant 写入 + BM25 关键词追加

### 检索流程

1. **Query 改写** — LLM 将原始问题改写为更利于检索的形式
2. **混合检索** — 向量检索（语义匹配）+ BM25 检索（关键词匹配）→ **RRF（k=60）融合**
3. **重排** — BGE-reranker-v2-m3 对候选结果精排（sigmoid 归一化到 0-1）
4. **上下组装** — 每个切片标注来源、页码、章节、类型
5. **LLM 生成** — DeepSeek API，强制引用格式的 QA 提示模板
6. **置信度** — 查询时：重排分 0.6 + 结果覆盖度 0.4 → 三级阈值

---

## 技术栈

| 模块 | 技术选型 |
|------|----------|
| 服务框架 | FastAPI 0.115+ / Uvicorn |
| 文档解析 | PyMuPDF（PDF）/ python-docx（Word） |
| 版面分析 | PP-DocLayoutV3 (PaddleX) → PyMuPDF 启发式降级 |
| OCR | EasyOCR（纯 Torch，GPU 加速） |
| Embedding | BAAI/bge-large-zh-v1.5（FlagEmbedding） |
| 向量存储 | Qdrant 本地模式（COSINE，1024 维） |
| 关键词索引 | rank-bm25（BM25Okapi + jieba 分词） |
| 重排 | BAAI/bge-reranker-v2-m3（transformers） |
| LLM | DeepSeek API（主）/ 阿里云百炼 Qwen API（多模态备用） |
| 混合检索 | RRF（Reciprocal Rank Fusion, k=60） |
| 配置 | Pydantic Settings |
| 测试 | pytest 215+ 项 / pytest-httpx（API Mock） |

---

## 配置项

所有配置通过 `.env` 文件或环境变量控制：

```ini
# === LLM ===
DEEPSEEK_API_KEY=sk-xxx                # DeepSeek API Key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
QWEN_API_KEY=sk-xxx                    # Qwen 多模态 API Key
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-7b-plus
QWEN_VL_MODEL=qwen2.5-vl-3b-instruct

# === 存储路径 ===
UPLOAD_DIR=./data/uploads
VECTOR_DB_DIR=./data/vector_db
MODEL_DIR=./data/models

# === GPU ===
DEVICE=cuda                             # 或 cpu

# === 置信度阈值 ===
CONFIDENCE_THRESHOLD_ACCEPT=0.75        # >= 此值直接返回
CONFIDENCE_THRESHOLD_REJECT=0.40        # < 此值拒绝回答
```

---

## 测试

```bash
# 运行全部测试（215+ 项，25-30 秒）
pytest

# 带覆盖率
pytest --cov=src

# 仅 API 集成测试
pytest tests/integration/test_api.py -v
```

---

## 作为其他 Agent 的文档解析中间服务

服务已经暴露 `POST /api/v1/documents/parse` 纯解析端点，其他 Agent 可以通过 HTTP 调用它来解析 PDF 文档，拿到结构化的页面和切片结果自己做后续处理，无需经过 RAG 链路。

```python
# 在另一个 Agent 中—— example-agent-sdk code
response = await client.post(
    f"{PARSER_URL}/api/v1/documents/parse",
    files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
)
pages = response.json()["pages"]
chunks = response.json()["chunks"]
# 自主处理 chunks...
```

---

## 许可证

MIT
