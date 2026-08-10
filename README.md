# RAG Pipeline — 工业级文档解析与检索增强生成系统

> 面向中文企业文档的 RAG 系统 · 无 LangChain/LlamaIndex 依赖 · 全部组件自主实现 · 214 项单元测试全绿

支持 **PDF / Word** 文档的解析、版面分析、公式/图表/表格识别、结构感知切片、混合检索（稠密 + 稀疏 + RRF 融合）、BGE 重排、LLM 生成与置信度审核闭环。GPU 加速（WSL2 实测版面分析 10 倍、整文档 8 倍）。

---

## 功能亮点

### 📄 文档解析与版面分析

| 能力 | 状态 |
|------|------|
| PDF 加载（PyMuPDF） | ✅ 文本块 + 字体 + 位置 + 图片流 |
| Word 加载（python-docx） | ✅ 段落/表格/图片 + VML 公式图检测 |
| **PP-DocLayoutV3 版面分析（25 类）** | ✅ PaddleX 深度学习模型 + 启发式降级，GPU 支持 |
| 扫描件整页识别 | ✅ Qwen-VL PageRecognizer 多模态 OCR |
| EasyOCR 缺文本页面补全 | ✅ 纯 Torch |
| 版面树构建 | ✅ 标题 → 子节层次组织 |
| 公式图片识别 | ✅ **pix2tex（LaTeX-OCR）**本地推理，结构正确 |
| PDF 图表描述 | ✅ Qwen-VL 架构图/流程图 → 可检索文字 |
| 表格结构恢复 | ✅ Qwen-VL 截图 → Markdown 结构化 |
| Word 内嵌公式（MathType/Equation） | ✅ VML WMF 高分辨率渲染 + 识别 |
| 纯解析 API（不索引） | ✅ 供其他 Agent 消费 |

### 🔍 检索增强

| 能力 | 状态 |
|------|------|
| **bge-base-zh-v1.5** 稠密向量化 | ✅ 768 维，ONNX 后端（导出已持久化，热启动 ~9s） |
| Qdrant 向量存储 | ✅ `documents_v2` collection，COSINE |
| BM25 关键词索引（rank_bm25） | ✅ jieba 分词，磁盘持久化 |
| **RRF 混合搜索融合** | ✅ k=60，向量 + 关键词联合排序 |
| BGE-reranker 重排 | ✅ sigmoid 归一化 0-1 分，Redis 缓存 |
| Query 改写 | ✅ LLM 驱动的同义词扩展检索优化 |

### ✂️ 结构感知切片

| 能力 | 状态 |
|------|------|
| 标题边界切片 + 版面树路径 | ✅ 段落/句子多粒度 |
| **考试题感知切片** | ✅ 一道题 = 一个块，ABCD 选项黏合题干 |
| 短标题合并 / 块内多题拆分 | ✅ 消除孤悬片段 |
| 切片重叠（128 字符 + 句末对齐） | ✅ 上下文不撕裂 |

### 🤖 LLM 生成

| 能力 | 状态 |
|------|------|
| **DeepSeek** 主语言模型 | ✅ |
| **Qwen VL** 多模态（图表/扫描件/表格） | ✅ |
| 结构感知上下文拼装 | ✅ 来源 + 页码 + 章节 + 类型标注 |
| 强制溯源引用 | ✅ `[来源: 文件 \| 第N页]` |
| 低置信度降级回答 | ✅ 自动空结果返回 |
| SSE 流式问答 | ✅ `POST /api/v1/query/stream` |

### 🎯 置信度 + 人工审核闭环

| 能力 | 状态 |
|------|------|
| 5 维评分引擎（布局/OCR/表格/切片/重排） | ✅ 查询时动态加权 |
| 三级阈值策略（accept / review / reject） | ✅ 可配置阈值 |
| **人工审核界面**（前端 Web） | ✅ 逐级展开页面/块/切片、内联编辑、通过/拒绝、重新处理 |
| 审核持久化 | ✅ `task_db.json` 重启不丢 |
| 反馈统计面板 | ✅ `GET /api/v1/review/stats` 错误率/问题类型分析 |

### 🏭 生产就绪

| 能力 | 状态 |
|------|------|
| Redis 三级缓存（embedding/reranker/answer） | ✅ 版本化 key，重启不失效 |
| Bearer Token 认证 | ✅ `api_auth_token`，可关闭 |
| API 限流 | ✅ slowapi |
| 结构化日志 | ✅ loguru + JSON 文件轮转 |
| Docker 部署 | ✅ 多阶段 Dockerfile + docker-compose |
| GPU 推理（WSL2） | ✅ 版面 167s→16s，整文档 8min→58s |

---

## 快速开始

### 1. 克隆并安装

```bash
git clone https://github.com/orange376/Document-processing-RAG-pipeline.git
cd rag-pipeline

# 创建虚拟环境（Python >= 3.10）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"
```

> 深度学习版面分析需要 PaddlePaddle：`pip install paddlepaddle-gpu paddlex`
> （GPU 部署完整步骤见 [docs/WSL2-GPU部署指南.md](docs/WSL2-GPU部署指南.md)）

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```ini
# DeepSeek（主语言模型）
DEEPSEEK_API_KEY=sk-xxx

# Qwen（多模态：图表/扫描件/表格）
QWEN_API_KEY=sk-xxx
```

### 3. 下载模型

```bash
python scripts/download_models.py
```

模型清单（存于 `data/models/`）：

| 模型 | 用途 | 说明 |
|------|------|------|
| bge-base-zh-v1.5 | Embedding（768 维） | ONNX 后端，导出自动持久化 |
| bge-reranker-base | 重排 | transformers 加载 |
| pix2tex 权重 | 公式识别 | 首次调用自动下载 |
| PP-DocLayoutV3 | 版面分析 | 首次版面分析自动下载 |

### 4. 启动服务

```bash
python run.py
# → http://localhost:8001
# → Web 界面（上传/问答/审核）: http://localhost:8001
# → API 文档: http://localhost:8001/docs
```

> 热重载默认关闭（会杀掉后台处理任务）；需要时 `RAG_RELOAD=1 python run.py`。

---

## API 端点

所有端点前缀 `/api/v1`，认证通过 `Authorization: Bearer <token>`（若配置了 `api_auth_token`）。

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/documents/upload` | 上传文档 → 全流程处理 |
| GET | `/documents/{task_id}/status` | 轮询处理状态 |
| DELETE | `/documents/{task_id}` | 删除文档及索引 |
| POST | `/documents/parse` | **纯解析**（不上索引，供 Agent 消费） |
| POST | `/query` | RAG 问答 |
| POST | `/query/stream` | RAG 流式问答（SSE） |
| GET | `/review/pending` | 待审核任务列表 |
| GET | `/review/{task_id}` | 任务详情（页面/块/切片/置信度） |
| POST | `/review/{task_id}/approve` | 审核通过/拒绝（含编辑 + 原因） |
| POST | `/review/{task_id}/reprocess` | 用原文件重新处理 |
| GET | `/review/stats` | 反馈统计（错误率/问题类型） |
| GET | `/review/feedback` | 最近反馈记录 |
| GET | `/health` | 健康检查 |

---

## 使用示例

### 纯解析（供其他 Agent 消费）

```python
import httpx

resp = httpx.post(
    "http://localhost:8001/api/v1/documents/parse",
    files={"file": ("report.pdf", open("report.pdf", "rb"), "application/pdf")},
)
data = resp.json()
print(f"共 {data['total_pages']} 页，{len(data['chunks'])} 个切片")
for chunk in data["chunks"]:
    print(f"  [{chunk['section']}] {chunk['content'][:80]}...")
```

### RAG 问答

```bash
curl -X POST http://localhost:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "文档中提到了什么技术方案？", "top_k": 10}'
```

响应包含 `answer`、`citations`（带来源页码）、`confidence` 评分。

---

## 项目结构

```
rag-pipeline/
├── run.py                   # 服务启动入口（端口 8001）
├── main.py                  # 旧版入口（兼容保留）
├── .env                     # 配置（API Key、路径、阈值）
├── src/
│   ├── api/                 # FastAPI 应用 + 路由 + Schemas
│   │   ├── app.py           #   应用工厂
│   │   ├── static/          #   Web 前端（上传/问答/审核界面）
│   │   ├── schemas/         #   Pydantic 请求/响应模型
│   │   └── routers/         #   upload, query, review, admin
│   ├── pipeline/            # 主编排器（全流程 + 纯解析）
│   ├── parser/
│   │   ├── loader/          #   PDF/Word 加载（含 VML 公式图）
│   │   ├── layout/          #   版面分析（PP-DocLayoutV3 + 启发式）
│   │   ├── ocr/             #   EasyOCR + pix2tex 公式识别
│   │   ├── table/           #   表格结构恢复（Qwen-VL）
│   │   └── chunker.py       #   结构感知切片（考试题感知）
│   ├── index/
│   │   ├── embedding.py     #   bge-base-zh ONNX 向量化
│   │   ├── vector_store.py  #   Qdrant 向量存储
│   │   ├── bm25_index.py    #   BM25 关键词索引
│   │   └── hybrid_search.py #   RRF 混合检索融合
│   ├── retrieval/           # retriever / reranker / query_rewriter
│   ├── generation/          # llm_client / context_builder / prompt_manager
│   ├── confidence/          # 5 维评分 + 三级阈值 + 降级路由
│   ├── cache/               # Redis 三级缓存
│   ├── domain/              # 领域模型（Document/Chunk/Layout/Table）
│   └── config/              # Pydantic Settings
├── tests/                   # 214 项测试，全绿
│   ├── unit/                #   单元测试
│   └── integration/         #   集成测试
├── data/                    # 运行时生成（models/uploads/vector_db/logs）
├── docs/                    # 迭代日志 / GPU 部署指南 / 设计文档
├── Dockerfile               # 多阶段构建
└── docker-compose.yml       # app + redis
```

---

## 架构概览

```
用户 / Agent / Web 界面
        │
        ▼
┌───────────────────┐
│   FastAPI 服务    │  ← /api/v1 (Bearer 认证 + 限流)
└─────────┬─────────┘
          │
          ├──▶ 全流程：上传 → 加载 → 版面分析 → OCR/公式/图表/表格 → 切片 → 索引
          │
          │        PipelineOrchestrator
          │        ① 加载 PDF/Word
          │        ② PP-DocLayoutV3 版面分析（GPU）
          │        ③ 公式(pix2tex) + 图表/表格(Qwen-VL) + OCR(EasyOCR)
          │        ④ 版面树构建
          │        ⑤ 结构感知切片（考试题感知）
          │        ⑥ Embedding(ONNX) + Qdrant + BM25 索引
          │
          ├──▶ 查询：改写 → 混合检索 → 重排 → LLM → 置信度 → 审核
          │
          │        Retriever ── HybridSearch ──┬─ VectorStore (Qdrant)
          │                                    └─ BM25Index
          │                        │
          │                        ▼
          │                 Reranker (bge-reranker)
          │                        │
          │                        ▼
          │        ContextBuilder ── LLMClient ── PromptManager
          │                        │
          │                        ▼
          │        ConfidenceScorer ── ThresholdStrategy ── Review API
          │
          └──▶ 纯解析：不上索引，返回结构化结果（供其他 Agent）
```

---

## 技术栈

| 模块 | 技术选型 |
|------|----------|
| 服务框架 | FastAPI + Uvicorn |
| 文档解析 | PyMuPDF（PDF）/ python-docx（Word） |
| 版面分析 | PP-DocLayoutV3（PaddleX，GPU） |
| OCR | EasyOCR + Qwen-VL（扫描件） |
| 公式识别 | pix2tex（LaTeX-OCR） |
| Embedding | bge-base-zh-v1.5（ONNX，768 维） |
| 向量存储 | Qdrant（`documents_v2`，COSINE） |
| 关键词索引 | rank-bm25 + jieba |
| 重排 | BGE-reranker-base |
| LLM | DeepSeek（主）/ 阿里云百炼 Qwen-VL（多模态） |
| 混合检索 | RRF（k=60） |
| 缓存 | Redis（embedding/reranker/answer） |
| 认证限流 | Bearer Token + slowapi |
| 日志 | loguru（JSON 轮转） |
| 部署 | Docker Compose + WSL2 GPU |
| 测试 | pytest 214 项 / pytest-httpx |

---

## 配置项

所有配置通过 `.env` 或环境变量控制（`src/config/settings.py`）：

```ini
# === LLM ===
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
QWEN_API_KEY=sk-xxx
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-7b-plus
QWEN_VL_MODEL=qwen-vl-plus            # 多模态（图表/扫描件/表格）

# === 存储路径 ===
UPLOAD_DIR=./data/uploads
VECTOR_DB_DIR=./data/vector_db
MODEL_DIR=./data/models

# === 向量存储 ===
QDRANT_COLLECTION=documents_v2        # 768 维 collection
EMBEDDING_DIM=768

# === GPU ===
DEVICE=cuda                           # 或 cpu

# === 置信度阈值 ===
CONFIDENCE_THRESHOLD_ACCEPT=0.75
CONFIDENCE_THRESHOLD_REJECT=0.40

# === 上下文窗口 ===
MAX_CONTEXT_TOKENS=3600

# === 认证（空 = 开放模式） ===
API_AUTH_TOKEN=

# === Redis ===
REDIS_URL=redis://localhost:6379/0
```

---

## 测试

```bash
# 运行全部测试（214 项）
pytest

# 带覆盖率
pytest --cov=src

# 仅 API 集成测试
pytest tests/integration/test_api.py -v
```

> 若本地服务器在运行，集成测试可能因 Qdrant 文件锁失败，先停服务器再跑。

---

## 部署

### Docker（CPU）

```bash
docker compose up -d --build
# app → :8001, redis → :6379
```

### WSL2 GPU 加速

完整步骤见 [docs/WSL2-GPU部署指南.md](docs/WSL2-GPU部署指南.md)。要点：

- WSL2 Ubuntu + RTX 4060，`uv` 管理 Python 3.12
- torch CUDA 从 pytorch.org 下载，其余依赖走清华镜像
- paddle GPU 3.3.1，`LD_LIBRARY_PATH` 指到 torch/lib 解决 libgomp
- 模型通过 HF 镜像 + ghproxy 下载

**实测性能**：

| 指标 | Windows CPU | WSL2 GPU |
|------|-------------|----------|
| 17 页 PDF 版面分析 | 167s | **16s（10 倍）** |
| 整文档处理 | ~8 分钟 | **58s（8 倍）** |
| Embedding（17 页） | 199s | **~10s** |

---

## 迭代记录

迭代历程、性能数据与决策见 [docs/迭代日志.md](docs/迭代日志.md)。

---

## 许可证

MIT
