# 处理速度 + 公式识别准确性 优化设计

> 日期：2026-07-31
> 状态：已批准（用户确认）

## 背景

基于当前 RAG Pipeline 的实测表现，两个方向需要优化：

1. **处理速度慢**：embedding 使用 bge-large-zh-v1.5（326M 参数）在纯 CPU 上跑，71 个切片耗时 199s。CUDA 不可用。
2. **公式识别不准**：Qwen-VL-plus 是通用视觉模型，对数学公式转 LaTeX 准确率约 60%，例如 ε₁ε₂ 被误识别为 `\mathcal{E}^{2}\subset\mathcal{E}^{3}`。

## 目标

- Embedding：71 块从 199s 降到 ~25-35s（6-9 倍提速）
- 公式识别：纯数学公式准确率从 ~60% 提到 ~90%
- 全程不中断流水线，失败自动回退

---

## 一、公式识别改造 → pix2tex (LaTeX-OCR)

### 技术选型

pix2tex（LaTeX-OCR）是专为「公式图片 → LaTeX」训练的开源模型：
- ViT encoder + Transformer decoder
- 纯 CPU 可跑，模型约 200MB
- 对纯公式的准确率远高于通用 VL 模型
- 离线可用（首次下载权重后）

### 实现

**新增模块** `src/ocr/latex_ocr.py`：

```python
class LatexOCREngine:
    """基于 pix2tex 的本地公式识别引擎（懒加载单例）。"""
    def __init__(self, model_dir: str | None = None): ...
    def recognize(self, image_bytes: bytes) -> tuple[str, float]:
        """image_bytes -> (latex_wrapped_in_$$, confidence)"""
```

- `recognize()`：图片字节 → numpy (PIL 转换) → `model(img_array)` → LaTeX → 包 `$$...$$`
- 空结果返回 `("", 0.0)`
- 非空 LaTeX 置信度记 0.85
- 模型缓存在 `data/models/latex-ocr`
- 首次下载失败 / 模型加载失败 → 抛异常给调用方回退

**接入 orchestrator** `_ocr_embedded_images()`：

```
pix2tex (LatexOCREngine)  ← 优先
   ↓ 失败 / 空结果
Qwen-VL (FormulaRecognizer)  ← 回退
   ↓ 失败 / 空结果
easyocr (OCR)  ← 最终回退
```

### 预期收益

| 指标 | Qwen-VL (现状) | pix2tex |
|------|---------------|---------|
| 纯公式准确率 | ~60% | ~90% |
| 依赖 | 需 Qwen API key + 联网 | 本地离线 |
| 单张耗时 | ~2-4s | ~0.5-1s (CPU) |

---

## 二、Embedding 加速 → bge-base-zh + ONNX

### 技术选型

- **模型**：`BAAI/bge-base-zh-v1.5`（102M 参数，输出 768 维）
- **推理后端**：`sentence-transformers` 的 `backend="onnx"` + onnxruntime
- 相比 bge-large (326M) 模型小 3 倍，ONNX 免去 PyTorch 动态图开销

### 实现

**重写** `src/index/embedding.py`：

```python
def _get_model(cache_dir: str) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(
            "BAAI/bge-base-zh-v1.5",
            cache_folder=cache_dir,
            backend="onnx",
            device="cpu",
        )
    return _MODEL
```

- `embed()` / `embed_batch()` / `embed_chunks()` 接口保持不变
- Redis 缓存逻辑保留（原 `embed_chunks` 中已有）

### 预期收益

| 指标 | bge-large torch (现状) | bge-base onnx |
|------|------------------------|---------------|
| 模型参数 | 326M | 102M (3x 小) |
| 推理方式 | PyTorch 动态图 | ONNX 静态图 |
| 71 块耗时 | 199s | ~25-35s (估算) |
| 维度 | 1024 | 768 |

---

## 三、向量库迁移（维度 1024→768）

### 问题

Qdrant collection `documents` 当前 vectors_config 为 size=1024。bge-base 输出 768 维，写入会报维度不匹配。

### 方案

1. 新增 collection `documents_v2`（size=768, distance=Cosine）
2. `VectorStore` 增加 `qdrant_collection` 配置项，默认 `documents_v2`
3. 旧 `documents`（1024 维）保留，历史数据不迁移
4. 重新上传文档即可重建索引

### 受影响文件

- `src/config/settings.py` — 加 `qdrant_collection: str = "documents_v2"`
- `src/index/vector_store.py` — 默认 collection 读取配置；`_get_qdrant_client` 创建时用配置名

### 备注

- BM25 索引不受影响（独立存储）
- 检索时 query embedding 也是 768 维，与 collection 匹配

---

## 四、容错与回退

| 场景 | 回退行为 |
|------|----------|
| pix2tex 下载失败 | 回退 Qwen-VL |
| pix2tex 返回空 LaTeX | 回退 Qwen-VL |
| ONNX 模型加载失败 | 回退原 FlagModel (bge-large-zh) |
| Qwen-VL 无 API key | 直接 easyocr |
| 所有识别都失败 | `[图片识别失败]` 占位符（现状已有） |

保证：任何组件故障不阻断整个文档处理流水线。

---

## 五、验证方案

1. **公式切片抽查**：上传高数模拟题 docx，抽查 3-5 条公式的 LaTeX，与 Qwen-VL 结果对比
2. **速度记录**：上传课程论文 PDF，对比 embedding 耗时（期望 199s → <60s）
3. **问答回归**：公式相关查询能命中正确切片
4. **回退测试**：临时禁用 pix2tex（改名模型目录）→ 确认回退 Qwen-VL 不报错

---

## 涉及文件清单

| 文件 | 改动 |
|------|------|
| `src/ocr/latex_ocr.py` | **新增** — pix2tex 引擎 |
| `src/ocr/__init__.py` | 导出 LatexOCREngine |
| `src/pipeline/orchestrator.py` | `_ocr_embedded_images` 加 pix2tex 优先链 |
| `src/index/embedding.py` | 重写 — bge-base + ONNX |
| `src/config/settings.py` | 加 `qdrant_collection` |
| `src/index/vector_store.py` | collection 名可配置 |
| `pyproject.toml` | 加 pix2tex 依赖 |

## 不做的事（YAGNI）

- 不迁移旧 1024 维向量数据（历史数据孤儿化，可重传重建）
- 不加本地 SQLite embedding 缓存（用户未选择此方向，Redis 缓存已存在）
- 不改 bge-base 为 fastembed（内置列表无 base，sentence-transformers ONNX 已满足）
- 不引入多模态 RAG（ColPali）——成本/延迟高，现阶段不需要
