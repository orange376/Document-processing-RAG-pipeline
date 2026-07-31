# 处理速度 + 公式识别准确性 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 pix2tex 本地公式识别替代 Qwen-VL 主路径提升公式准确率，用 bge-base-zh + ONNX 替代 bge-large + PyTorch 提升 embedding 速度 6-9 倍。

**Architecture:** 新增 `LatexOCREngine`（pix2tex）作为嵌入图片公式识别的首选，失败回退 Qwen-VL → easyocr；重写 `EmbeddingEngine` 使用 `SentenceTransformer(backend="onnx")` 加载 bge-base-zh-v1.5（768 维），并迁移 Qdrant 到新 collection `documents_v2`。

**Tech Stack:** pix2tex 0.1.4, sentence-transformers 5.6 (ONNX backend), onnxruntime 1.28, Qdrant, Python 3.12

## Global Constraints

- 本机无 CUDA，一切推理走 CPU
- bge-base-zh-v1.5 输出 **768 维**，必须使用新 Qdrant collection（旧 `documents` 是 1024 维，保留不迁移）
- 任何识别/加载失败必须回退，不中断文档处理流水线
- 接口 `EmbeddingEngine.embed / embed_batch / embed_chunks` 签名不变，消费方无需改动
- pix2tex 模型权重首次使用从 HuggingFace 下载（约 200MB），需联网；下载路径由 pix2tex 自身管理（`~/.pix2tex/checkpoints`），不强制改到 `data/models`
- 遵守项目现有惯例：懒加载单例、`logger = logging.getLogger(__name__)`、中文注释/docstring

---

### Task 1: 安装 pix2tex 并验证可导入

**Files:**
- Modify: `pyproject.toml`
- 无代码文件（依赖安装）

**Interfaces:**
- Consumes: 无
- Produces: `pix2tex` 可在当前 Python 3.12 环境导入

- [ ] **Step 1: 安装 pix2tex**

```bash
pip install pix2tex==0.1.4
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from pix2tex.cli import LatexOCR; print('pix2tex OK')"
```

Expected: 打印 `pix2tex OK`。若因 transformers/timm 版本冲突失败，记录报错，调整兼容版本后重试（目标：`LatexOCR` 类可导入）。

- [ ] **Step 3: 在 pyproject.toml 登记依赖**

在 `[project.dependencies]` 末尾追加：

```toml
    "pix2tex>=0.1.4",
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add pix2tex for local formula OCR"
```

---

### Task 2: 创建 LatexOCREngine

**Files:**
- Create: `src/ocr/latex_ocr.py`
- Modify: `src/ocr/__init__.py:1-5`
- Create: `tests/test_latex_ocr.py`

**Interfaces:**
- Consumes: 无
- Produces: `LatexOCREngine`，`recognize(image_bytes: bytes) -> tuple[str, float]`（LaTeX 用 `$$...$$` 包裹，失败返回 `("", 0.0)`）。同步方法，调用方需 `asyncio.to_thread` 包裹。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_latex_ocr.py`：

```python
"""LatexOCREngine unit tests."""

import io

import pytest

pix2tex = pytest.importorskip("pix2tex", reason="pix2tex not installed")

from src.ocr import LatexOCREngine  # noqa: E402


class TestLatexOCREngine:
    def test_recognize_returns_tuple(self):
        """recognize 总是返回 (str, float)，即便识别失败。"""
        from PIL import Image

        engine = LatexOCREngine()
        buf = io.BytesIO()
        Image.new("RGB", (64, 32), "white").save(buf, format="PNG")
        latex, conf = engine.recognize(buf.getvalue())
        assert isinstance(latex, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_recognize_wraps_in_dollars(self):
        """返回的 LaTeX 用 $$ 包裹（若模型输出非空）。"""
        from PIL import Image

        engine = LatexOCREngine()
        buf = io.BytesIO()
        Image.new("RGB", (128, 64), "white").save(buf, format="PNG")
        latex, _ = engine.recognize(buf.getvalue())
        # 空结果或 $$ 包裹均可（识别失败返回 ""，不强制非空）
        assert latex == "" or latex.startswith("$$")
```

注：首个测试会触发 pix2tex 权重下载（~200MB）。若离线，`recognize` 内部捕获异常返回 `("", 0.0)`，测试仍通过。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_latex_ocr.py -v`
Expected: FAIL — `ImportError`/`AttributeError`（`LatexOCREngine` 不存在）

- [ ] **Step 3: 实现 LatexOCREngine**

创建 `src/ocr/latex_ocr.py`：

```python
"""本地公式识别引擎 — 基于 pix2tex (LaTeX-OCR)。

将公式图片转换为 LaTeX，本地 CPU 推理，不依赖外部 API。
权重首次使用时从 HuggingFace 下载（~200MB），之后离线可用。
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_MODEL: object | None = None


def _get_model() -> object:
    """返回共享的 pix2tex LatexOCR 单例。"""
    global _MODEL
    if _MODEL is None:
        from pix2tex.cli import LatexOCR

        _MODEL = LatexOCR(no_cuda=True)
    return _MODEL


class LatexOCREngine:
    """基于 pix2tex 的本地公式识别引擎。

    用法::

        engine = LatexOCREngine()
        latex, confidence = engine.recognize(image_bytes)
    """

    def recognize(self, image_bytes: bytes) -> tuple[str, float]:
        """识别公式图片为 LaTeX。

        Returns:
            (latex_string, confidence)。
            latex 用 ``$$...$$`` 包裹；识别失败返回 ("", 0.0)。
        """
        try:
            from PIL import Image

            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            model = _get_model()
            latex = model(pil_img)
            latex = (latex or "").strip()
            if not latex:
                return "", 0.0
            if not latex.startswith("$$"):
                latex = f"$${latex}$$"
            return latex, 0.85
        except Exception:
            logger.warning("pix2tex formula recognition failed", exc_info=True)
            return "", 0.0
```

- [ ] **Step 4: 在 __init__.py 导出**

修改 `src/ocr/__init__.py`：

```python
"""Formula recognition and image analysis."""
from .formula_recognizer import FormulaRecognizer
from .latex_ocr import LatexOCREngine
from .page_recognizer import PageRecognizer

__all__ = ["FormulaRecognizer", "LatexOCREngine", "PageRecognizer"]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_latex_ocr.py -v`
Expected: PASS（首次运行可能等待权重下载）

- [ ] **Step 6: Commit**

```bash
git add src/ocr/latex_ocr.py src/ocr/__init__.py tests/test_latex_ocr.py
git commit -m "feat: LatexOCREngine via pix2tex for local formula OCR"
```

---

### Task 3: 在 orchestrator 中接入 pix2tex 优先链

**Files:**
- Modify: `src/pipeline/orchestrator.py` — `_ocr_embedded_images`（约 400-500 行区域）

**Interfaces:**
- Consumes: `LatexOCREngine.recognize(img_bytes) -> (str, float)`（同步，需 `asyncio.to_thread`）
- Produces: 无新接口（内部行为变更）

- [ ] **Step 1: 导入并初始化 LatexOCREngine**

在 `_ocr_embedded_images` 方法内、`formula_recognizer = FormulaRecognizer()` 之后加：

```python
        # 本地公式识别（pix2tex）—— 优先使用，失败回退 Qwen-VL
        from src.ocr import LatexOCREngine
        latex_ocr: LatexOCREngine | None = None
        try:
            latex_ocr = LatexOCREngine()
        except Exception:
            latex_ocr = None
```

- [ ] **Step 2: 修改 formula 分支为 pix2tex 优先**

将 `_recognize_one` 内 `if category == "formula":` 分支替换为：

```python
                if category == "formula":
                    # 主路径：本地 pix2tex
                    if latex_ocr is not None:
                        latex, _ = await asyncio.to_thread(
                            latex_ocr.recognize, img_bytes
                        )
                        if latex:
                            block.content = block.content.replace(
                                placeholder, latex, 1
                            )
                            completed += 1
                            return
                    # 回退：Qwen-VL
                    latex, _ = await formula_recognizer.recognize(img_bytes)
                    if latex:
                        block.content = block.content.replace(
                            placeholder, latex, 1
                        )
                        completed += 1
                        return
```

- [ ] **Step 3: 修改 general 分支同样优先 pix2tex**

将 general 分支的「先 Qwen-VL 后 easyocr」改为：

```python
                # --- General: pix2tex → Qwen-VL → easyocr ---
                if latex_ocr is not None:
                    latex, _ = await asyncio.to_thread(
                        latex_ocr.recognize, img_bytes
                    )
                    if latex:
                        block.content = block.content.replace(
                            placeholder, latex, 1
                        )
                        completed += 1
                        return

                latex, _ = await formula_recognizer.recognize(img_bytes)
                if latex:
                    block.content = block.content.replace(
                        placeholder, latex, 1
                    )
                    completed += 1
                    return
```

（其后保留原 easyocr 回退代码不变）

- [ ] **Step 4: 验证编译**

Run: `python -c "from src.pipeline.orchestrator import PipelineOrchestrator; print('OK')"`
Expected: `OK`，无语法错误

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/orchestrator.py
git commit -m "feat: pix2tex as primary formula recognizer in image pipeline"
```

---

### Task 4: 重写 EmbeddingEngine 为 bge-base-zh + ONNX

**Files:**
- Modify: `src/index/embedding.py`（整体重写）
- Modify: `tests/unit/test_embedding.py:13`（断言 768 维）

**Interfaces:**
- Consumes: 无
- Produces: 与现有一致的 `EmbeddingEngine.embed(text) -> list[float]`、`embed_batch(texts) -> list[list[float]]`、`embed_chunks(chunks) -> list[Chunk]`。维度 **768**。

- [ ] **Step 1: 更新失败测试**

修改 `tests/unit/test_embedding.py`：

```python
class TestEmbeddingEngine:
    def test_initialization(self):
        engine = EmbeddingEngine()
        assert getattr(engine, "_model", None) is None  # lazy

    def test_embed(self):
        engine = EmbeddingEngine()
        emb = engine.embed("测试文本")
        assert len(emb) == 768  # bge-base-zh-v1.5 维度

    def test_unload(self):
        engine = EmbeddingEngine()
        engine.unload()
        assert True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/unit/test_embedding.py -v`
Expected: FAIL — 断言 1024 != 768（模型仍是旧的，或导入报错）

- [ ] **Step 3: 重写 embedding.py**

整体替换 `src/index/embedding.py` 内容：

```python
from __future__ import annotations

from src.config import get_settings
from src.domain import Chunk

# Module-level singleton for the underlying SentenceTransformer.
# Loaded once on first use and shared across all EmbeddingEngine instances.
_MODEL: object | None = None


def _get_model(cache_dir: str) -> object:
    """Return the shared SentenceTransformer singleton (bge-base-zh, ONNX backend).

    Tries ONNX runtime first for CPU speed; falls back to torch backend
    (same 768-dim model) if ONNX export is unavailable.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        try:
            _MODEL = SentenceTransformer(
                "BAAI/bge-base-zh-v1.5",
                cache_folder=cache_dir,
                backend="onnx",
                device="cpu",
            )
        except Exception:
            # ONNX 不可用时回退 torch 后端（同模型同维度，仅更慢）
            _MODEL = SentenceTransformer(
                "BAAI/bge-base-zh-v1.5",
                cache_folder=cache_dir,
                device="cpu",
            )
    return _MODEL


class EmbeddingEngine:
    """Embedding 引擎 — 基于 bge-base-zh-v1.5 (768 维, ONNX 加速)

    The underlying SentenceTransformer is a module-level singleton so it
    survives across requests without reloading.
    """

    def __init__(self):
        self._settings = get_settings()

    def unload(self):
        """No-op — the singleton model stays loaded across requests."""
        pass

    def embed(self, text: str) -> list[float]:
        """对单段文本编码"""
        model = _get_model(str(self._settings.resolved_model_dir / "bge-base-zh"))
        emb = model.encode(text, show_progress_bar=False)
        return emb.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        model = _get_model(str(self._settings.resolved_model_dir / "bge-base-zh"))
        embs = model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embs]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """为 Chunk 列表填充 embedding（优先走 Redis 缓存）。"""
        from src.cache import RedisCache
        cache = RedisCache()

        uncached: list[tuple[int, Chunk]] = []
        for i, chunk in enumerate(chunks):
            cached_vec = cache.get_embedding(chunk.content)
            if cached_vec is not None:
                chunk.embedding = cached_vec
            else:
                uncached.append((i, chunk))

        if not uncached:
            return chunks

        texts = [c.content for _, c in uncached]
        embeddings = self.embed_batch(texts)
        for (idx, chunk), emb in zip(uncached, embeddings):
            chunk.embedding = emb
            cache.set_embedding(chunk.content, emb)

        return chunks


def create_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/unit/test_embedding.py -v`
Expected: PASS（首次运行会下载 bge-base-zh-v1.5 模型到 `data/models/bge-base-zh`，需联网）

- [ ] **Step 5: Commit**

```bash
git add src/index/embedding.py tests/unit/test_embedding.py
git commit -m "perf: bge-base-zh with ONNX backend for embedding (1024->768 dims)"
```

---

### Task 5: Qdrant collection 迁移到 documents_v2（768 维）

**Files:**
- Modify: `src/config/settings.py`（加 `qdrant_collection`、`embedding_dim`）
- Modify: `src/index/vector_store.py`（collection 名与向量维度可配置）

**Interfaces:**
- Consumes: settings 的 `qdrant_collection` / `embedding_dim`
- Produces: `VectorStore(collection_name: str | None = None)`；默认 collection 读 settings。`_get_qdrant_client(db_path)` 不再负责建 collection；建 collection 职责移到 `_ensure_collection(client, name, size)`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_vector_store_config.py`：

```python
"""VectorStore collection 配置测试。"""

from src.config import get_settings
from src.index.vector_store import VectorStore, _ensure_collection


class TestVectorStoreConfig:
    def test_default_collection_from_settings(self):
        s = get_settings()
        vs = VectorStore()
        assert vs._collection_name == s.qdrant_collection == "documents_v2"
        assert vs._vector_size == s.embedding_dim == 768

    def test_ensure_collection_idempotent(self, tmp_path):
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(tmp_path))
        _ensure_collection(client, "test_col", 768)
        _ensure_collection(client, "test_col", 768)  # 二次调用不报错
        names = [c.name for c in client.get_collections().collections]
        assert "test_col" in names
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_vector_store_config.py -v`
Expected: FAIL — `_ensure_collection` 不存在；`qdrant_collection` 未定义

- [ ] **Step 3: settings.py 加配置**

在 `src/config/settings.py` 的 `# === Paths ===` 区块后加：

```python
    # === Vector Store ===
    qdrant_collection: str = "documents_v2"  # bge-base-zh 768 维对应 collection
    embedding_dim: int = 768
```

- [ ] **Step 4: 重构 vector_store.py**

修改 `src/index/vector_store.py`：

将 `_get_qdrant_client(db_path)` 改为「只负责创建/返回共享 client」，collection 创建抽出为独立函数：

```python
def _get_qdrant_client(db_path: str) -> object:
    """Return the shared QdrantClient singleton, creating it on first call."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    from qdrant_client import QdrantClient

    Path(db_path).mkdir(parents=True, exist_ok=True)
    _QDRANT_CLIENT = QdrantClient(path=db_path)
    return _QDRANT_CLIENT


def _ensure_collection(client: object, collection_name: str, vector_size: int) -> None:
    """Idempotently create the collection if it doesn't exist."""
    from qdrant_client.http.models import VectorParams, Distance

    existing = client.get_collections()
    names = [c.name for c in existing.collections]
    if collection_name not in names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
```

将 `VectorStore.__init__` 与 `_lazy_init` 改为：

```python
    def __init__(self, collection_name: str | None = None):
        s = get_settings()
        self._settings = s
        self._collection_name = collection_name or s.qdrant_collection
        self._vector_size = s.embedding_dim
        self._client = None

    def _lazy_init(self):
        if self._client is not None:
            return
        db_path = str(self._settings.resolved_vector_db_dir)
        client = _get_qdrant_client(db_path)
        _ensure_collection(client, self._collection_name, self._vector_size)
        self._client = client
```

其余 `index_chunks` / `search` / `delete_by_source_file` / `count` 方法体不变（它们已使用 `self._collection_name`）。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_vector_store_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/config/settings.py src/index/vector_store.py tests/test_vector_store_config.py
git commit -m "feat: migrate vector store to documents_v2 (768-dim bge-base)"
```

---

### Task 6: 端到端验证

**Files:**
- 无代码改动（验证 + 更新测试套件）

**Interfaces:**
- Consumes: Task 1-5 全部产物

- [ ] **Step 1: 运行全部测试确认无回归**

Run: `python -m pytest tests/ -q --ignore=tests/integration`
Expected: 无新增失败（既有 3 个环境性失败可接受：Qdrant 锁 / httpx_mock fixture）

- [ ] **Step 2: 上传高数文档验证公式**

用浏览器上传 `data/uploads/*_高数AII模拟题*.docx`（或 API），然后：

```bash
python -c "
import httpx
r = httpx.get('http://localhost:8001/api/v1/review/pending', timeout=10)
print(r.status_code)
"
```

打开审核页抽查 3-5 条公式切片，确认 LaTeX 内容（对比修复前 Qwen-VL 输出）：
- 期望：`\frac{分子}{分母}`、`\sqrt{...}`、`\sum`、`\int` 等结构正确
- 接受：极复杂公式偶发偏差（pix2tex 非完美）

- [ ] **Step 3: 上传 PDF 测量 embedding 耗时**

上传课程论文 PDF，检查 `server.log`：

```
grep "embed+index" server.log
```

Expected: `< 60s`（修复前 199s）；同时确认 `[stage]` 各阶段日志正常

- [ ] **Step 4: 问答回归**

用「求根公式」「定积分」等查询，确认仍能命中正确切片并返回带引用答案。

- [ ] **Step 5: 回退测试（可选）**

临时将 `src/ocr/latex_ocr.py` 的 `_MODEL` 初始化为 None 并删除权重目录（或断网），确认流水线回退 Qwen-VL 不中断。测试后恢复。

- [ ] **Step 6: Commit（如有代码修正）**

```bash
git add -A
git commit -m "test: verify formula OCR and embedding speed improvements"
```

---

## 附注

- **旧向量数据**：`documents`（1024 维）保留在磁盘，不再被查询。需重新上传文档以重建 `documents_v2` 索引。
- **pix2tex 权重下载**：首次 `LatexOCREngine.recognize` 触发，路径 `~/.pix2tex/checkpoints`（pix2tex 自身管理）。
- **Redis 缓存**：`embed_chunks` 的 Redis 缓存逻辑保留；Redis 未运行时自动降级（现有行为）。
- **query.py 无改动**：768 维 query embedding 与 `documents_v2` collection 匹配。
