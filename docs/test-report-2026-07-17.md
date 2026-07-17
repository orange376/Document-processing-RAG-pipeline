# RAG Pipeline 阶段性测试报告（更新版）

> **日期:** 2026-07-17 (Phase 1 完成)
> **上次更新:** 2026-07-17 (PP-DocLayoutV3 恢复)

---

## 架构变更摘要

**当前版面分析:** PP-DocLayoutV3（PaddleX，25 类检测）
- doc_title, paragraph_title, text, table, figure, chart, header, footer, formula, reference 等
- 深度学习模型，基于渲染页面图像
- 降级路径：启发式（PyMuPDF 字号+位置）在模型不可用时自动切换

**OCR:** easyocr（纯 torch，支持中英，GPU 加速）

**关键修复:**
- paddle 2.6.2 → **3.0.0**（PIR JSON 格式需要）
- import torch 必须在 paddle 之前（DLL 冲突 WinError 127）
- `set_optimization_level` 兼容性 monkey-patch

---

## 测试总览

| 层级 | 测试项 | 状态 |
|------|--------|------|
| **单元测试** | 44 项 | ✅ **42 通过，2 跳过** |
| **集成测试** | 21 项 | ✅ **全部通过** |
| **PP-DocLayoutV3** | 模型推理 | ✅ **25 类版面分析正常** |

---

## 集成测试详细结果

### ✅ 测试 1：PDF 加载器 (PyMuPDF Dict 模式)

| 检查项 | 结果 |
|--------|------|
| 文件名 | 正确 |
| 文件类型 | pdf |
| 页数 | 4 页 |
| Block 解析 | 24 块，590 字符 |
| raw_dict 存在 | ✅ 全部 |
| 字号元数据 | ✅ 含 max_font_size |

### ✅ 测试 2：结构感知切片器

- 4 个切片，元数据完整性通过

### ✅ 测试 3：Qdrant 向量存储

- 写入 3 条 / 搜索返回 2 条 / Count = 3

### ✅ 测试 4：bge-large-zh Embedding (CPU)

- 14 个模型文件 (1.3GB)
- 编码维度 1024 ✅

### ✅ 测试 5：PP-DocLayoutV3 版面分析

| 分类 | PP-DocLayoutV3 |
|------|---------------|
| 文档标题 | `doc_title` ✅ |
| 段落标题 | `paragraph_title` ✅ |
| 正文 | `text` ✅ |
| 图片 | `image` ✅ |
| 表格 | `table` ✅ |
| 图表 | `chart` ✅ |
| 公式 | `display_formula` / `inline_formula` ✅ |
| 页眉/页脚 | `header` / `footer` ✅ |
| 参考 | `reference` ✅ |
| 合计 | **25 类** ✅ |

### ✅ 测试 6：easyocr 导入

- `import easyocr` 成功

### ✅ 测试 7：编排器端到端

```
Status: indexed
Chunks: 4
Layout tree: yes
Indexed: 4
```

---

## 依赖清单

```
paddlepaddle>=3.0.0 → PP-DocLayoutV3 版面分析
paddlex>=3.7.0     → PaddleX 推理框架
pymupdf>=1.24.0    → 文档加载 + 渲染
easyocr>=1.7.0     → OCR（扫描件页面备用）
FlagEmbedding>=1.3.0 → bge-large-zh 编码（CPU）
qdrant-client>=1.16.0 → 向量存储（本地模式，1.18 API 已适配）
rank-bm25>=0.2.2   → 稀疏检索（Phase 2）
```

---

## 已知问题

### P1 — easyocr 模型首次下载

**症状:** 首次 OCR 调用时下载模型权重（~100MB），可能耗时 30-60s
**解决:** 在空闲时预下载：`python -c "import easyocr; easyocr.Reader(['ch_sim','en'])"`

### P2 — DLL 加载顺序

**症状:** `import paddle` 必须在 `import torch` 之后，否则 `shm.dll` 报 WinError 127
**解决:** `detector.py` 中 `_lazy_load()` 已确保先 `import torch`

---

## 下一步

Phase 2：
1. **混合检索器** — BM25 + 向量 + BGE-Reranker
2. **LLM 客户端** — Qwen3.7-Plus API + 强制溯源引用
3. **置信度评分** — 5 维 + 三级阈值
4. **API 层** — FastAPI 路由

测试文档: `scripts/test_e2e_pipeline.py`
运行: `python scripts/test_e2e_pipeline.py`
