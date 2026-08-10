# WSL2 GPU 部署指南

> 让 RAG Pipeline 在 WSL2 Ubuntu 上使用 NVIDIA GPU 推理。已实测：17 页 PDF 版面分析从 CPU 167s → GPU 16s（10x），整文档处理 ~8 分钟 → 58s。

## 环境

- WSL2 Ubuntu 26.04，NVIDIA RTX 4060 (8GB)，CUDA 13.3 UMD 驱动
- 系统 Python 3.14（paddle 不支持）→ 用 `uv` 装 Python 3.12
- 国内网络：清华 PyPI 镜像 + HF 镜像 + ghproxy 代理

## 一键安装步骤（WSL2 内）

### 1. 基础工具
```bash
# uv (Python 版本管理, 免 sudo)
mkdir -p ~/.local/bin
curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz -o /tmp/uv.tar.gz
tar -xzf /tmp/uv.tar.gz -C /tmp/ && cp /tmp/uv-*/uv ~/.local/bin/uv && chmod +x ~/.local/bin/uv
export PATH=$HOME/.local/bin:$PATH

# 镜像配置
echo 'UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# Python 3.12 + 项目
uv python install 3.12
git clone https://github.com/orange376/Document-processing-RAG-pipeline.git ~/rag-pipeline
cd ~/rag-pipeline && uv venv --python 3.12 .venv && source .venv/bin/activate
```

### 2. torch CUDA（关键：pytorch.org 索引 + 清华补充源）
```bash
# 装 pip
uv pip install pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 先装 nvidia 依赖（清华镜像快），再 curl 下 torch wheel
python -m pip install nvidia-cudnn-cu12==9.10.2.21 nvidia-cusparselt-cu12==0.7.1 \
  nvidia-nccl-cu12==2.29.3 nvidia-nvshmem-cu12==3.4.5 \
  nvidia-cuda-runtime-cu12 nvidia-cufile-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-nvtx-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvcc-cu12 nvidia-nvjitlink-cu12 -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch wheel 必须 curl（uv/pip 直连 pytorch.org 会卡；curl 38MB/s）
curl -sL 'https://download.pytorch.org/whl/cu126/torch-2.13.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl' \
  -o ~/torch-2.13.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl
python -m pip install '/home/tom/torch-2.13.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl' --no-deps
python -m pip install filelock typing-extensions sympy networkx jinja2 fsspec -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
# 期望: True NVIDIA GeForce RTX 4060
```

### 3. torchvision
```bash
curl -sL 'https://download.pytorch.org/whl/cu126/torchvision-0.28.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl' \
  -o ~/torchvision-0.28.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl
python -m pip install '/home/tom/torchvision-0.28.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl' --no-deps
```

### 4. paddle GPU（3.3.1，官方 cu126 索引）
```bash
# 必须先装 paddle 再装 paddlex（否则拉 CPU 版）
python -m pip install 'paddlepaddle-gpu==3.3.1' -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# paddle 装完会降级 nccl/cusparselt，torch 会坏！升回 torch 要的版本（两者兼容）
python -m pip install 'nvidia-nccl-cu12==2.29.3' 'nvidia-cusparselt-cu12==0.7.1' \
  'nvidia-cudnn-cu12==9.10.2.21' 'nvidia-cublas-cu12==12.9.2.10' -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c 'import torch, paddle; print(torch.cuda.is_available(), paddle.is_compiled_with_cuda())'
# 期望: True True
```

### 5. libgomp（paddle 依赖，torch 自带）
```bash
# paddle 找不到 libgomp.so.1 —— 用 torch 自带的
echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH' >> .venv/bin/activate
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> .venv/bin/activate
```

### 6. 其余依赖（清华镜像）
```bash
export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR
python -m pip install fastapi uvicorn pydantic-settings pymupdf python-docx qdrant-client \
  rank-bm25 httpx python-multipart aiofiles jieba tiktoken loguru slowapi numpy celery redis \
  transformers sentence-transformers optimum-onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pillow easyocr pix2tex==0.1.4 FlagEmbedding -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install paddlex paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e . --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 7. 模型下载（HF 镜像）
```bash
source .venv/bin/activate
mkdir -p data/models
# embedding (768 维) + reranker
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5', cache_folder='data/models/bge-base-zh')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', cache_folder='data/models/bge-reranker')"
# ⚠️ reranker 文件必须拷到顶层（项目直接加载该目录）
cp data/models/bge-reranker/models--BAAI--bge-reranker-base/snapshots/*/* data/models/bge-reranker/

# pix2tex 权重（GitHub 慢，用 ghproxy 代理）
curl -sL 'https://ghproxy.net/https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/weights.pth' -o ~/weights.pth
curl -sL 'https://ghproxy.net/https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/image_resizer.pth' -o ~/image_resizer.pth
cp ~/weights.pth ~/image_resizer.pth .venv/lib/python3.12/site-packages/pix2tex/model/checkpoints/

# PP-DocLayoutV3：首次版面分析自动下载 (~/.paddlex/official_models)
```

### 8. API keys + 启动
```bash
cp /mnt/e/桌面/agent开发/rag-pipeline/.env ~/rag-pipeline/.env   # 从 Windows 复制
source .venv/bin/activate
python run.py
# WSL2 端口自动映射，浏览器访问 http://localhost:8001
```

## 启动服务器（保持 WSL 存活）

WSL 空闲会自动停止（清空 /tmp）。用后台任务保持会话：
```bash
wsl -d Ubuntu -e bash -c "cd ~/rag-pipeline && source .venv/bin/activate && python run.py"
```

## 实测性能

| 指标 | Windows CPU | WSL2 GPU |
|------|-------------|----------|
| 17页 PDF 版面分析 | 167s | **16s (10x)** |
| 整文档处理 | ~8 分钟 | **58s (8x)** |
| 查询 | — | 0.799 置信度，5 引用 ✓ |

## 已知限制

1. **ONNX 导出不持久化**：每次服务器启动首次 embed 重新导出 (~26s)。可优化：预导出保存 `model.onnx`。
2. **Redis 未运行**：缓存禁用（不影响功能）。
3. **Qwen-VL 图表描述**：需 `.env` 有 `qwen_api_key`（已从 Windows 复制）。
4. WSL 内存 7.6GB 偏紧，模型逐个加载（代码已 unload）。
