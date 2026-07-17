#!/usr/bin/env python3
"""下载所有本地推理所需的模型权重"""

# Windows 兼容：修复 PyTorch shm.dll 加载问题
import os
os.environ.setdefault("USE_WINDOWS_IPC", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_all():
    """下载全部模型"""
    model_dir = Path(__file__).resolve().parent.parent / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # [1/4] PP-DocLayoutV3 — 由 paddleocr 自动缓存
    print("[1/4] PP-DocLayoutV3 ... 首次使用时自动下载，缓存到 ~/.paddlex/")

    # [2/4] PaddleOCR-VL-1.5 — 首次使用时自动下载
    print("[2/4] PaddleOCR-VL-1.5 ... 首次使用时自动下载")

    # [3/4] bge-large-zh-v1.5
    print("[3/4] 下载 bge-large-zh-v1.5 ...")
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(
        'BAAI/bge-large-zh-v1.5',
        local_dir=str(model_dir / "bge-large-zh"),
    )
    print("  OK bge-large-zh-v1.5")

    # [4/4] bge-reranker-v2-m3
    print("[4/4] 下载 bge-reranker-v2-m3 ...")
    snapshot_download(
        'BAAI/bge-reranker-v2-m3',
        local_dir=str(model_dir / "bge-reranker"),
    )
    print("  OK bge-reranker-v2-m3")

    print(f"\n=== 下载完成！模型目录: {model_dir} ===")


if __name__ == "__main__":
    download_all()
