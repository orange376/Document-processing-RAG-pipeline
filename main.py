#!/usr/bin/env python3
"""RAG Pipeline — API 服务启动入口

Usage:
    python main.py                       # 开发模式 (localhost:8000, 热重载)
    python main.py --port 8080           # 自定义端口
    python main.py --host 0.0.0.0        # 监听所有地址
    python main.py --reload              # 显式开启热重载
    python main.py --no-reload           # 关闭热重载
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认 8000)")
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument(
        "--reload", action="store_true", default=None, help="开启热重载 (开发模式)"
    )
    reload_group.add_argument(
        "--no-reload", action="store_false", dest="reload", help="关闭热重载"
    )
    args = parser.parse_args()

    # 开发环境默认开启热重载，生产环境默认关闭
    reload = args.reload if args.reload is not None else True

    try:
        import uvicorn
    except ImportError:
        print("错误: 缺少 uvicorn，请执行: pip install uvicorn[standard]", file=sys.stderr)
        sys.exit(1)

    print(f"🚀 RAG Pipeline API 启动: http://{args.host}:{args.port}")
    print(f"📖 API 文档: http://{args.host}:{args.port}/docs")
    print(f"📋 健康检查: http://{args.host}:{args.port}/api/v1/health")
    if reload:
        print("⚡ 热重载已开启 (开发模式)")

    uvicorn.run(
        "src.api.app:create_app",
        host=args.host,
        port=args.port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
