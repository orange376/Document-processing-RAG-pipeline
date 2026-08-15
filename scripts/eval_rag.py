#!/usr/bin/env python3
"""RAG 检索质量评估脚本 — 跑 QA 测试集，输出正确率基线。

Usage:
    python scripts/eval_rag.py                  # 跑默认测试集（连 localhost:8001）
    python scripts/eval_rag.py --warmup          # 先发一个预热请求再测（避开首查模型加载）
    python scripts/eval_rag.py --set path.json   # 自定义测试集

判定规则（务实）：答案包含任一 key_term 且至少 1 个引用 → PASS。
输出：pass_rate、平均置信度、平均引用数、逐题明细。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

API = "http://localhost:8001/api/v1/query"
DEFAULT_SET = Path(__file__).resolve().parent / "qa_eval_set.json"


def run_one(query: str, top_k: int = 5, timeout: float = 120) -> dict:
    resp = httpx.post(API, json={"query": query, "top_k": top_k}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def passes(item: dict, result: dict) -> tuple[bool, str]:
    """判定一条 QA 是否通过（答案含任一关键术语 + 有引用）。"""
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    hits = [k for k in item["key_terms"] if k.lower() in answer.lower()]
    ok = bool(hits) and bool(citations)
    reason = f"命中[{','.join(hits) or '无'}] 引用{len(citations)}"
    return ok, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--warmup", action="store_true", help="先发预热请求")
    args = parser.parse_args()

    items = json.loads(Path(args.set).read_text(encoding="utf-8"))
    print(f"QA 测试集：{len(items)} 题\n")

    if args.warmup:
        t0 = time.time()
        run_one("预热", timeout=120)
        print(f"预热完成（{time.time()-t0:.1f}s）\n")

    results = []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        try:
            r = run_one(item["q"])
            ok, reason = passes(item, r)
            conf = r.get("confidence", 0.0)
            cites = len(r.get("citations", []))
            status = "PASS" if ok else "FAIL"
            print(f"[{i:2d}/{len(items)}] {status} conf={conf:.2f} cites={cites} {time.time()-t0:.1f}s")
            print(f"      Q: {item['q']}")
            print(f"      {reason}")
            results.append((ok, conf, cites))
        except Exception as e:
            print(f"[{i:2d}/{len(items)}] ERROR: {e}")
            results.append((False, 0.0, 0))

    passed = sum(1 for ok, _, _ in results if ok)
    avg_conf = sum(c for _, c, _ in results) / len(results)
    avg_cites = sum(c for _, _, c in results) / len(results)
    print("\n" + "=" * 40)
    print(f"PASS RATE : {passed}/{len(results)} = {passed/len(results)*100:.0f}%")
    print(f"AVG CONF  : {avg_conf:.3f}")
    print(f"AVG CITES : {avg_cites:.1f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
