"""
audit_kb_docs.py
================
排查知识库文档内容是否完整,不是查检索准不准,是查存进去的原始内容
本身是不是空的/异常短的。用于定位"某个 fault_type 的 AC@k 下降"到底是
内容层面的问题,还是 prompt 里某句启发式规则的问题。

用法:
    python audit_kb_docs.py --embed-model all-MiniLM-L6-v2
"""

import argparse
import chromadb
from build_knowledge_base import get_chroma_path, COLLECTION_NAME
import json
from collections import Counter, defaultdict

SUSPICIOUS_LEN = 50  # 正文短于这个字符数,标记为可疑(大概率生成失败或内容被截断)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    db = chromadb.PersistentClient(path=get_chroma_path(args.embed_model))
    collection = db.get_collection(COLLECTION_NAME)
    result = collection.get(include=["documents", "metadatas"])

    rows = []
    for doc_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        rows.append({
            "id": doc_id,
            "type": meta.get("type", "?"),
            "fault_type": meta.get("fault_type", ""),
            "metric": meta.get("metric", ""),
            "service": meta.get("service", ""),
            "source_category": meta.get("source_category", ""),
            "len": len(text or ""),
            "preview": (text or "")[:80].replace("\n", " "),
        })

    rows.sort(key=lambda r: (r["type"], r["len"]))

    print(f"{'type':<16} {'fault_type':<10} {'metric':<10} {'len':>6}  id / preview")
    print("-" * 100)
    for r in rows:
        flag = "⚠️ 可疑" if r["len"] < SUSPICIOUS_LEN else ""
        print(f"{r['type']:<16} {r['fault_type']:<10} {r['metric']:<10} {r['len']:>6}  "
              f"{r['id']}  {flag}")
        if flag:
            print(f"    内容: {r['preview']!r}")

    print("\n" + "=" * 100)
    print(f"总文档数: {len(rows)}, 可疑(短于{SUSPICIOUS_LEN}字符): "
          f"{sum(1 for r in rows if r['len'] < SUSPICIOUS_LEN)}")

    # 单独检查你现在最关心的两类: Delay 的 latency 覆盖, Loss 的 error 覆盖
    print("\n--- Delay/Loss 相关文档 ---")
    for r in rows:
        if r["fault_type"] in ("Delay", "Loss") or r["metric"] in ("latency", "error"):
            print(f"  [{r['type']}] fault_type={r['fault_type']} metric={r['metric']} "
                  f"len={r['len']}  {r['id']}")

    # 覆盖度统计: 每个 fault_type / metric 实际有几条文档,一眼看出哪类是"裸的"
    from collections import Counter

    print("\n" + "=" * 100)
    print("覆盖度统计 (playbook 按 fault_type,feature 按 metric)")
    print("=" * 100)

    playbook_rows = [r for r in rows if r["type"] == "playbook"]
    all_fts = sorted({r["fault_type"] or "(未标注)" for r in playbook_rows})
    all_sources = sorted({r["source_category"] or "(未标注)" for r in playbook_rows})

    pivot = Counter((r["fault_type"] or "(未标注)", r["source_category"] or "(未标注)")
                    for r in playbook_rows)

    print("\n[playbook] 按 fault_type × source_category 分表 (不合计):")
    header = f"{'fault_type':<10}" + "".join(f"{s:<18}" for s in all_sources)
    print(header)
    for ft in all_fts:
        row = f"{ft:<10}" + "".join(f"{pivot[(ft, s)]:<18}" for s in all_sources)
        print(row)

    feature_rows = [r for r in rows if r["type"] == "feature_semantic"]
    metric_counter = Counter(r["metric"] or "(未标注)" for r in feature_rows)
    print("\n[feature_semantic] 按 metric 计数:")
    for m in sorted(metric_counter):
        print(f"  {m:<12} 共 {metric_counter[m]} 条")

    # 全类型总览,一眼看出哪个 type 完全是空的(比如之前 algo_config/topology 没写进去)
    type_counter = Counter(r["type"] for r in rows)
    print("\n[全部类型] 总条数:")
    for t in sorted(type_counter):
        print(f"  {t:<20} 共 {type_counter[t]} 条")


def analyze_metadata(kb_cache_path="kb_docs_cache.json"):
    """从知识库缓存中统计元数据"""

    with open(kb_cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    # 只统计 playbook_ext（外部 runbook）
    playbooks = cache.get("playbook_ext", [])

    if not playbooks:
        print("没有找到 playbook_ext 数据")
        return

    # ==================== 1. 根因统计 ====================
    fault_type_counter = Counter()
    for p in playbooks:
        ft = p.get("fault_type", "unknown")
        fault_type_counter[ft] += 1

    print("=== 根因（fault_type）分布 ===")
    for ft, count in fault_type_counter.most_common():
        print(f"  {ft}: {count}")

    # ==================== 2. 指标统计 ====================
    trigger_counter = Counter()
    for p in playbooks:
        triggers = p.get("triggers", [])
        for t in triggers:
            trigger_counter[t] += 1

    print("\n=== 指标（triggers）分布 ===")
    for t, count in trigger_counter.most_common():
        print(f"  {t}: {count}")

    # ==================== 3. 根因×指标交叉统计 ====================
    # 每种根因下使用了哪些指标
    fault_triggers = defaultdict(Counter)
    for p in playbooks:
        ft = p.get("fault_type", "unknown")
        triggers = p.get("triggers", [])
        for t in triggers:
            fault_triggers[ft][t] += 1

    print("\n=== 根因 × 指标 交叉统计 ===")
    for ft, counter in fault_triggers.items():
        print(f"  {ft}:")
        for t, count in counter.most_common():
            print(f"      {t}: {count}")

    return {
        "fault_type": fault_type_counter,
        "triggers": trigger_counter,
        "fault_triggers": fault_triggers,
    }

if __name__ == "__main__":
    # main()
    analyze_metadata()
