"""
debug_playbook_recall.py
=========================
诊断 _retrieve_playbooks 的阶段1(精确过滤召回)为什么在某些 fault_type
上完全扑空,打印中间每一步的实际值,不猜,直接看数据。

用法:
    python debug_playbook_recall.py --l2-dir ./eval_outputs/l2_test_fold --fault-type disk
"""

import argparse
from l2_adapter import load_l2_dir
from build_knowledge_base import (
    KnowledgeBase, METRIC_TO_FAULT_TYPE, FAULT_TYPES, TYPE_TAG, _extract_metrics
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", default="eval_outputs/l2_test_fold")
    parser.add_argument("--fault-type", default="disk", help="只看这个 fault_type 的第一个 case")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    print("METRIC_TO_FAULT_TYPE 反查表:", METRIC_TO_FAULT_TYPE)
    print()

    # 1. 找一个目标 fault_type 的 case
    xai_reports = load_l2_dir(args.l2_dir)
    target = None
    for r in xai_reports:
        case_id = r.get("case_id", "")
        parts = case_id.split("_")
        ft = parts[-2] if len(parts) >= 2 else ""
        if ft.lower() == args.fault_type.lower():
            target = r
            break

    if target is None:
        print(f"没找到 fault_type={args.fault_type} 的 case")
        return

    chain = target["xai_analysis"]["root_cause_chain"]
    print(f"case_id: {target['case_id']}")
    print(f"root_cause_chain (query 用的候选链): {chain}")

    # 2. 看 _extract_metrics 抽出了什么
    metrics = _extract_metrics(chain)
    print(f"\n_extract_metrics 抽出的裸 metric: {metrics}")

    fault_type_candidates = {
        METRIC_TO_FAULT_TYPE[m] for m in metrics if m in METRIC_TO_FAULT_TYPE
    }
    print(f"反查出的 fault_type_candidates: {fault_type_candidates}")

    # 3. 看知识库里所有 playbook 文档的真实 metadata,逐条核对匹不匹配
    kb = KnowledgeBase(embed_model=args.embed_model)
    all_playbooks = kb.collection.get(
        where={"type": TYPE_TAG["playbook"]},
        include=["metadatas"],
    )

    print(f"\n知识库里全部 {len(all_playbooks['ids'])} 条 playbook 文档,逐条核对:")
    print(f"{'id':<35} {'fault_type':<10} {'triggers':<20} {'path_a':<8} {'path_b':<8}")
    for doc_id, meta in zip(all_playbooks["ids"], all_playbooks["metadatas"]):
        meta = meta or {}
        root_cause = meta.get("fault_type", "")
        triggers = [t.strip() for t in (meta.get("triggers") or "").split(",") if t.strip()]
        path_a = root_cause in fault_type_candidates
        path_b = bool(set(triggers) & set(metrics))
        flag = "✓ 召回" if (path_a or path_b) else ""
        print(f"{doc_id:<35} {root_cause!r:<10} {str(triggers):<20} "
              f"{path_a!s:<8} {path_b!s:<8} {flag}")


if __name__ == "__main__":
    main()
