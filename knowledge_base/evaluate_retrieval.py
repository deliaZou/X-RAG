"""
evaluate_retrieval.py
======================
只测检索,不调 LLM。用 case 自己的 fault_type (从 case_id 反推,不是拿
ground_truth 当查询输入) 去核对检索回来的文档 metadata 对不对,判断
"知识库+检索机制"这一步本身有没有在起作用,和后面 L3 的 LLM 推理质量
完全分开评估。

同时跑一个随机基线做对照: 如果随机抽文档的 Hit@k 和真实检索的 Hit@k
差不多,说明检索机制没有比瞎猜好,这一步必须做,不然单看 Hit@k 数字
本身不能证明检索"有效",只能证明"有命中",两者不是一回事。

用法:
    python evaluate_retrieval.py --l2-dir ./eval_outputs/l2_test_fold --embed-model all-MiniLM-L6-v2
"""

import argparse
import random
from collections import defaultdict

from l2_adapter import load_l2_dir
from build_knowledge_base import KnowledgeBase


def _fault_type_from_case_id(case_id: str) -> str:
    parts = case_id.split("_")
    return parts[-2] if len(parts) >= 2 else "unknown"


def _hit_playbook(docs, fault_type: str) -> bool:
    # 大小写不敏感比较: case_id 反推出来的是小写("cpu"),但知识库里
    # fault_type 字段存的是首字母大写("CPU"/"Delay"...),精确匹配会
    # 永远为 False,之前 Hit@k 全表 0.000(连随机基线也是 0.000)就是这个原因,
    # 不是真的没检索到东西,是这一步比较从没生效过
    target = fault_type.lower()
    return any((d.get("fault_type") or "").lower() == target for d in docs)


def evaluate(l2_dir: str, embed_model: str, top_k_each: int = 2, seed: int = 42):
    xai_reports = load_l2_dir(l2_dir)
    kb = KnowledgeBase(embed_model=embed_model)

    # 随机基线用同一批文档池,不经过检索,直接随机抽 top_k_each 条
    all_docs = kb.collection.get(include=["metadatas"])
    pool_by_type = defaultdict(list)
    for meta in all_docs["metadatas"]:
        pool_by_type[meta.get("type", "?")].append(meta)
    random.seed(seed)

    real_hits = defaultdict(list)   # {fault_type: [bool, ...]}
    random_hits = defaultdict(list)

    for report in xai_reports:
        case_id = report.get("case_id", "")
        fault_type = _fault_type_from_case_id(case_id)
        chain = report.get("xai_analysis", {}).get("root_cause_chain", [])

        # ctx = kb.retrieve_by_type(chain, top_k_each=top_k_each, current_case_id=case_id)
        # real_hits[fault_type].append(_hit_playbook(ctx["playbooks"], fault_type))

        ctx = kb._retrieve_playbooks(chain, top_k_each=top_k_each)
        real_hits[fault_type].append(_hit_playbook(ctx, fault_type))




        # 随机基线: 从同一个 type 的全量 metadata 池里随机抽 top_k_each 条
        pool = pool_by_type.get("playbook", [])
        random_sample = random.sample(pool, min(top_k_each, len(pool))) if pool else []
        random_hits[fault_type].append(_hit_playbook(random_sample, fault_type))

    print(f"{'fault_type':<10} {'n':>4} {'真实检索 Hit@k':>16} {'随机基线 Hit@k':>16} {'提升':>8}")
    print("-" * 60)
    for ft in sorted(real_hits):
        r_rate = sum(real_hits[ft]) / len(real_hits[ft])
        rand_rate = sum(random_hits[ft]) / len(random_hits[ft])
        print(f"{ft:<10} {len(real_hits[ft]):>4} {r_rate:>16.3f} {rand_rate:>16.3f} "
              f"{r_rate - rand_rate:>+8.3f}")

    macro_real = sum(sum(v) / len(v) for v in real_hits.values()) / len(real_hits)
    macro_random = sum(sum(v) / len(v) for v in random_hits.values()) / len(random_hits)
    print("-" * 60)
    print(f"{'macro avg':<10} {'':>4} {macro_real:>16.3f} {macro_random:>16.3f} "
          f"{macro_real - macro_random:>+8.3f}")

    print("\n判断标准: 提升列大于 0 且有实际差距(比如 >0.2),说明检索机制真的在"
          "起作用,不是随机命中;如果某个 fault_type 提升接近 0,说明检索对"
          "这一类没有实际帮助,该类的知识库内容或检索匹配逻辑需要重新看。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", default="eval_outputs/l2_test_fold")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k-each", type=int, default=2)
    args = parser.parse_args()
    evaluate(args.l2_dir, args.embed_model, args.top_k_each)
