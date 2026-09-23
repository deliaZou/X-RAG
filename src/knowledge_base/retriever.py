import json
import os
import chromadb

from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from configs.paths import VECTOR_DB_DIR, SERVICE_TOPOLOGY

EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "re2ob_knowledge"

ALL_TYPES = ["feature", "playbook", "playbook_ext", "alert", "algo", "topology"]
TYPE_TAG  = {  # 内部 type 字段命名,和 rag_llm_layer.py 的四类 (+topology) 对齐
    "feature":     "feature_semantic",
    "playbook":    "playbook",
    "playbook_ext": "playbook",  # 和 LLM 生成的 playbook 共用同一个检索 tag,
                                  # 靠 source_category 字段区分来源,不共用同一个 --only 开关
    "alert":       "historical_alert",
    "algo":        "algo_config",
    "topology":    "topology",
}

FAULT_TYPES = {
    "CPU":    {"metrics": ["cpu"],
               "pattern": "target service 的 cpu 指标持续显著升高,workload 未必同步升高,"
                          "是判断异常还是正常负载增长的关键区分点"},
    "Delay":  {"metrics": ["latency"],
               "pattern": "target 及下游 service 的 latency 指标升高,自身 cpu/mem 未必异常,"
                          "容易被 SHAP 误判为高权重的 root cause 而实际是传导结果"},
    "Disk":   {"metrics": ["diskio"],
               "pattern": "target service 的 diskio 指标异常,可能伴随 latency 上升"},
    "Loss":   {"metrics": ["latency"],
               "pattern": "网络丢包导致失败请求和错误响应码上升,root cause indicator是error指标,"
                          "不是socket(RCAEval论文3.2节原文定义,之前误把socket也算进Loss,已修正)"},
    "Mem":    {"metrics": ["mem"],
               "pattern": "target service 的 mem 指标持续升高,可能伴随 diskio (swap) 上升"},
    "Socket": {"metrics": ["socket"],
               "pattern": "target service 的 socket 连接数异常,是六类故障里归因质量最弱的一类,"
                          "PCC 重建残差对 socket 信号捕捉能力有限"},
}

# metric -> fault_type 反查表,严格一对一,workload 不在其中(它是观察指标,
# 不是任何 fault_type 的 root cause indicator,依据 RCAEval 论文 3.2 节原文)
METRIC_TO_FAULT_TYPE = {
    metric: fault_type
    for fault_type, info in FAULT_TYPES.items()
    for metric in info["metrics"]
}


def load_docs(path) -> Dict[str, List[Dict]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def get_chroma_path(embed_model: str) -> str:
    """
    按 embedding 模型区分向量库路径,避免换模型时互相覆盖,
    也让"固定知识库,换 embedding 模型对比"这个受控实验能同时保留多份结果。
    可用环境变量 XRAG_KB_ROOT 覆盖基准目录。
    """
    slug = embed_model.replace("/", "_").replace(":", "_")
    return os.path.join(VECTOR_DB_DIR, f"chroma_re2ob_kb__{slug}")


def _extract_services_and_metrics(metrics_list: List[str]) -> tuple[List[str], List[str]]:
    """从指标名列表（如 ["emailservice_latency", "checkoutservice_cpu"]）提取服务名和指标名"""
    services_out = []
    metrics_out = []
    services_seen = set()
    metrics_seen = set()

    for name in metrics_list:
        if "_" not in name:
            continue
        service, metric = name.rsplit("_", 1)
        if service not in services_seen:
            services_out.append(service)
            services_seen.add(service)
        if metric not in metrics_seen:
            metrics_out.append(metric)
            metrics_seen.add(metric)

    return services_out, metrics_out


def _build_topology_prompt(chain: list[str]) -> str:
    """根据 root_cause_chain 构建详细的拓扑描述"""
    if not chain:
        return "  (no service chain available)"

    lines = ["Service Topology (based on root cause chain):"]
    lines.append("")

    topology = load_docs(SERVICE_TOPOLOGY)
    # 构建涉及的服务的拓扑信息
    for i, svc in enumerate(chain, 1):
        info = topology.get(svc, {})
        role = info.get("role", "unknown")
        upstream = info.get("upstream", [])
        downstream = info.get("downstream", [])
        hint = info.get("diagnosis_hint", "")

        lines.append(f"  [{i}] {svc}")
        lines.append(f"      role: {role}")
        if upstream:
            lines.append(f"      upstream: {', '.join(upstream)}")
        if downstream:
            lines.append(f"      downstream: {', '.join(downstream)}")
        if hint:
            lines.append(f"      hint: {hint}")
        lines.append("")

    # 添加传播方向分析
    if len(chain) >= 2:
        lines.append("  Propagation direction:")
        lines.append(f"    {chain[0]} → {' → '.join(chain[1:])}")

        # 检查第一个服务是否是 leaf
        first_info = topology.get(chain[0], {})
        if first_info.get("role") == "leaf":
            lines.append(f"    ⚠️ {chain[0]} is a LEAF node. Its anomaly should NOT propagate to downstream.")
            lines.append(f"    If downstream services show anomalies, they may have separate root causes.")
        elif first_info.get("role") in ["entrypoint", "orchestrator"]:
            lines.append(f"    {chain[0]} is an {first_info.get('role')}. Anomaly may propagate to downstream.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 检索接口 (供 rag_llm_layer.py 直接替换 load_knowledge_base.KnowledgeBase)
# ─────────────────────────────────────────────
class KnowledgeBase:
    """
    向量检索版知识库,接口和 load_knowledge_base.py 的关键词版保持一致:
        kb.build_query(chain) -> str
        kb.retrieve_by_type(chain, top_k_each, current_case_id) -> dict

    返回结构统一为:
        {"feature_docs": [...], "playbooks": [...], "alert_history": [...],
         "algo_config": [...], "topology_docs": [...]}
    """

    def __init__(self, embed_model: str = EMBED_MODEL, chroma_path: Optional[str] = None):
        self.embed_model = embed_model
        resolved_path = chroma_path or get_chroma_path(embed_model)
        db = chromadb.PersistentClient(path=resolved_path)
        self.collection = db.get_collection(COLLECTION_NAME)
        self.embedder = SentenceTransformer(embed_model)

    def build_query(self, chain: List[str]) -> str:
        return " ".join(chain)

    def _query(self, query_emb, top_k: int, type_tag: str,
               current_case_id: Optional[str] = None) -> List[Dict]:
        where = {"type": type_tag}
        if type_tag == TYPE_TAG["alert"] and current_case_id:
            # leave-one-case-out: 排除当前正在诊断的 case
            where = {"$and": [{"type": type_tag}, {"case_id": {"$ne": current_case_id}}]}

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, dist, meta in zip(docs, dists, metas):
            item = {"text": doc, "similarity": round(1 - dist, 4)}
            item.update(meta or {})  # 带出 fault_type/metric/service/source 等字段,
                                      # 之前这里查了 metadatas 却没往外传,做 Hit@k
                                      # 评估时会缺数据,现在补上
            out.append(item)
        return out

    def _retrieve_playbooks(self, metrics: List[str], top_k_each: int) -> List[Dict]:
        """
        三阶段检索,取代原来"整条候选链拼一句话直接 embedding 搜索"的做法:

        阶段1 精确过滤(召回): 候选链里的裸 metric 通过 METRIC_TO_FAULT_TYPE
            反查出可能的 fault_type,连同该 metric 在候选链里的排名(rank
            越小越靠前,SHAP 越高)一起记下来,不是简单丢进一个无序集合。
            再用 trigger_metrics 字段和候选 metric 是否有交集做补充召回
            (路径B,排名记为候选链长度,优先级最低)。
        阶段2 余弦匹配: 只在阶段1召回的子集里算 query 向量和每条文档向量
            的 cosine 相似度,不再对全库搜索。
        阶段3 rerank: 先按"命中的 metric 在候选链里的最佳排名"排序(排名
            越靠前优先级越高),同排名内部再按相似度降序。之前这里用布尔值
            exact_match 当唯一优先级,导致候选链里 diskio/mem/cpu 同时出现
            时,CPU/Mem 类文档和真正该优先的 Disk 类文档优先级完全相等,
            最终排序完全交给余弦相似度决定,而余弦相似度会偏向内容短、
            关键词密度高的文档,导致真正相关的 Disk 文档反而挤不进
            top_k_each——这是之前 Hit@k 在 disk/loss/socket 上意外掉到 0
            的直接原因,不是文档缺失或知识库量不够。
        """
        if not metrics:
            return []

        # metric -> 它在候选链里的最佳(最小)排名,排名从 0 开始
        metric_rank = {}
        for i, m in enumerate(metrics):
            metric_rank.setdefault(m, i)

        # fault_type -> 支持它的 metric 里最靠前的排名,用来给 path_a 命中排优先级
        fault_type_best_rank = {}
        for m, rank in metric_rank.items():
            ft = METRIC_TO_FAULT_TYPE.get(m)
            if ft is not None:
                fault_type_best_rank[ft] = min(rank, fault_type_best_rank.get(ft, rank))

        NO_MATCH_RANK = len(metrics)  # 只命中路径B、没有精确 fault_type 匹配的,排在所有精确匹配之后

        # 阶段1: 精确过滤召回,先把 type=="playbook" 的全部文档拉出来
        # (含 metadata + embedding,子集不大,Chroma 支持一次性 get 全量)
        all_playbooks = self.collection.get(
            where={"type": TYPE_TAG["playbook"]},
            include=["documents", "metadatas", "embeddings"],
        )

        recalled = []
        for doc_id, doc, meta, emb in zip(
            all_playbooks["ids"], all_playbooks["documents"],
            all_playbooks["metadatas"], all_playbooks["embeddings"],
        ):
            meta = meta or {}
            root_cause = meta.get("fault_type", "")
            triggers = [t.strip() for t in (meta.get("triggers") or "").split(",") if t.strip()]

            path_a = root_cause in fault_type_best_rank
            path_b = bool(set(triggers) & set(metrics))
            if not (path_a or path_b):
                continue

            rank = fault_type_best_rank[root_cause] if path_a else NO_MATCH_RANK

            recalled.append({
                "id": doc_id, "text": doc, "meta": meta,
                "embedding": emb, "rank": rank,
            })

        if not recalled:
            return []

        # 3. BM25 重排
        # 3.1 准备语料：对每个候选文档的 text 做分词（英文按空格切分即可）
        tokenized_corpus = []
        for doc in recalled:
            # 只取 text 的前 500 字符（防止过长文本稀释分数）
            text = doc['text'][:500]
            # 转小写 + 按空格分词（英文场景够用）
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)

        # 3.2 构建 BM25 索引
        bm25 = BM25Okapi(tokenized_corpus)

        # 3.3 用 query（metrics）计算每个文档的 BM25 分数
        # query 也用同样的分词方式
        query_tokens = [m.lower() for m in metrics]
        scores = bm25.get_scores(query_tokens)

        # 3.4 把分数存回候选文档
        for i, doc in enumerate(recalled):
            doc['bm25_score'] = scores[i]

        # 4. 排序：先按 rank，再按 BM25 分数
        recalled.sort(key=lambda r: (r['rank'], -r['bm25_score']))

        # 5. 返回 top_k
        out = []
        for r in recalled[:top_k_each]:
            item = {"text": r["text"]}
            item.update(r["meta"])
            out.append(item)
        return out
        # return recalled[:top_k_each]

        # 阶段2: 只在召回子集里算 cosine 相似度
        # query_text = " ".join(metrics)
        # query_vec = np.array(self.embedder.encode([query_text])[0])
        # for r in recalled:
        #     doc_vec = np.array(r["embedding"])
        #     sim = float(np.dot(query_vec, doc_vec) /
        #                 (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-9))
        #     r["similarity"] = round(sim, 4)
        #
        # # 阶段3: 先按候选链排名排序(排名小的优先,即命中的 metric 在 L2
        # # 候选链里 SHAP 越高越靠前),同排名内部再按相似度降序
        # recalled.sort(key=lambda r: (r["rank"], -r["similarity"]))
        #
        # out = []
        # for r in recalled[:top_k_each]:
        #     item = {"text": r["text"], "similarity": r["similarity"]}
        #     item.update(r["meta"])
        #     out.append(item)
        # return out

    def retrieve_by_type(self, chain: list[str], top_k_each: int = 2,
                          current_case_id: Optional[str] = None) -> Dict[str, List[Dict]]:
        # query = self.build_query(chain)
        # query_emb = self.embedder.encode([query]).tolist()
        # 结构化查询：拓扑
        services, metrics = _extract_services_and_metrics(chain)
        return {
            # "feature_docs":  self._query(query_emb, top_k_each, TYPE_TAG["feature"]),
            "playbooks":     self._retrieve_playbooks(metrics, top_k_each),
            # "alert_history": self._query(query_emb, top_k_each, TYPE_TAG["alert"], current_case_id),
            # "algo_config":   self._query(query_emb, top_k_each, TYPE_TAG["algo"]),
            "topology_docs": _build_topology_prompt(services)
        }

def main():
    kb = KnowledgeBase(embed_model="all-MiniLM-L6-v2")
    ctx = kb.retrieve_by_type(["cartservice_diskio", "checkoutservice_cpu"])
    print(ctx)


if __name__ == "__main__":
    main()
