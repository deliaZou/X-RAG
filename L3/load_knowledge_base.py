"""
load_knowledge_base.py
========================
Layer 3 知识库最小实现。四类知识:
  A. feature_docs   : metric 语义卡
  B. playbooks      : 按 fault type 的排查步骤
  C. alert_history  : 历史真阳性案例摘要 (对当前 case 做 leave-one-case-out 隔离)
  D. algo_config    : 算法配置说明 (对应 configs/selection.yaml)

当前用关键词 Jaccard 相似度做检索,不依赖 embedding 服务,
目的是先跑通 Layer3 调用链路。后续要换成向量检索时,
只需要替换 _retrieve_pool 的打分逻辑,其余接口不用变。
"""

import re
import random
from typing import List, Dict, Optional

# 已知的 7 个 metric 类型,用于把 "checkoutservice_cpu" 这种复合名字拆成
# service + metric 两部分,分别喂给不同类型的检索
KNOWN_METRIC_TYPES = {"cpu", "mem", "socket", "workload", "diskio", "latency", "error"}


def _tokenize(text: str) -> List[str]:
    # 先把下划线换成空格再切词,否则 "checkoutservice_cpu" 会被当成一个
    # 无法拆分的完整 token,永远匹配不上 feature_docs 里单独的 "cpu" 关键词,
    # 这是之前检索一直失效但分数全是 0、看不出异常的根源
    text = text.replace("_", " ")
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _split_name(name: str) -> tuple:
    """把 "checkoutservice_cpu" 拆成 ("checkoutservice", "cpu");
    拆不出已知 metric 类型时,整个字符串当作 service,metric 返回 None。"""
    if "_" in name:
        prefix, suffix = name.rsplit("_", 1)
        if suffix in KNOWN_METRIC_TYPES:
            return prefix, suffix
    return name, None


def _keyword_overlap_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not doc_tokens:
        return 0.0
    q, d = set(query_tokens), set(doc_tokens)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


class KnowledgeBase:
    """
    四类文档统一存储为 list[dict]:
        {"id": str, "keywords": [...], "text": str, "case_id": Optional[str]}
    alert_history 的每条必须带 case_id,用于隔离。
    """

    def __init__(self,
                 feature_docs: Optional[List[Dict]] = None,
                 playbooks: Optional[List[Dict]] = None,
                 alert_history: Optional[List[Dict]] = None,
                 algo_config: Optional[List[Dict]] = None,
                 topology_docs: Optional[List[Dict]] = None):
        self.feature_docs  = feature_docs  or self._default_feature_docs()
        self.playbooks     = playbooks     or self._default_playbooks()
        self.alert_history = alert_history or []
        self.algo_config   = algo_config   or self._default_algo_config()
        self.topology_docs = topology_docs or self._default_topology_docs()

    # ── 占位数据,先用最少条目跑通链路,后续逐类替换成真实内容 ──
    @staticmethod
    def _default_feature_docs():
        return [
            {"id": "f_cpu", "keywords": ["cpu"],
             "text": "cpu 指标反映服务进程的 CPU 占用率,持续升高常见于计算密集型"
                      "请求增多或资源限制过低,也可能是正常流量增长,需结合 workload"
                      "判断是否为因果关系。"},
            {"id": "f_mem", "keywords": ["mem", "memory"],
             "text": "mem 指标反映服务内存占用,持续升高可能是内存泄漏、缓存未回收"
                      "或批量请求堆积,常伴随响应延迟上升。"},
            {"id": "f_socket", "keywords": ["socket"],
             "text": "socket 指标反映网络连接数或文件描述符占用,异常升高常见于连接"
                      "未正常关闭或下游响应慢导致连接堆积,归因上容易与 cpu/mem 的"
                      "重建残差信号混淆。"},
            {"id": "f_latency", "keywords": ["latency"],
             "text": "latency 指标反映请求响应时间,通常是故障的结果而非原因,排名"
                      "靠前时优先怀疑是资源类指标传导的下游表现。"},
            {"id": "f_diskio", "keywords": ["diskio", "disk"],
             "text": "diskio 指标反映磁盘 I/O 吞吐或延迟,升高常见于磁盘空间不足、"
                      "I/O 争用或日志写入异常。"},
            {"id": "f_workload", "keywords": ["workload"],
             "text": "workload 指标反映请求量或并发量,是判断资源类指标升高是正常"
                      "负载还是异常的关键参照。"},
        ]

    @staticmethod
    def _default_playbooks():
        return [
            {"id": "p_cpu", "keywords": ["cpu"],
             "text": "CPU 故障排查: 确认是否伴随 workload 同步升高,若无则怀疑异常;"
                      "检查该服务是否有慢查询或死循环;检查容器 CPU limit 是否过低。"},
            {"id": "p_mem", "keywords": ["mem", "memory"],
             "text": "Mem 故障排查: 判断是渐进增长(疑似泄漏)还是突增;检查 GC 或"
                      "缓存策略;检查是否有批量任务堆积。"},
            {"id": "p_socket", "keywords": ["socket"],
             "text": "Socket 故障排查: 检查连接数是否单调上升;检查下游服务是否响应"
                      "变慢导致连接未释放;检查连接池配置。"},
            {"id": "p_disk", "keywords": ["diskio", "disk"],
             "text": "Disk 故障排查: 检查磁盘剩余空间;检查是否有异常大量写入;检查"
                      "I/O 是否被其他进程争用。"},
            {"id": "p_delay", "keywords": ["latency", "delay"],
             "text": "Delay 故障排查: latency 通常是结果,优先定位上游资源类指标;"
                      "检查网络层是否丢包或限速;检查依赖服务健康状态。"},
            {"id": "p_loss", "keywords": ["loss", "error"],
             "text": "Loss 故障排查: 检查网络丢包率和重传;检查服务错误率是否同步"
                      "升高;检查是否为下游服务不可用导致的级联失败。"},
        ]

    @staticmethod
    def _default_algo_config():
        return [
            {"id": "a_l1", "keywords": ["l1", "detection", "pcc"],
             "text": "L1 检测算法: PCC + KernelSHAP,分位数阈值 q=0.999,配置见"
                      "configs/selection.yaml,在 RE2-OB 上 AUC-PR 约 0.991。"},
            {"id": "a_l2", "keywords": ["l2", "attribution", "shap"],
             "text": "L2 归因: 基于 PCC 重建残差的 KernelSHAP 排名,coarse AC@1=0.789,"
                      "Avg@5=0.924,socket 类故障需结合 z-score 补充候选覆盖。"},
        ]

    @staticmethod
    def _default_topology_docs():
        # 和 build_knowledge_base.py 里的 SERVICE_TOPOLOGY 保持同一份内容,
        # 这里是关键词版占位,向量版上线后以那边为准
        topo = [
            ("frontend", ["checkoutservice", "cartservice", "productcatalogservice",
                          "recommendationservice", "adservice", "currencyservice", "shippingservice"],
             "对外 HTTP 网关,渲染网页,自动为所有用户生成 session"),
            ("checkoutservice", ["cartservice", "paymentservice", "shippingservice",
                                 "emailservice", "currencyservice"],
             "订单编排入口,取出购物车、生成订单,协调支付、物流、邮件三个下游服务"),
            ("cartservice", ["redis"], "购物车存取,数据落在 redis 里"),
            ("productcatalogservice", [], "商品目录查询,不依赖其他服务"),
            ("currencyservice", [], "货币换算,QPS 最高的服务之一"),
            ("paymentservice", [], "模拟信用卡扣款,返回交易号"),
            ("shippingservice", [], "根据购物车估算运费并模拟发货"),
            ("emailservice", [], "发送订单确认邮件(模拟)"),
            ("recommendationservice", ["productcatalogservice"], "根据购物车内容推荐其他商品"),
            ("adservice", [], "根据上下文关键词返回广告文案"),
            ("redis", [], "cartservice 的后端存储,本身不是业务服务"),
        ]
        return [
            {"id": f"topo_{service}", "keywords": [service],
             "text": f"[Service Topology: {service}] {role}。"
                      f"调用下游: {', '.join(calls) if calls else '(无)'}。"}
            for service, calls, role in topo
        ]

    # ── 检索接口 ─────────────────────────────────────
    def build_query(self, chain: List[str]) -> str:
        return " ".join(chain)

    def _split_chain(self, chain: List[str]) -> Dict[str, List[str]]:
        """
        把候选链拆成三份,分别喂给不同类型的检索,不再五类共用同一句拼接字符串:
          services   : 去重的 service 名,给 topology_docs 用
          metrics    : 去重的 metric 类型,给 feature_docs / playbook / algo_config 用
          full_names : 原始复合名 (不去重,保留 rank 顺序),给 alert_history 用
        """
        services, metrics, full_names = [], [], []
        seen_s, seen_m = set(), set()
        for name in chain:
            full_names.append(name)
            service, metric = _split_name(name)
            if service not in seen_s:
                services.append(service)
                seen_s.add(service)
            if metric and metric not in seen_m:
                metrics.append(metric)
                seen_m.add(metric)
        return {"services": services, "metrics": metrics, "full_names": full_names}

    def _retrieve_pool(self, pool: List[Dict], query_tokens: List[str],
                        top_k: int) -> List[Dict]:
        scored = []
        for doc in pool:
            score = _keyword_overlap_score(
                query_tokens, doc.get("keywords", []) + _tokenize(doc["text"])
            )
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": doc["text"], "similarity": round(score, 4), "id": doc["id"]}
            for score, doc in scored[:top_k]
        ]

    def retrieve_by_type(self, chain: List[str], top_k_each: int = 2,
                          current_case_id: Optional[str] = None) -> Dict[str, List[Dict]]:
        parts = self._split_chain(chain)
        metric_tokens   = _tokenize(" ".join(parts["metrics"]))
        service_tokens  = _tokenize(" ".join(parts["services"]))
        fullname_tokens = _tokenize(" ".join(parts["full_names"]))

        # leave-one-case-out: alert_history 必须排除当前诊断的 case,
        # 否则测试集正例会泄漏进检索结果
        history_pool = [
            d for d in self.alert_history
            if d.get("case_id") != current_case_id
        ]

        return {
            # feature/playbook/algo 按 metric 类型匹配,不关心是哪个 service
            "feature_docs":  self._retrieve_pool(self.feature_docs, metric_tokens, top_k_each),
            "playbooks":     self._retrieve_pool(self.playbooks, metric_tokens, top_k_each),
            "algo_config":   self._retrieve_pool(self.algo_config, metric_tokens, top_k_each),
            # topology 按 service 匹配,不关心具体 metric
            "topology_docs": self._retrieve_pool(self.topology_docs, service_tokens, top_k_each),
            # alert_history 用完整复合名整体比对历史候选链像不像
            "alert_history": self._retrieve_pool(history_pool, fullname_tokens, top_k_each),
        }


def stratified_sample_for_layer3(cases: List[Dict], n_per_fault_type: int = 2,
                                  seed: int = 42) -> List[Dict]:
    """
    按 fault type 分层抽样,用于 L3 小批量调试 / RAGAS 评估。
    cases: L2 输出 json 列表,每条含 _meta.case_id,格式形如 <service>_<faulttype>_<idx>
    """
    random.seed(seed)
    buckets: Dict[str, List[Dict]] = {}
    for c in cases:
        case_id = c.get("_meta", {}).get("case_id", "")
        parts = case_id.split("_")
        fault_type = parts[-2] if len(parts) >= 2 else "unknown"
        buckets.setdefault(fault_type, []).append(c)

    sample = []
    for items in buckets.values():
        random.shuffle(items)
        sample.extend(items[:n_per_fault_type])
    return sample
