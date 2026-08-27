"""
build_knowledge_base.py
========================
RE2-OB (Online Boutique) Layer3 知识库构建脚本

五类文档写入 ChromaDB:
  Type A feature   — Feature 语义卡  (7条, 按 metric 类型: cpu/mem/socket/workload/diskio/latency/error)
  Type B alert     — 历史告警        (从 train-fold 的 L2 case json 提取,打 case_id 供 leave-one-case-out)
  Type C playbook  — 运维 Playbook   (6条, 按 RE2-OB 的 6 个 fault type: CPU/Delay/Disk/Loss/Mem/Socket)
  Type D algo      — 算法配置        (从 configs/selection.yaml 解析,不调 LLM)
  Type E topology  — 服务拓扑        (11条, Online Boutique 服务职责与调用关系,人工改写,不调 LLM)

增量构建:
  --only feature,playbook   只重新生成/写入这两类,不影响集合里已有的其他类型
  不传 --only               全量重建 (清空整个集合重新生成)

用法示例:
  python build_knowledge_base.py --only algo,topology
  python build_knowledge_base.py --only alert --l2-dir ./data/re2ob/train_fold
  python build_knowledge_base.py                      # 全量重建 (需要 ANTHROPIC_API_KEY)

依赖:
  pip install anthropic chromadb sentence-transformers pyyaml
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict, Optional

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "re2ob_knowledge"

# 用脚本所在目录当基准,不用 cwd 相对路径。否则从别的目录(比如跑
# rag_llm_layer.py 时的 cwd 不是这里)导入本模块,相对路径会解析到一个
# 不存在的空目录,PersistentClient 不会报路径错误,只会在 get_collection()
# 时报 NotFoundError,看起来像"数据库不存在",其实是路径解析错了。
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_CACHE_PATH = os.path.join(_MODULE_DIR, "kb_docs_cache.json")


def get_chroma_path(embed_model: str) -> str:
    """
    按 embedding 模型区分向量库路径,避免换模型时互相覆盖,
    也让"固定知识库,换 embedding 模型对比"这个受控实验能同时保留多份结果。
    可用环境变量 XRAG_KB_ROOT 覆盖基准目录。
    """
    slug = embed_model.replace("/", "_").replace(":", "_")
    base = os.getenv("XRAG_KB_ROOT", _MODULE_DIR)
    return os.path.join(base, f"chroma_re2ob_kb__{slug}")

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

# 增量重建时用于精确删除旧文档的 where 条件,playbook/playbook_ext 共用检索 tag
# 但归属不同的 cache key,删除时必须用 source_category 再过滤一层,
# 否则重建 playbook 会把 playbook_ext 的文档也一起清掉 (反之亦然)
DELETE_WHERE = {
    "feature":     {"type": TYPE_TAG["feature"]},
    "playbook":    {"$and": [{"type": TYPE_TAG["playbook"]}, {"source_category": "llm_generated"}]},
    "playbook_ext": {"$and": [{"type": TYPE_TAG["playbook_ext"]}, {"source_category": "external_runbook"}]},
    "alert":       {"type": TYPE_TAG["alert"]},
    "algo":        {"type": TYPE_TAG["algo"]},
    "topology":    {"type": TYPE_TAG["topology"]},
}

# ─────────────────────────────────────────────
# RE2-OB 领域知识 (静态定义,不依赖数据集本身)
# ─────────────────────────────────────────────

# 8 个 metric 类型,语义在所有 service 上通用,不按 service 展开生成
METRIC_TYPES = ["cpu", "mem", "socket", "workload", "diskio", "latency", "error", "delay"]

# 6 个 fault type,对应 RCAEval RE2-OB 的故障注入类型
FAULT_TYPES = {
    "CPU":    {"metrics": ["cpu"],
               "pattern": "target service 的 cpu 指标持续显著升高,workload 未必同步升高,"
                          "是判断异常还是正常负载增长的关键区分点"},
    "Delay":  {"metrics": ["latency", "delay"],
               "pattern": "target 及下游 service 的 latency 指标升高,自身 cpu/mem 未必异常,"
                          "容易被 SHAP 误判为高权重的 root cause 而实际是传导结果"},
    "Disk":   {"metrics": ["diskio"],
               "pattern": "target service 的 diskio 指标异常,可能伴随 latency 上升"},
    "Loss":   {"metrics": ["error", "loss"],
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


def _extract_services_and_metrics_old(chain: List[str]) -> tuple[List[str], List[str]]:
    """
    从候选链中拆出服务名和指标名

    例如: ["checkoutservice_latency", "emailservice_latency", "frontend_socket"]
    返回: (["checkoutservice", "emailservice", "frontend"], ["latency", "latency", "socket"])

    按候选链原有顺序去重返回
    """
    services_out = []
    metrics_out = []
    services_seen = set()
    metrics_seen = set()

    for name in chain:
        if "_" not in name:
            continue

        # 拆出 service 和 metric
        # 注意: service 可能包含下划线，但微服务名通常没有
        # 用 rsplit 从右边分割，只分割一次
        service, metric = name.rsplit("_", 1)

        # 去重添加服务
        if service not in services_seen:
            services_out.append(service)
            services_seen.add(service)

        # 去重添加指标
        if metric not in metrics_seen:
            metrics_out.append(metric)
            metrics_seen.add(metric)

    return services_out, metrics_out


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

# Online Boutique 服务拓扑,依据官方 README 改写 (转述,非原文摘抄) 废弃
SERVICE_TOPOLOGY_abandom = [
    {"service": "frontend", "language": "Go",
     "role": "对外 HTTP 网关,渲染网页,自动为所有用户生成 session,不需要登录",
     "calls": ["checkoutservice", "cartservice", "productcatalogservice",
               "recommendationservice", "adservice", "currencyservice", "shippingservice"]},
    {"service": "checkoutservice", "language": "Go",
     "role": "订单编排入口,取出用户购物车、生成订单,并协调支付、物流、邮件三个下游服务",
     "calls": ["cartservice", "paymentservice", "shippingservice", "emailservice", "currencyservice"]},
    {"service": "cartservice", "language": "C#",
     "role": "购物车存取,数据落在 redis 里",
     "calls": ["redis"]},
    {"service": "productcatalogservice", "language": "Go",
     "role": "商品目录查询与搜索,数据来自本地 JSON 文件,不依赖其他服务",
     "calls": []},
    {"service": "currencyservice", "language": "Node.js",
     "role": "货币换算,是整个系统里 QPS 最高的服务之一",
     "calls": []},
    {"service": "paymentservice", "language": "Node.js",
     "role": "模拟信用卡扣款,返回交易号",
     "calls": []},
    {"service": "shippingservice", "language": "Go",
     "role": "根据购物车内容估算运费并模拟发货",
     "calls": []},
    {"service": "emailservice", "language": "Python",
     "role": "发送订单确认邮件 (模拟)",
     "calls": []},
    {"service": "recommendationservice", "language": "Python",
     "role": "根据购物车内容推荐其他商品",
     "calls": ["productcatalogservice"]},
    {"service": "adservice", "language": "Java",
     "role": "根据上下文关键词返回广告文案",
     "calls": []},
    {"service": "redis", "language": "-",
     "role": "cartservice 的后端存储,本身不是业务逻辑服务",
     "calls": []},
]

# Online Boutique 服务拓扑新 来源https://github.com/GoogleCloudPlatform/microservices-demo
SERVICE_TOPOLOGY = {
    "frontend": {
        "role": "entrypoint",
        "downstream": [
            "checkoutservice", "adservice", "cartservice",
            "recommendationservice", "productcatalogservice", "currencyservice"
        ],
        "upstream": [],
        "diagnosis_hint": "Entrypoint for all traffic. If anomalous, all downstream services are affected."
    },
    "checkoutservice": {
        "role": "orchestrator",
        "downstream": ["emailservice", "shippingservice", "paymentservice"],
        "upstream": ["frontend"],
        "diagnosis_hint": "Orchestrator. If anomalous, email/shipping/payment services are affected."
    },
    "cartservice": {
        "role": "stateful",
        "downstream": ["redis"],
        "upstream": ["frontend"],
        "diagnosis_hint": "Depends on Redis. Redis failure causes cartservice anomalies."
    },
    "emailservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["checkoutservice"],
        "diagnosis_hint": "Leaf node. Anomaly is usually self-contained, does not affect other services."
    },
    "shippingservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["checkoutservice"],
        "diagnosis_hint": "Leaf node. Anomaly is usually self-contained, does not affect other services."
    },
    "paymentservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["checkoutservice"],
        "diagnosis_hint": "Leaf node. Anomaly is usually self-contained, does not affect other services."
    },
    "currencyservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["frontend"],
        "diagnosis_hint": "High QPS service, prone to become a performance bottleneck."
    },
    "adservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["frontend"],
        "diagnosis_hint": "Leaf node. Anomaly is usually self-contained."
    },
    "recommendationservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["frontend"],
        "diagnosis_hint": "Leaf node. Anomaly is usually self-contained."
    },
    "productcatalogservice": {
        "role": "leaf",
        "downstream": [],
        "upstream": ["frontend"],
        "diagnosis_hint": "Leaf node. Anomaly is usually self-contained."
    },
    "redis": {
        "role": "infrastructure",
        "downstream": [],
        "upstream": ["cartservice"],
        "diagnosis_hint": "Cache service. If anomalous, cartservice is affected."
    }
}


def _build_topology_prompt_back(services: list[str]) -> str:
    """
    根据受影响的服务列表，生成拓扑上下文
    """
    lines = ["Service Topology:"]

    for service in services:
        if service not in SERVICE_TOPOLOGY:
            continue
        info = SERVICE_TOPOLOGY[service]

        if info["downstream"]:
            lines.append(f"  - {service} calls: {', '.join(info['downstream'])}")
        if info["upstream"]:
            lines.append(f"  - {service} is called by: {', '.join(info['upstream'])}")
        if info["diagnosis_hint"]:
            lines.append(f"    Hint: {info['diagnosis_hint']}")

    return "\n".join(lines)

def _build_topology_prompt(chain: list[str]) -> str:
    """根据 root_cause_chain 构建详细的拓扑描述"""
    if not chain:
        return "  (no service chain available)"

    lines = ["Service Topology (based on root cause chain):"]
    lines.append("")

    # 构建涉及的服务的拓扑信息
    for i, svc in enumerate(chain, 1):
        info = SERVICE_TOPOLOGY.get(svc, {})
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
        first_info = SERVICE_TOPOLOGY.get(chain[0], {})
        if first_info.get("role") == "leaf":
            lines.append(f"    ⚠️ {chain[0]} is a LEAF node. Its anomaly should NOT propagate to downstream.")
            lines.append(f"    If downstream services show anomalies, they may have separate root causes.")
        elif first_info.get("role") in ["entrypoint", "orchestrator"]:
            lines.append(f"    {chain[0]} is an {first_info.get('role')}. Anomaly may propagate to downstream.")

    return "\n".join(lines)

# ─────────────────────────────────────────────
# LLM 调用工具 (只用于 Type A / C 生成)
# ─────────────────────────────────────────────
def _llm_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("未检测到环境变量 LLM_API_KEY,检查 .env 文件")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client: OpenAI, prompt: str, max_tokens: int = 400) -> str:
    model = os.getenv("LLM_MODEL_ID")
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Type A — Feature 语义卡 (按 metric 类型,不按 service 展开)
# ─────────────────────────────────────────────
FEATURE_DOC_PROMPT = """You are an SRE expert on a Kubernetes-based microservice system
(Online Boutique / RCAEval RE2-OB benchmark).

Write a concise diagnostic reference for the metric type: **{metric_type}**
(this metric exists per-service, e.g. "checkoutservice_{metric_type}").

Cover exactly these three points in plain text, no markdown headers or bullets:
1. What this metric measures at the service/pod level.
2. Whether it is typically a CAUSE metric (root cause of a fault) or a RESULT
   metric (a symptom that propagates from an upstream cause) — be explicit,
   e.g. latency/error are usually results, cpu/mem/socket/diskio are usually causes.
3. Which other metric types on the SAME service or DOWNSTREAM services tend to
   move together with it, and in what direction.

Keep it under 100 words. Write in English.
"""


def generate_feature_docs() -> List[Dict]:
    client = _llm_client()
    docs = []
    for i, metric_type in enumerate(METRIC_TYPES, 1):
        print(f"  [{i}/{len(METRIC_TYPES)}] 生成 feature 语义卡: {metric_type} ...", end=" ")
        text = call_llm(client, FEATURE_DOC_PROMPT.format(metric_type=metric_type))
        docs.append({
            "id": f"feat_{metric_type}",
            "type": TYPE_TAG["feature"],
            "metric": metric_type,
            "text": f"[Feature: {metric_type}]\n{text}",
        })
        print("✓")
        time.sleep(0.3)
    return docs


# ─────────────────────────────────────────────
# Type C — Playbook (按 RE2-OB 6 个 fault type)
# ─────────────────────────────────────────────
PLAYBOOK_PROMPT = """You are a senior SRE writing an internal runbook for a
Kubernetes-based microservice system (Online Boutique / RCAEval RE2-OB benchmark).

Fault type: **{fault_type}**
Triggering metric types: {metrics}
Observed pattern: {pattern}

Write a concise playbook entry covering:
1. Likely root causes (2-3 most probable, ranked)
2. Immediate triage steps (what to check first, at pod/service level)
3. Mitigation actions (ordered, with conditions)
4. A note on how to distinguish this fault type from a metric that is merely
   a downstream RESULT rather than the actual cause.

Keep it under 180 words. Plain text, no markdown symbols.
"""


def generate_playbook_docs() -> List[Dict]:
    client = _llm_client()
    docs = []
    for i, (fault_type, info) in enumerate(FAULT_TYPES.items(), 1):
        print(f"  [{i}/{len(FAULT_TYPES)}] 生成 playbook: {fault_type} ...", end=" ")
        prompt = PLAYBOOK_PROMPT.format(
            fault_type=fault_type,
            metrics=", ".join(info["metrics"]),
            pattern=info["pattern"],
        )
        text = call_llm(client, prompt, max_tokens=350)
        docs.append({
            "id": f"play_{fault_type}",
            "type": TYPE_TAG["playbook"],
            "source_category": "llm_generated",
            "fault_type": fault_type,
            "triggers": info["metrics"],
            "text": f"[Playbook: {fault_type} fault]\nTrigger metrics: {', '.join(info['metrics'])}\n{text}",
        })
        print("✓")
        time.sleep(0.3)
    return docs


# ─────────────────────────────────────────────
# Type C-ext — 外部 runbook (人工审核过的 manifest,直接用本机 local_path 读取)
# ─────────────────────────────────────────────
def _clean_markdown(raw: str, max_chars: int = 1500) -> str:
    """去掉图片/badge/<details>折叠标签等噪音,保留标题和正文结构,不做语义改写。"""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", raw)  # 图片
    text = re.sub(r"<details>|</details>|<summary>.*?</summary>", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def ingest_curated_runbooks(manifest_path: str) -> List[Dict]:
    """
    读取人工审核过的外部 runbook manifest (is_relevant/mapped_fault_type/trigger_metrics
    那份清单),只收 is_relevant==true 的条目,直接用 local_path 读取本机文件内容,
    不做路径可移植性处理。
    """
    if not os.path.exists(manifest_path):
        print(f"  ⚠️  {manifest_path} 不存在,跳过外部 runbook")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = []
    skipped = 0
    for entry in manifest:
        if not entry.get("is_relevant"):
            skipped += 1
            continue

        local_path = entry.get("local_path", "")
        if not os.path.exists(local_path):
            print(f"  ⚠️  文件不存在,跳过: {local_path}")
            continue

        with open(local_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        text = _clean_markdown(raw_text)

        fault_type = entry.get("mapped_fault_type", "unknown")
        triggers = entry.get("trigger_metrics", [])
        doc_id = f"playbook_ext_{entry['file_name'].replace('.md', '')}"

        docs.append({
            "id": doc_id,
            "type": TYPE_TAG["playbook_ext"],
            "source_category": "external_runbook",
            "fault_type": fault_type,
            "triggers": triggers,
            "source": entry.get("sub_url", ""),
            "source_url": entry.get("github_url", ""),
            "text": f"[Playbook: {fault_type} fault | source: prometheus-operator/runbooks]\n{text}",
        })

    print(f"  已收录 {len(docs)} 条外部 runbook (跳过 {skipped} 条 is_relevant=false)")
    return docs


# ─────────────────────────────────────────────
# Type B — 历史告警 (从 train-fold 的 L2 case json 提取)
# ─────────────────────────────────────────────
def _top_root_cause_from_l2(l2_output: Dict, top_k: int = 3) -> List[str]:
    flat = []
    for cand in l2_output.get("candidates", []):
        for m in cand.get("metrics", []):
            if m.get("shap", 0) > 0:
                flat.append((m["shap"], m["name"]))
    flat.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in flat[:top_k]]


def load_alert_history_from_l2_dir(l2_dir: str) -> List[Dict]:
    """
    读取一个目录下所有 L2 输出 json (train-fold),转成 alert_history 文档。
    每条打上 case_id,供检索时做 leave-one-case-out 过滤。

    调用方必须保证传入的是 train-fold 目录,不能包含当前测试 case,
    这是防泄漏的第一层;检索时按 current_case_id 排除是第二层。
    """
    docs = []
    if not os.path.isdir(l2_dir):
        print(f"  ⚠️  {l2_dir} 不是目录,跳过历史告警加载")
        return docs

    for fname in sorted(os.listdir(l2_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(l2_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            l2_output = json.load(f)

        meta = l2_output.get("_meta", {})
        case_id = meta.get("case_id", fname.replace(".json", ""))
        gt = meta.get("ground_truth", {})
        fidelity = l2_output.get("attribution_fidelity")
        chain = _top_root_cause_from_l2(l2_output)

        text = (
            f"[Historical Alert | case_id={case_id}]\n"
            f"Ground truth: service={gt.get('service')}, metric={gt.get('metric')}\n"
            f"Top attributed metrics (SHAP desc): {', '.join(chain)}\n"
            f"Attribution fidelity: {fidelity}"
        )
        docs.append({
            "id": f"alert_{case_id}",
            "type": TYPE_TAG["alert"],
            "case_id": case_id,
            "text": text,
        })

    print(f"  已加载 {len(docs)} 条历史告警 (train-fold) ✓")
    return docs


# ─────────────────────────────────────────────
# Type D — 算法配置 (从 configs/selection.yaml 解析,不调 LLM)
# ─────────────────────────────────────────────
def generate_algo_config_docs(yaml_path: Optional[str]) -> List[Dict]:
    config_text = None
    if yaml_path and os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            config_text = json.dumps(cfg, ensure_ascii=False, indent=2)
        except ImportError:
            print("  ⚠️  未安装 pyyaml,回退到默认算法说明")
        except Exception as e:
            print(f"  ⚠️  解析 {yaml_path} 失败: {e},回退到默认算法说明")

    if config_text:
        docs = [{
            "id": "algo_selection_yaml",
            "type": TYPE_TAG["algo"],
            "text": f"[Algorithm Config | source: {yaml_path}]\n{config_text}",
        }]
    else:
        docs = [
            {"id": "algo_l1", "type": TYPE_TAG["algo"],
             "text": "[Algorithm Config: L1]\nDetection: PCC (PCA-based reconstruction) "
                      "+ KernelSHAP, quantile threshold q=0.999. "
                      "AUC-PR ~0.991 on RE2-OB."},
            {"id": "algo_l2", "type": TYPE_TAG["algo"],
             "text": "[Algorithm Config: L2]\nAttribution: KernelSHAP on PCC reconstruction "
                      "residuals, coarse AC@1=0.789, Avg@5=0.924. Socket faults need z-score "
                      "union to raise candidate coverage."},
        ]
    print(f"  已生成 {len(docs)} 条算法配置文档 ✓")
    return docs


# ─────────────────────────────────────────────
# Type E — 服务拓扑 (静态改写,不调 LLM)
# ─────────────────────────────────────────────
def generate_topology_docs(services: list[str]) -> List[Dict]:
    # docs = []
    # for s in SERVICE_TOPOLOGY:
    #     calls_str = ", ".join(s["calls"]) if s["calls"] else "(no downstream calls)"
    #     text = (
    #         f"[Service Topology: {s['service']}]\n"
    #         f"Language: {s['language']}. Role: {s['role']}. "
    #         f"Calls downstream: {calls_str}."
    #     )
    #     docs.append({
    #         "id": f"topo_{s['service']}",
    #         "type": TYPE_TAG["topology"],
    #         "service": s["service"],
    #         "calls": s["calls"],
    #         "text": text,
    #     })
    # print(f"  已生成 {len(docs)} 条服务拓扑文档 ✓")
    # return docs
    lines = ["Service Topology:"]

    for service in services:
        if service not in SERVICE_TOPOLOGY:
            continue
        info = SERVICE_TOPOLOGY[service]

        if info["downstream"]:
            lines.append(f"  - {service} calls: {', '.join(info['downstream'])}")
        if info["upstream"]:
            lines.append(f"  - {service} is called by: {', '.join(info['upstream'])}")
        if info["diagnosis_hint"]:
            lines.append(f"    Hint: {info['diagnosis_hint']}")

    return "\n".join(lines)

# ─────────────────────────────────────────────
# 缓存读写 (避免每次调试都重新调 LLM)
# ─────────────────────────────────────────────
def load_docs_cache() -> Dict[str, List[Dict]]:
    if os.path.exists(DOCS_CACHE_PATH):
        with open(DOCS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {t: [] for t in ALL_TYPES}


def save_docs_cache(cache: Dict[str, List[Dict]]):
    with open(DOCS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  文档缓存已保存到 {DOCS_CACHE_PATH}")


# ─────────────────────────────────────────────
# 向量化 + 写入 ChromaDB (支持增量: 只替换指定 type 的文档)
# ─────────────────────────────────────────────
def upsert_chroma(all_docs_by_type: Dict[str, List[Dict]], only_types: Optional[List[str]] = None,
                   embed_model: str = EMBED_MODEL):
    chroma_path = get_chroma_path(embed_model)
    print(f"\n[ChromaDB] 打开持久化存储: {chroma_path} (embed_model={embed_model})")
    db = chromadb.PersistentClient(path=chroma_path)

    if only_types is None:
        try:
            db.delete_collection(COLLECTION_NAME)
            print("  全量重建: 旧集合已清除")
        except Exception:
            pass
        collection = db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        types_to_write = ALL_TYPES
    else:
        collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        types_to_write = only_types
        # 增量模式: 先删掉这几类旧文档,再重新写入,避免重复 id 冲突
        # playbook/playbook_ext 共用同一个检索 tag,delete 必须再叠加 source_category
        # 过滤,否则重建 playbook 会把 playbook_ext 的文档一并清掉
        for t in types_to_write:
            try:
                collection.delete(where=DELETE_WHERE[t])
                print(f"  增量模式: 已清除旧的 {t} 类文档")
            except Exception:
                pass

    print(f"[Embedding] 加载模型: {embed_model} ...")
    embedder = SentenceTransformer(embed_model)

    docs_to_add = []
    for t in types_to_write:
        docs_to_add.extend(all_docs_by_type.get(t, []))

    if not docs_to_add:
        print("  没有文档需要写入")
        return

    texts = [d["text"] for d in docs_to_add]
    ids = [d["id"] for d in docs_to_add]
    metadatas = [
        {k: (", ".join(v) if isinstance(v, list) else str(v))
         for k, v in d.items() if k not in ("text", "id")}
        for d in docs_to_add
    ]

    print(f"[Embedding] 正在嵌入 {len(texts)} 条文档...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    BATCH = 50
    for start in range(0, len(texts), BATCH):
        end = min(start + BATCH, len(texts))
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"[ChromaDB] 已写入 {len(texts)} 条文档 ✓")


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


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                         help=f"逗号分隔,只重新生成/写入这些类型,可选: {','.join(ALL_TYPES)}。"
                              "不传则全量重建。")
    parser.add_argument("--l2-dir", default="./data/re2ob/train_fold",
                         help="Type B 用,train-fold 的 L2 输出 json 所在目录")
    parser.add_argument("--config-yaml", default="./configs/selection.yaml",
                         help="Type D 用,算法配置 yaml 路径")
    parser.add_argument("--runbook-manifest", default="./runbook_manifest.json",
                         help="Type C-ext 用,人工审核过的外部 runbook 清单 (is_relevant/"
                              "mapped_fault_type/trigger_metrics/local_path)")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                         help="embedding 模型名,不同模型会写入不同的 Chroma 路径 "
                              "(./chroma_re2ob_kb__<model>),互不覆盖,方便做模型对比实验")
    args = parser.parse_args()

    only_types = args.only.split(",") if args.only else None
    if only_types:
        invalid = [t for t in only_types if t not in ALL_TYPES]
        if invalid:
            raise ValueError(f"未知类型: {invalid},可选: {ALL_TYPES}")

    types_to_build = only_types or ALL_TYPES
    print("=" * 55)
    print(f"RE2-OB 知识库构建: {'全量重建' if only_types is None else f'增量 ({types_to_build})'}")
    print("=" * 55)

    cache = load_docs_cache()

    # if "feature" in types_to_build:
    #     print("\n[Type A] 生成 feature 语义卡...")
    #     cache["feature"] = generate_feature_docs()
    #
    # if "playbook" in types_to_build:
    #     print("\n[Type C] 生成 playbook...")
    #     cache["playbook"] = generate_playbook_docs()

    if "playbook_ext" in types_to_build:
        print("\n[Type C-ext] 收录外部 runbook...")
        cache["playbook_ext"] = ingest_curated_runbooks(args.runbook_manifest)

    # if "alert" in types_to_build:
    #     print("\n[Type B] 加载历史告警 (train-fold)...")
    #     cache["alert"] = load_alert_history_from_l2_dir(args.l2_dir)
    #
    # if "algo" in types_to_build:
    #     print("\n[Type D] 生成算法配置文档...")
    #     cache["algo"] = generate_algo_config_docs(args.config_yaml)
    #
    # if "topology" in types_to_build:
    #     print("\n[Type E] 生成服务拓扑文档...")
        cache["topology"] = generate_topology_docs()

    save_docs_cache(cache)
    upsert_chroma(cache, only_types=only_types, embed_model=args.embed_model)

    total = sum(len(cache.get(t, [])) for t in ALL_TYPES)
    print("\n" + "=" * 55)
    print(f"✅ 知识库构建完成,当前缓存共 {total} 条文档 (含未在本次重建的旧类型)")
    print(f"   Chroma 路径: {get_chroma_path(args.embed_model)}")
    print(f"   Embedding 模型: {args.embed_model}")
    print("=" * 55)
    print("\n使用示例:")
    print("  from build_knowledge_base import KnowledgeBase")
    print(f'  kb = KnowledgeBase(embed_model="{args.embed_model}")')
    print('  ctx = kb.retrieve_by_type(["checkoutservice_cpu"], current_case_id="checkoutservice_cpu_1")')

def main2():
    kb = KnowledgeBase(embed_model="all-MiniLM-L6-v2")
    ctx = kb.retrieve_by_type(["a_diskio"])
    print(ctx)


if __name__ == "__main__":
    # main()
    main2()
