"""
l2_adapter.py
=============
把 Layer2 输出的 json (如 checkoutservice_cpu_1.json) 转换成
RAGDiagnosticLayer.diagnose() 期望的 xai_report 结构。

Layer2 json 结构参考:
    {
      "_meta": {"case_id": ..., "ground_truth": {"service":..., "metric":...}},
      "attribution_fidelity": 0.9719,
      "candidates": [{"service":..., "service_shap":..., "metrics":[{name, metric, shap, direction}, ...]}, ...]
    }
"""

from typing import Dict, List
import json
import os


def _top_root_cause_chain(l2_output: Dict, top_k: int = 5) -> List[str]:
    """按 shap 值降序抽取 metric name,忽略 shap == 0 的候选。"""
    flat = []
    for cand in l2_output.get("candidates", []):
        for m in cand.get("metrics", []):
            if m.get("shap", 0) > 0:
                flat.append((m["shap"], m["name"]))
    flat.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in flat[:top_k]]


def _l2_candidates_ranked(l2_output: Dict, top_k: int = 10) -> List[Dict]:
    """
    L2 按 shap 值排序的候选 root cause 列表,供 L3 重排使用。
    每条: {rank, name, service, metric, shap}
    name 格式与 metric semantic 卡片对齐,如 "checkoutservice_cpu"

    top_k 是候选池大小,不等于评估窗口 (AC@1/3/5)。候选池应比评估窗口宽,
    否则真实 root cause 若排在 L2 的第 6 名以后,会在这一步被直接砍掉,
    LLM 之后再怎么重排也救不回来。默认 10 只是起点,建议在 train-fold 上
    对 AC@K_train 做 K=5/8/10/15/20 的 sweep,取边际收益开始趋平的点。
    """
    flat = []
    for cand in l2_output.get("candidates", []):
        for m in cand.get("metrics", []):
            if m.get("shap", 0) > 0:
                flat.append({
                    "name": m["name"],
                    "service": cand.get("service"),
                    "metric": m.get("metric"),
                    "shap": m["shap"],
                })
    flat.sort(key=lambda x: x["shap"], reverse=True)
    for i, item in enumerate(flat[:top_k], 1):
        item["rank"] = i
    return flat[:top_k]


def convert_l2_to_xai_report(l2_output: Dict, candidate_pool_size: int = 10,
                              query_chain_size: int = 5) -> Dict:
    """
    query_chain_size: 用于构建检索 query 的 metric 数量,保持小一些 (默认5),
        避免候选池变大后连带把检索 query 稀释成一长串关键词。
    candidate_pool_size: 喂给 LLM 重排的候选池大小,和评估窗口 (AC@1/3/5) 是两件事,
        建议 >= 评估窗口的 2 倍,具体数值应在 train-fold 上对 L2 的 AC@K_train 做
        K=5/8/10/15/20 sweep,取边际收益趋平的点,而不是固定写死。
    """
    meta = l2_output.get("_meta", {})
    chain = _top_root_cause_chain(l2_output, top_k=query_chain_size)
    fidelity = l2_output.get("attribution_fidelity")

    if fidelity is None:
        fidelity_str = "N/A"
    elif fidelity >= 0.8:
        fidelity_str = f"High (Fidelity: {fidelity}) -> Strong evidence"
    elif fidelity >= 0.5:
        fidelity_str = f"Medium (Fidelity: {fidelity}) -> Moderate evidence"
    else:
        fidelity_str = f"Low (Fidelity: {fidelity}) -> Evidence is weak"

    top_candidate = l2_output.get("candidates", [{}])[0]

    return {
        "timestamp": meta.get("case_id", "unknown"),
        "detection_result": "Anomaly",
        "model_score": top_candidate.get("service_shap", 0.0),
        "xai_analysis": {
            "root_cause_chain": chain,
            "fidelity_assessment": fidelity_str,
            "stability_assessment": "N/A (未接入 stability 指标)",
        },
        "xai_gateway_suggestion": (
            "Escalate: High-Confidence Anomaly" if (fidelity or 0) >= 0.8
            else "Monitor: Moderate-Confidence Anomaly"
        ),
        # L2 的候选排序,供 L3 重排,是 AC@k/Avg@k 评估的对比基准
        "l2_candidates": _l2_candidates_ranked(l2_output, top_k=candidate_pool_size),
        "ground_truth_meta": meta.get("ground_truth", {}),
        "case_id": meta.get("case_id"),
    }


def load_l2_dir(dir_path: str, candidate_pool_size: int = 10,
                 query_chain_size: int = 5) -> List[Dict]:
    """
    读取一个目录下所有 L2 输出 json (一个文件 = 一个 case),
    逐个转换成 xai_report 列表,供 diagnose_batch 使用。
    """
    reports = []
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{dir_path} 不是一个目录")

    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(dir_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            l2_output = json.load(f)
        reports.append(convert_l2_to_xai_report(
            l2_output,
            candidate_pool_size=candidate_pool_size,
            query_chain_size=query_chain_size,
        ))

    if not reports:
        print(f"  ⚠️  {dir_path} 下没有找到任何 .json 文件")
    else:
        print(f"  从 {dir_path} 加载了 {len(reports)} 条 L2 case")
    return reports
