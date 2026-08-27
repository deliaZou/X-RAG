"""
l2_adapter.py
=============
把 Layer2 输出的 json (如 checkoutservice_cpu_1.json) 转换成
RAGDiagnosticLayer.diagnose() 期望的 xai_report 结构。

Layer2 json 结构参考 (新格式):
    {
      "_meta": {"case_id": ..., "ground_truth": {"service":..., "fault":..., "metric":...}},
      "attribution_fidelity": 0.9719,
      "n_explained_points": 15,
      "service_list": [{"service": "checkoutservice", "service_shap": 314345.03}, ...],
      "metrics_list": [{"name": "checkoutservice_cpu", "shap": 299179.99, "direction": "up"}, ...]
    }
"""

from typing import Dict, List
import json
import os


def _top_root_cause_chain(l2_output: Dict, top_k: int = 5) -> List[str]:
    """
    按 shap 值降序抽取 metric name，用于构建检索 query。
    从 metrics_list 中取 top_k 个（忽略 shap == 0 的候选）。
    """
    metrics_list = l2_output.get("metrics_list", [])
    # 过滤 shap > 0 的指标，按 shap 降序排序
    positive_metrics = [m for m in metrics_list if m.get("shap", 0) > 0]
    sorted_metrics = sorted(positive_metrics, key=lambda x: x["shap"], reverse=True)
    return [m["name"] for m in sorted_metrics[:top_k]]


def _l2_candidates_ranked(l2_output: Dict, top_k: int = 10) -> List[Dict]:
    """
    L2 按 shap 值排序的候选 root cause 列表，供 L3 重排使用。
    每条: {rank, name, service, metric, shap, direction}
    name 格式与 metric semantic 卡片对齐，如 "checkoutservice_cpu"

    top_k 是候选池大小，不等于评估窗口 (AC@1/3/5)。候选池应比评估窗口宽，
    否则真实 root cause 若排在 L2 的第 6 名以后，会在这一步被直接砍掉，
    LLM 之后再怎么重排也救不回来。默认 10 只是起点，建议在 train-fold 上
    对 AC@K_train 做 K=5/8/10/15/20 的 sweep，取边际收益开始趋平的点。
    """
    metrics_list = l2_output.get("metrics_list", [])
    # 过滤 shap > 0 的指标，按 shap 降序排序
    positive_metrics = [m for m in metrics_list if m.get("shap", 0) > 0]
    sorted_metrics = sorted(positive_metrics, key=lambda x: x["shap"], reverse=True)
    
    l2_candidates = []
    for i, m in enumerate(sorted_metrics[:top_k], 1):
        name = m["name"]
        # 从 name 提取 service 和 metric_type
        if "_" in name:
            service, metric_type = name.rsplit("_", 1)
        else:
            service, metric_type = name, "unknown"
        
        l2_candidates.append({
            "rank": i,
            "name": name,
            "service": service,
            "metric": metric_type,
            "shap": m["shap"],
            "direction": m.get("direction", "unknown"),
        })
    return l2_candidates


def convert_l2_to_xai_report(l2_output: Dict, candidate_pool_size: int = 10,
                              query_chain_size: int = 5) -> Dict:
    """
    将 Layer2 新格式 (service_list + metrics_list) 转换为 xai_report
    
    query_chain_size: 用于构建检索 query 的 metric 数量，保持小一些 (默认5)，
        避免候选池变大后连带把检索 query 稀释成一长串关键词。
    candidate_pool_size: 喂给 LLM 重排的候选池大小，和评估窗口 (AC@1/3/5) 是两件事，
        建议 >= 评估窗口的 2 倍，具体数值应在 train-fold 上对 L2 的 AC@K_train 做
        K=5/8/10/15/20 sweep，取边际收益趋平的点，而不是固定写死。
    """
    meta = l2_output.get("_meta", {})
    ground_truth = meta.get("ground_truth", {})
    service_list = l2_output.get("service_list", [])
    metrics_list = l2_output.get("metrics_list", [])
    fidelity = l2_output.get("attribution_fidelity")
    n_points = l2_output.get("n_explained_points", 0)
    
    # 构建 root_cause_chain: 从 service_list 取前 query_chain_size 个服务（用于拓扑）
    service_chain = [item["service"] for item in service_list[:query_chain_size]]

    # 指标链（用于 playbook 检索）- 从 metrics_list 提取前几个指标名
    metric_chain = [m["name"] for m in metrics_list[:query_chain_size]]
    
    # 评估 fidelity
    if fidelity is None:
        fidelity_str = "N/A"
    elif fidelity >= 0.8:
        fidelity_str = f"High (Fidelity: {fidelity:.4f}) -> Strong evidence"
    elif fidelity >= 0.5:
        fidelity_str = f"Medium (Fidelity: {fidelity:.4f}) -> Moderate evidence"
    else:
        fidelity_str = f"Low (Fidelity: {fidelity:.4f}) -> Evidence is weak"
    
    # 获取 top service 的 service_shap 作为 model_score 的代理
    top_service = service_list[0] if service_list else {}
    model_score = top_service.get("service_shap", 0.0)
    
    # xai_gateway_suggestion 综合考虑 fidelity 和 n_explained_points
    if fidelity is not None and fidelity >= 0.8 and n_points >= 10:
        xai_gateway_suggestion = "Escalate: High-Confidence Anomaly"
    elif fidelity is not None and fidelity >= 0.5:
        xai_gateway_suggestion = "Monitor: Moderate-Confidence Anomaly"
    else:
        xai_gateway_suggestion = "Suppress LLM Report"
    
    return {
        "timestamp": meta.get("case_id", "unknown"),
        "model_score": model_score,
        "xai_analysis": {
            "service_chain": service_chain,
            "metric_chain": metric_chain,
            "fidelity_assessment": fidelity_str,
            "stability_assessment": "N/A (未接入 stability 指标)",
        },
        "xai_gateway_suggestion": xai_gateway_suggestion,
        # L2 的候选排序，供 L3 重排，是 AC@k/Avg@k 评估的对比基准
        "l2_candidates": _l2_candidates_ranked(l2_output, top_k=candidate_pool_size),
        "ground_truth_meta": {
            "service": ground_truth.get("service", ""),
            "metric": ground_truth.get("metric", ""),  # 完整名称如 "checkoutservice_cpu"
            "fault": ground_truth.get("fault", ""),
        },
        "case_id": meta.get("case_id"),
        # 保留一些元数据供调试
        "n_explained_points": n_points,
        "attribution_fidelity": fidelity,
    }


def load_l2_dir(dir_path: str, candidate_pool_size: int = 10,
                 query_chain_size: int = 5) -> List[Dict]:
    """
    读取一个目录下所有 L2 输出 json (一个文件 = 一个 case)，
    逐个转换成 xai_report 列表，供 diagnose_batch 使用。
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
        
        # 兼容旧格式：如果有 candidates 字段但没有 service_list，使用旧格式转换
        if "candidates" in l2_output and "service_list" not in l2_output:
            reports.append(_convert_legacy_l2_to_xai_report(
                l2_output,
                candidate_pool_size=candidate_pool_size,
                query_chain_size=query_chain_size,
            ))
        else:
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


def _convert_legacy_l2_to_xai_report(l2_output: Dict, candidate_pool_size: int = 10,
                                      query_chain_size: int = 5) -> Dict:
    """
    兼容旧格式 (candidates 嵌套结构) 的转换函数
    """
    meta = l2_output.get("_meta", {})
    ground_truth = meta.get("ground_truth", {})
    fidelity = l2_output.get("attribution_fidelity")
    
    # 从旧格式提取 chain
    flat = []
    for cand in l2_output.get("candidates", []):
        for m in cand.get("metrics", []):
            if m.get("shap", 0) > 0:
                flat.append((m["shap"], m["name"]))
    flat.sort(key=lambda x: x[0], reverse=True)
    chain = [name for _, name in flat[:query_chain_size]]
    
    # 从旧格式提取 l2_candidates
    l2_candidates = []
    for i, (shap, name) in enumerate(flat[:candidate_pool_size], 1):
        if "_" in name:
            service, metric_type = name.rsplit("_", 1)
        else:
            service, metric_type = name, "unknown"
        l2_candidates.append({
            "rank": i,
            "name": name,
            "service": service,
            "metric": metric_type,
            "shap": shap,
            "direction": "unknown",
        })
    
    top_candidate = l2_output.get("candidates", [{}])[0]
    
    if fidelity is None:
        fidelity_str = "N/A"
    elif fidelity >= 0.8:
        fidelity_str = f"High (Fidelity: {fidelity:.4f}) -> Strong evidence"
    elif fidelity >= 0.5:
        fidelity_str = f"Medium (Fidelity: {fidelity:.4f}) -> Moderate evidence"
    else:
        fidelity_str = f"Low (Fidelity: {fidelity:.4f}) -> Evidence is weak"
    
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
        "l2_candidates": l2_candidates,
        "ground_truth_meta": {
            "service": ground_truth.get("service", ""),
            "metric": ground_truth.get("metric", ""),
            "fault": ground_truth.get("fault", ""),
        },
        "case_id": meta.get("case_id"),
    }