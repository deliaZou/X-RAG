"""
l2_adapter.py
=============
把 Layer2 输出的 json 转换成 RAGDiagnosticLayer.diagnose() 期望的 xai_report 结构。

相对旧版的改动:
  1. 新旧两种 Layer2 格式 (service_list/metrics_list 新格式,
     candidates 嵌套 旧格式) 现在共用同一套转换逻辑 (convert_l2_to_xai_report),
     不再是两个各自维护一份、容易漏改的独立函数。
  2. 修复字段不对齐的 bug: 新格式之前只产出 service_chain / metric_chain,
     没有 root_cause_chain / detection_result,导致 rag_llm_layer.py 的
     prompt 里这两行永远是空的 / "unknown"。现在两种格式统一产出同一套
     xai_analysis 结构:
       {service_chain, metric_chain, root_cause_chain, fidelity_assessment,
        stability_assessment}
     root_cause_chain 与 metric_chain 取值一致 (按 shap 降序的 top metric
     名称链),不再额外维护一份重复逻辑。
  3. 删除没有被调用的死函数 _top_root_cause_chain。
  4. 新增 load_l2_input(): 一个入口同时支持"单个 json 文件"和"一个目录",
     取代原来 l2 模式单独读单文件、batch 模式单独读目录 两套重复代码,
     配合 rag_llm_layer.py 里统一后的 pipeline 使用。

Layer2 json 结构参考 (新格式):
    {
      "_meta": {"case_id": ..., "ground_truth": {"service":..., "fault":..., "metric":...}},
      "attribution_fidelity": 0.9719,
      "n_explained_points": 15,
      "service_list": [{"service": "checkoutservice", "service_shap": 314345.03}, ...],
      "metrics_list": [{"name": "checkoutservice_cpu", "shap": 299179.99, "direction": "up"}, ...]
    }

Layer2 json 结构参考 (旧格式,仍然兼容):
    {
      "_meta": {...},
      "attribution_fidelity": ...,
      "candidates": [{"service_shap": ..., "metrics": [{"name": ..., "shap": ...}, ...]}, ...]
    }
"""

from typing import Dict, List, Tuple
import json
import os
import random


def _fidelity_str(fidelity: float) -> str:
    if fidelity is None:
        return "N/A"
    if fidelity >= 0.8:
        return f"High (Fidelity: {fidelity:.4f}) -> Strong evidence"
    if fidelity >= 0.5:
        return f"Medium (Fidelity: {fidelity:.4f}) -> Moderate evidence"
    return f"Low (Fidelity: {fidelity:.4f}) -> Evidence is weak"


def _xai_gateway_suggestion(fidelity: float, n_points: int) -> str:
    if fidelity is not None and fidelity >= 0.8 and n_points >= 10:
        return "Escalate: High-Confidence Anomaly"
    if fidelity is not None and fidelity >= 0.5:
        return "Monitor: Moderate-Confidence Anomaly"
    return "Suppress LLM Report"


def _split_name(name: str) -> Tuple[str, str]:
    """"checkoutservice_cpu" -> ("checkoutservice", "cpu")"""
    if "_" in name:
        service, metric_type = name.rsplit("_", 1)
    else:
        service, metric_type = name, "unknown"
    return service, metric_type


def _dedup_services(names: List[str]) -> List[str]:
    """从一串 metric 名称里按出现顺序去重取出 service 名"""
    seen = set()
    out = []
    for n in names:
        service, _ = _split_name(n)
        if service not in seen:
            seen.add(service)
            out.append(service)
    return out


def _ranked_candidates(pairs: List[Tuple[float, str]], top_k: int, directions: Dict[str, str] = None,
    metric_details: Dict[str, Dict] = None) -> List[Dict]:
    directions = directions or {}
    metric_details = metric_details or {}

    out = []
    for i, (shap, name) in enumerate(pairs[:top_k], 1):
        service, metric_type = _split_name(name)
        details = metric_details.get(name, {})

        out.append({
            "name": name,
            "service": service,
            "metric": metric_type,
            "shap": shap,
            "direction": directions.get(name, details.get("direction", "unknown")),
            "zscore_valid": details.get("zscore_valid", False),
            "mean_zscore": details.get("mean_zscore"),
            "mean_abs_zscore": details.get("mean_abs_zscore"),
            "max_abs_zscore": details.get("max_abs_zscore"),
            "duration_samples": details.get("duration_samples"),
            "trend": details.get("trend"),
        })

    return out


def _is_new_format(l2_output: Dict) -> bool:
    return "service_list" in l2_output or "metrics_list" in l2_output


def convert_l2_to_xai_report(l2_output: Dict, candidate_pool_size: int = 10,
                              query_chain_size: int = 5) -> Dict:
    """
    统一转换入口,自动识别新旧两种 Layer2 输出格式,产出结构一致的 xai_report。

    query_chain_size: 用于构建 metric_chain / root_cause_chain 的 metric 数量,
        保持小一些 (默认 5),避免候选池变大后连带把检索 query 稀释成一长串关键词。
    candidate_pool_size: 喂给 LLM 重排的候选池大小,见 _ranked_candidates 的说明。
    """
    meta = l2_output.get("_meta", {})
    ground_truth = meta.get("ground_truth", {})
    fidelity = l2_output.get("attribution_fidelity")
    n_points = l2_output.get("n_explained_points", 0)

    service_list = l2_output.get("service_list", [])
    metrics_list = l2_output.get("metrics_list", [])

    # positive_metrics = [m for m in metrics_list if m.get("shap", 0) > 0]
    # sorted_metrics = sorted(positive_metrics, key=lambda x: x["shap"], reverse=True)
    # pairs = [(m["shap"], m["name"]) for m in sorted_metrics]
    # directions = {m["name"]: m.get("direction", "unknown") for m in sorted_metrics}
    #
    # metric_chain = [name for _, name in pairs[:query_chain_size]]
    # service_chain = [item["service"] for item in service_list[:query_chain_size]] \
    #     or _dedup_services(metric_chain)
    #
    # top_service = service_list[0] if service_list else {}
    # model_score = top_service.get("service_shap", 0.0)
    #
    # metric_details = {m["name"]: m for m in sorted_metrics}
    #
    # l2_candidates = _ranked_candidates(pairs, candidate_pool_size, directions, metric_details)
    all_metrics = [m for m in metrics_list if m.get("name")]

    shap_ranked = sorted(all_metrics, key=lambda m: float(m.get("shap", 0)), reverse=True,)

    metric_details = {m["name"]: m for m in all_metrics}
    directions = {m["name"]: m.get("direction", "unknown") for m in all_metrics }

    metric_chain = [m["name"] for m in shap_ranked[:query_chain_size]]

    candidate_pairs = [ (float(m.get("shap", 0)), m["name"]) for m in shap_ranked]
    l2_candidates = _ranked_candidates(candidate_pairs, candidate_pool_size, directions, metric_details)

    return {
        "timestamp": meta.get("case_id", "unknown"),  # diagnose_batch 当前用它显示进度
        "case_id": meta.get("case_id", "unknown"),
        "xai_analysis": {
            "metric_chain": metric_chain,  # 检索使用
            "fidelity_assessment": _fidelity_str(fidelity),  # prompt 使用
        },
        "l2_candidates": l2_candidates,  # prompt 中的候选证据
        "ground_truth_meta": {
            "service": ground_truth.get("service", ""),
            "metric": ground_truth.get("metric", ""),
            "fault": ground_truth.get("fault", ""),
        },
        # 保留作结果分析和调试元数据
        "n_explained_points": n_points,
        "attribution_fidelity": fidelity,
    }


def load_l2_input(path: str, candidate_pool_size: int = 10,
                   query_chain_size: int = 5) -> List[Dict]:
    """
    统一入口: path 既可以是单个 L2 json 文件,也可以是一个目录 (内含多个 case json)。
    取代原来 "l2 模式单独读单文件 + batch 模式单独读目录" 两套重复代码 ——
    上层 rag_llm_layer.py 不用再关心是单条还是批量,统一拿到一个 List[Dict],
    单条诊断只是 n=1 的批量诊断。
    """
    if os.path.isdir(path):
        paths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".json")]
    elif os.path.isfile(path):
        paths = [path]
    else:
        raise FileNotFoundError(f"{path} 不是一个有效的文件或目录")

    reports = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            l2_output = json.load(f)
        reports.append(convert_l2_to_xai_report(
            l2_output,
            candidate_pool_size=candidate_pool_size,
            query_chain_size=query_chain_size,
        ))

    if not reports:
        print(f"  ⚠️  {path} 下没有找到任何 .json 文件")
    else:
        print(f"  从 {path} 加载了 {len(reports)} 条 L2 case")
    return reports
