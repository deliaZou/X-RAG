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


def _ranked_candidates(pairs: List[Tuple[float, str]], top_k: int,
                        directions: Dict[str, str] = None) -> List[Dict]:
    """
    pairs: [(shap, name), ...] 已按 shap 降序排好序
    -> l2_candidates: 供 L3 重排使用的候选列表, 每条 {rank, name, service, metric, shap, direction}

    top_k 是候选池大小,不等于评估窗口 (AC@1/3/5)。候选池应比评估窗口宽,
    否则真实 root cause 若排在 L2 的第 6 名以后,会在这一步被直接砍掉,
    LLM 之后再怎么重排也救不回来。默认 10 只是起点,建议在 train-fold 上
    对 AC@K_train 做 K=5/8/10/15/20 的 sweep,取边际收益开始趋平的点。
    """
    directions = directions or {}
    # # top_k 的选择仍然基于 shap 数值大小（这一步不能打乱，否则会漏掉真正高 shap 的候选）
    # top_pairs = list(pairs[:top_k])
    # # 只打乱这 top_k 个候选“展示给 LLM 时”的顺序
    # random.shuffle(top_pairs)
    out = []
    for i, (shap, name) in enumerate(pairs[:top_k], 1):
        service, metric_type = _split_name(name)
        out.append({
            "name": name,
            "service": service,
            "metric": metric_type,
            "shap": shap,
            "direction": directions.get(name, "unknown"),
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

    if _is_new_format(l2_output):
        service_list = l2_output.get("service_list", [])
        metrics_list = l2_output.get("metrics_list", [])

        positive_metrics = [m for m in metrics_list if m.get("shap", 0) > 0]
        sorted_metrics = sorted(positive_metrics, key=lambda x: x["shap"], reverse=True)
        pairs = [(m["shap"], m["name"]) for m in sorted_metrics]
        directions = {m["name"]: m.get("direction", "unknown") for m in sorted_metrics}

        metric_chain = [name for _, name in pairs[:query_chain_size]]
        service_chain = [item["service"] for item in service_list[:query_chain_size]] \
            or _dedup_services(metric_chain)

        top_service = service_list[0] if service_list else {}
        model_score = top_service.get("service_shap", 0.0)

        l2_candidates = _ranked_candidates(pairs, candidate_pool_size, directions)
    else:
        # 旧格式: candidates -> metrics 嵌套结构
        flat = []
        for cand in l2_output.get("candidates", []):
            for m in cand.get("metrics", []):
                if m.get("shap", 0) > 0:
                    flat.append((m["shap"], m["name"]))
        flat.sort(key=lambda x: x[0], reverse=True)

        metric_chain = [name for _, name in flat[:query_chain_size]]
        service_chain = _dedup_services(metric_chain)

        top_candidate = l2_output.get("candidates", [{}])[0]
        model_score = top_candidate.get("service_shap", 0.0)

        l2_candidates = _ranked_candidates(flat, candidate_pool_size)

    return {
        "timestamp": meta.get("case_id", "unknown"),
        "detection_result": "Anomaly",
        "model_score": model_score,
        "xai_analysis": {
            "service_chain": service_chain,
            "metric_chain": metric_chain,
            # 与 metric_chain 同源,专供 prompt 里 "Root cause chain" 那一行使用,
            # 避免像之前那样 prompt 读的 key 和这里产出的 key 对不上导致永远为空
            "root_cause_chain": metric_chain,
            "fidelity_assessment": _fidelity_str(fidelity),
            "stability_assessment": "N/A (未接入 stability 指标)",
        },
        "xai_gateway_suggestion": _xai_gateway_suggestion(fidelity, n_points),
        # L2 的候选排序,供 L3 重排,是 AC@k/Avg@k 评估的对比基准
        "l2_candidates": l2_candidates,
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
