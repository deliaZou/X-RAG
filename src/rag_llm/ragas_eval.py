"""
ragas_eval.py
=============
独立于诊断流程的 RAGAS 评估脚本。

之前 run_ragas() 挂在 RAGDiagnosticLayer 上,靠 diagnose(collect_eval=True)
在诊断过程中顺便攒 eval_record,导致"只想换个 ground_truth 拼法、只想
重跑某个指标"也得连诊断 LLM 一起重新调一遍,浪费钱也浪费时间。

现在拆开: RAGDiagnosticLayer.diagnose() 只管出诊断报告 (retrieved_context /
root_cause_explanation / source_xai_report 这些字段已经够用),评估阶段
从已经存盘的报告 json 里现取数据拼 eval_record,不需要重新调用诊断 LLM。
两者只通过"报告 json 文件"这一份数据耦合,互不感知对方的实现。

用法:
    # 对着 run_pipeline() 存的报告文件单独跑评估
    python ragas_eval.py --path output/20260101_120000_reports.json

    # 只跑其中某几个指标 (省调用次数)
    python ragas_eval.py --path output/xxx_reports.json --metrics faithfulness,answer_relevance
"""

import argparse
import json
import os
import sys
import types

from l3_config import load_l3_config

# ragas 内部某些版本的导入链还在找 langchain_community.chat_models.vertexai,
# 新版 langchain_community 已经把 VertexAI 挪到独立包,老路径下这个模块已经
# 不存在。这里放一个占位模块骗过导入检查即可 —— 本项目诊断/评估全程用的是
# ChatOpenAI,不会真的用到 VertexAI,占位类永远不会被实例化/调用。
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub


# 任务本身是固定的 (给定证据判断根因),不是"每条样本问题都不同"的开放式 QA,
# 所以 question 用同一句话即可,不用从 metric chain 硬拼一句"看起来像问题"
# 的句子 —— 那样反而会让 Answer Relevancy 的相似度算法受用词影响、不稳定。
DIAGNOSIS_QUESTION = "What is the most likely root cause of this anomaly, and why?"

ALL_METRIC_NAMES = ("faithfulness", "answer_relevance", "context_precision")


def _context_as_texts(retrieved_context: dict) -> list[str]:
    """
    retrieved_context 里每个 value 可能是 list[str] (playbooks 已经在
    diagnose() 里截断成字符串列表了) 也可能是纯字符串 (topology_docs),
    统一拍平成一个字符串列表。
    """
    texts = []
    for docs in retrieved_context.values():
        if isinstance(docs, str):
            if docs:
                texts.append(docs)
        else:
            texts.extend(docs)
    return texts


def _ground_truth_text(gt: dict) -> str:
    """
    把 ground_truth_meta (service/metric/fault) 拼成一句陈述句,供
    context_precision 判断"检索到的文档是否支撑得起正确答案"。
    顺带把 FAULT_TYPES 里的故障模式描述也接进来,让 ground_truth 本身
    带上判断依据,context_precision 的打分会更准。
    """
    from src.knowledge_base.retriever import FAULT_TYPES  # 延迟导入,避免只想单独跑评估脚本时也强依赖整个检索模块

    fault_key = gt.get("fault", "").capitalize()  # L2 json 里是 "delay" 这种全小写,FAULT_TYPES 的 key 是 "Delay"
    pattern = FAULT_TYPES.get(fault_key, {}).get("pattern", "")

    text = (
        f"The root cause is {gt.get('metric', 'unknown')} "
        f"in the {gt.get('service', 'unknown')} service, "
        f"caused by a {gt.get('fault', 'unknown')} fault."
    )
    if pattern:
        text += f" Expected pattern: {pattern}"
    return text


def build_eval_records(reports: list[dict]) -> list[dict]:
    """
    从已经生成好的诊断报告 (diagnose_batch() 的返回值,或者存盘的
    xxx_reports.json 读回来的内容) 里提取 RAGAS 需要的
    question/answer/contexts/ground_truth,不需要重新调用诊断 LLM。
    """
    records = []
    for r in reports:
        if "error" in r:
            continue
        src = r.get("source_xai_report", {})
        gt = src.get("ground_truth_meta") or {}
        records.append({
            "case_id": src.get("case_id"),
            "question": DIAGNOSIS_QUESTION,
            "answer": r.get("root_cause_explanation", ""),
            "contexts": _context_as_texts(r.get("retrieved_context", {})),
            "ground_truth": _ground_truth_text(gt) if gt else "",
        })
    return records


def run_ragas(records: list[dict], metrics: list[str] = None, config: dict = None) -> dict:
    """
    对 build_eval_records() 产出的 records 跑 RAGAS 评估。
    不依赖 RAGDiagnosticLayer 实例,是纯函数 —— 输入 records,输出分数。

    metrics: 子集,取值来自 ALL_METRIC_NAMES,默认三个都跑。
        只想重跑某个指标时传这个,能省下不需要的那部分 LLM 调用。
    """
    config = config or load_l3_config()
    ragas_sample_min = config["ragas_sample_min"]
    if len(records) < ragas_sample_min:
        print(f"  ⚠️  RAGAS 需要至少 {ragas_sample_min} 条样本,"
              f"当前只有 {len(records)} 条,跳过")
        return {}

    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"  ⚠️  RAGAS 相关依赖缺失: {e}")
        return {}

    from src.config import llm_config  # 复用项目里已有的 .env 加载,拿 api_key/base_url

    all_metrics = {
        "faithfulness": Faithfulness(),
        "answer_relevance": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
    }
    metric_names = [m for m in (metrics or list(all_metrics.keys())) if m in all_metrics]
    selected = [all_metrics[m] for m in metric_names]

    print(f"\n[RAGAS] 评估 {len(records)} 条样本,指标: {metric_names} ...")

    judge_chat = ChatOpenAI(
        model=llm_config.ragas_model,
        openai_api_key=llm_config.ragas_api_key,
        openai_api_base=llm_config.ragas_base_url,
        temperature=0.2,
        n=1,
    )
    llm = LangchainLLMWrapper(judge_chat)

    local_model = HuggingFaceEmbeddings(
        model_name=os.getenv("XRAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(local_model)

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    scores = evaluate(dataset, metrics=selected, embeddings=embeddings_wrapper, llm=llm)
    df_res = scores.to_pandas()

    # 逐条明细,按 records 原本的顺序和 case_id 对齐 (evaluate 内部不保证
    # 会把 case_id 这种额外字段带出来,所以用位置对齐更稳妥)
    case_ids = [r.get("case_id") for r in records]
    per_case = []
    for i, row in df_res.iterrows():
        entry = {"case_id": case_ids[i] if i < len(case_ids) else None}
        for m in metric_names:
            if m in df_res.columns:
                entry[m] = round(float(row[m]), 4)
        per_case.append(entry)

    summary = {"n_samples": len(df_res)}
    for m in metric_names:
        summary[m] = round(float(df_res[m].mean()), 4) if m in df_res.columns else None

    weights = {"faithfulness": 0.4, "answer_relevance": 0.4, "context_precision": 0.2}
    used_weight = sum(weights[m] for m in metric_names if m in weights)
    if used_weight > 0:
        summary["overall_confidence"] = round(
            sum(summary[m] * weights[m] for m in metric_names if m in weights) / used_weight, 4
        )

    print(f"\n{'─' * 45}")
    print(f"  RAGAS 评估结果")
    print(f"{'─' * 45}")
    for m in metric_names:
        print(f"  {m:<20}: {summary.get(m)}")
    if "overall_confidence" in summary:
        print(f"  {'overall_confidence':<20}: {summary['overall_confidence']}")
    print(f"  样本数              : {summary['n_samples']}")
    print(f"{'─' * 45}")

    return {"summary": summary, "per_case": per_case}


def load_reports(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", required=True,
    #                      help="诊断报告 json 文件路径 (run_pipeline() 存的 xxx_reports.json)")
    # parser.add_argument("--metrics", default=None,
    #                      help="逗号分隔,取值来自 faithfulness,answer_relevance,context_precision;"
    #                           "不传则三个都跑")
    # parser.add_argument("--output", default=None,
    #                      help="评估结果保存路径,默认在报告同目录下把 _reports.json 换成 _ragas.json")
    # args = parser.parse_args()

    # metrics = args.metrics.split(",") if args.metrics else None
    # reports = load_reports(args.path)
    # records = build_eval_records(reports)
    # result = run_ragas(records, metrics=metrics)

    l3_path = "D:\\projects\\X-RAG\\src\\rag_llm\output\\20260923_180802_reports.json"
    metrics = ["context_precision"]  # 取值来自 faithfulness,answer_relevance,context_precision;"
    reports = load_reports(l3_path)
    records = build_eval_records(reports)
    result = run_ragas(records, metrics=metrics)

    if result:
        out_path = l3_path.replace("_reports.json", "_ragas.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 评估结果已保存到 {out_path}")
