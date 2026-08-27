"""
rag_llm_layer.py
================
Layer 3 — RAG-LLM 诊断推理层

流程:
  Pre-Retrieval  → query 构建 + 子系统扩展
  Retrieval      → 四类知识库分别检索 (含 alert_history leave-one-case-out 隔离)
  Post-Retrieval → 相关性过滤 + token 截断 + 分块组装
  Generation     → LLM 诊断报告
  Evaluation     → RAGAS 独立评估 Faithfulness / Answer Relevancy / Context Precision

本版本相对之前的修改:
  1. 补上缺失的 load_knowledge_base 依赖 (最小实现,关键词检索)
  2. compute_confusion_matrix 补回 self 参数
  3. 统一 xai_gateway_suggestion 字段名,去掉死代码 system_action
  4. full 模式不再重复调用 diagnose_batch_with_ragas (原来调了两次,多打一轮 LLM)
  5. API key 只读环境变量,不再硬编码
  6. 新增 --mode l2: 直接读 Layer2 输出 json,过 l2_adapter 转换后跑单条诊断,
     作为知识库/RAGAS 接入之前的最小可执行单元

依赖:
  pip install openai ragas langchain-openai langchain-community datasets sentence-transformers
"""

import json
import re
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 如果 ragas 执着于老路径,手动把它导向新路径
# try:
#     import langchain_google_vertexai
#     sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
# except ImportError:
#     pass

from pathlib import Path
import sys

# 1. 获取当前文件的父目录的父目录（即项目根目录）
project_root = Path(__file__).resolve().parent.parent

# 2. 将项目根目录添加到系统路径中
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 3. 正常导入 knowledge_base 中的模块或函数
from knowledge_base.build_knowledge_base import KnowledgeBase
# from load_knowledge_base import KnowledgeBase, stratified_sample_for_layer3

MAX_CONTEXT_CHARS = 400
RAGAS_SAMPLE_MIN = 3

DIAGNOSIS_SYSTEM = """You are an expert AI for root cause localization in microservice observability.
Output valid JSON only. No preamble, no markdown fences.
"""

DIAGNOSIS_PROMPT = """## Anomaly Report

## Pipeline Context
- **Layer 1 (Detection)**: PCC (Principal Component Classifier) detects anomalies in multivariate time-series metrics.
- **Layer 2 (Explanation)**: KernelSHAP explains each anomaly by computing Shapley values for all metrics.
- **Shapley Value Definition**: Each metric's Shapley value represents its **marginal contribution** to the anomaly score. Higher values indicate the metric contributed more to the anomaly being flagged.
- **⚠️ SHAP Limitations**: 
  - SHAP measures statistical correlation, NOT causation
  - A high SHAP metric could be the root cause, OR a symptom/downstream effect of the true root cause
  - SHAP does not capture temporal propagation or service dependency relationships
  - Use SHAP as a starting point, but validate against fault patterns and service topology

### Anomaly Details
Timestamp        : {timestamp}
Detection result : {detection_result}
Model score      : {model_score}  (higher = more anomalous)
Root cause chain : {root_cause_chain}
Fidelity         : {fidelity}
Stability        : {stability}
XAI gate action  : {xai_gateway_suggestion}

---
## L2 Candidate Root Causes (ranked by SHAP magnitude)
{l2_candidates}

---
## Retrieved Knowledge

### [A] Ops Playbooks
{playbooks}

### [B] Service Topology
{topology_docs}

---

## Known Fault Patterns (from microservice fault injection research)

**Resource faults (CPU / MEM / DISK / SOCKET):**
- The fault manifests directly in the resource metric
- Example: cpu fault → emailservice_cpu shows high SHAP
- Other metrics (latency) may show secondary effects

**Network fault (DELAY):**
- The fault manifests directly in latency metric
- Example: delay fault → checkoutservice_latency shows high SHAP
- Error metric is usually unaffected

**Network fault (LOSS):**
- PRIMARY indicator: latency metric (currencyservice_latency shows high SHAP)
  - Timeouts and retries cause latency elevation
- SECONDARY indicator: error metric may show minor elevation
  - Some requests may fail completely

**Propagation pattern:**
- A resource fault (CPU/MEM) in an upstream service → downstream services show latency elevation
- A latency anomaly in a leaf service → anomaly is self-contained, not propagated
- Use the Service Topology to determine propagation direction

---

## Task

Using the Shapley values (L2 candidates), known fault patterns, and service topology above,
identify the most likely root cause from the L2 candidates.

---

## Output Format

Produce a diagnosis JSON with EXACTLY these keys:

{{
  "root_cause": "<service_metric>",
  "root_cause_ranking": [
    {{"rank": 1, "name": "<service_metric>", "service": "<service>", "metric": "<metric>", "reason": "<why this is ranked here>"}},
    ...
  ],
  "confidence": <float 0.0-1.0>,
  "severity": "critical" | "high" | "medium" | "low",
  "root_cause_explanation": "<2-3 sentences explaining your reasoning>",
  "evidence_used": ["<playbook or topology evidence>", ...],
  "recommended_actions": ["<action>", ...],
  "escalation_required": true | false,
  "xai_quality_flag": "<note on fidelity/stability>"
}}

**IMPORTANT:**
- `root_cause` must be a single service_metric combination
- All candidates MUST come from the L2 candidates list

Output JSON only.
"""

from datetime import datetime

def generate_output_filename(input_file: str) -> str:
    """生成带时间戳的输出文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return f"output\\{base_name}_{timestamp}.json"

class RAGDiagnosticLayer:
    """
    快速测试 (1-2 条 / 真实 L2 case):
        layer3 = RAGDiagnosticLayer()
        report = layer3.diagnose(xai_report_dict)

    批量评估 + RAGAS:
        reports, ragas_scores = layer3.diagnose_batch_with_ragas(
            test_samples, run_ragas=True,
        )
    """

    def __init__(self, top_k_each: int = 2, kb: KnowledgeBase = None):
        self.kb = kb or KnowledgeBase()
        self.top_k = top_k_each
        self._eval_records: list[dict] = []

        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL_ID")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        if not self.api_key:
            raise RuntimeError("未检测到环境变量 LLM_API_KEY,检查 .env 文件")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    # ────────────────────────────────────────────────────
    # Pre-Retrieval
    # ────────────────────────────────────────────────────
    def _pre_retrieval(self, xai_report: dict) -> tuple[list[str], str]:
        chain = xai_report.get("xai_analysis", {}).get("metric_chain", [])
        # service_chain = xai_report.get("xai_analysis", {}).get("service_chain", [])
        # if not chain:
        #     chain = ["server", "anomaly"]
        # query = self.kb.build_query(chain)
        return chain

    # ────────────────────────────────────────────────────
    # Retrieval (含 alert_history 的 leave-one-case-out 隔离)
    # ────────────────────────────────────────────────────
    def _retrieve(self, chain: list[str], current_case_id: str = None) -> dict:
        return self.kb.retrieve_by_type(
            chain, top_k_each=self.top_k, current_case_id=current_case_id
        )

    # ────────────────────────────────────────────────────
    # Post-Retrieval: 格式化为 Prompt 区块
    # ────────────────────────────────────────────────────
    @staticmethod
    def _format_block(docs: list[dict]) -> str:
        if not docs:
            return "  (no relevant documents retrieved)"
        lines = []
        for i, d in enumerate(docs, 1):
            sim = d.get("similarity", "?")
            text = d["text"][:MAX_CONTEXT_CHARS]
            lines.append(f"[{i}] sim={sim}\n{text}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_l2_candidates(candidates: list[dict]) -> str:
        if not candidates:
            return "  (no L2 candidates provided)"
        lines = []
        for c in candidates:
            lines.append(
                f"[rank {c.get('rank')}] {c.get('name')} "
                f"(service={c.get('service')}, metric={c.get('metric')}, "
                f"shap={c.get('shap')})"
            )
        return "\n".join(lines)

    def _build_prompt(self, xai_report: dict, context: dict) -> str:
        xai = xai_report.get("xai_analysis", {})
        return DIAGNOSIS_PROMPT.format(
            timestamp=xai_report.get("timestamp", "unknown"),
            detection_result=xai_report.get("detection_result", "unknown"),
            model_score=xai_report.get("model_score", 0.0),
            root_cause_chain=" → ".join(xai.get("root_cause_chain", [])),
            fidelity=xai.get("fidelity_assessment", "N/A"),
            stability=xai.get("stability_assessment", "N/A"),
            xai_gateway_suggestion=xai_report.get("xai_gateway_suggestion", "N/A"),
            l2_candidates=self._format_l2_candidates(xai_report.get("l2_candidates", [])),
            # feature_docs=self._format_block(context["feature_docs"]),
            playbooks=self._format_block(context["playbooks"]),
            # alert_history=self._format_block(context["alert_history"]),
            # algo_config=self._format_block(context["algo_config"]),
            topology_docs=context.get("topology_docs"),
        )

    # ────────────────────────────────────────────────────
    # Generation
    # ────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> dict:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DIAGNOSIS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            print(f"[-] LLM 调用失败: {e}")
            raise

    # ────────────────────────────────────────────────────
    # 单条诊断主入口
    # ────────────────────────────────────────────────────
    def diagnose(self, xai_report: dict, collect_eval: bool = True) -> dict:
        case_id = xai_report.get("case_id")
        chain = self._pre_retrieval(xai_report)
        context = self._retrieve(chain, current_case_id=case_id)
        prompt = self._build_prompt(xai_report, context)
        result = self._call_llm(prompt)

        result["source_xai_report"] = xai_report
        result["retrieved_context"] = {
            cat: (docs[:200] if isinstance(docs, str) else [d["text"][:200] for d in docs])
            for cat, docs in context.items()
        }
        # result["pre_retrieval_query"] = query

        if collect_eval:
            all_context_texts = [d["text"] for docs in context.values() for d in docs]
            self._eval_records.append({
                "question": query,
                "answer": result.get("root_cause_explanation", ""),
                "contexts": all_context_texts,
                "ground_truth": "",
            })

        return result

    # ────────────────────────────────────────────────────
    # 批量诊断
    # ────────────────────────────────────────────────────
    def diagnose_batch(self, xai_reports: list[dict], collect_eval: bool = True) -> list[dict]:
        results = []
        for i, report in enumerate(xai_reports, 1):
            ts = report.get("timestamp", i)
            print(f"  [{i}/{len(xai_reports)}] ts={ts} ...", end=" ", flush=True)
            try:
                r = self.diagnose(report, collect_eval=collect_eval)
                print(f"→ {r.get('conclusion', '?')} (conf={r.get('confidence', '?')})")
            except Exception as e:
                print(f"✗ {e}")
                r = {"error": str(e), "source_xai_report": report}
            results.append(r)
        return results

    # ────────────────────────────────────────────────────
    # RAGAS 评估
    # ────────────────────────────────────────────────────
    def run_ragas(self) -> dict:
        """
        对已收集的 eval_records 跑 RAGAS 评估。
        需要先调用 diagnose() 或 diagnose_batch(collect_eval=True)。
        """
        records = self._eval_records
        if len(records) < RAGAS_SAMPLE_MIN:
            print(f"  ⚠️  RAGAS 需要至少 {RAGAS_SAMPLE_MIN} 条样本,"
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

        print(f"\n[RAGAS] 评估 {len(records)} 条样本...")

        faithfulness = Faithfulness()
        answer_relevance = AnswerRelevancy()
        context_precision = ContextPrecision()

        judge_chat = ChatOpenAI(
            model=os.getenv("RAGAS_JUDGE_MODEL", self.model),
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
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

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance, context_precision],
            embeddings=embeddings_wrapper,
            llm=llm,
        )
        df_res = scores.to_pandas()

        f = float(df_res["faithfulness"].mean()) if "faithfulness" in df_res.columns else 0.0
        ar = float(df_res["answer_relevance"].mean()) if "answer_relevance" in df_res.columns else 0.0
        cp = float(df_res["context_precision"].mean()) if "context_precision" in df_res.columns else 0.0
        overall = round(0.4 * f + 0.4 * ar + 0.2 * cp, 4)

        result = {
            "faithfulness": round(f, 4),
            "answer_relevance": round(ar, 4),
            "context_precision": round(cp, 4),
            "overall_confidence": overall,
            "n_samples": len(df_res),
        }

        print(f"\n{'─' * 45}")
        print(f"  RAGAS 评估结果 (RO3)")
        print(f"{'─' * 45}")
        print(f"  Faithfulness      : {result['faithfulness']}")
        print(f"  Answer Relevance  : {result['answer_relevance']}")
        print(f"  Context Precision : {result['context_precision']}")
        print(f"  Overall Confidence: {result['overall_confidence']}")
        print(f"  样本数            : {result['n_samples']}")
        print(f"{'─' * 45}")

        return result

    # ────────────────────────────────────────────────────
    # 一步完成: 批量诊断 + RAGAS (只跑一次 diagnose_batch)
    # ────────────────────────────────────────────────────
    def diagnose_batch_with_ragas(self, xai_reports: list[dict],
                                   run_ragas: bool = True) -> tuple[list[dict], dict]:
        self._eval_records = []
        results = self.diagnose_batch(xai_reports, collect_eval=True)
        ragas_scores = self.run_ragas() if run_ragas else {}
        return results, ragas_scores

    # ────────────────────────────────────────────────────
    def compute_root_cause_accuracy(self, results: list[dict],
                                     k_list: tuple = (1, 3, 5)) -> dict:
        """
        对每个 case 用 root_cause_ranking 和 ground_truth_meta (service+metric) 算:
          AC@k (coarse) : 前 k 名里有没有 service 命中
          AC@k (fine)   : 前 k 名里有没有 service+metric 都命中
          Avg@k         : 1..k 各自 AC@j 的均值 (RCAEval/GALA 口径)
        按 fault_type (从 case_id 反推) 分组算完再对 fault_type 取均值,
        不对原始 case 直接 pool。
        """
        max_k = max(k_list)
        per_case = []

        for r in results:
            if "error" in r:
                continue
            src = r.get("source_xai_report", {})
            gt = src.get("ground_truth_meta", {})
            if not gt:
                continue
            ranking = r.get("root_cause_ranking", [])
            case_id = src.get("case_id", "")
            parts = case_id.split("_")
            fault_type = parts[-2] if len(parts) >= 2 else "unknown"

            gt_service = gt.get("service")
            # gt_metric = gt.get("metric")
            gt_name = gt.get("metric")  # 这个字段存的其实是完整复合名

            coarse_hits, fine_hits = [], []
            for k in range(1, max_k + 1):
                top_k = ranking[:k]
                coarse_hits.append(any(c.get("service") == gt_service for c in top_k))
                # fine_hits.append(any(
                #     # c.get("service") == gt_service and c.get("metric") == gt_metric
                #     # for c in top_k
                # ))
                fine_hits.append(any(c.get("name") == gt_name for c in top_k))

            row = {"case_id": case_id, "fault_type": fault_type}
            for k in k_list:
                row[f"AC@{k}_coarse"] = int(any(coarse_hits[:k]))
                row[f"AC@{k}_fine"] = int(any(fine_hits[:k]))
            row[f"Avg@{max_k}_coarse"] = sum(
                1 for j in range(max_k) if any(coarse_hits[:j + 1])
            ) / max_k
            row[f"Avg@{max_k}_fine"] = sum(
                1 for j in range(max_k) if any(fine_hits[:j + 1])
            ) / max_k
            per_case.append(row)

        if not per_case:
            return {}

        metric_keys = [k for k in per_case[0].keys() if k not in ("case_id", "fault_type")]
        by_fault: dict = {}
        for row in per_case:
            by_fault.setdefault(row["fault_type"], []).append(row)

        per_fault_type = {
            ft: {mk: sum(r[mk] for r in rows) / len(rows) for mk in metric_keys}
            for ft, rows in by_fault.items()
        }
        overall_macro = {
            mk: sum(per_fault_type[ft][mk] for ft in per_fault_type) / len(per_fault_type)
            for mk in metric_keys
        }

        return {
            "per_case": per_case,
            "per_fault_type": per_fault_type,
            "overall_macro": overall_macro,
        }


# ════════════════════════════════════════════════════════
def _print_report(report: dict):
    print(f"\n  根因排序   :")
    for c in report.get("root_cause_ranking", [])[:5]:
        print(f"    [{c.get('rank')}] {c.get('name')} "
              f"(service={c.get('service')}, metric={c.get('metric')})")
    print(f"  结论       : {report.get('conclusion')}")
    print(f"  置信度     : {report.get('confidence')}")
    print(f"  严重等级   : {report.get('severity')}")
    print(f"  根因解释   : {report.get('root_cause_explanation', '')[:120]}...")
    print(f"  处置建议   :")
    for step in report.get("recommended_actions", [])[:3]:
        print(f"    • {step}")
    print(f"  升级告警   : {report.get('escalation_required')}")
    print(f"  XAI 质量   : {report.get('xai_quality_flag', '')[:80]}")
    print(f"  检索 query : {report.get('pre_retrieval_query', '')}")


def _print_accuracy(acc: dict):
    if not acc:
        print("\n  ⚠️  没有可用于计算 AC@k 的样本 (缺 ground_truth_meta 或 root_cause_ranking)")
        return

    print(f"\n{'─' * 60}")
    print("  Root Cause AC@k / Avg@k (按 fault_type, macro-average)")
    print(f"{'─' * 60}")
    per_fault = acc.get("per_fault_type", {})
    for fault_type, metrics in per_fault.items():
        coarse = {k: round(v, 4) for k, v in metrics.items() if k.endswith("_coarse")}
        fine = {k: round(v, 4) for k, v in metrics.items() if k.endswith("_fine")}
        n = sum(1 for r in acc.get("per_case", []) if r["fault_type"] == fault_type)
        print(f"\n  [{fault_type}]  (n={n})")
        print(f"    coarse (service 命中)      : {coarse}")
        print(f"    fine   (service+metric 命中): {fine}")

    overall = {k: round(v, 4) for k, v in acc.get("overall_macro", {}).items()}
    print(f"\n  整体 macro-average (跨 fault_type 取均值,不 pool):")
    print(f"    coarse: { {k: v for k, v in overall.items() if k.endswith('_coarse')} }")
    print(f"    fine  : { {k: v for k, v in overall.items() if k.endswith('_fine')} }")
    print(f"{'─' * 60}")


BUILTIN_SAMPLES = [
    {
        "timestamp": 15852.0,
        "detection_result": "Anomaly (FP candidate)",
        "model_score": -0.0003,
        "xai_analysis": {
            "root_cause_chain": ["CPU_Usage_Rate", "System_Load_1min"],
            "fidelity_assessment": "Low (Drop: 0.0279) -> Evidence is weak",
            "stability_assessment": "Unstable (Robustness: 0.1344)",
        },
        "xai_gateway_suggestion": "Suppress LLM Report",
        "ground_truth_label": 0,
    },
    {
        "timestamp": 28400.0,
        "detection_result": "Anomaly",
        "model_score": 0.142,
        "xai_analysis": {
            "root_cause_chain": ["Disk_Util_Rate", "Disk_Queue_Length", "Disk_Await_Time"],
            "fidelity_assessment": "High (Drop: 0.1820) -> Strong evidence",
            "stability_assessment": "Stable (Robustness: 0.8821)",
        },
        "xai_gateway_suggestion": "Escalate: High-Confidence Anomaly",
        "ground_truth_label": 1,
    },
    {
        "timestamp": 41200.0,
        "detection_result": "Anomaly",
        "model_score": 0.087,
        "xai_analysis": {
            "root_cause_chain": ["Memory_Usage_Rate", "Swap_Usage_Rate", "Page_Faults_Rate"],
            "fidelity_assessment": "Medium (Drop: 0.0612) -> Moderate evidence",
            "stability_assessment": "Moderate (Robustness: 0.6130)",
        },
        "xai_gateway_suggestion": "Monitor: Moderate-Confidence Anomaly",
        "ground_truth_label": 1,
    },
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["single", "l2", "batch", "full"],
        default="single",
        help=(
            "single: 内置样本测试单条 (不需要真实数据)\n"
            "l2:     读一条 Layer2 输出 json,过 adapter 转换后跑单条诊断\n"
            "batch:  读 --l2-dir 下所有 Layer2 case json,批量诊断 + AC@k 统计\n"
            "full:   batch + RAGAS 评估 (需要 ≥3 条样本)"
        ),
    )
    parser.add_argument("--l2-json", default="checkoutservice_cpu_1.json",
                         help="--mode l2 时使用,Layer2 输出的单个 json 路径")
    parser.add_argument("--l2-dir", default="eval_outputs/l2_test_fold",
                         help="--mode batch/full 时使用,一个目录,里面每个 *.json 是一条 Layer2 case")
    parser.add_argument("--candidate-pool-size", type=int, default=10,
                         help="喂给 LLM 重排的 L2 候选池大小,和评估窗口 AC@1/3/5 是两件事,"
                              "建议先在 train-fold 上对 L2 做 K sweep 再定这个数")
    args = parser.parse_args()

    layer3 = RAGDiagnosticLayer(top_k_each=2)

    if args.mode == "single":
        print("=" * 55)
        print("Layer 3 单条测试 (内置样本)")
        print("=" * 55)
        report = layer3.diagnose(BUILTIN_SAMPLES[0], collect_eval=False)
        _print_report(report)
        with open("sample_l3_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("\n完整报告已保存到 sample_l3_report.json")

    elif args.mode == "l2":
        from l2_adapter import convert_l2_to_xai_report

        print("=" * 55)
        print(f"Layer 3 最小可执行单元: 读取 {args.l2_json}")
        print("=" * 55)
        with open(args.l2_json, "r", encoding="utf-8") as f:
            l2_output = json.load(f)
        xai_report = convert_l2_to_xai_report(l2_output, candidate_pool_size=args.candidate_pool_size)
        print(f"  转换后 metric_chain: {xai_report['xai_analysis']['metric_chain']}")
        print(f"  ground_truth          : {xai_report['ground_truth_meta']}")

        report = layer3.diagnose(xai_report, collect_eval=False)
        _print_report(report)

        # 单条 case 也能算 AC@1/AC@5,只是 fault_type macro-average 这里只有 1 类
        acc = layer3.compute_root_cause_accuracy([report])
        _print_accuracy(acc)

        with open(generate_output_filename(args.l2_json.split("\\")[-1]), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("\n完整报告已保存到 sample_l3_report_l2.json")

    elif args.mode == "batch":
        from l2_adapter import load_l2_dir

        print("=" * 55)
        print(f"Layer 3 批量诊断: {args.l2_dir}")
        print("=" * 55)
        xai_reports = load_l2_dir(args.l2_dir, candidate_pool_size=args.candidate_pool_size)

        results = layer3.diagnose_batch(xai_reports, collect_eval=False)
        with open("batch_l3_reports.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ {len(results)} 条报告已保存到 batch_l3_reports.json")
        for r in results:
            _print_report(r)

        acc = layer3.compute_root_cause_accuracy(results)
        _print_accuracy(acc)
        with open("l3_accuracy.json", "w", encoding="utf-8") as f:
            json.dump(acc, f, indent=2, ensure_ascii=False)
        print("\n✅ AC@k 统计已保存到 l3_accuracy.json")

    elif args.mode == "full":
        from l2_adapter import load_l2_dir

        print("=" * 55)
        print(f"Layer 3 完整评估 (批量诊断 + RAGAS): {args.l2_dir}")
        print("=" * 55)
        xai_reports = load_l2_dir(args.l2_dir, candidate_pool_size=args.candidate_pool_size)

        results, ragas_scores = layer3.diagnose_batch_with_ragas(xai_reports, run_ragas=True)

        with open("full_l3_reports.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 诊断报告已保存到 full_l3_reports.json")

        if ragas_scores:
            with open("ragas_scores.json", "w", encoding="utf-8") as f:
                json.dump(ragas_scores, f, indent=2)
            print(f"✅ RAGAS 结果已保存到 ragas_scores.json")

        acc = layer3.compute_root_cause_accuracy(results)
        _print_accuracy(acc)
        with open("l3_accuracy.json", "w", encoding="utf-8") as f:
            json.dump(acc, f, indent=2, ensure_ascii=False)
        print("\n✅ AC@k 统计已保存到 l3_accuracy.json")
