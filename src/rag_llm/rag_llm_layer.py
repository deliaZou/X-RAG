"""
rag_llm_layer.py
================
Layer 3 — RAG-LLM 诊断推理层

流程:
  Pre-Retrieval  → query 构建 + 子系统扩展
  Retrieval      → 知识库检索 (playbooks + 拓扑,含 alert_history 预留的
                    leave-one-case-out 隔离接口)
  Post-Retrieval → 相关性过滤 + token 截断 + 分块组装
  Generation     → LLM 诊断报告
  Evaluation     → RAGAS 独立评估 Faithfulness / Answer Relevancy / Context Precision

本版本相对之前的修改 (在上一版"1-6"的基础上):
  7. 修复 collect_eval=True (即 --ragas) 路径下必然抛异常的 bug:
     - context 里 topology_docs 是字符串、playbooks 是 list[dict],混在一起
       用同一种方式取 ["text"] 字段会报 TypeError。
     - 引用了一个从未被定义过的 query 变量 (NameError)。
     之前这两个 bug 因为被 diagnose_batch 的 try/except 吞掉,表现为
     "full 模式每条 case 都诊断失败但不报错",很难发现。
  8. 修复字段名不对齐: 之前 prompt 里读的是 xai_analysis["root_cause_chain"],
     但 l2_adapter (新格式路径) 从来没产出过这个 key,导致 prompt 里
     "Root cause chain" 那一行永远是空的,detection_result 同理永远是
     "unknown"。现在 l2_adapter 统一产出对齐的字段 (见 l2_adapter.py)。
  9. 打印用的字段名和 LLM 输出 schema 对齐: 之前打印 report.get('conclusion'),
     但 schema 里定义的 key 是 root_cause,打印出来一直是 None/'?'。
  10. prompt 文本抽到 prompts/*.txt,用 prompt_loader 加载 (string.Template,
      不再需要手动转义 JSON 示例里的花括号)。
  11. 超参数 (MAX_CONTEXT_CHARS / RAGAS_SAMPLE_MIN / temperature / top_k_each /
      candidate_pool_size / query_chain_size) 抽到 configs/l3_config.yaml,
      由 l3_config.load_l3_config() 统一加载。
  12. single/l2/batch/full 四个模式的重复代码 (诊断 -> 存报告 -> 打印 ->
      算 accuracy -> 存 accuracy) 合并成一个 run_pipeline() 函数,
      --mode 拆成 --input {builtin,l2-json,l2-dir} + --ragas 两个正交参数;
      l2-json 现在直接复用 l2-dir 同一条代码路径 (n=1 的批量),不再单独
      写一套。同时把之前硬编码的 Windows 路径分隔符 (output\\...) 换成
      os.path.join,并在写文件前 os.makedirs 保证目录存在。
  13. RAGAS 评估整个拆到独立文件 ragas_eval.py,RAGDiagnosticLayer 不再
      持有 _eval_records 状态、也不再有 run_ragas()/diagnose_batch_with_ragas()
      方法。诊断和评估现在只通过"存盘的报告 json"耦合:diagnose_batch()
      的结果里已经带了评估需要的全部原料 (source_xai_report/retrieved_context/
      root_cause_explanation),ragas_eval.build_eval_records() 直接从这些
      字段里取数据拼评估样本,不需要重新调用诊断 LLM。好处是:换个
      ground_truth 拼法、单独重跑某个指标,都可以对着已经存盘的报告文件
      直接跑 `python ragas_eval.py --path xxx_reports.json`,不用再花钱
      重新诊断一遍。

依赖:
  pip install openai pyyaml
  # 只有需要跑 --ragas / ragas_eval.py 时才用得到:
  pip install ragas langchain-openai langchain-community datasets sentence-transformers
"""

import json
import os
from datetime import datetime

from openai import OpenAI
from src.config import llm_config  # 导入即完成 .env 加载
llm_config.show()

from pathlib import Path
import sys

# 1. 获取当前文件的父目录的父目录（即项目根目录）
project_root = Path(__file__).resolve().parent.parent

# 2. 将项目根目录添加到系统路径中
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 3. 正常导入 knowledge_base 中的模块或函数
from src.knowledge_base.retriever import KnowledgeBase

from l2_adapter import load_l2_input, _split_name
from l3_config import load_l3_config
from prompt_loader import load_prompt


class RAGDiagnosticLayer:
    """
    只管"诊断"这一件事:检索 -> 拼 prompt -> 调 LLM -> 出报告。
    不再兼管 RAGAS 评估相关的状态和逻辑 —— 评估是独立的下游步骤,
    见 ragas_eval.py: 它直接读 diagnose_batch() 存盘的报告 json 来打分,
    不需要 RAGDiagnosticLayer 实例参与,详见该文件顶部说明。

    快速测试 (1-2 条 / 真实 L2 case):
        layer3 = RAGDiagnosticLayer()
        report = layer3.diagnose(xai_report_dict)

    批量诊断:
        reports = layer3.diagnose_batch(test_samples)
    """

    def __init__(self, top_k_each: int = None, kb: KnowledgeBase = None, config: dict = None):
        self.config = config or load_l3_config()
        self.kb = kb or KnowledgeBase()
        self.top_k = top_k_each if top_k_each is not None else self.config["top_k_each"]

        self.api_key = llm_config.api_key
        self.base_url = llm_config.base_url
        self.model = llm_config.llm_model
        self.timeout = llm_config.timeout
        if not self.api_key:
            raise RuntimeError("未检测到环境变量 LLM_API_KEY,检查 .env 文件")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        self._system_prompt = load_prompt("diagnosis_system").template
        self._diagnosis_template = load_prompt("diagnosis_prompt")
        self._call_stats = {"total": 0, "api_error": 0, "json_parse_error": 0, "schema_incomplete": 0}

    # ────────────────────────────────────────────────────
    # Pre-Retrieval
    # ────────────────────────────────────────────────────
    def _pre_retrieval(self, xai_report: dict) -> list[str]:
        return xai_report.get("xai_analysis", {}).get("metric_chain", [])

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
    def _format_block(self, docs: list[dict]) -> str:
        if not docs:
            return "  (no relevant documents retrieved)"
        max_chars = self.config["max_context_chars"]
        lines = []
        for i, d in enumerate(docs, 1):
            sim = d.get("similarity", "?")
            text = d["text"][:max_chars]
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
                f"shap={c.get('shap')}, direction={c.get('direction')}))"
            )
        return "\n".join(lines)

    def _build_prompt(self, xai_report: dict, context: dict) -> str:
        xai = xai_report.get("xai_analysis", {})
        return self._diagnosis_template.substitute(
            timestamp=xai_report.get("timestamp", "unknown"),
            detection_result=xai_report.get("detection_result", "unknown"),
            model_score=xai_report.get("model_score", 0.0),
            root_cause_chain=" → ".join(xai.get("root_cause_chain", [])),
            fidelity=xai.get("fidelity_assessment", "N/A"),
            stability=xai.get("stability_assessment", "N/A"),
            xai_gateway_suggestion=xai_report.get("xai_gateway_suggestion", "N/A"),
            l2_candidates=self._format_l2_candidates(xai_report.get("l2_candidates", [])),
            playbooks=self._format_block(context.get("playbooks", [])),
            topology_docs=context.get("topology_docs", ""),
        )

    # ────────────────────────────────────────────────────
    # Generation
    # ────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> dict:
        self._call_stats["total"] += 1
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self._system_prompt},
                          {"role": "user", "content": prompt},],
                temperature=self.config["temperature"],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self._call_stats["api_error"] += 1
            print(f"[-] LLM 调用失败 (API): {e}")
            raise

        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self._call_stats["json_parse_error"] += 1
            print(f"[-] LLM output not illegal JSON: {e}\nOriginal output's first 200 characters: {raw[:200]}")
            raise

    # ────────────────────────────────────────────────────
    # 单条诊断主入口
    # ────────────────────────────────────────────────────
    def diagnose(self, xai_report: dict) -> dict:
        case_id = xai_report.get("case_id")
        chain = self._pre_retrieval(xai_report)
        context = self._retrieve(chain, current_case_id=case_id)
        prompt = self._build_prompt(xai_report, context)
        result = self._call_llm(prompt)
        # Schema / 字段不完整 校验
        ranking = result.get("root_cause_ranking")
        problems = []
        if not result.get("root_cause_explanation"):
            problems.append("missing_explanation")
        if not isinstance(ranking, list) or not ranking:
            problems.append("empty_ranking")
        else:
            for c in ranking:           # 解析result
                service, metric = _split_name(c.get("name", ""))
                c["service"] = service
                c["metric"] = metric
                if metric == "unknown":
                    problems.append(f"malformed_name:{c.get('name')}")

        if problems:
            self._call_stats["schema_incomplete"] += 1
            print(f"schema incomplete: {problems}")

        # 解析result
        result["root_cause"] = result.get("root_cause_ranking", [])[0]["name"] if result["root_cause_ranking"] else None
        result["source_xai_report"] = xai_report
        max_chars = self.config["max_context_chars"]
        # retrieved_context 存的是截断后的纯文本,既用于打印/调试,也是
        # ragas_eval.build_eval_records() 拼 RAGAS contexts 的原料 ——
        # 诊断和评估之间唯一的耦合点就是这份存盘的 json,不用共享任何状态。
        result["retrieved_context"] = {
            cat: (docs[:max_chars] if isinstance(docs, str) else [d["text"][:max_chars] for d in docs])
            for cat, docs in context.items()
        }
        result["pre_retrieval_query"] = " ".join(chain)

        return result

    # ────────────────────────────────────────────────────
    # 批量诊断
    # ────────────────────────────────────────────────────
    def diagnose_batch(self, xai_reports: list[dict]) -> list[dict]:
        results = []
        for i, report in enumerate(xai_reports, 1):
            ts = report.get("timestamp", i)
            print(f"  [{i}/{len(xai_reports)}] ts={ts} ...", end=" ", flush=True)
            try:
                r = self.diagnose(report)
                print(f"→ {r.get('root_cause', '?')} (conf={r.get('confidence', '?')})")
            except Exception as e:
                print(f"✗ {e}")
                r = {"error": str(e), "source_xai_report": report}
            results.append(r)
        return results

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
            gt_name = gt.get("metric")  # 这个字段存的其实是完整复合名

            coarse_hits, fine_hits = [], []
            for k in range(1, max_k + 1):
                top_k = ranking[:k]
                coarse_hits.append(any(c.get("service") == gt_service for c in top_k))
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
    print(f"  根因结论   : {report.get('root_cause')}")
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


# ════════════════════════════════════════════════════════
# 统一 pipeline: l2 / batch / full 三种旧模式现在都走这一个函数,
# 区别只在 run_ragas 开关和输入数据量 (n=1 还是 n=多)
# ════════════════════════════════════════════════════════
def run_pipeline(layer3: RAGDiagnosticLayer, xai_reports: list[dict],
                  run_ragas: bool = False, ragas_metrics: list[str] = None,
                  verbose: bool = True, print_limit: int = 20,
                  output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = layer3.diagnose_batch(xai_reports)

    # RAGAS 现在是独立模块 (ragas_eval.py),诊断结果已经存盘的话也可以事后
    # 单独对着报告文件跑 `python ragas_eval.py --path xxx_reports.json`,
    # 这里图方便直接在同一次运行里顺带调用,但两者不共享任何状态。
    ragas_result = {}
    if run_ragas:
        from ragas_eval import build_eval_records, run_ragas as _run_ragas
        records = build_eval_records(results)
        ragas_result = _run_ragas(records, metrics=ragas_metrics, config=layer3.config)

    reports_path = os.path.join(output_dir, f"{run_id}_reports.json")
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ {len(results)} 条报告已保存到 {reports_path}")

    if verbose:
        for r in results[:print_limit]:
            _print_report(r)
        if len(results) > print_limit:
            print(f"\n  ...(其余 {len(results) - print_limit} 条报告已省略,完整内容见 {reports_path})")

    if ragas_result:
        ragas_path = os.path.join(output_dir, f"{run_id}_ragas.json")
        with open(ragas_path, "w", encoding="utf-8") as f:
            json.dump(ragas_result, f, indent=2, ensure_ascii=False)
        print(f"✅ RAGAS 结果已保存到 {ragas_path}")

    acc = layer3.compute_root_cause_accuracy(results)
    _print_accuracy(acc)
    acc_path = os.path.join(output_dir, f"{run_id}_accuracy.json")
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=2, ensure_ascii=False)
    print(f"✅ AC@k 统计已保存到 {acc_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", choices=["l2-json", "l2-dir"], default="builtin",
        help=(
            "l2-json: 读 --path 指定的单个 Layer2 输出 json\n"
            "l2-dir:  读 --path 指定的目录,里面每个 *.json 是一条 Layer2 case"
        ),
    )
    parser.add_argument("--path", default="eval_outputs/l2_test_fold",
                         help="--input l2-json/l2-dir 时使用,单个 json 文件或目录路径")
    parser.add_argument("--ragas", action="store_true",
                         help="额外跑 RAGAS 评估 (样本数需 >= 配置里的 ragas_sample_min)")
    parser.add_argument("--ragas-metrics", default=None,
                         help="--ragas 时生效,逗号分隔子集,如 faithfulness,answer_relevance;"
                              "不传则三个指标都跑")
    parser.add_argument("--candidate-pool-size", type=int, default=None,
                         help="喂给 LLM 重排的 L2 候选池大小,默认读配置文件里的值")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", default=True,
                         help="关闭逐条打印诊断报告 (默认打印)")
    parser.add_argument("--print-limit", type=int, default=20,
                         help="verbose 模式下最多打印多少条,避免样本多时刷屏")
    parser.add_argument("--output-dir", default="output",
                         help="报告/accuracy/ragas 结果的保存目录")
    args = parser.parse_args()

    # l2_path = args.path
    # output = args.output_dir
    ragas = None
    verbose = None
    print_limit = None

    # l2_path = "D:\\projects\\X-RAG\\RCAEval_ds\\trans-OB\\layer2_output"
    l2_path = "D:\\projects\\X-RAG\\src\\rag_llm\\test\\checkoutservice_disk_3.json"
    output = "output"

    config = load_l3_config()
    pool_size = args.candidate_pool_size or config["candidate_pool_size"]

    layer3 = RAGDiagnosticLayer(config=config)

    xai_reports = load_l2_input(
        l2_path,
        candidate_pool_size=pool_size,
        query_chain_size=config["query_chain_size"],
    )

    run_pipeline(
        layer3, xai_reports,
        run_ragas=ragas,
        verbose=verbose,
        print_limit=print_limit,
        output_dir=output,
    )
