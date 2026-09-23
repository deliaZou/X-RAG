"""
stage_a_preprocess.py
======================
阶段A：预处理

职责：
  - 遍历指定目录下的源文件
  - 根据 file_type ("instance" 或 "text") 决定是否需要先做案例级切片
      instance : 一个文件天然就是一个案例（如 prometheus-operator/runbooks 的单个 md）
                 —— 不切片，直接整篇丢给 LLM 做分类
      text     : 一个文件可能包含 0~N 个案例（如 SRE Book 的一整章）
                 —— 先让 LLM 抽取出案例级片段（原文，不改写），再对每个片段分类
  - is_relevant=False 的直接丢弃
  - mapped_fault_type 允许多选；多选时在本阶段就展开成多条 record（每条单一 fault_type）
  - 输出：cases_intermediate.json （案例级中间产物，尚未去重、尚未正式入库）

与阶段B/C的接口：
  本脚本只产出 cases_intermediate.json，不碰正式知识库文件，也不碰 embeddings_cache.pkl。

用法：
  python stage_a_preprocess.py --config source_configs.json
  （或直接在下方 SOURCE_CONFIGS / 命令行参数中配置）
"""

import os
import re
import json
import glob
import argparse
from pathlib import Path   # 顶部导入
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

from configs.paths import KNOWLEDGEBASE_RAW_STAGE_A_SOURCECONFIGS
from src.config import llm_config   # 导入即完成 .env 加载
llm_config.show()

client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

# ==================== 分类体系（沿用你现有 runbook 脚本的定义） ====================
FAULT_TYPES: list[str] = ["CPU", "Delay", "Disk", "Loss", "Mem", "Socket"]
TRIGGER_METRICS: list[str] = ["cpu", "mem", "socket", "workload", "diskio", "latency", "error"]

CLASSIFY_RULES = f"""
【粗细粒度分类体系规则】：
1. `is_relevant` (布尔值)：
   - 若内容属于工具内部配置/测试/组织管理/流程规范等，与微服务/节点级物理性能故障无关，设为 false。
   - 涉及 TLS/SSL 证书过期、配置校验错误、版本不兼容、凭证失效等管理类问题，同样设为 false。
   - 只有明确描述了微服务/节点级物理性能故障的因果机制或排查方法时，才设为 true。

2. `mapped_fault_type`：
   - 从这 6 个词中选择 1 个或多个（数组）：{FAULT_TYPES}
   - CPU: CPU 满载、CPU throttling、计算资源竞争
   - Mem: 内存溢出(OOM)、内存泄漏、Swap 交换瓶颈
   - Disk: 磁盘空间爆满、磁盘 IO 读写瓶颈/延迟
   - Loss: 网络丢包、包损坏、网络断连
   - Delay: 网络延迟突增、RTT 变长
   - Socket: TCP/UDP 连接数耗尽、端口占用、Socket 缓冲区溢出
   - 若因果链条横跨多个类型（如 CPU→内存的级联链条），可多选。

3. `trigger_metrics`：
   - 从这 7 个小写单词中选择（可多选）：{TRIGGER_METRICS}
   - 把文中因果链条涉及到的全部指标都列出，用于后续细粒度检索匹配。
"""

# ==================== Pydantic 结构 ====================
class ClassifiedCase(BaseModel):
    case_title: str = Field(default="", description="该案例的简短标题，几个字即可")
    case_text: str = Field(description="该案例对应的原文片段，禁止改写/概括，必须是原文摘录")
    is_relevant: bool = False
    mapped_fault_type: List[str] = Field(default_factory=list)
    trigger_metrics: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    cases: List[ClassifiedCase] = Field(default_factory=list)


# ==================== LLM 调用 ====================
def call_llm_extract_and_classify(content: str, file_type: str, filename: str) -> ExtractionResult:
    """
    单次 LLM 调用，同时完成 (instance模式) 分类 或 (text模式) 切片+分类。
    """
    if file_type == "instance":
        task_instruction = f"""
输入的整篇内容本身就是【一个完整案例】，不要切分，不要概括。
请直接输出一个只包含 1 个元素的 cases 数组：
  - case_title: 给这篇文档起一个简短标题
  - case_text: 必须是输入原文本身（可以适当裁剪代码块/无关格式，但不要改写叙述内容）
  - is_relevant / mapped_fault_type / trigger_metrics: 按下方规则分类
"""
    else:  # text
        task_instruction = f"""
输入的是一整篇文档（可能包含多个小节），其中可能包含 0 到多个"具备明确故障因果机制描述"的案例级片段
（例如：某个具体的资源耗尽链条、某个具体的级联故障案例、某个具体的排障案例研究）。

请你：
  1. 找出所有这样的案例级片段，逐条抽取
  2. 每条 case_text 必须是【原文摘录】，禁止改写、禁止概括、禁止用自己的话转述——
     只允许做必要的裁剪（比如去掉与该案例无关的图片说明、脚注编号），
     但案例本身的叙述文字必须逐字保留原文
  3. 纯组织管理、流程规范、背景介绍、总结性内容——不要抽取，直接跳过
  4. 如果整篇文档里一个合格案例都没有，返回空数组 cases: []
"""

    prompt = f"""你是一个 AIOps 根因分析知识库预处理专家。

{task_instruction}

{CLASSIFY_RULES}

【文件名】：{filename}
【文档内容】：
{content[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model=llm_config.llm_model,
            messages=[
                {"role": "system", "content": "你是一个严格输出 JSON 的 AIOps 知识库预处理工具，绝不改写原文案例文本。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        res_json = json.loads(response.choices[0].message.content)
        # 兼容 LLM 直接返回 list 或 {"cases": [...]}
        if isinstance(res_json, list):
            res_json = {"cases": res_json}

        raw_cases = res_json.get("cases", [])
        valid_cases: List[ClassifiedCase] = []
        for idx, raw_case in enumerate(raw_cases):
            try:
                valid_cases.append(ClassifiedCase(**raw_case))
            except Exception as ce:
                print(f"\n[Warning] {filename} 第 {idx} 个 case 字段不合规，已跳过此条（不影响同文件其他case）: {ce}")
        return ExtractionResult(cases=valid_cases)

    except Exception as e:
        print(f"\n[Warning] 处理文件 {filename} 解析/请求失败: {e}")
        return ExtractionResult(cases=[])


# ==================== 展开多 fault_type ====================
def slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_")


def expand_case_to_records(case: ClassifiedCase, source_category: str,
                            source: str, source_url: str, case_index: int) -> List[dict]:
    """
    一个 case（可能多 fault_type）展开成多条 record，每条单一 fault_type。
    多条 record 共享同一个 case_id，text 完全相同（供阶段B判重时识别同源）。
    """
    if not case.is_relevant or not case.mapped_fault_type:
        return []

    case_id = f"{source_category}_{slugify(source)}_{case_index}"
    records = []
    for ft in case.mapped_fault_type:
        records.append({
            "id": f"playbook_ext_{case_id}_{ft}",
            "case_id": case_id,          # 供阶段B识别"同一案例展开出的多条"
            "case_title": case.case_title,
            "type": "playbook",
            "source_category": source_category,
            "fault_type": ft,
            "triggers": case.trigger_metrics,
            "source": source,
            "source_url": source_url,
            "text": case.case_text,
        })
    return records


# ==================== 主流程 ====================
def process_source(source_cfg: dict) -> List[dict]:
    """
    source_cfg 示例：
    {
        "name": "sre_book",
        "file_type": "text",                 # "instance" | "text"
        "local_dir": "D:/.../google_sre_book",
        "source_category": "sre_book_theory",
        "url_base": "https://sre.google/sre-book",
        "glob_pattern": "*.md",
        "include_files": ["12_effective-troubleshooting.md", "21_handling-overload.md", ...]  # 可选：只处理这些文件
    }
    """
    name = source_cfg["name"]
    file_type = source_cfg["file_type"]
    local_dir = source_cfg["local_dir"]
    source_category = source_cfg["source_category"]
    url_base = source_cfg.get("url_base", "")
    include_files = source_cfg.get("include_files")  # None 表示全部处理

    md_files = sorted(glob.glob(os.path.join(local_dir, "**", source_cfg.get("glob_pattern", "*.md")), recursive=True))
    if include_files:
        md_files = [f for f in md_files if os.path.basename(f) in include_files]

    print(f"[{name}] 扫描到 {len(md_files)} 个待处理文件")

    all_records = []
    for file_path in tqdm(md_files, desc=f"处理 {name}"):
        filename = os.path.basename(file_path)
        if filename in ["_index.md", "README.md", "00_table-of-contents.md"]:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, local_dir)
        source_url = f"{url_base}/{rel_path.replace(os.sep, '/')}" if url_base else rel_path

        result = call_llm_extract_and_classify(content, file_type, filename)

        for i, case in enumerate(result.cases):
            records = expand_case_to_records(
                case, source_category=source_category,
                source=rel_path.replace(os.sep, "/"),
                source_url=source_url, case_index=i,
            )
            all_records.extend(records)

    return all_records


def main():

    stage_a_source_configs = KNOWLEDGEBASE_RAW_STAGE_A_SOURCECONFIGS

    if not os.path.exists(stage_a_source_configs):
        print(f"[ERROR] Configuration file does not exist: {stage_a_source_configs}")
        print("请参考脚本内 process_source() 的 docstring 创建配置文件，示例见 EXAMPLE_CONFIG。")
        return

    with open(stage_a_source_configs, "r", encoding="utf-8") as f:
        source_configs = json.load(f)

    all_records = []
    all_records.extend(process_source(source_configs))
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = f"{source_configs['output_dir']}{source_configs['run_tag']}_{timestamp}.json"

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n阶段A完成，共产出 {len(all_records)} 条案例级记录，已写入 {output}")
    print("（此时尚未去重、尚未正式入库，请接着运行 stage_b_dedup.py）")


# ==================== 示例配置（可直接复制为 source_configs.json） ====================
EXAMPLE_CONFIG =    {
        "name": "sre_book",
        "file_type": "text",   # 最重要的参数instance/text, 区分是不是直接是案例或者要分片
        "local_dir": "D:\\projects\\knowledgeBase_raw\\google_sre_book",
        "source_category": "sre_book_theory",
        "url_base": "https://sre.google/sre-book",
        "glob_pattern": "*.md",
        "output_dir": "D:\\projects\\X-RAG\\data\\KnowledgeBase_processed\\google_sre_book\\",
        "run_tag": "test",  # 运行标签,结合上面output_dir，运行结果会存到output_dir+run_tag+时间戳的json文件中
        "include_files": [
            "12_effective-troubleshooting.md",
            "19_load-balancing-frontend.md",
            "20_load-balancing-datacenter.md",
            "21_handling-overload.md",
            "22_addressing-cascading-failures.md",
            "23_managing-critical-state.md",
        ]
    }


if __name__ == "__main__":
    main()