import os
import glob
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm
"""
这个脚本会遍历本地克隆/下载的 prometheus-operator/runbooks 文件夹，提取每个 Markdown 文件，
并调用 DeepSeek 模型进行分类与关联评估，最后输出一个标准的 JSON 清单。
"""
# ==================== 1. 配置项 ====================
LLM_MODEL_ID = "deepseek-v4-flash"
LLM_API_KEY="sk-f8d2ebc3a0614642b31fb02b356d0c9d"
LLM_BASE_URL = "https://api.deepseek.com"

# 本地 Runbooks 仓库地址 (根据实际修改)
LOCAL_RUNBOOKS_DIR = "D:\\projects\\runbooks\\content\\runbooks"
# GitHub 相对根路径
GITHUB_BASE_URL = "https://github.com/prometheus-operator/runbooks/blob/main/docs"

# 输出结果清单路径
OUTPUT_FILE = "runbook_preprocessing_manifest.json"

# 初始化 Client
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)


# ==================== 2. 定义严格对齐的 Pydantic 结构 ====================
class RunbookAnalysis(BaseModel):
    is_relevant: bool = Field(
        description="是否属于微服务/节点级别的物理性能故障"
    )
    mapped_fault_type: Optional[str] = Field(
        default=None,
        description="粗粒度: ['CPU', 'Delay', 'Disk', 'Loss', 'Mem', 'Socket'] 之一，若无关则为 null"
    )
    trigger_metrics: List[str] = Field(
        default=[],
        description="细粒度: ['cpu', 'mem', 'socket', 'workload', 'diskio', 'latency', 'error']"
    )


# ==================== 3. 核心提取逻辑（仅请求 1 次 LLM） ====================
def analyze_runbook_with_llm(filename: str, content: str) -> RunbookAnalysis:
    """一次性调用 DeepSeek 提取粗细粒度元数据"""

    # 将标准和契约直接写入 Prompt 约束
    prompt = f"""你是一个 AIOps 运维知识库预处理专家。请分析以下 Runbook Markdown 文档，提取元数据。
1. `is_relevant` (布尔值):
   - 若属于工具内部配置/测试等与微服务/节点故障无关的 Runbook（如 AlertmanagerFailedReload），设为 false；否则为 true。
   ⚠️ 注意：
    1. 涉及 TLS/SSL 证书过期、配置校验错误、版本不兼容、凭证失效等管理类问题，不属于上述 6 种微服务物理故障，请将 is_relevant 设为 false，mapped_fault_type 设为 null！
    2. 只有文档明确属于微服务/节点级物理性能指标故障时，才设 is_relevant 为 true。
    
【粗细粒度分类体系规则】：
2. `mapped_fault_type` :
   - 必须且只能从这 6 个词中选择 1 个：['CPU', 'Delay', 'Disk', 'Loss', 'Mem', 'Socket']
   【粗粒度故障分类（mapped_fault_type）的严格定义】：
    - CPU: CPU 满载、CPU throttling、计算资源竞争。
    - Mem: 内存溢出 (OOM)、内存泄漏、Swap 交换瓶颈。
    - Disk: 磁盘空间爆满、磁盘 IO 读写瓶颈/延迟。
    - Loss: 网络丢包、包损坏、网络断连。
    - Delay: 网络延迟突增、RTT 变长。
    - Socket: TCP/UDP 连接数耗尽、端口占用、 Socket 缓冲区溢出。

3. `trigger_metrics` :
   - 必须且只能从这 7 个小写单词中选择（可多选）：['cpu', 'mem', 'socket', 'workload', 'diskio', 'latency', 'error']
   - 作用：当文档提到多种指标或排查步骤涉及多指标时全部列出，用于细粒度 Metric 匹配。



【文件名】：{filename}
【文档内容（前 2500 字）】：
{content[:2500]}
"""

    try:
        # 单次 API 请求，利用 JSON Mode 一次性返回结果
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": "你是一个严格输出 JSON 的 AIOps 元数据提取工具。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        res_json = json.loads(response.choices[0].message.content)
        # 用 Pydantic 校验格式是否合规
        return RunbookAnalysis(**res_json)

    except Exception as e:
        print(f"\n[Warning] 处理文件 {filename} 解析/请求失败: {e}")
        return RunbookAnalysis(is_relevant=False, mapped_fault_type=None, trigger_metrics=[])


# ==================== 4. 主遍历流程 ====================
def main():
    results = []
    md_files = glob.glob(os.path.join(LOCAL_RUNBOOKS_DIR, "**/*.md"), recursive=True)
    print(f"扫描到 {len(md_files)} 个 Runbook 文件，开始单次请求预处理...")

    for file_path in tqdm(md_files):
        filename = os.path.basename(file_path)

        if filename in ["_index.md", "README.md"]:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, LOCAL_RUNBOOKS_DIR)
        github_url = f"{GITHUB_BASE_URL}/{rel_path.replace(os.sep, '/')}"

        # 仅发起 1 次 LLM 调用
        analysis = analyze_runbook_with_llm(filename, content)

        record = {
            "file_name": filename,
            "local_path": file_path,
            "sub_url": rel_path.replace(os.sep, "/"),  # 子文件夹/相对路径
            "github_url": github_url,  # GitHub 地址
            "is_relevant": analysis.is_relevant,  # 是否相关
            "mapped_fault_type": analysis.mapped_fault_type,  # 粗粒度: 6 类之一
            "trigger_metrics": analysis.trigger_metrics  # 细粒度: 7 类列表中若干个
        }
        results.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n预处理完成！清单存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()