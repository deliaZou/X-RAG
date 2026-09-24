"""
prompt_loader.py
================
从 prompts/ 目录加载 prompt 模板。

用 string.Template ($var 语法) 而不是原来的 str.format:
DIAGNOSIS_PROMPT 里有一段 JSON 输出格式示例,示例本身就要用花括号 {}。
用 str.format 时,这段示例里所有 {} 都得手动转义成 {{ }},容易漏改、
后续改 prompt 时也容易忘记转义。string.Template 只认 $var,天然不和
JSON 里的花括号冲突,prompts/*.txt 里可以直接原样贴 JSON 示例。
"""

from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str, version: str = None) -> Template:
    """
    name: 不带扩展名的文件名,如 "diagnosis_system"、"diagnosis_prompt"
    version: 可选。若传入,优先找 "{name}_{version}.txt",找不到再退回
        "{name}.txt"。用于切换/对比不同版本的 prompt,比如做
        prompt engineering 消融实验时跑 diagnosis_prompt_v2.txt。
    """
    candidates = []
    if version:
        candidates.append(PROMPTS_DIR / f"{name}_{version}.txt")
    candidates.append(PROMPTS_DIR / f"{name}.txt")

    for path in candidates:
        if path.exists():
            return Template(path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"未找到 prompt 文件,依次尝试过: {[str(c) for c in candidates]}")
