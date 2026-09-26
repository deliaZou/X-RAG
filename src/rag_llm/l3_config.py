"""
l3_config.py
============
统一加载 Layer3 的超参数配置 (configs/l3_config.yaml)。

之前 MAX_CONTEXT_CHARS、RAGAS_SAMPLE_MIN、temperature、top_k_each、
candidate_pool_size 这些数字散落在 rag_llm_layer.py 代码里,想复现某次
实验的完整设置得回头翻 git log。现在集中到一份配置文件,和 prompt 版本
一起,构成一次实验的完整、可复现的设置。
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "l3_config.yaml"

_DEFAULTS = {
    "max_context_chars": 400,
    "ragas_sample_min": 3,
    "temperature": 0.1,
    "top_k_each": 2,
    "candidate_pool_size": 10,
    "query_chain_size": 5,
    "prompt": "diagnosis_prompt.txt"
}


def load_l3_config(path: Path = None) -> dict:
    """
    读取配置文件并和默认值合并 (文件里没写的字段回退默认值)。
    找不到配置文件时不报错,直接用默认值 —— 方便在还没建配置文件的
    环境里也能跑起来。
    """
    path = path or CONFIG_PATH
    config = dict(_DEFAULTS)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
    return config
