"""
stage_b_dedup.py
==================
阶段B：查重

职责：
  - 读取阶段A产出的 cases_intermediate.json
  - 读取（或首次生成）embeddings_cache.pkl —— 已有知识库（playbook_ext 全量）的 embedding 缓存
  - 对中间json里的每条候选 record，逐条计算 embedding，与"已有库 + 本批次中已确认为新的候选"
    做 cosine similarity 比较；相似度超过阈值判定为重复，跳过
  - 输出：to_insert.json （确定要写入正式知识库的记录清单）

注意：
  - 本脚本不直接写入正式知识库 json，也不落盘更新 embeddings_cache.pkl ——
    这两件事交给 stage_c_insert.py 统一做（写入知识库的同时才是"这条记录真正存在"的时刻，
    避免脚本2和正式库之间产生不一致）。
  - 本脚本运行期间会在内存中维护一份"缓存+本批次已接受记录"的合集，用于捕获同一批次内部的重复
    （比如第21章和第22章切出的相似案例），这部分不落盘，仅供本次运行内部比较使用。
  - embedding 只在候选记录首次出现时计算一次，之后直接复用同一个向量做后续比较，不重复计算。

用法：
  python stage_b_dedup.py \
      --intermediate cases_intermediate.json \
      --existing_kb knowledgeBase_raw/rag_knowledge_base.json \
      --cache embeddings_cache.pkl \
      --output to_insert.json \
      --threshold 0.85
"""

import os
import re
import json
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from configs.paths import KNOWLEDGEBASE_RAW_STAGE_B_SOURCECONFIGS

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    print(f"加载本地嵌入模型: {EMBEDDING_MODEL_NAME} ...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_cache(cache_path: str, existing_kb_path: str, model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """
    加载 embeddings_cache.pkl；若不存在，则基于现有知识库（playbook_ext）全量生成一份并保存。
    cache 的 key 用 record 的 id，value 是 embedding 向量。
    """

    print(f"基于现有知识库 {existing_kb_path} 首次生成 ...")
    cache: Dict[str, np.ndarray] = {}

    if os.path.exists(existing_kb_path):
        with open(existing_kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        existing_records = kb.get("playbook_ext", []) if isinstance(kb, dict) else kb

        texts = [r["text"] for r in existing_records]
        ids = [r["id"] for r in existing_records]
        if texts:
            vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            for rid, vec in zip(ids, vectors):
                cache[rid] = vec

    # 首次生成后落盘，避免下次重算
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"已生成初始缓存，共 {len(cache)} 条，写入 {cache_path}")
    return cache


def clean_text_for_embedding(text: str) -> str:
    """
    去除代码块 / markdown 标题符号等噪声，避免稀释语义相似度信号。
    仅用于计算 embedding，不影响正式入库的 text 字段本身。
    """
    import re
    text = re.sub(r"```[\s\S]*?```", " ", text)   # 去掉代码块
    text = re.sub(r"`[^`]*`", " ", text)            # 去掉行内代码
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # 去掉标题符号
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # 向量已 normalize，点积即余弦相似度


KEEP_KEYS = ["id", "case_id", "type", "source_category", "fault_type",
             "triggers", "source", "source_url", "text"]


def strip_internal_fields(record: dict) -> dict:
    """正式合并进总库前，去掉 _duplicate_of / _similarity 等阶段B调试字段。"""
    return {k: record[k] for k in KEEP_KEYS if k in record}


def merge_and_save_versioned(existing_kb_path: str, to_insert: List[dict]) -> str:
    """
    把去重后确认要入库的记录追加进现有总库，另存为带时间戳的新文件
    （不覆盖原 docs_cache.json，保留历史版本可回滚）。
    阶段C 只需要读这一个合并后的文件即可完成入库。
    """
    if os.path.exists(existing_kb_path):
        with open(existing_kb_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    cache.setdefault("playbook_ext", [])
    clean_records = [strip_internal_fields(r) for r in to_insert]
    cache["playbook_ext"].extend(clean_records)

    base_dir = os.path.dirname(existing_kb_path) or "."
    # base_name = os.path.splitext(os.path.basename(existing_kb_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(base_dir, f"kb_docs_cache_{timestamp}.json")

    with open(versioned_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return versioned_path


def find_most_similar(query_vec: np.ndarray, pool: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """在 pool 中找相似度最高的一条，返回 (id, similarity)。pool 为空时返回 (None, 0.0)"""
    best_id, best_sim = None, 0.0
    for rid, vec in pool.items():
        sim = cosine_sim(query_vec, vec)
        if sim > best_sim:
            best_id, best_sim = rid, sim
    return best_id, best_sim


def main():
    stage_b_source_configs = KNOWLEDGEBASE_RAW_STAGE_B_SOURCECONFIGS

    if not os.path.exists(stage_b_source_configs):
        print(f"[ERROR] Configuration file does not exist: {stage_b_source_configs}")
        print("请参考脚本内 process_source() 的 docstring 创建配置文件，示例见 EXAMPLE_CONFIG。")
        return

    with open(stage_b_source_configs, "r", encoding="utf-8") as f:
        source_configs = json.load(f)

    intermediate = source_configs["input_dir"] + source_configs["file_name"]

    with open(intermediate, "r", encoding="utf-8") as f:
        candidates: List[dict] = json.load(f)

    print(f"读取到 {len(candidates)} 条候选记录（阶段A产出）")

    model = load_model()
    existing_kb = source_configs["input_dir"] + source_configs["existing_kb"]
    cache = source_configs["input_dir"] + source_configs["cache"]
    threshold = source_configs["threshold"]
    _cache = build_cache(cache, existing_kb, model)

    # 内存中的比较池：从已有缓存出发，运行期间逐条扩充（用于捕获批内重复），不落盘
    comparison_pool: Dict[str, np.ndarray] = dict(_cache)

    to_insert = []
    skipped_duplicates = []

    for record in tqdm(candidates, desc="查重中"):
        clean_text = clean_text_for_embedding(record["text"])
        vec = model.encode(clean_text, convert_to_numpy=True, normalize_embeddings=True)

        best_id, best_sim = find_most_similar(vec, comparison_pool)

        if best_sim >= threshold:
            record["_duplicate_of"] = best_id
            record["_similarity"] = round(best_sim, 4)
            skipped_duplicates.append(record)
            continue

        to_insert.append(record)
        # 立即加入内存比较池，供本批次后续候选比较（同一 case 展开的多条 record 也会互相比较，
        # 但因 fault_type 不同、text 相同，会被判定为"相似度=1.0 的重复"而误删——
        # 因此同一 case_id 内的记录不参与彼此的比较，见下方特殊处理）
        comparison_pool[record["id"]] = vec

    # ---- 修正：同一 case_id 展开出的多条 record 不应互相判重 ----
    # 上面的循环里，若同一 case 的第二条 fault_type record 紧跟第一条之后处理，
    # 会在 comparison_pool 中找到第一条（相似度=1.0）而被误判为重复。
    # 这里做一次二次检查：把因"同 case_id"而被跳过的记录捞回来。
    reclaimed = []
    remaining_dupes = []
    for record in skipped_duplicates:
        dup_of_id = record.get("_duplicate_of", "")
        same_case = any(
            r["id"] == dup_of_id and r.get("case_id") == record.get("case_id")
            for r in to_insert
        )
        if same_case:
            record.pop("_duplicate_of", None)
            record.pop("_similarity", None)
            reclaimed.append(record)
        else:
            remaining_dupes.append(record)

    to_insert.extend(reclaimed)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # output = f"{source_configs['output_dir']}{source_configs['run_tag']}_{timestamp}.json"
    # with open(output, "w", encoding="utf-8") as f:
    #     json.dump(to_insert, f, ensure_ascii=False, indent=2)

    # dupe_log_path = output.replace(".json", "_skipped_duplicates.json")
    # with open(dupe_log_path, "w", encoding="utf-8") as f:
    #     json.dump(remaining_dupes, f, ensure_ascii=False, indent=2)

    versioned_path = merge_and_save_versioned(existing_kb, to_insert)

    print(f"\n阶段B完成：")
    print(f"  待入库记录: {len(to_insert)} 条（原始调试版，含_duplicate_of等字段） → {source_configs["input_dir"]}")
    # print(f"  判重跳过:   {len(remaining_dupes)} 条 → {dupe_log_path}（供人工核查）")
    print(f"  同case误判修正: {len(reclaimed)} 条")
    print(f"  合并后总库(新版本): {versioned_path}")
    print(f"\n阶段C只需要传这一个文件即可完成入库：--kb_docs_cache {versioned_path}")
    print(f"（embeddings_cache.pkl 仍未更新——它只用于查重比对，正式入库后由阶段C按实际写入ChromaDB的记录更新）")


# ==================== 示例配置（可直接复制为 source_configs.json） ====================
EXAMPLE_CONFIG =    {
    "input_dir": "D:\\projects\\X-RAG\\data\\KnowledgeBase_processed\\",
    "file_name": "google_sre_book\\test_20260920_1707.json", # stageA 输出的结果和上面dir拼接
    "existing_kb": "kb_docs_cache.json",  # 已知的全量文件， 要改成最新的
    "cache": "embeddings_cache.pkl",   # 已知全量文件对应的pkl，每次都重新生成
    "output_dir": "D:\\projects\\X-RAG\\data\\KnowledgeBase_processed\\",
    "threshold" : 0.85    # 余弦相似度对比
}


if __name__ == "__main__":
    main()