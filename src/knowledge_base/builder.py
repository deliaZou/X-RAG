import argparse
import os
import json
import re
from typing import List, Dict, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from configs.paths import DOCS_CACHE_PATH, VECTOR_DB_DIR


EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "re2ob_knowledge"


ALL_TYPES = ["feature", "playbook", "playbook_ext", "alert", "algo", "topology"]
TYPE_TAG  = {  # 内部 type 字段命名,和 rag_llm_layer.py 的四类 (+topology) 对齐
    "feature":     "feature_semantic",
    "playbook":    "playbook",
    "playbook_ext": "playbook",  # 和 LLM 生成的 playbook 共用同一个检索 tag,
                                  # 靠 source_category 字段区分来源,不共用同一个 --only 开关
    "alert":       "historical_alert",
    "algo":        "algo_config",
    "topology":    "topology",
}
DELETE_WHERE = {
    "feature":     {"type": TYPE_TAG["feature"]},
    "playbook":    {"$and": [{"type": TYPE_TAG["playbook"]}, {"source_category": "llm_generated"}]},
    "playbook_ext": {"$and": [{"type": TYPE_TAG["playbook_ext"]}, {"source_category": "external_runbook"}]},
    "alert":       {"type": TYPE_TAG["alert"]},
    "algo":        {"type": TYPE_TAG["algo"]},
    "topology":    {"type": TYPE_TAG["topology"]},
}

def load_docs(path) -> Dict[str, List[Dict]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # return {t: [] for t in ALL_TYPES}

def save_docs_cache(cache: Dict[str, List[Dict]]):
    with open(DOCS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  文档缓存已保存到 {DOCS_CACHE_PATH}")

def get_chroma_path(embed_model: str) -> str:
    """
    按 embedding 模型区分向量库路径,避免换模型时互相覆盖,
    也让"固定知识库,换 embedding 模型对比"这个受控实验能同时保留多份结果。
    可用环境变量 XRAG_KB_ROOT 覆盖基准目录。
    """
    slug = embed_model.replace("/", "_").replace(":", "_")
    # base = os.getenv("XRAG_KB_ROOT", VECTOR_DB_DIR)
    return os.path.join(VECTOR_DB_DIR, f"chroma_re2ob_kb__{slug}")

def _clean_markdown(raw: str, max_chars: int = 1500) -> str:
    """去掉图片/badge/<details>折叠标签等噪音,保留标题和正文结构,不做语义改写。"""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", raw)  # 图片
    text = re.sub(r"<details>|</details>|<summary>.*?</summary>", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def ingest_curated_runbooks(manifest_path: str) -> List[Dict]:
    """
    读取人工审核过的外部 runbook manifest (is_relevant/mapped_fault_type/trigger_metrics
    那份清单),只收 is_relevant==true 的条目,直接用 local_path 读取本机文件内容,
    不做路径可移植性处理。
    """
    if not os.path.exists(manifest_path):
        print(f"  ⚠️  {manifest_path} 不存在,跳过外部 runbook")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = []
    skipped = 0
    for entry in manifest:
        if not entry.get("is_relevant"):
            skipped += 1
            continue

        local_path = entry.get("local_path", "")
        if not os.path.exists(local_path):
            print(f"  ⚠️  文件不存在,跳过: {local_path}")
            continue

        with open(local_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        text = _clean_markdown(raw_text)

        fault_type = entry.get("mapped_fault_type", "unknown")
        triggers = entry.get("trigger_metrics", [])
        doc_id = f"playbook_ext_{entry['file_name'].replace('.md', '')}"

        docs.append({
            "id": doc_id,
            "type": TYPE_TAG["playbook_ext"],
            "source_category": "external_runbook",
            "fault_type": fault_type,
            "triggers": triggers,
            "source": entry.get("sub_url", ""),
            "source_url": entry.get("github_url", ""),
            "text": f"[Playbook: {fault_type} fault | source: prometheus-operator/runbooks]\n{text}",
        })

    print(f"  已收录 {len(docs)} 条外部 runbook (跳过 {skipped} 条 is_relevant=false)")
    return docs


# ─────────────────────────────────────────────
# 向量化 + 写入 ChromaDB (支持增量: 只替换指定 type 的文档)
# ─────────────────────────────────────────────
def upsert_chroma(all_docs_by_type: Dict[str, List[Dict]], only_types: Optional[List[str]] = None,
                   embed_model: str = EMBED_MODEL):
    chroma_path = get_chroma_path(embed_model)
    print(f"\n[ChromaDB] 打开持久化存储: {chroma_path} (embed_model={embed_model})")
    db = chromadb.PersistentClient(path=chroma_path)

    if only_types is None:
        try:
            db.delete_collection(COLLECTION_NAME)
            print("  全量重建: 旧集合已清除")
        except Exception:
            pass
        collection = db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        types_to_write = ALL_TYPES
    else:
        collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        types_to_write = only_types
        # 增量模式: 先删掉这几类旧文档,再重新写入,避免重复 id 冲突
        # playbook/playbook_ext 共用同一个检索 tag,delete 必须再叠加 source_category
        # 过滤,否则重建 playbook 会把 playbook_ext 的文档一并清掉
        for t in types_to_write:
            try:
                collection.delete(where=DELETE_WHERE[t])
                print(f"  增量模式: 已清除旧的 {t} 类文档")
            except Exception:
                pass

    print(f"[Embedding] 加载模型: {embed_model} ...")
    embedder = SentenceTransformer(embed_model)

    docs_to_add = []
    for t in types_to_write:
        docs_to_add.extend(all_docs_by_type.get(t, []))

    if not docs_to_add:
        print("  没有文档需要写入")
        return

    texts = [d["text"] for d in docs_to_add]
    ids = [d["id"] for d in docs_to_add]
    metadatas = [
        {k: (", ".join(v) if isinstance(v, list) else str(v))
         for k, v in d.items() if k not in ("text", "id")}
        for d in docs_to_add
    ]

    print(f"[Embedding] 正在嵌入 {len(texts)} 条文档...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    BATCH = 50
    for start in range(0, len(texts), BATCH):
        end = min(start + BATCH, len(texts))
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"[ChromaDB] 已写入 {len(texts)} 条文档 ✓")


# ─────────────────────────────────────────────
# 轻量入口：只 upsert playbook_ext（目前向量库里唯一有数据的类型）
# ─────────────────────────────────────────────
def upsert_playbook_ext_from_file(docs_cache_path: str, embed_model: str = EMBED_MODEL):
    """
    专给 stage_b_dedup.py 产出的合并总库文件（docs_cache_<timestamp>.json）用。

    现状：向量库里目前只有 playbook_ext 这一类数据，其他类型(feature/alert/algo/topology)
    都还没有内容，所以不需要走 upsert_chroma() 里那套"先delete指定source_category再add"
    的通用多类型重建逻辑——直接把文件里的 playbook_ext 列表整批 upsert 进 ChromaDB 即可。

    用 upsert 而不是 add：对已存在的 id 会原地更新，对新 id 才插入，天然幂等——
    重复运行本函数、或者传入的文件里混着历史记录+新增记录，都不会报错或产生副作用。
    """
    with open(docs_cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    docs = cache.get("playbook_ext", [])
    if not docs:
        print("文件中没有 playbook_ext 数据，跳过")
        return

    chroma_path = get_chroma_path(embed_model)
    print(f"\n[ChromaDB] 打开持久化存储: {chroma_path} (embed_model={embed_model})")
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    print(f"[Embedding] 加载模型: {embed_model} ...")
    embedder = SentenceTransformer(embed_model)

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [
        {k: (", ".join(v) if isinstance(v, list) else str(v))
         for k, v in d.items() if k not in ("text", "id")}
        for d in docs
    ]

    print(f"[Embedding] 正在嵌入 {len(texts)} 条 playbook_ext 文档...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    BATCH = 50
    for start in range(0, len(texts), BATCH):
        end = min(start + BATCH, len(texts))
        collection.upsert(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"[ChromaDB] 已 upsert {len(texts)} 条 playbook_ext 文档 ✓（幂等：重复id会被更新而非报错）")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                         help=f"逗号分隔,只重新生成/写入这些类型,可选: {','.join(ALL_TYPES)}。"
                              "不传则全量重建。")
    parser.add_argument("--l2-dir", default="./data/re2ob/train_fold",
                         help="Type B 用,train-fold 的 L2 输出 json 所在目录")
    parser.add_argument("--config-yaml", default="./configs/selection.yaml",
                         help="Type D 用,算法配置 yaml 路径")
    parser.add_argument("--runbook-manifest", default="./runbook_manifest.json",
                         help="Type C-ext 用,人工审核过的外部 runbook 清单 (is_relevant/"
                              "mapped_fault_type/trigger_metrics/local_path)")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                         help="embedding 模型名,不同模型会写入不同的 Chroma 路径 "
                              "(./chroma_re2ob_kb__<model>),互不覆盖,方便做模型对比实验")
    parser.add_argument("--playbook-ext-cache", default=None,
                         help="直接传入阶段B产出的合并总库文件(docs_cache_<timestamp>.json)，"
                              "只 upsert 其中的 playbook_ext 列表，跳过其他类型的通用重建流程。"
                              "传了这个参数时，忽略 --only/--runbook-manifest 等其他入库相关参数。")
    args = parser.parse_args()

    if args.playbook_ext_cache:
        print("=" * 55)
        print("RE2-OB 知识库更新: playbook_ext 轻量 upsert 模式")
        print("=" * 55)
        upsert_playbook_ext_from_file(args.playbook_ext_cache, embed_model=args.embed_model)
        return

    only_types = args.only.split(",") if args.only else None
    if only_types:
        invalid = [t for t in only_types if t not in ALL_TYPES]
        if invalid:
            raise ValueError(f"未知类型: {invalid},可选: {ALL_TYPES}")

    types_to_build = only_types or ALL_TYPES
    print("=" * 55)
    print(f"RE2-OB 知识库构建: {'全量重建' if only_types is None else f'增量 ({types_to_build})'}")
    print("=" * 55)

    cache = load_docs(DOCS_CACHE_PATH)

    if "playbook_ext" in types_to_build:
        print("\n[Type C-ext] 收录外部 runbook...")
        # 注意：只替换 source_category=="external_runbook" 的部分，
        # 保留其他来源（如 stage_c_insert.py 追加进来的 sre_book_theory 等）不被清空。
        other_sources = [d for d in cache.get("playbook_ext", [])
                          if d.get("source_category") != "external_runbook"]
        cache["playbook_ext"] = other_sources + ingest_curated_runbooks(args.runbook_manifest)

    # if "topology" in types_to_build:
    #     print("\n[Type E] 生成服务拓扑文档...")
    #     cache["topology"] = generate_topology_docs()

    save_docs_cache(cache)
    upsert_chroma(cache, only_types=only_types, embed_model=args.embed_model)

    total = sum(len(cache.get(t, [])) for t in ALL_TYPES)
    print("\n" + "=" * 55)
    print(f"✅ 知识库构建完成,当前缓存共 {total} 条文档 (含未在本次重建的旧类型)")
    print(f"   Chroma 路径: {get_chroma_path(args.embed_model)}")
    print(f"   Embedding 模型: {args.embed_model}")
    print("=" * 55)
    print("\n使用示例:")
    print("  from build_knowledge_base import KnowledgeBase")
    print(f'  kb = KnowledgeBase(embed_model="{args.embed_model}")')
    print('  ctx = kb.retrieve_by_type(["checkoutservice_cpu"], current_case_id="checkoutservice_cpu_1")')


if __name__ == "__main__":
    # main()
    main()