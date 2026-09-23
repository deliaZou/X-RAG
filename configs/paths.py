import os
from pathlib import Path

# 自动获取项目根目录 (X-RAG/)
BASE_DIR = Path(__file__).resolve().parents[1]

# 1. 初始文件路径 (data/raw)
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROMETHEUS_RAW_DIR = RAW_DATA_DIR / "prometheus_runbooks"
SRE_BOOK_RAW_DIR = RAW_DATA_DIR / "sre_book"

# 2. 中间文件路径 (data/processed)
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MANIFEST_PROMETHEUS = PROCESSED_DATA_DIR / "manifest_prometheus.json"
MANIFEST_SRE_BOOK = PROCESSED_DATA_DIR / "manifest_sre_book.json"
MERGED_MANIFEST = PROCESSED_DATA_DIR / "merged_manifest.json"

# 3. 知识库路径
KNOWLEDGEBASE_PROCESSED = BASE_DIR / "data" / "KnowledgeBase_processed"
KNOWLEDGEBASE_RAW = BASE_DIR / "data" / "KnowledgeBase_raw"
DOCS_CACHE_PATH = KNOWLEDGEBASE_PROCESSED / "kb_docs_cache.json"
SERVICE_TOPOLOGY = KNOWLEDGEBASE_PROCESSED / "SERVICE_TOPOLOGY.json"
KNOWLEDGEBASE_RAW_STAGE_A_SOURCECONFIGS = KNOWLEDGEBASE_RAW / "knowledgeBase_stageA_sourceConfigs.json"
KNOWLEDGEBASE_RAW_STAGE_B_SOURCECONFIGS = KNOWLEDGEBASE_PROCESSED / "knowledgeBase_stageB_sourceConfigs.json"


# 3. Vector DB 持久化目录 (data/vector_db)
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

# 4.



def get_chroma_db_path(model_name: str) -> str:
    """根据 Embedding 模型名称动态生成向量库存储路径"""
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    path = VECTOR_DB_DIR / f"chroma_re2ob_kb__{clean_model_name}"
    os.makedirs(path, exist_ok=True)
    return str(path)