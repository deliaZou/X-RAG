# X-RAG: Explainable RAG-Enhanced Anomaly Detection for Network Traffic

Three-layer end-to-end anomaly detection architecture: detection, attribution, diagnosis, with two false positive suppressions throughout the process.

## Architecture
![img.png](img.png)
- **Layer 1 — Detection Engine**: iForest, Train each of the 28 SMD machines separately and output the abnormal scores
- **Layer 2 — XAI Attribution**: TreeSHAP Feature attribution + Top-K sorting;Stability test (1% Gaussian noise + Spearman ρ) and Fidelity Gate (first false positive suppression)
- **Layer 3 — RAG-LLM Diagnosis**: Retrieval-Augmented Generation(RAG)  search for diagnosis + confidence level filtering (second suppression of false positives)

### Layer 1 — Detection
| Component | Choice | Status | Notes |
|---|---|---|---|
| Detector | iForest | ✅ Selected | 在检测能力、效率、FP率间最优平衡；树模型天然兼容 TreeSHAP |
| Baselines compared | LOF, HBOS, COPOD, KNN, CBLOF, PCC | ✅ Done (preliminary) | 基于 SMD 单台代表机器的初步实验，非最终性能结论 |

### Layer 2 — XAI Attribution
| Component | Choice | Status | Notes |
|---|---|---|---|
| Attribution | TreeSHAP | ✅ Selected | 树结构精确 Shapley 值，零近似误差 |
| Stability check | 1% Gaussian noise + Spearman ρ | 🔨 Implementing | |
| Fidelity Gate | Top-K 置零重打分 | 🔨 Implementing | 阈值待实验确定；第一次FP抑制 |

### Layer 3 — RAG-LLM Diagnosis
| Component | Choice | Status | Notes |
|---|---|---|---|
| LLM candidates | Mistral-7B / Llama-3.1-8B / DeepSeek-R1 (distilled) | 📋 Planned | 端到端管线已用大规模API验证，7B/8B本地候选需重调prompt |
| Embedding | all-MiniLM-L6-v2 (sentence-transformers) | 📋 Planned | 基于 MiniLM 架构 (Wang et al., 2020) |
| Knowledge base | SMD 派生：训练集统计画像 + interpretation_labels 历史案例 | 📋 Planned | 不使用AI生成的语义命名（方法论诚实性） |
| Confidence filter | Confidence-based Filter | 📋 Planned | 第二次FP抑制 |
| Eval judge | ⚠️ TBD (GPT-4 class API preferred) | ❓ Open question | 必须独立于候选模型，避免自评 |

图例：✅ 已确定 | 🔨 开发中 | 📋 已规划 | ❓ 未决

## Dataset
Server Machine Dataset (SMD), Su et al. (2019).

## Quick Start
pip install -r requirements.txt
- [ ] python scripts/run_layer1_2.py --config configs/default.yaml

## Project Info
Capstone project. Author: Zou Ting. 

## Model & Algorithm Registry

