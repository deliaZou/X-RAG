"""
Layer 2 输出固化
==============================================
用固定配置 (PCC + KernelSHAP, q=0.999) 对每个案例产出候选证据 JSON,
作为 Layer 3 的唯一输入.

每案例输出:
  评测元数据 _meta:      case_id, fold, ground_truth   仅评测用, 不进模型
  候选证据 candidates:   全部服务的服务级 SHAP 分数 + 每服务的指标明细
  案例级 fidelity:       归因忠实度, 去掉 Top-K 特征后分数下降比例, 不依赖答案

Top-N 在下游读取时决定, 此处存满全部服务, 一次固化任意调 N.

配置优先读 configs/selection.yaml, 缺省用内置 DEFAULT.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import shap
from pyod.models.pca import PCA as PyodPCA

# ============================ CONFIG ============================

BASE = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"
CONFIG = r"D:\projects\X-RAG\notebooks\RCAEval_ds\selection.yaml"   # 可缺省
OUT = os.path.join(BASE, "layer2_output")

DEFAULT = {
    "pca_n_components": 0.95,
    "random_state": 0,
    "eval_quantile": 0.999,
    "kernel_background": 50,
    "max_explain": 15,          # 每案例最多解释的异常点数
    "fidelity_topk": 5,         # 案例级 fidelity 去掉的 Top-K 特征数
}

FAULT_TO_FAMILY = {"cpu": "cpu", "mem": "mem", "disk": "diskio",
                   "socket": "socket", "delay": "latency", "loss": "latency"}

# ================================================================


def load_config():
    cfg = dict(DEFAULT)
    if os.path.exists(CONFIG):
        try:
            import yaml
            y = yaml.safe_load(open(CONFIG, encoding="utf-8"))
            l1 = (y or {}).get("layer1", {})
            l2 = (y or {}).get("layer2", {})
            for k in ("pca_n_components", "random_state", "eval_quantile"):
                if k in l1:
                    cfg[k] = l1[k]
            for k in ("kernel_background", "max_explain", "fidelity_topk"):
                if k in l2:
                    cfg[k] = l2[k]
            print(f"已读配置 {CONFIG}")
        except Exception as e:
            print(f"读配置失败, 用内置默认: {e}")
    else:
        print("未找到配置文件, 用内置默认")
    return cfg


def split_service(col):
    return col.rsplit("_", 1)[0] if "_" in col else col


# def process_case(cid, row, features, cfg):
#     cdir = os.path.join(BASE, "cases", cid)
#     Xtr = np.load(os.path.join(cdir, "train.npy"))
#     Xte = np.load(os.path.join(cdir, "test.npy"))
#     y = np.load(os.path.join(cdir, "test_label.npy"))
#     meta = json.load(open(os.path.join(cdir, "meta.json")))
#
#     sc = StandardScaler().fit(Xtr)
#     Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
#
#     clf = PyodPCA(n_components=cfg["pca_n_components"],
#                   random_state=cfg["random_state"]).fit(Ztr)
#     score = lambda X: clf.decision_function(X)   # 越大越异常
#     s_tr, s_te = score(Ztr), score(Zte)
#     thr = np.quantile(s_tr, cfg["eval_quantile"])
#
#     # 选要解释的异常点: 异常段内超阈值的, 取分数最高的若干
#     anom = np.where(y == 1)[0]
#     sel = anom[s_te[anom] > thr]
#     if len(sel) == 0:
#         sel = anom
#     sel = sel[np.argsort(-s_te[sel])[:cfg["max_explain"]]]
#     Zsel = Zte[sel]
#
#     # KernelSHAP 归因
#     bg = shap.sample(Ztr, min(cfg["kernel_background"], len(Ztr)),
#                      random_state=cfg["random_state"])
#     sv = np.asarray(shap.KernelExplainer(score, bg).shap_values(Zsel, silent=True))
#     if sv.ndim == 3:
#         sv = sv[..., 0]
#
#     # 聚合: 逐点绝对值均值, 压成特征级分数
#     feat_score = np.abs(sv).mean(axis=0)                 # (D,)
#     # 原始偏移方向: 异常段均值相对训练段均值
#     direction = np.where(Xte[sel].mean(0) >= Xtr.mean(0), "up", "down")
#
#     # 服务级聚合: 同服务特征分数求和
#     svc_score = {}
#     for i, col in enumerate(features):
#         svc = split_service(col)
#         svc_score[svc] = svc_score.get(svc, 0.0) + float(feat_score[i])
#     svc_rank = sorted(svc_score, key=svc_score.get, reverse=True)
#
#     # 构造候选证据: 全部服务, 每服务列其指标
#     candidates = []
#     for svc in svc_rank:
#         metrics = []
#         for i, col in enumerate(features):
#             if split_service(col) != svc:
#                 continue
#             metrics.append(dict(
#                 name=col,
#                 metric=col.rsplit("_", 1)[1],
#                 shap=round(float(feat_score[i]), 5),
#                 direction=str(direction[i])))
#         metrics.sort(key=lambda m: m["shap"], reverse=True)
#         candidates.append(dict(service=svc,
#                                service_shap=round(svc_score[svc], 5),
#                                metrics=metrics))
#
#     # 案例级 fidelity: 去掉 Top-K 特征后, 分数下降比例
#     topk_idx = np.argsort(-feat_score)[:cfg["fidelity_topk"]]
#     Zmask = Zsel.copy()
#     Zmask[:, topk_idx] = 0.0                             # 用背景值 0 替换 (已标准化)
#     s_full = score(Zsel).mean()
#     s_mask = score(Zmask).mean()
#     base = score(np.zeros((1, Zsel.shape[1]))).item()
#     denom = abs(s_full - base) + 1e-9
#     fidelity = float(max(0.0, (s_full - s_mask) / denom))  # 越接近 1 越忠实
#
#     gt_metric = meta["gt_metric"]
#     out = {
#         "_meta": {
#             "case_id": cid,
#             "fold": int(row.fold),
#             "ground_truth": {"service": meta["root_cause_service"],
#                              "fault": meta["fault_type"],
#                              "metric": gt_metric},
#         },
#         "inject_relative": True,          # 不传绝对时间戳
#         "n_explained_points": int(len(sel)),
#         "attribution_fidelity": round(fidelity, 4),
#         "candidates": candidates,         # 全部服务, 下游按 Top-N 截断
#     }
#     return out, dict(case_id=cid, fault=row.fault,
#                      fidelity=round(fidelity, 4),
#                      n_points=int(len(sel)),
#                      gt_in_top5_service=meta["root_cause_service"] in svc_rank[:5])


def process_case(cid, row, features, cfg):
    cdir = os.path.join(BASE, "cases", cid)
    Xtr = np.load(os.path.join(cdir, "train.npy"))
    Xte = np.load(os.path.join(cdir, "test.npy"))
    y = np.load(os.path.join(cdir, "test_label.npy"))
    meta = json.load(open(os.path.join(cdir, "meta.json")))

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    clf = PyodPCA(n_components=cfg["pca_n_components"],
                  random_state=cfg["random_state"]).fit(Ztr)
    score = lambda X: clf.decision_function(X)   # 越大越异常
    s_tr, s_te = score(Ztr), score(Zte)
    thr = np.quantile(s_tr, cfg["eval_quantile"])

    # 选要解释的异常点: 异常段内超阈值的, 取分数最高的若干
    anom = np.where(y == 1)[0]
    sel = anom[s_te[anom] > thr]
    if len(sel) == 0:
        sel = anom
    sel = sel[np.argsort(-s_te[sel])[:cfg["max_explain"]]]
    Zsel = Zte[sel]

    # KernelSHAP 归因
    bg = shap.sample(Ztr, min(cfg["kernel_background"], len(Ztr)),
                     random_state=cfg["random_state"])
    sv = np.asarray(shap.KernelExplainer(score, bg).shap_values(Zsel, silent=True))
    if sv.ndim == 3:
        sv = sv[..., 0]

    # 聚合: 逐点绝对值均值, 压成特征级分数
    feat_score = np.abs(sv).mean(axis=0)                 # (D,)
    # 原始偏移方向: 异常段均值相对训练段均值
    direction = np.where(Xte[sel].mean(0) >= Xtr.mean(0), "up", "down")

    # 服务级聚合: 同服务特征分数求和
    svc_score = {}
    for i, col in enumerate(features):
        svc = split_service(col)
        svc_score[svc] = svc_score.get(svc, 0.0) + float(feat_score[i])
    svc_rank = sorted(svc_score, key=svc_score.get, reverse=True)

    # ========== 新格式: 两个独立列表 ==========
    # 1. service_list: 服务名 + service_shap
    service_list = [
        {
            "service": svc,
            "service_shap": round(svc_score[svc], 5)
        }
        for svc in svc_rank
    ]

    # 2. metrics_list: 所有指标, 包含 name, shap, direction
    metrics_list = []
    for i, col in enumerate(features):
        metrics_list.append({
            "name": col,
            "shap": round(float(feat_score[i]), 5),
            "direction": str(direction[i])
        })
    # 按 shap 降序排列
    metrics_list.sort(key=lambda m: m["shap"], reverse=True)

    # 案例级 fidelity: 去掉 Top-K 特征后, 分数下降比例
    topk_idx = np.argsort(-feat_score)[:cfg["fidelity_topk"]]
    Zmask = Zsel.copy()
    Zmask[:, topk_idx] = 0.0
    s_full = score(Zsel).mean()
    s_mask = score(Zmask).mean()
    base = score(np.zeros((1, Zsel.shape[1]))).item()
    denom = abs(s_full - base) + 1e-9
    fidelity = float(max(0.0, (s_full - s_mask) / denom))

    gt_metric = meta["gt_metric"]
    out = {
        "_meta": {
            "case_id": cid,
            "fold": int(row.fold),
            "ground_truth": {
                "service": meta["root_cause_service"],
                "fault": meta["fault_type"],
                "metric": gt_metric,
            },
        },
        "inject_relative": True,
        "n_explained_points": int(len(sel)),
        "attribution_fidelity": round(fidelity, 4),
        "service_list": service_list,      # ← 新格式
        "metrics_list": metrics_list,      # ← 新格式
    }
    return out, dict(case_id=cid, fault=row.fault,
                     fidelity=round(fidelity, 4),
                     n_points=int(len(sel)),
                     gt_in_top5_service=meta["root_cause_service"] in svc_rank[:5])


def run():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    print("配置:", cfg)

    man = pd.read_csv(os.path.join(BASE, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)
    features = json.load(open(os.path.join(BASE, "feature_names.json")))

    summary = []
    for i, r in man.iterrows():
        out, s = process_case(r.case_id, r, features, cfg)
        with open(os.path.join(OUT, f"{r.case_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        summary.append(s)
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(man)}")

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, "_summary.csv"), index=False)

    print(f"\n完成, 输出 {len(sm)} 份 JSON 到 {OUT}")
    print("\n案例级 fidelity 分布, 按故障类型:")
    print(sm.groupby("fault")["fidelity"].agg(["mean", "min", "max"]).round(3).to_string())
    print(f"\nsocket 类 fidelity 明显偏低则印证 PCC 残差不覆盖 socket 指标")


if __name__ == "__main__":
    run()
