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

BASE = r"D:\projects\X-RAG\RCAEval_ds\trans-OB"
CONFIG = r"D:\projects\X-RAG\RCAEval_ds\selection.yaml"   # 可缺省
# OUT = os.path.join(BASE, "layer2_output")
OUT = "D:\projects\X-RAG\src\pcc_kernelSHAP\layer2_output"

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
    # direction = np.where(Xte[sel].mean(0) >= Xtr.mean(0), "up", "down")

    # 按异常标签时间点计算候选的时间序列证据
    anomaly_indices = np.where(y == 1)[0]

    def candidate_evidence(i):
        mu = Xtr[:, i].mean()
        sd = Xtr[:, i].std()
        anomaly_values = Xte[anomaly_indices, i]

        if sd < 1e-9:
            mean_delta = (
                float(anomaly_values.mean() - mu)
                if len(anomaly_values) else 0.0
            )
            return {
                "zscore_valid": False,
                "mean_zscore": None,
                "mean_abs_zscore": None,
                "max_abs_zscore": None,
                "direction": (
                    "up" if mean_delta > 0
                    else "down" if mean_delta < 0
                    else "stable"
                ),
                "duration_samples": None,
                "trend": "unknown",
            }

        z = (Xte[:, i] - mu) / sd
        anomaly_z = z[anomaly_indices]

        # 只统计异常标签范围内 |z| >= 2 的最长连续区间
        active = (y == 1) & (np.abs(z) >= 2)
        longest = current = 0
        for is_active in active:
            current = current + 1 if is_active else 0
            longest = max(longest, current)

        # 在线性趋势计算前保留时间顺序
        if len(anomaly_indices) >= 2:
            slope = np.polyfit(anomaly_indices, Xte[anomaly_indices, i], 1)[0]
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        else:
            trend = "unknown"

        mean_z = float(anomaly_z.mean()) if len(anomaly_z) else 0.0
        return {
            "zscore_valid": True,
            "mean_zscore": round(mean_z, 5),
            "mean_abs_zscore": round(float(np.abs(anomaly_z).mean()), 5)
            if len(anomaly_z) else 0.0,
            "max_abs_zscore": round(float(np.abs(anomaly_z).max()), 5)
            if len(anomaly_z) else 0.0,
            "direction": "up" if mean_z > 0 else "down" if mean_z < 0 else "stable",
            "duration_samples": int(longest),
            "trend": trend,
        }

    allowed_metrics = {"cpu", "latency", "diskio", "mem", "socket"}
    eligible_indices = [
        i for i, col in enumerate(features)
        if col.rsplit("_", 1)[-1] in allowed_metrics
    ]

    # service_list 只根据允许的根因指标聚合 SHAP
    svc_score = {}
    for i in eligible_indices:
        svc = split_service(features[i])
        svc_score[svc] = svc_score.get(svc, 0.0) + float(feat_score[i])
    svc_rank = sorted(svc_score, key=svc_score.get, reverse=True)

    service_list = [
        {"service": svc, "service_shap": round(svc_score[svc], 5)}
        for svc in svc_rank
    ]

    # metrics_list 只输出允许的根因指标
    metrics_list = []
    for i in eligible_indices:
        evidence = candidate_evidence(i)
        metrics_list.append({
            "name": features[i],
            "shap": round(float(feat_score[i]), 5),
            "zscore_valid": evidence["zscore_valid"],
            "direction": evidence["direction"],
            "mean_zscore": evidence["mean_zscore"],
            "mean_abs_zscore": evidence["mean_abs_zscore"],
            "max_abs_zscore": evidence["max_abs_zscore"],
            "duration_samples": evidence["duration_samples"],
            "trend": evidence["trend"],
        })

    # 保持 metrics_list 主顺序为 SHAP 排名，便于兼容现有 adapter
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
