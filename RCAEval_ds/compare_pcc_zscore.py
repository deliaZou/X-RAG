"""
PCC+SHAP 对 z-score, 全案例归因对比
==============================================
两种归因排序:
  PCC:     KernelSHAP 对 PCC 分数, 已固化在 layer2_output
  zscore:  特征偏离训练均值的标准化幅度, 不依赖模型

对每案例算根因指标位次, 汇总细粒度与粗粒度 AC@k, 按故障类型分.
目的: 定位 PCC 在哪些故障类型输给 z-score, 为是否引入 z-score 补充提供依据.

输出
    compare_pcc_zscore/per_case_his.csv
    compare_pcc_zscore/summary.csv
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyod.models.pca import PCA as PyodPCA

# ============================ CONFIG ============================

BASE = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"
L2OUT = os.path.join(BASE, "layer2_output")     # PCC 固化输出
OUT = os.path.join(BASE, "compare_pcc_zscore")

Q = 0.999
PCA_NCOMP = 0.95
SEED = 0
K_MAX = 5

FAULT_TO_FAMILY = {"cpu": "cpu", "mem": "mem", "disk": "diskio",
                   "socket": "socket", "delay": "latency", "loss": "latency"}

# ================================================================


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def hit(ranked, ans, k=K_MAX):
    for i, x in enumerate(ranked[:k], 1):
        if x == ans:
            return i
    return None


def ranks_from_feature_scores(scores, features):
    order = np.argsort(-scores)
    cols = [features[i] for i in order]
    coarse = dedup([c.split("_")[0] for c in cols])
    fine = [(c.split("_")[0], c.split("_", 1)[1] if "_" in c else "unknown")
            for c in cols]
    return cols, coarse, fine


def pcc_scores_from_json(cid, features):
    """从固化的 layer2 JSON 还原 PCC 的特征级分数"""
    d = json.load(open(os.path.join(L2OUT, f"{cid}.json"), encoding="utf-8"))
    m = {}
    for c in d["candidates"]:
        for me in c["metrics"]:
            m[me["name"]] = me["shap"]
    return np.array([m.get(f, 0.0) for f in features]), d["_meta"]


def zscore_feature_scores(cid, features):
    """z-score: 异常段被标记点上, 特征标准化偏离的均值"""
    cdir = os.path.join(BASE, "cases", cid)
    Xtr = np.load(os.path.join(cdir, "train.npy"))
    Xte = np.load(os.path.join(cdir, "test.npy"))
    y = np.load(os.path.join(cdir, "test_label.npy"))

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    clf = PyodPCA(n_components=PCA_NCOMP, random_state=SEED).fit(Ztr)
    s_tr, s_te = clf.decision_function(Ztr), clf.decision_function(Zte)
    thr = np.quantile(s_tr, Q)

    anom = np.where(y == 1)[0]
    sel = anom[s_te[anom] > thr]
    if len(sel) == 0:
        sel = anom

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return np.abs((Xte[sel] - mu) / sd).mean(0)


def ac(ranks, k):
    r = pd.Series(ranks)
    return float(((r.notna()) & (r <= k)).mean())


def summarize(df, label):
    out = []
    for m, g in df.groupby("method"):
        rec = {"scope": label, "method": m, "n": len(g)}
        for gran in ["coarse", "fine"]:
            acs = [ac(g[f"rank_{gran}"], k) for k in range(1, K_MAX + 1)]
            rec[f"AC1_{gran}"] = acs[0]
            rec[f"AC3_{gran}"] = acs[2]
            rec[f"AC5_{gran}"] = acs[4]
            rec[f"Avg5_{gran}"] = float(np.mean(acs))
        out.append(rec)
    return pd.DataFrame(out)


def run():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(BASE, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)
    features = json.load(open(os.path.join(BASE, "feature_names.json")))

    rows = []
    for _, r in man.iterrows():
        cid = r.case_id
        pcc_s, meta = pcc_scores_from_json(cid, features)
        z_s = zscore_feature_scores(cid, features)
        svc = meta["ground_truth"]["service"]
        fine_ans = (svc, FAULT_TO_FAMILY[r.fault])

        for name, sc_arr in [("PCC", pcc_s), ("zscore", z_s)]:
            _, coarse, fine = ranks_from_feature_scores(sc_arr, features)
            rows.append(dict(case_id=cid, fault=r.fault, method=name,
                             rank_coarse=hit(coarse, svc),
                             rank_fine=hit(fine, fine_ans)))

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "per_case_his.csv"), index=False)

    parts = [summarize(per, "overall")]
    for f, g in per.groupby("fault"):
        parts.append(summarize(g, f"fault={f}"))
    summ = pd.concat(parts, ignore_index=True)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=== 总体 ===")
    print(summ[summ.scope == "overall"].round(4).to_string(index=False))

    print("\n=== 分故障类型 细粒度 AC@1 / AC@5 ===")
    piv1 = per.assign(h1=per.rank_fine == 1).pivot_table(
        index="fault", columns="method", values="h1")
    piv5 = per.assign(h5=(per.rank_fine.notna()) & (per.rank_fine <= 5)).pivot_table(
        index="fault", columns="method", values="h5")
    comp = piv1.round(3).add_suffix("_AC1").join(piv5.round(3).add_suffix("_AC5"))
    print(comp.to_string())

    print("\n=== union 潜力: 两者取并集, 根因是否进 top5 ===")
    # 并集覆盖: 根因指标在 PCC top5 或 zscore top5
    cov = []
    for _, r in man.iterrows():
        cid = r.case_id
        pcc_s, meta = pcc_scores_from_json(cid, features)
        z_s = zscore_feature_scores(cid, features)
        gt = meta["ground_truth"]["metric"]
        pcc_top5 = set(np.array(features)[np.argsort(-pcc_s)[:5]])
        z_top5 = set(np.array(features)[np.argsort(-z_s)[:5]])
        cov.append(dict(fault=r.fault,
                        in_pcc=gt in pcc_top5,
                        in_z=gt in z_top5,
                        in_union=gt in (pcc_top5 | z_top5)))
    cov = pd.DataFrame(cov)
    print(cov.groupby("fault")[["in_pcc", "in_z", "in_union"]].mean().round(3).to_string())
    print("\noverall:", cov[["in_pcc", "in_z", "in_union"]].mean().round(3).to_dict())
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    run()
