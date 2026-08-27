"""
算法选择表, RE2-OB, 一次跑齐
==============================================
一张表定算法. 每个候选算出:
  检测质量:   AUC-ROC, AUC-PR
  归因质量:   AC@1_fine, AC@5_fine   (归因排序命中根因指标)
  归因成本:   Explain/s 每条 SHAP 耗时
  XAI 兼容性: TreeSHAP 精确 / KernelSHAP 近似

判据: 检测饱和使 AUC 区分度低, 决定权在归因质量 AC@1_fine.
     iForest 虽 TreeSHAP 精确但归因差, PCC 虽 KernelSHAP 近似但归因优.

阈值固定 Q=0.99, 因本表列均不依赖 q:
  AUC / AC@k / 耗时 都是免阈值或排序类指标.

输出
    algo_select/table_his.csv        论文用的主表
    algo_select/per_case_his.csv     每案例每算法明细
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import shap

from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.pca import PCA as PyodPCA

# ============================ CONFIG ============================

DATA = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"
OUT = os.path.join(DATA, "algo_select")

Q = 0.99
MAX_EXPLAIN = 5       # 每案例解释的点数, 控制 KernelSHAP 成本
KERNEL_BG = 50
K_MAX = 5
SEED = 42
N_CASES = None         # 调试设 6

FAULT_TO_FAMILY = {"cpu": "cpu", "mem": "mem", "disk": "diskio",
                   "socket": "socket", "delay": "latency", "loss": "latency"}

# 算法注册: 名字 -> (构造, 类型, SHAP 方式, XAI 兼容)
def registry():
    return {
        "iForest": (lambda: IsolationForest(n_estimators=100, max_samples=256,
                    random_state=SEED, n_jobs=-1), "Trees", "TreeSHAP", "Exact", "tree"),
        "EIF": (lambda: IsolationForest(n_estimators=100, max_samples=256,
                max_features=0.8, random_state=SEED, n_jobs=-1),
                "Trees", "TreeSHAP", "Exact", "tree"),
        "LOF": (lambda: LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
                "Distance", "KernelSHAP", "Approximate", "kernel"),
        "KNN": (lambda: KNN(n_neighbors=5), "Distance", "KernelSHAP", "Approximate", "kernel"),
        "HBOS": (lambda: HBOS(n_bins=10), "Distribution", "KernelSHAP", "Approximate", "kernel"),
        "CBLOF": (lambda: CBLOF(n_clusters=8, alpha=0.75, beta=3, random_state=SEED),
                  "Distance", "KernelSHAP", "Approximate", "kernel"),
        "COPOD": (lambda: COPOD(), "Distribution", "KernelSHAP", "Approximate", "kernel"),
        "PCC": (lambda: PyodPCA(n_components=0.95, random_state=SEED),
                "Reconstruction", "KernelSHAP", "Approximate", "kernel"),
    }


def fit_model(name, build, Ztr):
    if name == "CBLOF":
        for nc in [8, 5, 10, 3, 12]:
            for a, b in [(0.75, 3), (0.9, 5), (0.6, 2)]:
                try:
                    m = CBLOF(n_clusters=nc, alpha=a, beta=b, random_state=SEED)
                    m.fit(Ztr)
                    return m
                except Exception:
                    continue
        raise RuntimeError("CBLOF fit failed")
    m = build()
    m.fit(Ztr)
    return m


def scorer(name, m):
    if name in ("iForest", "EIF"):
        return lambda X: -m.score_samples(X)
    if name == "LOF":
        return lambda X: -m.decision_function(X)
    return lambda X: m.decision_function(X)


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


def eval_case(cid, row, features):
    cdir = os.path.join(DATA, "cases", cid)
    Xtr = np.load(os.path.join(cdir, "train.npy"))
    Xte = np.load(os.path.join(cdir, "test.npy"))
    y = np.load(os.path.join(cdir, "test_label.npy"))
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    svc = meta["root_cause_service"]
    fam = FAULT_TO_FAMILY[row.fault]

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    anom = np.where(y == 1)[0]

    rows = []
    for name, (build, kind, stype, compat, mode) in registry().items():
        try:
            t_fit = time.time()
            m = fit_model(name, build, Ztr)
            train_s = time.time() - t_fit
            f = scorer(name, m)
            s_tr, s_te = f(Ztr), f(Zte)

            aucroc = roc_auc_score(y, s_te)
            aucpr = average_precision_score(y, s_te)

            thr = np.quantile(s_tr, Q)
            pred = (s_te > thr).astype(int)
            fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            tp = int(((pred == 1) & (y == 1)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            fpr_q = fp / max(fp + tn, 1)
            recall_q = tp / max(tp + fn, 1)

            sel = anom[s_te[anom] > thr]
            if len(sel) == 0:
                sel = anom
            sel = sel[np.argsort(-s_te[sel])[:MAX_EXPLAIN]]
            Zsel = Zte[sel]

            t0 = time.time()
            if mode == "tree":
                sv = np.asarray(shap.TreeExplainer(m).shap_values(
                    Zsel, check_additivity=False))
                if sv.ndim == 3:
                    sv = sv[..., 0]
                sv = -sv
            else:
                bg = shap.sample(Ztr, min(KERNEL_BG, len(Ztr)), random_state=SEED)
                sv = np.asarray(shap.KernelExplainer(f, bg).shap_values(Zsel, silent=True))
                if sv.ndim == 3:
                    sv = sv[..., 0]
            explain_s = (time.time() - t0) / max(len(sel), 1)

            order = np.argsort(-np.abs(sv).mean(axis=0))
            cols = [features[i] for i in order]
            coarse = dedup([c.split("_")[0].replace("-db", "") for c in cols])
            fine = [(c.split("_")[0], c.split("_", 1)[1] if "_" in c else "unknown")
                    for c in cols]

            rows.append(dict(case_id=cid, fault=row.fault, fold=int(row.fold),
                             algo=name, type=kind, shap=stype, xai=compat,
                             aucroc=aucroc, aucpr=aucpr,
                             rank_coarse=hit(coarse, svc),
                             rank_fine=hit(fine, (svc, fam)),
                             fpr_q=fpr_q, recall_q=recall_q,
                             train_s=train_s, explain_s=explain_s))
        except Exception as e:
            rows.append(dict(case_id=cid, algo=name, error=str(e)[:100]))
    return rows


def ac(ranks, k):
    r = pd.Series(ranks)
    return float(((r.notna()) & (r <= k)).mean())


def run():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)
    if N_CASES:
        man = man.head(N_CASES)
    features = json.load(open(os.path.join(DATA, "feature_names.json")))

    print(f"案例 {len(man)}, 算法 8 个")
    rows = []
    for i, r in man.iterrows():
        rows += eval_case(r.case_id, r, features)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(man)}")

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "per_case_his.csv"), index=False)
    ok = per[per.get("error").isna()] if "error" in per else per

    recs = []
    for name, g in ok.groupby("algo"):
        recs.append(dict(
            Algor=name, Type=g.type.iloc[0],SHAP=g.shap.iloc[0],
            AUC_ROC=g.aucroc.mean(), AUC_PR=g.aucpr.mean(),
            AC1_fine=ac(g.rank_fine, 1), AC5_fine=ac(g.rank_fine, 5),
            Recall_q=g.recall_q.mean(), FPR_q=g.fpr_q.mean(),
            Train_s=g.train_s.mean(), Explain_s=g.explain_s.mean(),
            # XAI=g.xai.iloc[0],
            n=len(g)))
    tab = pd.DataFrame(recs).sort_values("AUC_ROC")
    tab.to_csv(os.path.join(OUT, "table_his.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== 算法选择表, 按 AUC_ROC 升序 ===")
    print(tab.round(4).to_string(index=False))

    if "error" in per and per.error.notna().any():
        print("\n失败:")
        print(per[per.error.notna()].groupby("algo").size().to_string())
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    run()
