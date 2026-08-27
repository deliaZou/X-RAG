"""
Layer 1 阈值标定, PCC 版
==============================================
在 PCC 分数上扫 q 网格, 折感知选取 eval_quantile

规则: 在满足目标召回的前提下, 取最大的 q, 使 FPR 最低
     检测已饱和, 召回在很宽 q 范围接近 1, 判据实际落在 FPR

折感知: 每折在其余 4 折上选 q, 留出折上报告, 避免选择泄漏

输出
    calib/sweep_macro.csv      全网格宏平均, 论文附录
    calib/fold_selection.csv   每折选出的 q 与留出折表现
    calib/score_hist.png       抽样案例的 train 分数分布
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.pca import PCA as PyodPCA
from sklearn.preprocessing import StandardScaler

# ============================ CONFIG ============================

BASE = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"
OUT = os.path.join(BASE, "calib")

Q_GRID = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
TARGET_RECALL = 0.95
PCA_NCOMP = 0.95
SEED = 0

# ================================================================


def macro(df, cols):
    g = df.groupby("q")[cols]
    return (g.mean().add_suffix("_mean")
            .join(g.std().add_suffix("_sd"))
            .join(g.median().add_suffix("_med")).reset_index())


def run():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(BASE, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)

    rows, hist = [], {}
    for _, r in man.iterrows():
        cdir = os.path.join(BASE, "cases", r.case_id)
        Xtr = np.load(os.path.join(cdir, "train.npy"))
        Xte = np.load(os.path.join(cdir, "test.npy"))
        y = np.load(os.path.join(cdir, "test_label.npy"))

        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        clf = PyodPCA(n_components=PCA_NCOMP, random_state=SEED).fit(Ztr)
        s_tr = clf.decision_function(Ztr)     # 越大越异常
        s_te = clf.decision_function(Zte)

        if len(hist) < 6:
            hist[r.case_id] = s_tr

        for q in Q_GRID:
            thr = np.quantile(s_tr, q)
            pred = (s_te > thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            rows.append(dict(case_id=r.case_id, fault=r.fault, fold=int(r.fold), q=q,
                             recall=tp / max(tp + fn, 1),
                             fpr=fp / max(fp + tn, 1),
                             precision=tp / max(tp + fp, 1)))

    per = pd.DataFrame(rows)
    METRICS = ["recall", "fpr", "precision"]
    sweep = macro(per, METRICS)
    sweep.to_csv(os.path.join(OUT, "sweep_macro.csv"), index=False)

    # 折感知选 q
    sel = []
    for f in sorted(per.fold.unique()):
        tr = per[per.fold != f]
        ho = per[per.fold == f]
        cand = macro(tr, METRICS)
        okq = cand[cand.recall_mean >= TARGET_RECALL]
        q_pick = float(okq.q.max()) if len(okq) else float(cand.q.min())
        h = ho[ho.q == q_pick]
        sel.append(dict(fold=f, held_out_service=ho.case_id.iloc[0].split("_")[0], q=q_pick,
                        recall=h.recall.mean(), fpr=h.fpr.mean(),
                        precision=h.precision.mean()))
    sel = pd.DataFrame(sel)
    sel.to_csv(os.path.join(OUT, "fold_selection.csv"), index=False)

    # 分数分布图
    n = len(hist)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (cid, s) in zip(axes.ravel(), hist.items()):
        ax.hist(s, bins=50)
        for q, c in [(0.99, "red"), (0.999, "orange")]:
            ax.axvline(np.quantile(s, q), color=c, lw=1)
        ax.set_title(cid, fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.suptitle("PCC train score distribution, red q=0.99 orange q=0.999", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "score_hist.png"), dpi=120)

    pd.set_option("display.width", 200)
    print("=== 全网格宏平均 ===")
    print(sweep[["q", "recall_mean", "recall_sd", "fpr_mean", "fpr_sd",
                 "precision_mean"]].round(4).to_string(index=False))
    print("\n=== 折感知选取 ===")
    print(sel.round(4).to_string(index=False))
    print(f"\n建议 eval_quantile = {sel.q.mode().iloc[0]}"
          f"   五折选出: {sorted(sel.q.tolist())}")
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    run()
