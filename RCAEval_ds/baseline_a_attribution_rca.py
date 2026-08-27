"""
基线 A: 不用 LLM, 直接用 Layer 2 的归因排序当作根因排序

目的: 确定 Layer 3 是否还有提升空间
如果本基线的 AC@1 已经接近饱和, RAG 加 LLM 在 AC@k 这条轴上无贡献可言,
Layer 3 的衡量维度必须改为诊断文本质量

四种排序方式, 全部按 RCAEval 官方口径评测
  shap_abs     TreeSHAP 绝对值均值, 主方案
  shap_signed  只累计推向异常方向的贡献
  zscore       特征在异常段相对 train 的标准化偏移绝对值, 不用模型的平凡基线
  random       随机排序, 给出下界参照

评测口径完全对齐 RCAEval main.py
  粗粒度: 候选串按第一个下划线取服务名, 去重后保序, 答案是根因服务
  细粒度: 候选串拆成 服务 与 指标, 答案是 (根因服务, 故障族)
  AC@k 为真根因落在前 k 的比例, Avg@k 为 AC@1 到 AC@k 的均值

输出
    baselineA/per_case_his.csv     每案例每方法的命中位次
    baselineA/summary.csv      AC@1 AC@3 AC@5 Avg@5, 总体与分故障类型
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================ CONFIG ============================

DATA = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"
OUT = os.path.join(DATA, "baselineA")

Q = 0.99                # Layer 1 阈值分位数
TOPN_FALLBACK = 20      # 异常段内没有点被标记时, 取分数最高的前 N 个点
K_MAX = 5
SEED = 42

IFOREST = dict(n_estimators=100, max_samples=256, random_state=SEED, n_jobs=-1)

# 故障类型 -> ground truth 指标族, 与 RCAEval main.py 一致
FAULT_TO_FAMILY = {"cpu": "cpu", "mem": "mem", "disk": "diskio",
                   "socket": "socket", "delay": "latency", "loss": "latency"}

# ================================================================

import shap


def dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def hit_rank(ranked, answer, k_max=K_MAX):
    """答案在 ranked 中的 1-based 位次, 不在前 k_max 内返回 None"""
    for i, x in enumerate(ranked[:k_max], start=1):
        if x == answer:
            return i
    return None


def rank_from_scores(scores, features):
    """按打分降序返回列名列表"""
    order = np.argsort(-scores)
    return [features[i] for i in order]


def eval_case(cid, row, features, rng):
    cdir = os.path.join(DATA, "cases", cid)
    Xtr = np.load(os.path.join(cdir, "train.npy"))
    Xte = np.load(os.path.join(cdir, "test.npy"))
    y = np.load(os.path.join(cdir, "test_label.npy"))
    meta = json.load(open(os.path.join(cdir, "meta.json")))

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    clf = IsolationForest(**IFOREST).fit(Ztr)
    s_tr = -clf.score_samples(Ztr)
    s_te = -clf.score_samples(Zte)
    thr = np.quantile(s_tr, Q)
    pred = (s_te > thr).astype(int)

    # 只解释异常段内被标记的点. inject_time 是 RCAEval 官方交给方法的输入, 合规
    anom_idx = np.where(y == 1)[0]
    flagged = anom_idx[pred[anom_idx] == 1]
    if len(flagged) == 0:
        flagged = anom_idx[np.argsort(-s_te[anom_idx])[:TOPN_FALLBACK]]

    Zsel = Zte[flagged]

    # ---- TreeSHAP ----
    sv = shap.TreeExplainer(clf).shap_values(Zsel, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0]

    # sklearn IsolationForest 的输出越低越异常, 因此负贡献推向异常
    score_abs = np.abs(sv).mean(axis=0)
    score_signed = np.clip(-sv, 0, None).mean(axis=0)

    # ---- z-score 平凡基线 ----
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    score_z = np.abs((Xte[flagged] - mu) / sd).mean(axis=0)

    # ---- 随机下界 ----
    score_rand = rng.random(len(features))

    rankings = {
        "shap_abs": rank_from_scores(score_abs, features),
        "shap_signed": rank_from_scores(score_signed, features),
        "zscore": rank_from_scores(score_z, features),
        "random": rank_from_scores(score_rand, features),
    }

    svc_ans = meta["root_cause_service"]
    fam_ans = FAULT_TO_FAMILY[row.fault]
    fine_ans = (svc_ans, fam_ans)

    rows = []
    for name, cols in rankings.items():
        coarse = dedup_keep_order([c.split("_")[0].replace("-db", "") for c in cols])
        fine = [(c.split("_")[0], c.split("_", 1)[1] if "_" in c else "unknown")
                for c in cols]

        rows.append(dict(
            case_id=cid, service=svc_ans, fault=row.fault, fold=int(row.fold),
            method=name,
            gt_metric=meta["gt_metric"],
            n_flagged=int(len(flagged)),
            rank_coarse=hit_rank(coarse, svc_ans),
            rank_fine=hit_rank(fine, fine_ans),
            top1_coarse=coarse[0],
            top1_fine=f"{fine[0][0]}_{fine[0][1]}",
        ))
    return rows


def ac_at_k(ranks, k):
    """AC@k: 真根因落在前 k 的比例. 单一根因下 min(k,|V|)=1"""
    r = pd.Series(ranks)
    return float(((r.notna()) & (r <= k)).mean())


def summarize(df, label):
    out = []
    for m, g in df.groupby("method"):
        rec = {"scope": label, "method": m, "n": len(g)}
        for gran in ["coarse", "fine"]:
            acs = [ac_at_k(g[f"rank_{gran}"], k) for k in range(1, K_MAX + 1)]
            rec[f"AC@1_{gran}"] = acs[0]
            rec[f"AC@3_{gran}"] = acs[2]
            rec[f"AC@5_{gran}"] = acs[4]
            rec[f"Avg@5_{gran}"] = float(np.mean(acs))
        out.append(rec)
    return pd.DataFrame(out)


def run():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)
    features = json.load(open(os.path.join(DATA, "feature_names.json")))
    rng = np.random.default_rng(SEED)

    rows = []
    for i, r in man.iterrows():
        rows += eval_case(r.case_id, r, features, rng)
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(man)}")

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "per_case_his.csv"), index=False)

    parts = [summarize(per, "overall")]
    for f, g in per.groupby("fault"):
        parts.append(summarize(g, f"fault={f}"))
    summ = pd.concat(parts, ignore_index=True)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== 总体 ===")
    print(summ[summ.scope == "overall"].round(4).to_string(index=False))

    print("\n=== 分故障类型, 只看 shap_abs 与 zscore ===")
    sub = summ[(summ.scope != "overall") & (summ.method.isin(["shap_abs", "zscore"]))]
    print(sub[["scope", "method", "AC@1_coarse", "AC@1_fine",
               "Avg@5_coarse", "Avg@5_fine"]].round(3).to_string(index=False))

    print(f"\n每案例被解释的点数: 中位 {per.n_flagged.median():.0f}, "
          f"最小 {per.n_flagged.min()}, 最大 {per.n_flagged.max()}")
    print(f"输出目录: {OUT}")


if __name__ == "__main__":
    run()
