"""
基线 A 扩展: 比较不同检测算法的归因排序能力

iForest / EIF 用 TreeSHAP 精确解
CBLOF / PCC   用 KernelSHAP 近似解
zscore        平凡基线, 不依赖任何模型, 只算一次

每个算法用自己的检测结果挑要解释的点, 贴近实际部署
为控制 KernelSHAP 成本, 每案例最多解释 MAX_EXPLAIN 个点, 取分数最高的

【本版修改】
fine-grained ground truth 不再用 FAULT_TO_FAMILY（故障类型->指标族）做近似映射，
改为直接读取每个 case 的 meta.json 里的 gt_metric 字段（数据集自带的逐案例精确
标注），做字符串级精确匹配。
    - 规避了 delay/loss 是否都映射到 "latency" 家族的语义歧义
    - 规避了 socket/disk 等 family 名称与实际列名切分方式不一致的问题
    - 若某案例 gt_metric_in_space=false（ground truth 列本身不在特征空间里，
      例如某些服务缺 _error 列），rank_fine 必为 None，此类案例在汇总时会被
      单独统计，不与"模型没找到"混为一谈

输出
    baselineA_algos/per_case.csv
    baselineA_algos/summary.csv
    baselineA_algos/unreachable_cases.csv   (gt_metric_in_space=false 的案例清单)
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import shap
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA as PyodPCA

# ============================ CONFIG ============================

DATA = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"
OUT = os.path.join(DATA, "baselineA_algos")

# ALGOS = ["iForest", "EIF", "CBLOF", "PCC"]
ALGOS = ["PCC"]

Q = 0.99
MAX_EXPLAIN = 15        # 每案例解释的点数上限, 直接决定 KernelSHAP 总耗时
KERNEL_BG = 50          # KernelSHAP 背景集大小
K_MAX = 5
SEED = 42
N_CASES = None          # 调试时设成 6

# 【已删除】FAULT_TO_FAMILY 映射表 —— 不再需要，gt_metric 直接给出精确答案

# ================================================================


def build(name):
    """返回 (模型, 打分函数构造器, 归因方式). 打分函数统一为越大越异常"""
    if name == "iForest":
        return IsolationForest(n_estimators=100, max_samples=256,
                               random_state=SEED, n_jobs=-1), "tree"
    if name == "EIF":
        return IsolationForest(n_estimators=100, max_samples=256,
                               max_features=0.8, random_state=SEED,
                               n_jobs=-1), "tree"
    if name == "PCC":
        return PyodPCA(n_components=0.95, random_state=SEED), "kernel"
    if name == "CBLOF":
        return CBLOF(n_clusters=8, alpha=0.75, beta=3, random_state=SEED), "kernel"
    raise ValueError(name)


def fit_model(name, Ztr):
    m, mode = build(name)
    if name == "CBLOF":
        for nc in [8, 5, 10, 3, 12]:
            for a, b in [(0.75, 3), (0.9, 5), (0.6, 2)]:
                try:
                    m = CBLOF(n_clusters=nc, alpha=a, beta=b, random_state=SEED)
                    m.fit(Ztr)
                    return m, mode
                except Exception:
                    continue
        raise RuntimeError("CBLOF 拟合失败")
    m.fit(Ztr)
    return m, mode


def score_fn(name, m):
    """统一成越大越异常"""
    if name in ("iForest", "EIF"):
        return lambda X: -m.score_samples(X)
    return lambda X: m.decision_function(X)


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def hit_rank(ranked, answer, k_max=K_MAX):
    for i, x in enumerate(ranked[:k_max], start=1):
        if x == answer:
            return i
    return None


def rank_cols(scores, features):
    return [features[i] for i in np.argsort(-scores)]


def make_rows(cid, row, meta, name, cols):
    """
    cols: 该算法对这个 case 输出的、按重要性从高到低排好的完整指标名列表
          (不是只取前 5，是完整排序；hit_rank 内部再截断到 k_max)

    fine-grained 比对：
        gt_metric              -- meta.json 里逐 case 精确标注的答案列名
                                   (如 "checkoutservice_diskio")
        gt_metric_in_space     -- 这个答案列本身是否存在于该 case 的特征空间里
                                   若为 False，rank_fine 必然是 None，
                                   属于"结构性不可达"，不是模型没找到
    """
    svc = meta["root_cause_service"]
    gt_metric = meta["gt_metric"]
    gt_in_space = meta.get("gt_metric_in_space", True)

    coarse = dedup([c.split("_")[0].replace("-db", "") for c in cols])

    return dict(
        case_id=cid,
        service=svc,
        fault=row.fault,
        fold=int(row.fold),
        method=name,
        gt_metric=gt_metric,
        gt_metric_in_space=gt_in_space,
        rank_coarse=hit_rank(coarse, svc),
        rank_fine=hit_rank(cols, gt_metric),   # 直接字符串精确匹配，不经过 family
        top1_fine=cols[0],
    )


def eval_case(cid, row, features, timer):
    cdir = os.path.join(DATA, "cases", cid)
    Xtr = np.load(os.path.join(cdir, "train.npy"))
    Xte = np.load(os.path.join(cdir, "test.npy"))
    y = np.load(os.path.join(cdir, "test_label.npy"))
    meta = json.load(open(os.path.join(cdir, "meta.json")))

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    anom = np.where(y == 1)[0]

    rows = []
    for name in ALGOS:
        try:
            t0 = time.time()
            m, mode = fit_model(name, Ztr)
            f = score_fn(name, m)
            s_tr, s_te = f(Ztr), f(Zte)
            thr = np.quantile(s_tr, Q)

            sel = anom[s_te[anom] > thr]
            if len(sel) == 0:
                sel = anom
            sel = sel[np.argsort(-s_te[sel])[:MAX_EXPLAIN]]
            Zsel = Zte[sel]

            if mode == "tree":
                sv = np.asarray(shap.TreeExplainer(m).shap_values(
                    Zsel, check_additivity=False))
                if sv.ndim == 3:
                    sv = sv[..., 0]
                sv = -sv          # sklearn 输出越低越异常, 取负后正值推向异常
            else:
                bg = shap.sample(Ztr, min(KERNEL_BG, len(Ztr)),
                                 random_state=SEED)
                sv = np.asarray(shap.KernelExplainer(f, bg).shap_values(
                    Zsel, silent=True))
                if sv.ndim == 3:
                    sv = sv[..., 0]

            r = make_rows(cid, row, meta, name,
                          rank_cols(np.abs(sv).mean(axis=0), features))
            r["n_explained"] = int(len(sel))
            r["secs"] = round(time.time() - t0, 2)
            rows.append(r)
            timer[name] = timer.get(name, 0.0) + (time.time() - t0)
        except Exception as e:
            rows.append(dict(case_id=cid, method=name, error=str(e)[:120]))

    # zscore 只依赖数据, 用 iForest 的选点以保持可比
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    zsel = anom[np.argsort(-(-IsolationForest(**dict(
        n_estimators=100, max_samples=256, random_state=SEED, n_jobs=-1)
    ).fit(Ztr).score_samples(Zte))[anom])[:MAX_EXPLAIN]]
    zs = np.abs((Xte[zsel] - mu) / sd).mean(axis=0)
    r = make_rows(cid, row, meta, "zscore", rank_cols(zs, features))
    r["n_explained"] = int(len(zsel))
    rows.append(r)
    return rows


def ac_at_k(ranks, k):
    r = pd.Series(ranks)
    return float(((r.notna()) & (r <= k)).mean())


def summarize(df, label):
    out = []
    for m, g in df.groupby("method"):
        rec = {"scope": label, "method": m, "n": len(g)}
        for gran in ["coarse", "fine"]:
            acs = [ac_at_k(g[f"rank_{gran}"], k) for k in range(1, K_MAX + 1)]
            rec[f"AC@1_{gran}"] = round(acs[0], 2)
            rec[f"AC@3_{gran}"] = round(acs[2], 2)
            rec[f"AC@5_{gran}"] = round(acs[4], 2)
            rec[f"Avg@5_{gran}"] = round(float(np.mean(acs)), 2)
        out.append(rec)
    return pd.DataFrame(out)


def run():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)
    if N_CASES:
        man = man.head(N_CASES)
    features = json.load(open(os.path.join(DATA, "feature_names.json")))

    print(f"案例 {len(man)}, 算法 {ALGOS}, 每案例最多解释 {MAX_EXPLAIN} 点")
    timer, rows = {}, []
    for i, r in man.iterrows():
        rows += eval_case(r.case_id, r, features, timer)
        if (i + 1) % 10 == 0:
            el = ", ".join(f"{k} {v/60:.1f}min" for k, v in timer.items())
            print(f"  {i + 1}/{len(man)}   累计: {el}")

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "per_case.csv"), index=False)
    ok = per[per.get("error").isna()] if "error" in per else per

    # ── 结构性不可达案例：gt_metric 本身不在特征空间里，与模型能力无关 ──
    if "gt_metric_in_space" in ok:
        unreachable = ok[ok.gt_metric_in_space == False]
        if len(unreachable):
            unreachable.drop_duplicates("case_id").to_csv(
                os.path.join(OUT, "unreachable_cases.csv"), index=False)
            print(f"\n[提示] {unreachable.case_id.nunique()} 个案例的 gt_metric "
                  f"不在特征空间内（结构性不可达，已单独存入 unreachable_cases.csv，"
                  f"不建议与其余案例混合统计 fine-grained 分数）")

    parts = [summarize(ok, "overall")]
    for f, g in ok.groupby("fault"):
        parts.append(summarize(g, f"fault={f}"))
    summ = pd.concat(parts, ignore_index=True)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n=== 总体 ===")
    print(summ[summ.scope == "overall"].round(4).to_string(index=False))

    print("\n=== 分故障类型的 AC@1_fine ===")
    print(ok.assign(hit1=(ok.rank_fine == 1))
          .pivot_table(index="fault", columns="method", values="hit1")
          .round(3).to_string())

    if "error" in per and per.error.notna().any():
        print("\n失败:")
        print(per[per.error.notna()].groupby("method").size().to_string())

    if "secs" in ok:
        print("\n=== 每案例归因耗时, 秒 ===")
        print(ok.groupby("method")["secs"].mean().round(2).to_string())
    print(f"\n输出目录: {OUT}")


if __name__ == "__main__":
    run()
