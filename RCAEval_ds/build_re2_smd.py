"""
RE2 -> SMD 格式转换器

产出结构:
    out/RE2-OB/
        feature_names.json          全局固定列空间, 顺序即数组列序
        config_dump.json            本次转换的全部参数
        manifest.csv                90 个案例的清单, 含分组与实际长度
        report_columns.txt          列空间与数据的差异报告, 需人工过目
        cases/adservice_cpu_1/
            train.npy               (FIT_LEN, D)      注入前早段, 无标签
            test.npy                (N_test, D)       注入前晚段 + 注入后
            test_label.npy          (N_test,)         0 正常 1 异常
            meta.json               根因服务/根因指标/inject_time/切分索引

切分为不重叠设计: train 与 test 的正常部分没有交集, 避免在训练数据上测 FPR

池化协议用法:
    np.vstack([np.load(f"cases/{c}/train.npy") for c in 训练组案例])
"""

import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

# ============================ CONFIG ============================

ROOT = r"D:\projects\X-RAG\notebooks\RCAEval_ds\RE2-OB"
OUTDIR = r"D:\projects\X-RAG\notebooks\RCAEval_ds\trans-OB"

FIT_LEN = 480        # train: 注入前最早的 480 行
TEST_NORMAL = 240    # test 正常部分: 紧邻注入时刻之前的 240 行
TEST_ANOMALY = 600   # test 异常部分: 注入后最多 600 行

METRIC_FAMILIES = ["cpu", "mem", "diskio", "socket", "latency", "error", "workload"]

PRUNE_NEVER_OBSERVED = True

# 应用服务白名单, 依据被测系统的服务清单规则化定义, 与案例数据无关
SERVICE_WHITELIST = {
    "RE2-OB": [
        "adservice", "cartservice", "checkoutservice", "currencyservice",
        "emailservice", "frontend", "paymentservice", "productcatalogservice",
        "recommendationservice", "shippingservice", "redis",
    ],
    # SS 与 TT 未定义时走自动推导 + 黑名单
}

# 服务网格与基础设施组件, 不是应用服务, 不可能是根因
INFRA_BLACKLIST_SUBSTR = [
    "istio", "passthrough", "PassthroughCluster", "InboundPassthrough",
    "frontend-check", "frontend-external", "unknown", "BlackHole",
]

# 故障类型 -> ground truth 指标族, 与 RCAEval main.py 一致
FAULT_TO_FAMILY = {
    "cpu": "cpu", "mem": "mem", "disk": "diskio",
    "socket": "socket", "delay": "latency", "loss": "latency",
}

# ================================================================


def clean_columns(cols):
    """丢掉 latency-50, 把 latency-90 改名为 latency, 与官方预处理一致"""
    out = []
    for c in cols:
        if c.endswith("_latency-50"):
            continue
        out.append(c.replace("_latency-90", "_latency"))
    return out


def split_col(c):
    """列名拆成 (服务, 指标族). 服务名可含连字符, 分隔符是最后一个下划线"""
    if "_" not in c:
        return None, None
    svc, fam = c.rsplit("_", 1)
    return svc, fam


def is_infra(svc):
    low = svc.lower()
    return any(b.lower() in low for b in INFRA_BLACKLIST_SUBSTR)


def scan_all_columns(paths):
    """扫一遍全部案例的列名, 只用于生成差异报告和自动推导服务清单"""
    seen = Counter()
    for p in paths:
        for c in clean_columns(pd.read_csv(p, nrows=0).columns):
            seen[c] += 1
    return seen


def build_feature_space(system, seen):
    """规则化定义列空间: 服务白名单 x 指标族 的笛卡尔积"""
    if system in SERVICE_WHITELIST:
        services = list(SERVICE_WHITELIST[system])
        source = "配置白名单"
    else:
        services = sorted({
            s for c in seen
            for s in [split_col(c)[0]]
            if s and not is_infra(s)
        })
        source = "自动推导加黑名单过滤"

    features = [f"{s}_{f}" for s in services for f in METRIC_FAMILIES]
    if PRUNE_NEVER_OBSERVED:
        pruned = [f for f in features if f not in seen]
        features = [f for f in features if f in seen]
        print(f"[剪枝] 全数据集从未出现, 已移除 {len(pruned)} 列: {pruned}")
    return services, features, source


def write_column_report(path, system, services, features, source, seen):
    """把列空间与数据的差异写成报告, 需人工过目确认没有误删"""
    in_data = set(seen) - {"time"}
    in_space = set(features)

    dropped = sorted(in_data - in_space)
    filled = sorted(in_space - in_data)

    fam_in_data = Counter()
    for c in in_data:
        svc, fam = split_col(c)
        if svc and not is_infra(svc):
            fam_in_data[fam] += 1
    unused_fams = sorted(set(fam_in_data) - set(METRIC_FAMILIES))

    lines = [
        f"系统: {system}",
        f"服务清单来源: {source}",
        f"服务数: {len(services)}   指标族数: {len(METRIC_FAMILIES)}   列空间维度: {len(features)}",
        f"数据中出现过的列总数: {len(in_data)}",
        "",
        "[需确认 1] 数据里有但被列空间丢弃的列, 确认这些确实不是应用服务指标:",
    ]
    for c in dropped:
        lines.append(f"    {c}   出现于 {seen[c]} 个案例")
    lines += [
        "",
        "[需确认 2] 列空间里有但数据从未出现的列, 这些会全程填 0:",
    ]
    for c in filled:
        lines.append(f"    {c}")
    lines += [
        "",
        "[需确认 3] 非基础设施服务上出现过但不在 METRIC_FAMILIES 里的指标族:",
    ]
    for f in unused_fams:
        lines.append(f"    {f}   涉及 {fam_in_data[f]} 个列名")
    if not unused_fams:
        lines.append("    无")
    lines += ["", "服务清单:"] + [f"    {s}" for s in services]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return dropped, filled, unused_fams


def load_case(path, features):
    """读一个案例, 对齐到固定列空间, 返回 (time, X) 与坏行数"""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
    df = df.rename(columns={c: c.replace("_latency-90", "_latency")
                            for c in df.columns})

    t = pd.to_numeric(df["time"], errors="coerce")
    bad = int((~np.isfinite(t.to_numpy(dtype="float64", na_value=np.nan))).sum())
    df = df.loc[np.isfinite(t.to_numpy(dtype="float64", na_value=np.nan))].copy()
    time = df["time"].astype("int64").to_numpy()

    # 数值化, inf 转 nan, 前向填充, 剩余填 0
    vals = df.drop(columns=["time"]).apply(pd.to_numeric, errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    # 对齐固定列空间: 缺列填 0, 多余列丢弃, 顺序固定
    X = vals.reindex(columns=features).fillna(0.0).to_numpy(dtype="float32")
    return time, X, bad


def convert(system=None):
    system = system or os.path.basename(ROOT.rstrip("\\/"))
    paths = sorted(glob.glob(os.path.join(ROOT, "*", "*", "simple_metrics.csv")))
    if not paths:
        raise FileNotFoundError(f"在 {ROOT} 下没找到 simple_metrics.csv")

    os.makedirs(os.path.join(OUTDIR, "cases"), exist_ok=True)

    seen = scan_all_columns(paths)
    services, features, source = build_feature_space(system, seen)

    dropped, filled, unused = write_column_report(
        os.path.join(OUTDIR, "report_columns.txt"),
        system, services, features, source, seen)

    with open(os.path.join(OUTDIR, "feature_names.json"), "w") as fh:
        json.dump(features, fh, indent=1)

    cfg = dict(root=ROOT, system=system, fit_len=FIT_LEN,
               test_normal=TEST_NORMAL, test_anomaly=TEST_ANOMALY,
               metric_families=METRIC_FAMILIES, services=services,
               service_source=source, n_features=len(features))
    with open(os.path.join(OUTDIR, "config_dump.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)

    rows = []
    for p in paths:
        d = os.path.dirname(p)
        svc, fault = os.path.basename(os.path.dirname(d)).split("_", 1)
        run = os.path.basename(d)
        case_id = f"{svc}_{fault}_{run}"

        inject = int(open(os.path.join(d, "inject_time.txt")).read().strip())
        time, X, bad = load_case(p, features)

        pre_mask = time < inject
        n_pre = int(pre_mask.sum())
        X_pre, X_post = X[pre_mask], X[~pre_mask]

        # 不重叠切分
        need = FIT_LEN + TEST_NORMAL
        if n_pre < need:
            print(f"[跳过] {case_id}: 注入前只有 {n_pre} 行, 需要 {need} 行")
            rows.append(dict(case_id=case_id, service=svc, fault=fault, run=run,
                             status="skipped_short_pre", n_pre=n_pre))
            continue

        train = X_pre[:FIT_LEN]                       # 最早的一段
        test_n = X_pre[n_pre - TEST_NORMAL:]          # 紧邻注入之前
        test_a = X_post[:TEST_ANOMALY]                # 注入之后

        test = np.vstack([test_n, test_a])
        label = np.concatenate([np.zeros(len(test_n), dtype="int8"),
                                np.ones(len(test_a), dtype="int8")])

        gt_family = FAULT_TO_FAMILY[fault]
        gt_metric = f"{svc}_{gt_family}"

        cdir = os.path.join(OUTDIR, "cases", case_id)
        os.makedirs(cdir, exist_ok=True)
        np.save(os.path.join(cdir, "train.npy"), train)
        np.save(os.path.join(cdir, "test.npy"), test)
        np.save(os.path.join(cdir, "test_label.npy"), label)

        meta = dict(
            case_id=case_id, system=system,
            root_cause_service=svc, fault_type=fault, run=run,
            gt_metric=gt_metric,
            gt_metric_in_space=gt_metric in features,
            gt_metric_col_index=features.index(gt_metric) if gt_metric in features else -1,
            inject_time=inject,
            n_features=len(features),
            n_train=int(len(train)), n_test=int(len(test)),
            n_test_normal=int(len(test_n)), n_test_anomaly=int(len(test_a)),
            anomaly_ratio=round(float(label.mean()), 4),
            n_pre_available=n_pre, n_post_available=int(len(X_post)),
            n_bad_time_rows=bad,
            truncated=bool(len(test_a) < TEST_ANOMALY),
        )
        with open(os.path.join(cdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)

        row = dict(meta)
        row.update(status="ok", service=svc, fault=fault, run=run)
        rows.append(row)

    man = pd.DataFrame(rows)
    # 留一服务分组: 以注入服务为折, 同服务的全部案例同折
    man["fold"] = man["service"].astype("category").cat.codes
    man.to_csv(os.path.join(OUTDIR, "manifest.csv"), index=False)

    ok = man[man.status == "ok"] if "status" in man else man
    print(f"完成. 输出目录: {OUTDIR}")
    print(f"案例数: {len(man)}   成功: {len(ok)}   列空间维度: {len(features)}")
    if len(ok):
        print(f"每案例 train {FIT_LEN} 行, test {ok.n_test.min()} 到 {ok.n_test.max()} 行")
        print(f"异常占比 {ok.anomaly_ratio.min():.2f} 到 {ok.anomaly_ratio.max():.2f}")
        print(f"窗口被截断的案例: {int(ok.truncated.sum())} 个")
        print(f"ground truth 列不在列空间的案例: {int((~ok.gt_metric_in_space).sum())} 个")
        print(f"留一服务折数: {ok.fold.nunique()}")
    print(f"\n列报告: 丢弃 {len(dropped)} 列, 全程填 0 {len(filled)} 列, "
          f"未纳入的指标族 {len(unused)} 个")
    print("请打开 report_columns.txt 过目三处确认项")


if __name__ == "__main__":
    convert()
