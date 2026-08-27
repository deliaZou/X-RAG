"""
RO3-1 Candidate Algorithm Comparison
====================================
Orchestrates: load data -> fit each candidate -> score -> detection metrics
-> sampled explanation-time -> collect into a table.

Debug/run directly by editing the CONFIG block in __main__ below.
No CLI args and no interactive menu required.
"""

import os
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# make src/ importable when running this file directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.utils.data_loader import load_smd, make_synthetic_smd            # noqa: E402
from src.utils.data_loader import read_metadata                          # noqa: E402
from src.selection.algorithms.candidates import get_candidates, EIF_MODE  # noqa: E402
from src.selection.algorithms.metrics import (                            # noqa: E402
    detection_metrics, measure_explain_time, predict_by_rate,
)

COL_ORDER = [
    "Machine", "Algorithm", "Type", "SHAP Type", "XAI Compat", "Parameters",
    "Eval Rate", "AUC-ROC", "AUC-PR", "F1", "FPR", "Confusion Matrix",
    "Train Time (s)", "Explain Time (s)", "Explain Time/sample (s)",
]


def evaluate_one(machine, cand, X_train, X_test, y_test,
                 explain_samples, kernel_background, eval_rate=None):
    """
    Run a single candidate on a single machine, return one result row.

    eval_rate : test-set true anomaly rate (from metadata). If given, the
                binary prediction is set by flagging the top eval_rate fraction
                of scores (unified across algorithms, evaluation-side only).
                If None, falls back to the algorithm's own predict().
    """
    t0 = time.time()
    cand.fit(X_train)
    train_time = round(time.time() - t0, 4)

    y_score = cand.score(X_test)
    if eval_rate is not None:
        y_pred = predict_by_rate(y_score, eval_rate)
    else:
        y_pred = cand.predict(X_test)

    row = {
        "Machine": machine,
        "Algorithm": cand.name,
        "Type": cand.type,
        "SHAP Type": "TreeSHAP" if cand.shap_type == "tree" else "KernelSHAP",
        "XAI Compat": cand.compat,
        "Parameters": str(cand.params),
        "Eval Rate": round(eval_rate, 4) if eval_rate is not None else "own",
        "Train Time (s)": train_time,
    }
    row.update(detection_metrics(y_test, y_score, y_pred))

    et, ep = measure_explain_time(cand, X_train, X_test,
                                  n_sample=explain_samples,
                                  n_background=kernel_background)
    row["Explain Time (s)"] = et
    row["Explain Time/sample (s)"] = ep
    return row


def run(data_dir, machines=None, algos=None,
        config=None, quick=False, save_path=None,
        contamination=None, explain_samples=None, kernel_background=None):
    """
    data_dir     : path to SMD
    machines     : None (all) | "machine-1-1" | ["machine-1-1", ...]
    algos        : None (all) | list of names, e.g. ["iForest", "EIF", "LOF"]
    config       : path to selection.yaml (None = use the default file)
    quick        : ignore data_dir, use synthetic data (offline smoke test)
    save_path    : if set, write the results table to a timestamped CSV
    contamination / explain_samples / kernel_background:
                   optional overrides; if None, taken from the config file.
    """
    from src.selection.algorithms.candidates import _load_config

    cfg = _load_config(config)
    if contamination is not None:
        cfg["shared"]["contamination"] = contamination
    contamination = cfg["shared"]["contamination"]
    explain_samples = explain_samples or cfg["experiment"]["explain_samples"]
    kernel_background = kernel_background or cfg["experiment"]["kernel_background"]

    print(f"[INFO] EIF mode: {EIF_MODE} "
          f"({'real eif package' if EIF_MODE == 'real' else 'sklearn approximation'})")
    print(f"[INFO] contamination={contamination} "
          f"explain_samples={explain_samples} kernel_background={kernel_background}")

    # decide data source
    if quick:
        print("[INFO] Quick mode: synthetic SMD-like data")
        sources = [("synthetic", *make_synthetic_smd(contamination=contamination))]
    else:
        sources = load_smd(data_dir, machines)   # generator: (m, Xtr, Xte, y)

    all_rows = []
    for machine, X_train, X_test, y_test in sources:
        # evaluation-side true anomaly rate: prefer metadata, else derive
        # from the test labels directly (identical value, just a fallback)
        # eval_rate = None if quick else read_metadata(data_dir, machine)
        eval_rate = 0.1
        if eval_rate is None:
            eval_rate = float(y_test.mean()) if len(y_test) else None

        rate_str = f" eval_rate={eval_rate:.4f}" if eval_rate else ""
        print(f"\n[MACHINE] {machine}  "
              f"train={X_train.shape} test={X_test.shape} "
              f"anomalies={int(y_test.sum())} ({y_test.mean():.1%}){rate_str}")

        registry = get_candidates(cfg)
        if algos:
            registry = {k: v for k, v in registry.items() if k in set(algos)}

        for i, (name, cand) in enumerate(registry.items(), 1):
            print(f"  [{i}/{len(registry)}] {name} ...", end=" ", flush=True)
            try:
                row = evaluate_one(machine, cand, X_train, X_test, y_test,
                                   explain_samples, kernel_background,
                                   eval_rate=eval_rate)
                all_rows.append(row)
                print(f"AUC-ROC={row.get('AUC-ROC')} F1={row.get('F1')} "
                      f"FPR={row.get('FPR')} explain/s={row.get('Explain Time/sample (s)')}")
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] {e}")

    df = pd.DataFrame(all_rows)
    df = df[[c for c in COL_ORDER if c in df.columns]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 50)
    print("\n" + "=" * 80)
    print("  RO3-1 RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    if save_path:
        ts = datetime.now().strftime("%y%d%m%H%M")
        root, ext = os.path.splitext(save_path)
        stamped = f"{root}_{ts}{ext or '.csv'}"
        df.to_csv(stamped, index=False)
        print(f"\n[SAVED] {stamped}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG / RUN HERE — edit CONFIG and press run
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    CONFIG = dict(
        data_dir="./data/ServerMachineDataset",

        # data source: pick ONE style
        #   None                           -> every machine in data_dir
        #   "machine-1-1"                  -> a single machine
        #   ["machine-1-1", "machine-2-1"] -> several machines
        machines="machine-1-1",

        # algorithms: None = all, or a subset e.g. ["iForest", "EIF", "LOF"]
        # algos=["iForest","CBLOF", "PCC"],
        algos=None,

        # all hyperparameters live in configs/selection.yaml
        #   config=None -> use the default selection.yaml
        #   config="path/to/other.yaml" -> use a different one
        config=None,

        # quick=True -> synthetic data, no real files needed (smoke test)
        quick=False,

        # set a path to save, or None to skip
        save_path="results/RO3-1_results",

        # optional one-off overrides (leave None to use the yaml values)
        contamination=None,
        explain_samples=None,
        kernel_background=None,
    )

    run(**CONFIG)