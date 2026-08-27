"""
SMD Data Loader
===============
Shared across selection experiments, Layer 1, and Layer 2.
Supports a single machine, a list of machines, or scanning an entire folder.

Two on-disk layouts are supported:

Structure A (original SMD .txt):
    data_dir/train/<machine>.txt
    data_dir/test/<machine>.txt
    data_dir/test_label/<machine>.txt

Structure B (CSV in root):
    data_dir/<machine>.train.csv
    data_dir/<machine>.test.csv   (label = last column or a *.test.label.csv file)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_LABEL_CANDIDATES = ["is_anomaly", "label", "anomaly", "class"]


def list_machines(data_dir):
    """
    Return a sorted list of every machine id found under data_dir,
    regardless of whether the layout is Structure A or B.
    """
    machines = set()

    # Structure A: look inside data_dir/train/*.txt
    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir):
        for p in glob.glob(os.path.join(train_dir, "*.txt")):
            machines.add(os.path.splitext(os.path.basename(p))[0])

    # Structure B: look for *.train.csv in root
    for p in glob.glob(os.path.join(data_dir, "*.train.csv")):
        machines.add(os.path.basename(p).replace(".train.csv", ""))

    return sorted(machines)


def load_smd_machine(data_dir, machine, scale=True):
    """
    Load one machine. Returns (X_train, X_test, y_test).
    Set scale=False to skip StandardScaler (e.g. if you scale elsewhere).
    """
    train_csv = os.path.join(data_dir, f"{machine}.train.csv")
    test_csv = os.path.join(data_dir, f"{machine}.test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        X_train, X_test, y_test = _load_structure_b(data_dir, machine,
                                                     train_csv, test_csv)
    else:
        X_train, X_test, y_test = _load_structure_a(data_dir, machine)

    print(f"  [DATA] {machine}: train={X_train.shape}, test={X_test.shape}, "
          f"anomaly rate={y_test.mean():.1%}")

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_test


def load_smd(data_dir, machines=None, scale=True):
    """
    Flexible entry point.
      machines=None        -> load every machine found in data_dir
      machines="machine-1-1" -> load that one machine
      machines=[...]       -> load the listed machines
    Yields (machine_id, X_train, X_test, y_test) one at a time so large
    multi-machine runs don't hold all data in memory at once.
    """
    if machines is None:
        machines = list_machines(data_dir)
    elif isinstance(machines, str):
        machines = [machines]

    if not machines:
        raise FileNotFoundError(f"No machines found under '{data_dir}'.")

    for m in machines:
        yield (m, *load_smd_machine(data_dir, m, scale=scale))


# ── internal helpers ──────────────────────────────────────────────────────────
def read_metadata(data_dir, machine):
    """
    Read <machine>.metadata.json and return the test-set anomaly rate.
    This value is derived from TEST labels, so it must ONLY be used on the
    evaluation side (threshold setting), never as a training parameter.
    Returns None if no metadata file exists.
    """
    import json
    path = os.path.join(data_dir, f"{machine}.metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    m = meta[0] if isinstance(meta, list) else meta
    return m.get("contamination")


def _load_structure_b(data_dir, machine, train_csv, test_csv):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    for df in (df_train, df_test):
        if "timestamp" in df.columns:
            df.drop(columns=["timestamp"], inplace=True)

    for c in _LABEL_CANDIDATES:
        if c in df_train.columns:
            df_train.drop(columns=[c], inplace=True)
            break

    label_col = next((c for c in _LABEL_CANDIDATES if c in df_test.columns), None)
    if label_col:
        y_test = df_test[label_col].values.astype(int)
        X_test = df_test.drop(columns=[label_col]).values.astype(float)
    elif df_test.shape[1] == df_train.shape[1] + 1:
        X_test = df_test.iloc[:, :-1].values.astype(float)
        y_test = df_test.iloc[:, -1].values.astype(int)
    else:
        label_csv = os.path.join(data_dir, f"{machine}.test.label.csv")
        if os.path.exists(label_csv):
            y_test = pd.read_csv(label_csv).iloc[:, -1].values.astype(int)
        else:
            print(f"  [WARN] No label for {machine}; metrics will be invalid.")
            y_test = np.zeros(len(df_test), dtype=int)
        X_test = df_test.values.astype(float)

    X_train = df_train.values.astype(float)
    return X_train, X_test, y_test


def _load_structure_a(data_dir, machine):
    train_path = os.path.join(data_dir,  f"{machine}.txt")
    test_path = os.path.join(data_dir,  f"{machine}.txt")
    label_path = os.path.join(data_dir,  f"{machine}.txt")
    X_train = np.loadtxt(train_path, delimiter=",")
    X_test = np.loadtxt(test_path, delimiter=",")
    y_test = np.loadtxt(label_path, delimiter=",").astype(int)
    return X_train, X_test, y_test


def make_synthetic_smd(n_train=2000, n_test=500, n_features=38,
                       contamination=0.05, random_state=42):
    """Synthetic SMD-like data for quick offline testing."""
    rng = np.random.default_rng(random_state)
    X_train = rng.standard_normal((n_train, n_features))
    X_test = rng.standard_normal((n_test, n_features))
    n_anom = int(n_test * contamination)
    y_test = np.zeros(n_test, dtype=int)
    anom_idx = rng.choice(n_test, n_anom, replace=False)
    y_test[anom_idx] = 1
    X_test[anom_idx] += rng.uniform(3, 6, (n_anom, n_features))
    return X_train, X_test, y_test