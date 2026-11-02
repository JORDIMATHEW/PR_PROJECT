"""
Evaluation harness for multi-class scene classification.
Leakage-safe cross-validation and metric reporting.

Every experiment in the project goes through this file, so all results use the
same 3 seeds x 5 stratified folds stored in data/splits.npy (created by
prepare_data.py). A fresh copy of the model is fitted on the training folds of
every split, so scaling, PCA, LDA and Fisher selection never see test data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
CM_DIR = FIGURES_DIR / "confusion_matrices"
for d in (TABLES_DIR, FIGURES_DIR, CM_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_labels():
    """Return (y, class_names) from the shared dataset file."""
    data = np.load(DATA_DIR / "intel_subset.npz")
    return data["y"], [str(c) for c in data["class_names"]]


def load_splits():
    """Return the (n_seeds, n_samples) fold-assignment matrix."""
    return np.load(DATA_DIR / "splits.npy")


def load_feature(name: str) -> np.ndarray:
    """Load a cached feature matrix produced by features.py."""
    path = FEATURES_DIR / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/features.py first.")
    return np.load(path)


def iter_folds(n_seeds: int = 3):
    """Yield (seed_idx, fold_idx, train_mask, test_mask) for every split."""
    splits = load_splits()
    n_seeds = min(n_seeds, splits.shape[0])
    for s in range(n_seeds):
        for f in np.unique(splits[s]):
            test = splits[s] == f
            yield s, int(f), ~test, test


# ---------------------------------------------------------------------------
# Metric aggregation (shared by classical models and the CNN)
# ---------------------------------------------------------------------------

class RunCollector:
    """Accumulates per-fold predictions and summarises them as mean +/- std."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.accs, self.f1s = [], []
        self.precisions, self.recalls, self.class_f1s = [], [], []
        self.cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.fit_times = []

    def add(self, y_true, y_pred, fit_time: float = 0.0):
        labels = np.arange(self.n_classes)
        self.accs.append(accuracy_score(y_true, y_pred))
        self.f1s.append(f1_score(y_true, y_pred, average="macro"))
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        self.precisions.append(p)
        self.recalls.append(r)
        self.class_f1s.append(f)
        self.cm += confusion_matrix(y_true, y_pred, labels=labels)
        self.fit_times.append(fit_time)

    def summary(self) -> dict:
        P, R, F = (np.array(a) for a in (self.precisions, self.recalls, self.class_f1s))
        return {
            "acc_mean": float(np.mean(self.accs)),
            "acc_std": float(np.std(self.accs)),
            "f1_mean": float(np.mean(self.f1s)),
            "f1_std": float(np.std(self.f1s)),
            "n_runs": len(self.accs),
            "fit_time_s": float(np.mean(self.fit_times)),
            "precision_mean": P.mean(0), "precision_std": P.std(0),
            "recall_mean": R.mean(0), "recall_std": R.std(0),
            "class_f1_mean": F.mean(0), "class_f1_std": F.std(0),
            "cm": self.cm.copy(),
        }


def run_cv(X: np.ndarray, model, n_seeds: int = 3, label: str = "", verbose: bool = True) -> dict:
    """
    Evaluate an sklearn-compatible model over the shared splits.

    The model is cloned and refitted on every training fold, so any preprocessing
    inside it (scaler, PCA, selector) is learned from training data only.
    """
    y, class_names = load_labels()
    runs = RunCollector(len(class_names))
    t0 = time.time()
    for s, f, tr, te in iter_folds(n_seeds):
        start = time.time()
        m = clone(model).fit(X[tr], y[tr])
        fit_time = time.time() - start
        runs.add(y[te], m.predict(X[te]), fit_time)
    res = runs.summary()
    res["label"] = label
    if verbose:
        print(f"  {label:45s} acc {res['acc_mean']:.3f} ± {res['acc_std']:.3f}   "
              f"macro-F1 {res['f1_mean']:.3f} ± {res['f1_std']:.3f}   "
              f"({time.time() - t0:.0f}s)")
    return res


def run_cv_shared(X: np.ndarray, reducer, classifiers: dict, n_seeds: int = 3,
                  prefix: str = "", verbose: bool = True) -> dict:
    """
    Evaluate several classifiers that share one reduction step.

    In every fold the reducer (e.g. scaler + PCA) is fitted on the training fold
    only, then each classifier is trained on the reduced training data. This is
    exactly equivalent to running Pipeline(reducer, clf) through run_cv for each
    classifier, but fits the expensive reducer once per fold instead of once
    per classifier.
    """
    y, class_names = load_labels()
    runs = {name: RunCollector(len(class_names)) for name in classifiers}
    t0 = time.time()
    for s, f, tr, te in iter_folds(n_seeds):
        start = time.time()
        red = clone(reducer).fit(X[tr], y[tr])
        Z_tr, Z_te = red.transform(X[tr]), red.transform(X[te])
        red_time = time.time() - start
        for name, clf in classifiers.items():
            start = time.time()
            m = clone(clf).fit(Z_tr, y[tr])
            runs[name].add(y[te], m.predict(Z_te), red_time + time.time() - start)
    out = {}
    for name, rc in runs.items():
        res = rc.summary()
        res["label"] = f"{prefix} + {name}" if prefix else name
        out[name] = res
        if verbose:
            print(f"  {res['label']:45s} acc {res['acc_mean']:.3f} ± {res['acc_std']:.3f}   "
                  f"macro-F1 {res['f1_mean']:.3f} ± {res['f1_std']:.3f}")
    if verbose:
        print(f"  {'':45s} ({time.time() - t0:.0f}s for {len(classifiers)} classifiers)")
    return out


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def plot_confusion_matrix(cm: np.ndarray, class_names, title: str, path: Path):
    """Row-normalised confusion matrix (recall per true class) with raw counts."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_detailed(res: dict, name: str | None = None) -> None:
    """Write the per-class table and confusion matrix for one result."""
    _, class_names = load_labels()
    name = name or _slug(res["label"])
    per_class = pd.DataFrame({
        "class": class_names,
        "precision": [f"{m:.3f} ± {s:.3f}" for m, s in zip(res["precision_mean"], res["precision_std"])],
        "recall": [f"{m:.3f} ± {s:.3f}" for m, s in zip(res["recall_mean"], res["recall_std"])],
        "f1": [f"{m:.3f} ± {s:.3f}" for m, s in zip(res["class_f1_mean"], res["class_f1_std"])],
    })
    per_class.to_csv(TABLES_DIR / f"per_class_{name}.csv", index=False, encoding="utf-8-sig")
    plot_confusion_matrix(
        res["cm"], class_names,
        f"{res['label']}\nacc {res['acc_mean']:.3f} ± {res['acc_std']:.3f}, "
        f"macro-F1 {res['f1_mean']:.3f} ± {res['f1_std']:.3f}  (summed over {res['n_runs']} runs)",
        CM_DIR / f"cm_{name}.png",
    )


def to_row(res: dict, **extra) -> dict:
    """Flatten a result into one row for a summary table."""
    row = dict(extra)
    row.update({
        "accuracy": f"{res['acc_mean']:.4f} ± {res['acc_std']:.4f}",
        "macro_f1": f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}",
        "acc_mean": round(res["acc_mean"], 4),
        "f1_mean": round(res["f1_mean"], 4),
        "runs": res["n_runs"],
        "fit_time_s": round(res["fit_time_s"], 2),
    })
    return row


def save_table(rows: list, filename: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    # Quick self-test: LDA on the compact handcrafted features, one seed (prints only).
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = load_feature("handcrafted_compact")
    run_cv(X, make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
           n_seeds=1, label="self-test: compact + LDA")
