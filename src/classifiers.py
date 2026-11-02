"""
Classical classifiers and the proposed fusion method.

Experiments
-----------
1. Feature comparison : kNN / LDA / Naive Bayes / SVM on every feature set
                        (raw pixels -> single handcrafted features -> combined).
2. Reduction study    : no reduction vs PCA vs LDA projection vs Fisher score
                        (all fitted inside each training fold).
3. Fusion (proposed)  : one SVM "expert" per feature family, combined by
                        skill-weighted probability averaging. Compared against
                        concatenation, majority voting and equal-weight averaging.

Run:  python src/classifiers.py            (3 seeds x 5 folds, final numbers)
      python src/classifiers.py --seeds 1  (quick check)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluate import (
    FIGURES_DIR, TABLES_DIR, load_feature, run_cv, run_cv_shared, save_detailed, save_table, to_row,
)
from src.reduction import get_fisher_pipeline, get_lda_pipeline, get_pca_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# Column layout of features/handcrafted_all.npy (see features.py):
#   hsv_hist (128) + color_moments (9) | lbp (112) | gabor (24) | hog (1764)
FEATURE_GROUPS = {
    "color": slice(0, 137),
    "lbp": slice(137, 249),
    "gabor": slice(249, 273),
    "hog": slice(273, 2037),
}

FEATURE_SETS = [
    "raw_gray", "raw_rgb_small",
    "hsv_hist", "color_moments", "lbp", "gabor", "hog",
    "handcrafted_compact", "handcrafted_all",
]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def get_classifiers() -> dict:
    """Fresh, untrained classical classifiers with fixed hyperparameters."""
    return {
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "LDA": LinearDiscriminantAnalysis(),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
    }


def make_reducer(reduction: str = "pca", k: int = 100) -> Pipeline:
    """Person 1's reduction pipeline on its own (used when classifiers share it)."""
    return Pipeline(make_model(None, reduction, k).steps[:-1])


def make_model(clf, reduction: str = "pca", k: int = 100) -> Pipeline:
    """
    Chain Person 1's reduction pipeline with a classifier.
    reduction: "none" (scaling only), "pca", "lda" or "fisher".
    """
    if reduction == "none":
        steps = [("scaler", StandardScaler())]
    elif reduction == "pca":
        steps = list(get_pca_pipeline(0.95).steps)
    elif reduction == "lda":
        steps = list(get_lda_pipeline(5).steps)
    elif reduction == "fisher":
        steps = list(get_fisher_pipeline(k).steps)
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    return Pipeline(steps + [("clf", clone(clf) if clf is not None else "passthrough")])


def default_reduction(n_features: int) -> str:
    """PCA for high-dimensional sets, scaling only for small ones."""
    return "pca" if n_features > 300 else "none"


# ---------------------------------------------------------------------------
# Proposed method: skill-weighted fusion of per-feature experts
# ---------------------------------------------------------------------------

class FeatureFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Late fusion of per-feature-family SVM experts.

    Concatenating all features lets the 1764-dim HOG block dominate the 137-dim
    colour block. Instead, each feature family gets its own expert (scaler ->
    PCA -> RBF-SVM with probability outputs), and the experts' class
    probabilities are combined.

    Combination rules
    -----------------
    "vote"     : majority vote over the experts' predicted labels.
    "equal"    : average of the experts' probability vectors.
    "weighted" : (proposed) each expert's probabilities are weighted by its skill
                 above chance, w_g = max(acc_g - 1/C, 0), measured on a held-out
                 validation split carved out of the TRAINING fold only. Experts
                 are then refitted on the full training fold.
    """

    def __init__(self, groups: dict | None = None, rule: str = "weighted",
                 C: float = 10.0, val_size: float = 0.2, random_state: int = 0):
        self.groups = groups
        self.rule = rule
        self.C = C
        self.val_size = val_size
        self.random_state = random_state

    def _expert(self):
        return Pipeline(list(get_pca_pipeline(0.95).steps) + [
            ("clf", SVC(kernel="rbf", C=self.C, gamma="scale",
                        probability=True, random_state=self.random_state)),
        ])

    def fit(self, X, y):
        groups = self.groups or FEATURE_GROUPS
        self.classes_ = np.unique(y)
        chance = 1.0 / len(self.classes_)
        self.weights_ = {g: 1.0 for g in groups}

        if self.rule == "weighted":
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
            )
            for g, cols in groups.items():
                acc = self._expert().fit(X_tr[:, cols], y_tr).score(X_val[:, cols], y_val)
                self.weights_[g] = max(acc - chance, 0.0)
            total = sum(self.weights_.values()) or 1.0
            self.weights_ = {g: w / total for g, w in self.weights_.items()}

        self.experts_ = {g: self._expert().fit(X[:, cols], y) for g, cols in groups.items()}
        self.groups_ = groups
        return self

    def predict_proba(self, X):
        P = np.zeros((X.shape[0], len(self.classes_)))
        for g, cols in self.groups_.items():
            P += self.weights_[g] * self.experts_[g].predict_proba(X[:, cols])
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        if self.rule == "vote":
            votes = np.stack([self.experts_[g].predict(X[:, c]) for g, c in self.groups_.items()], axis=1)
            counts = np.stack([(votes == k).sum(axis=1) for k in self.classes_], axis=1)
            # Ties are broken by the summed probabilities.
            counts = counts + 1e-3 * self.predict_proba(X)
            return self.classes_[counts.argmax(axis=1)]
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_feature_comparison(n_seeds: int) -> pd.DataFrame:
    print("\n[1/3] Feature comparison: every classifier on every feature set")
    rows = []
    for feat in FEATURE_SETS:
        X = load_feature(feat)
        red = default_reduction(X.shape[1])
        results = run_cv_shared(X, make_reducer(red), get_classifiers(), n_seeds, f"{feat} + {red}")
        for name, res in results.items():
            rows.append(to_row(res, features=feat, dims=X.shape[1], reduction=red, classifier=name))
            if feat in ("raw_gray", "handcrafted_all"):
                save_detailed(res)
    return save_table(rows, "feature_comparison.csv")


def experiment_reduction(n_seeds: int) -> pd.DataFrame:
    print("\n[2/3] Reduction study on handcrafted_all (fitted inside each fold)")
    X = load_feature("handcrafted_all")
    configs = [("none", None), ("pca", None), ("lda", None),
               ("fisher", 50), ("fisher", 100), ("fisher", 200)]
    rows = []
    for red, k in configs:
        red_label = f"fisher(k={k})" if k else red
        clfs = {n: c for n, c in get_classifiers().items() if n in ("kNN", "SVM")}
        results = run_cv_shared(X, make_reducer(red, k or 100), clfs, n_seeds, f"handcrafted_all + {red_label}")
        for name, res in results.items():
            rows.append(to_row(res, reduction=red_label, classifier=name))
    return save_table(rows, "reduction_comparison.csv")


def experiment_fusion(n_seeds: int) -> pd.DataFrame:
    print("\n[3/3] Fusion study on handcrafted_all")
    X = load_feature("handcrafted_all")
    candidates = {
        "Concatenation + SVM (early fusion)": make_model(get_classifiers()["SVM"], "pca"),
        "Majority vote of experts": FeatureFusionClassifier(rule="vote"),
        "Equal-weight probability fusion": FeatureFusionClassifier(rule="equal"),
        "Skill-weighted probability fusion (proposed)": FeatureFusionClassifier(rule="weighted"),
    }
    rows = []
    for label, model in candidates.items():
        res = run_cv(X, model, n_seeds, label)
        rows.append(to_row(res, method=label))
        save_detailed(res)

    # Learned weights from one full-data fit, for the report discussion only
    # (no accuracy is reported from this fit).
    from src.evaluate import load_labels
    y, _ = load_labels()
    w = FeatureFusionClassifier(rule="weighted").fit(X, y).weights_
    pd.DataFrame([w]).to_csv(TABLES_DIR / "fusion_weights.csv", index=False, encoding="utf-8-sig")
    print("  learned expert weights:", {g: round(v, 3) for g, v in w.items()})
    return save_table(rows, "fusion_comparison.csv")


def plot_summary(feat_df: pd.DataFrame, fusion_df: pd.DataFrame):
    """Bar chart: best classical result per feature set, then fusion methods, then CNN."""
    best = feat_df.loc[feat_df.groupby("features")["acc_mean"].idxmax()]
    best = best.set_index("features").loc[[f for f in FEATURE_SETS if f in best.index]]
    labels = [f"{f}\n({c})" for f, c in zip(best.index, best["classifier"])]
    accs = list(best["acc_mean"])
    stds = [float(s.split("±")[1]) for s in best["accuracy"]]
    colors = ["#9aa5b1"] * 2 + ["#5b8def"] * 5 + ["#2d5bd1"] * 2

    for _, r in fusion_df.iterrows():
        if "Concatenation" in r["method"]:
            continue
        labels.append(r["method"].replace(" probability", "\n").replace(" of ", "\nof "))
        accs.append(r["acc_mean"])
        stds.append(float(r["accuracy"].split("±")[1]))
        colors.append("#e07b39" if "proposed" in r["method"] else "#f0b27a")

    cnn_path = TABLES_DIR / "cnn_results.csv"
    if cnn_path.exists():
        c = pd.read_csv(cnn_path).iloc[0]
        labels.append("Small CNN\n(DL reference)")
        accs.append(c["acc_mean"])
        stds.append(float(c["accuracy"].split("±")[1]))
        colors.append("#6c757d")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(accs)), accs, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(range(len(accs)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (mean ± std over folds)")
    ax.set_ylim(0, 1)
    ax.set_title("Raw pixels → handcrafted features → fusion → CNN")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_results_comparison.png", dpi=200)
    plt.close(fig)


def run_all(n_seeds: int = 3):
    feat_df = experiment_feature_comparison(n_seeds)
    experiment_reduction(n_seeds)
    fusion_df = experiment_fusion(n_seeds)
    plot_summary(feat_df, fusion_df)
    print(f"\n[✓] Tables in {TABLES_DIR}, figures in {FIGURES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="number of CV seeds (1-3)")
    ap.add_argument("--only", choices=["features", "reduction", "fusion"], default=None)
    args = ap.parse_args()
    if args.only == "features":
        experiment_feature_comparison(args.seeds)
    elif args.only == "reduction":
        experiment_reduction(args.seeds)
    elif args.only == "fusion":
        experiment_fusion(args.seeds)
    else:
        run_all(args.seeds)
