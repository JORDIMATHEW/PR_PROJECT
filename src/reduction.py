"""
Dimensionality Reduction and Feature Selection Pipeline.
Person 1 Deliverable: Custom Fisher Score selector and leakage-safe pipelines.
"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FisherScoreSelector(BaseEstimator, TransformerMixin):
    """
    Supervised univariate feature selector using the Fisher Score criterion:
    Ratio of between-class variance to within-class variance.
    
    Parameters
    ----------
    k : int
        Number of top scoring features to retain.
    """

    def __init__(self, k: int = 100):
        self.k = k
        self.scores_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute Fisher scores for each feature based on training fold data only.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        # Overall mean for each feature across the entire training split
        overall_mean = np.mean(X, axis=0)

        between_class_var = np.zeros(n_features, dtype=np.float32)
        within_class_var = np.zeros(n_features, dtype=np.float32)

        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            if n_c == 0:
                continue

            mean_c = np.mean(X_c, axis=0)
            var_c = np.var(X_c, axis=0)

            between_class_var += n_c * ((mean_c - overall_mean) ** 2)
            within_class_var += n_c * var_c

        # Numerical stabilizer to prevent division by zero for constant features
        epsilon = 1e-12
        self.scores_ = between_class_var / (within_class_var + epsilon)

        # Select indices of the top k highest scoring features
        k_capped = min(self.k, n_features)
        # np.argsort sorts ascending; take the last k_capped elements and reverse
        self.selected_indices_ = np.argsort(self.scores_)[-k_capped:][::-1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X to the top k features determined during fit().
        """
        if self.selected_indices_ is None:
            raise RuntimeError("FisherScoreSelector has not been fitted yet. Call fit() before transform().")
        
        X = np.asarray(X, dtype=np.float32)
        return X[:, self.selected_indices_]


# ---------------------------------------------------------------------------
# Pipeline Factory Helpers (Leakage-Safe Interface for Person 2)
# ---------------------------------------------------------------------------

def get_pca_pipeline(variance_ratio: float = 0.95) -> Pipeline:
    """
    Returns a standard scaler + PCA pipeline maintaining specified cumulative variance.
    Fits strictly on training folds inside cross-validation.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=variance_ratio, svd_solver="full", random_state=42)),
    ])


def get_lda_pipeline(n_components: int = 5) -> Pipeline:
    """
    Returns a standard scaler + LDA projection pipeline.
    Projects 6 classes down to at most 5 linear discriminant axes.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=n_components)),
    ])


def get_fisher_pipeline(k: int = 100) -> Pipeline:
    """
    Returns a standard scaler + custom Fisher score selector pipeline.
    Filters down to the top k discriminative features.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("fisher", FisherScoreSelector(k=k)),
    ])


# ---------------------------------------------------------------------------
# Verification Routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[*] Verifying reduction module with synthetic validation batch...")

    rng = np.random.default_rng(42)
    # Synthetic dataset: 100 samples, 50 features, 6 classes
    X_synthetic = rng.standard_normal((100, 50)).astype(np.float32)
    y_synthetic = rng.integers(0, 6, size=100)

    # 1. Test FisherScoreSelector standalone
    fisher = FisherScoreSelector(k=10)
    fisher.fit(X_synthetic, y_synthetic)
    X_fisher = fisher.transform(X_synthetic)
    print(f"[✓] FisherScoreSelector(k=10) transformed shape: {X_fisher.shape} (Expected: (100, 10))")
    print(f"    Top 5 feature indices: {fisher.selected_indices_[:5]}")

    # 2. Test PCA Pipeline
    pca_pipe = get_pca_pipeline(variance_ratio=0.95)
    X_pca = pca_pipe.fit_transform(X_synthetic)
    n_components_kept = pca_pipe.named_steps["pca"].n_components_
    print(f"[✓] PCA Pipeline transformed shape: {X_pca.shape} (Kept {n_components_kept} components for 95% variance)")

    # 3. Test LDA Pipeline
    lda_pipe = get_lda_pipeline(n_components=5)
    X_lda = lda_pipe.fit_transform(X_synthetic, y_synthetic)
    print(f"[✓] LDA Pipeline transformed shape: {X_lda.shape} (Expected: (100, 5))")

    print("[*] All reduction transformers operational and compliant with sklearn API.")