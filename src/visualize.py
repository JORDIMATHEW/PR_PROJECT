"""
Visualisations for multi-class scene classification.

Generates the report figures: sample images, HOG / LBP / Gabor illustrations,
per-class HSV histograms, PCA and t-SNE embeddings, and the top Fisher-score
features. Figures are saved to results/figures/.

Note: the embeddings and Fisher ranking here are fitted on the whole dataset
for illustration only. All reported accuracies come from the cross-validated
pipelines in classifiers.py, where every fitted step sees training data only.
"""

import sys
from pathlib import Path

# Automatically inject the project root directory into Python's search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent display errors
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.reduction import FisherScoreSelector


# Resolve project directories relative to script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Loads image data and metadata."""
    npz_path = DATA_DIR / "intel_subset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run prepare_data.py first.")
    data = np.load(npz_path)
    return data["X_img"], data["y"], [str(c) for c in data["class_names"]]


# ---------------------------------------------------------------------------
# 1. Dataset Sample Grid
# ---------------------------------------------------------------------------
def plot_sample_grid(X_img: np.ndarray, y: np.ndarray, class_names: list):
    """Generates a 6x5 grid of sample images per class."""
    print("[*] Generating sample image grid...")
    fig, axes = plt.subplots(len(class_names), 5, figsize=(10, 12))
    rng = np.random.default_rng(42)

    for c_idx, c_name in enumerate(class_names):
        indices = np.where(y == c_idx)[0]
        selected = rng.choice(indices, size=5, replace=False)
        for col_idx, img_idx in enumerate(selected):
            ax = axes[c_idx, col_idx]
            ax.imshow(X_img[img_idx])
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(c_name, fontsize=11, fontweight="bold", loc="left")

    plt.suptitle("Intel Scene Classification: Dataset Samples", fontsize=14, y=0.92)
    out_path = FIGURES_DIR / "01_sample_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 2. HOG Feature Visualizations
# ---------------------------------------------------------------------------
def plot_hog_visualizations(X_img: np.ndarray, y: np.ndarray, class_names: list):
    """Plots original image vs. HOG gradient representation for selected classes."""
    print("[*] Generating HOG visualizations...")
    target_classes = ["buildings", "forest", "sea"]
    fig, axes = plt.subplots(len(target_classes), 2, figsize=(6, 9))

    for row_idx, c_name in enumerate(target_classes):
        c_idx = class_names.index(c_name)
        img_idx = np.where(y == c_idx)[0][0]
        img = X_img[img_idx]
        gray = color.rgb2gray(img)

        _, hog_img = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{c_name.capitalize()} (Original)", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(hog_img, cmap="inferno")
        axes[row_idx, 1].set_title(f"{c_name.capitalize()} (HOG Gradients)", fontsize=10)
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "02_hog_visualizations.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 3. LBP Texture Code Maps
# ---------------------------------------------------------------------------
def plot_lbp_maps(X_img: np.ndarray, y: np.ndarray, class_names: list):
    """Plots LBP code maps for high-texture (forest) vs. low-texture (glacier)."""
    print("[*] Generating LBP texture maps...")
    target_classes = ["forest", "glacier"]
    fig, axes = plt.subplots(len(target_classes), 2, figsize=(6, 6))

    for row_idx, c_name in enumerate(target_classes):
        c_idx = class_names.index(c_name)
        img_idx = np.where(y == c_idx)[0][0]
        img = X_img[img_idx]
        gray = color.rgb2gray(img)

        lbp_map = local_binary_pattern(gray, P=8, R=1, method="uniform")

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{c_name.capitalize()} (Original)", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(lbp_map, cmap="magma")
        axes[row_idx, 1].set_title(f"{c_name.capitalize()} (LBP Map P=8, R=1)", fontsize=10)
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "03_lbp_maps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 4. Gabor Filter Orientation Responses
# ---------------------------------------------------------------------------
def plot_gabor_responses(X_img: np.ndarray, y: np.ndarray, class_names: list):
    """Plots Gabor filter magnitude responses across 4 orientations."""
    print("[*] Generating Gabor response bank...")
    mountain_idx = class_names.index("mountain")
    img_idx = np.where(y == mountain_idx)[0][0]
    gray = color.rgb2gray(X_img[img_idx])

    thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    theta_labels = ["0°", "45°", "90°", "135°"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(X_img[img_idx])
    axes[0].set_title("Mountain (Original)", fontsize=10)
    axes[0].axis("off")

    for idx, (th, label) in enumerate(zip(thetas, theta_labels), start=1):
        real_resp, imag_resp = gabor(gray, frequency=0.2, theta=th)
        magnitude = np.hypot(real_resp, imag_resp)
        axes[idx].imshow(magnitude, cmap="viridis")
        axes[idx].set_title(f"Gabor θ = {label}", fontsize=10)
        axes[idx].axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "04_gabor_responses.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 5. Class Average HSV Profiles
# ---------------------------------------------------------------------------
def plot_hsv_histograms(y: np.ndarray, class_names: list):
    """Plots the mean HSV histogram vector per class."""
    print("[*] Generating HSV average profiles...")
    hsv_path = FEATURES_DIR / "hsv_hist.npy"
    if not hsv_path.exists():
        print("[-] Skipping HSV plot: features/hsv_hist.npy not found.")
        return

    hsv_features = np.load(hsv_path)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for c_idx, c_name in enumerate(class_names):
        class_mean = np.mean(hsv_features[y == c_idx], axis=0)
        axes[c_idx].bar(range(len(class_mean)), class_mean, color="teal", alpha=0.8, width=1.0)
        axes[c_idx].set_title(f"Class: {c_name.capitalize()}", fontsize=10, fontweight="bold")
        axes[c_idx].grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Mean 3D HSV Histogram Distribution (128 Bins) per Class", fontsize=13)
    fig.text(0.5, 0.04, "Histogram Bin Index", ha="center", fontsize=11)
    fig.text(0.06, 0.5, "Normalized Frequency", va="center", rotation="vertical", fontsize=11)

    out_path = FIGURES_DIR / "05_hsv_histograms.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 6. 2D Manifold Embeddings (PCA & t-SNE)
# ---------------------------------------------------------------------------
def plot_embeddings(y: np.ndarray, class_names: list):
    """Compares 2D PCA & t-SNE projections: raw pixels vs. handcrafted features."""
    print("[*] Generating 2D PCA & t-SNE manifold projections (1,500 samples)...")
    raw_path = FEATURES_DIR / "raw_gray.npy"
    handcrafted_path = FEATURES_DIR / "handcrafted_all.npy"

    if not raw_path.exists() or not handcrafted_path.exists():
        print("[-] Skipping manifold plots: features files missing.")
        return

    X_raw = np.load(raw_path)
    X_hand = np.load(handcrafted_path)

    # Subsample 250 instances per class (1,500 total) for visual legibility
    rng = np.random.default_rng(42)
    sample_indices = []
    for c in range(len(class_names)):
        idx = np.where(y == c)[0]
        sample_indices.extend(rng.choice(idx, size=250, replace=False))
    sample_indices = np.array(sample_indices)

    y_sub = y[sample_indices]
    X_raw_sub = StandardScaler().fit_transform(X_raw[sample_indices])
    X_hand_sub = StandardScaler().fit_transform(X_hand[sample_indices])

    # 1. 2D PCA Projections
    pca = PCA(n_components=2, random_state=42)
    raw_pca = pca.fit_transform(X_raw_sub)
    hand_pca = pca.fit_transform(X_hand_sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for c_idx, c_name in enumerate(class_names):
        mask = (y_sub == c_idx)
        ax1.scatter(raw_pca[mask, 0], raw_pca[mask, 1], label=c_name, alpha=0.6, s=18, color=colors[c_idx])
        ax2.scatter(hand_pca[mask, 0], hand_pca[mask, 1], label=c_name, alpha=0.6, s=18, color=colors[c_idx])

    ax1.set_title("PCA: Raw Grayscale Pixels (4,096 dims)", fontsize=11)
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_title("PCA: Handcrafted Features (2,037 dims)", fontsize=11)
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", markerscale=1.5, framealpha=0.9)

    plt.suptitle("PCA Comparison: Feature Representation vs. Class Separability", fontsize=13)
    out_pca = FIGURES_DIR / "06_pca_comparison.png"
    plt.savefig(out_pca, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_pca.name}")

    # 2. 2D t-SNE Projections
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    raw_tsne = tsne.fit_transform(X_raw_sub)
    hand_tsne = tsne.fit_transform(X_hand_sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for c_idx, c_name in enumerate(class_names):
        mask = (y_sub == c_idx)
        ax1.scatter(raw_tsne[mask, 0], raw_tsne[mask, 1], label=c_name, alpha=0.6, s=18, color=colors[c_idx])
        ax2.scatter(hand_tsne[mask, 0], hand_tsne[mask, 1], label=c_name, alpha=0.6, s=18, color=colors[c_idx])

    ax1.set_title("t-SNE: Raw Grayscale Pixels (Unseparated Clusters)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_title("t-SNE: Handcrafted Features (Defined Class Clusters)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", markerscale=1.5, framealpha=0.9)

    plt.suptitle("t-SNE Manifold Visualization (Visualization Only - No Leakage)", fontsize=13)
    out_tsne = FIGURES_DIR / "07_tsne_comparison.png"
    plt.savefig(out_tsne, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_tsne.name}")


# ---------------------------------------------------------------------------
# 7. Fisher Score Feature Ranking
# ---------------------------------------------------------------------------
def plot_fisher_top_features(y: np.ndarray):
    """Identifies and plots top 20 discriminative features categorized by type."""
    print("[*] Computing Fisher Score ranking across handcrafted representations...")
    handcrafted_path = FEATURES_DIR / "handcrafted_all.npy"
    if not handcrafted_path.exists():
        print("[-] Skipping Fisher plot: features/handcrafted_all.npy missing.")
        return

    X_hand = np.load(handcrafted_path)

    # Dimensional boundary partitions:
    # 0..127: HSV Hist (128)
    # 128..136: Color Moments (9)
    # 137..248: LBP (112)
    # 249..272: Gabor (24)
    # 273..2036: HOG (1764)
    def identify_feature_type(idx: int) -> str:
        if idx < 128:
            return "Color (HSV Hist)"
        elif idx < 137:
            return "Color (Moments)"
        elif idx < 249:
            return "Texture (LBP)"
        elif idx < 273:
            return "Texture (Gabor)"
        else:
            return "Shape (HOG)"

    fisher = FisherScoreSelector(k=20)
    fisher.fit(X_hand, y)

    top_indices = fisher.selected_indices_[:20]
    top_scores = fisher.scores_[top_indices]
    feature_labels = [f"F_{idx} ({identify_feature_type(idx)})" for idx in top_indices]

    # Color bar assignments by family
    color_map = {
        "Color (HSV Hist)": "#2b5c8f",
        "Color (Moments)": "#4a90e2",
        "Texture (LBP)": "#e67e22",
        "Texture (Gabor)": "#d35400",
        "Shape (HOG)": "#27ae60",
    }
    bar_colors = [color_map[identify_feature_type(idx)] for idx in top_indices]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_indices))
    plt.barh(y_pos, top_scores[::-1], color=bar_colors[::-1], align="center")
    plt.yticks(y_pos, feature_labels[::-1], fontsize=9)
    plt.xlabel("Fisher Score (Between-Class / Within-Class Variance Ratio)", fontsize=10)
    plt.title("Top 20 Features Ranked by Fisher Discriminant Criterion", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Add custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in color_map.values()]
    plt.legend(handles, color_map.keys(), loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "08_fisher_top_features.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Master Visualization Driver
# ---------------------------------------------------------------------------
def run_all_visualizations():
    """Generates all visualization artifacts."""
    print("==================================================")
    print("      INITIALIZING VISUALIZATION SUITE           ")
    print("==================================================")

    X_img, y, class_names = load_dataset()

    plot_sample_grid(X_img, y, class_names)
    plot_hog_visualizations(X_img, y, class_names)
    plot_lbp_maps(X_img, y, class_names)
    plot_gabor_responses(X_img, y, class_names)
    plot_hsv_histograms(y, class_names)
    plot_embeddings(y, class_names)
    plot_fisher_top_features(y)

    print("==================================================")
    print(f" [✓] Visualization suite complete. Figures saved to: {FIGURES_DIR}")
    print("==================================================")


if __name__ == "__main__":
    run_all_visualizations()