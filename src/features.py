"""
Feature Extraction Pipeline for Intel Scene Classification.
Person 1 Deliverable: Generates and caches all feature representations.
"""

from pathlib import Path
import numpy as np
from scipy import stats
from skimage import color, transform
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor


# ---------------------------------------------------------------------------
# Individual Extraction Subroutines (Single Image)
# ---------------------------------------------------------------------------

def extract_raw_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Grayscale pixel baseline, flattened and scaled to [0, 1]. Dims: 4096."""
    gray = color.rgb2gray(img_rgb)  # Already scaled in [0, 1]
    return gray.flatten().astype(np.float32)


def extract_raw_rgb_small(img_rgb: np.ndarray) -> np.ndarray:
    """Downsampled 32x32 RGB baseline, flattened. Dims: 3072."""
    small = transform.resize(
        img_rgb, (32, 32, 3), anti_aliasing=True, preserve_range=False
    )
    return small.flatten().astype(np.float32)


def extract_hsv_hist(img_rgb: np.ndarray) -> np.ndarray:
    """Joint 3D HSV histogram (8 H x 4 S x 4 V bins), normalized. Dims: 128."""
    img_hsv = color.rgb2hsv(img_rgb)
    h = img_hsv[:, :, 0].ravel()
    s = img_hsv[:, :, 1].ravel()
    v = img_hsv[:, :, 2].ravel()

    # Joint 3D histogram
    hist, _ = np.histogramdd(
        sample=np.column_stack([h, s, v]),
        bins=(8, 4, 4),
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    hist = hist.astype(np.float32)
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist /= total_pixels
    return hist.flatten()


def extract_color_moments(img_rgb: np.ndarray) -> np.ndarray:
    """Mean, standard deviation, and skewness across H, S, V channels. Dims: 9."""
    img_hsv = color.rgb2hsv(img_rgb)
    moments = []
    for c in range(3):
        channel = img_hsv[:, :, c].ravel()
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        skew_val = stats.skew(channel, bias=False)
        # Numerical guard for zero-variance flat channels
        if np.isnan(skew_val):
            skew_val = 0.0
        moments.extend([mean_val, std_val, skew_val])
    return np.array(moments, dtype=np.float32)


def extract_lbp(img_rgb: np.ndarray) -> np.ndarray:
    """
    Uniform LBP over a 2x2 spatial grid.
    Scale 1: P=8, R=1 (10 bins). Scale 2: P=16, R=2 (18 bins).
    Dims: 4 cells * (10 + 18) = 112.
    """
    gray = color.rgb2gray(img_rgb)
    h, w = gray.shape
    half_h, half_w = h // 2, w // 2

    # Precompute LBP maps across entire image
    lbp_scale1 = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_scale2 = local_binary_pattern(gray, P=16, R=2, method="uniform")

    features = []
    # 2x2 grid partitions
    quadrants = [
        (0, half_h, 0, half_w),
        (0, half_h, half_w, w),
        (half_h, h, 0, half_w),
        (half_h, h, half_w, w),
    ]

    for r_start, r_end, c_start, c_end in quadrants:
        cell_s1 = lbp_scale1[r_start:r_end, c_start:c_end].ravel()
        cell_s2 = lbp_scale2[r_start:r_end, c_start:c_end].ravel()

        # Scale 1: P=8 uniform -> bins 0 to 9 + outlier bin 10 (10 uniform patterns + 1 non-uniform)
        hist_s1, _ = np.histogram(cell_s1, bins=10, range=(0, 10), density=True)
        # Scale 2: P=16 uniform -> bins 0 to 17 (18 patterns)
        hist_s2, _ = np.histogram(cell_s2, bins=18, range=(0, 18), density=True)

        features.extend(hist_s1)
        features.extend(hist_s2)

    return np.array(features, dtype=np.float32)


def extract_gabor(img_rgb: np.ndarray) -> np.ndarray:
    """
    Gabor filter bank: 4 angles x 3 frequencies = 12 filters.
    Mean and std of response magnitude recorded. Dims: 24.
    """
    gray = color.rgb2gray(img_rgb)
    thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    frequencies = [0.1, 0.2, 0.3]

    features = []
    for th in thetas:
        for freq in frequencies:
            real_resp, imag_resp = gabor(gray, frequency=freq, theta=th)
            magnitude = np.hypot(real_resp, imag_resp)
            features.append(float(np.mean(magnitude)))
            features.append(float(np.std(magnitude)))

    return np.array(features, dtype=np.float32)


def extract_hog(img_rgb: np.ndarray) -> np.ndarray:
    """
    HOG descriptors: 9 orientations, 8x8 pixels per cell, 2x2 cells per block.
    Dims: 7 * 7 * 4 * 9 = 1764.
    """
    gray = color.rgb2gray(img_rgb)
    descriptors = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )
    return descriptors.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch Extraction & Caching Driver
# ---------------------------------------------------------------------------

def extract_all_and_cache(data_path: str = "data/intel_subset.npz", output_dir: str = "features"):
    """Loads dataset, computes all feature sets, and caches them to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Accessing dataset: {data_path}")
    raw_data = np.load(data_path)
    X_img = raw_data["X_img"]
    n_images = len(X_img)
    print(f"[*] Total instances identified: {n_images}")

    extractors = {
        "raw_gray": extract_raw_gray,
        "raw_rgb_small": extract_raw_rgb_small,
        "hsv_hist": extract_hsv_hist,
        "color_moments": extract_color_moments,
        "lbp": extract_lbp,
        "gabor": extract_gabor,
        "hog": extract_hog,
    }

    feature_matrices = {}

    for name, func in extractors.items():
        dest = out_dir / f"{name}.npy"
        if dest.exists():
            print(f"[-] Feature set '{name}' already present in cache. Loading directly.")
            feature_matrices[name] = np.load(dest)
            continue

        print(f"[+] Processing feature set: '{name}' across {n_images} instances...")
        matrix = np.zeros((n_images, len(func(X_img[0]))), dtype=np.float32)
        for i in range(n_images):
            matrix[i] = func(X_img[i])
            if (i + 1) % 1500 == 0:
                print(f"    - Completed {i + 1}/{n_images} instances.")

        np.save(dest, matrix)
        feature_matrices[name] = matrix
        print(f"[✓] Successfully saved {name}.npy with shape: {matrix.shape}")

    # Build Combined Representation Sets
    # 1. handcrafted_compact = hsv_hist + color_moments + lbp + gabor (273 dims)
    compact_dest = out_dir / "handcrafted_compact.npy"
    if not compact_dest.exists():
        compact_matrix = np.hstack([
            feature_matrices["hsv_hist"],
            feature_matrices["color_moments"],
            feature_matrices["lbp"],
            feature_matrices["gabor"],
        ]).astype(np.float32)
        np.save(compact_dest, compact_matrix)
        print(f"[✓] Successfully saved handcrafted_compact.npy with shape: {compact_matrix.shape}")

    # 2. handcrafted_all = compact + hog (2037 dims)
    all_dest = out_dir / "handcrafted_all.npy"
    if not all_dest.exists():
        all_matrix = np.hstack([
            np.load(compact_dest),
            feature_matrices["hog"],
        ]).astype(np.float32)
        np.save(all_dest, all_matrix)
        print(f"[✓] Successfully saved handcrafted_all.npy with shape: {all_matrix.shape}")

    print("\n[+] All feature matrices generated and cached in accordance with protocol.")


if __name__ == "__main__":
    extract_all_and_cache()