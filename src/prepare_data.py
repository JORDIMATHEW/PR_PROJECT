import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# Configuration
# Path to your extracted 'seg_train' folder:
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "seg_train" / "seg_train"

# Output directory automatically created inside your project:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (64, 64)
SAMPLES_PER_CLASS = 1000
RANDOM_SEED = 42

CLASSES = sorted([
    d.name for d in RAW_DATA_DIR.iterdir() 
    if d.is_dir() and not d.name.startswith(".")
])

print(f"Discovered {len(CLASSES)} classes: {CLASSES}")

images_list = []
labels_list = []

rng = np.random.default_rng(RANDOM_SEED)

for label_idx, class_name in enumerate(CLASSES):
    class_folder = RAW_DATA_DIR / class_name
    all_files = sorted([
        f for f in class_folder.iterdir() 
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    
    if len(all_files) < SAMPLES_PER_CLASS:
        raise ValueError(f"Class '{class_name}' has only {len(all_files)} images, needed {SAMPLES_PER_CLASS}.")
    
    # Deterministic sampling of 1000 images per class
    selected_files = rng.choice(all_files, size=SAMPLES_PER_CLASS, replace=False)
    
    for filepath in selected_files:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            images_list.append(np.array(img, dtype=np.uint8))
            labels_list.append(label_idx)

# Pack into contiguous NumPy arrays
X_img = np.stack(images_list, axis=0)  # Shape: (6000, 64, 64, 3)
y = np.array(labels_list, dtype=np.int64)  # Shape: (6000,)
class_names = np.array(CLASSES)

npz_path = OUTPUT_DIR / "intel_subset.npz"
np.savez_compressed(npz_path, X_img=X_img, y=y, class_names=class_names)
print(f"Saved dataset: {npz_path} ({npz_path.stat().st_size / (1024 * 1024):.2f} MB)")

# Generate 3 seeds x 5 stratified folds
# Matrix shape: (3, 6000) where entry (s, i) is the fold index (0 to 4) for image i
n_seeds = 3
n_splits = 5
splits_matrix = np.zeros((n_seeds, len(y)), dtype=np.int32)

for seed_idx, seed in enumerate([42, 123, 999]):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, test_indices) in enumerate(skf.split(X_img, y)):
        splits_matrix[seed_idx, test_indices] = fold_idx

splits_path = OUTPUT_DIR / "splits.npy"
np.save(splits_path, splits_matrix)
print(f"Saved split matrix: {splits_path} with shape {splits_matrix.shape}")