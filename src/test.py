import numpy as np

data = np.load("data/intel_subset.npz")
print("Images shape:", data["X_img"].shape)      # Must be (6000, 64, 64, 3)
print("Labels shape:", data["y"].shape)          # Must be (6000,)
print("Classes:", data["class_names"])           # 6 classes

splits = np.load("data/splits.npy")
print("Splits shape:", splits.shape)             # Must be (3, 6000)
print("Folds in seed 0:", np.unique(splits[0]))  # Must be array([0, 1, 2, 3, 4])