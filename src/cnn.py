"""
Small CNN baseline (deep learning used for comparison only).

Trains on the same 64x64 images and the same folds as the classical models,
and reports the same metrics through evaluate.RunCollector.

Results are saved to results/tables/cnn_results.csv, and main.py loads them
instead of retraining. Training takes about 1 minute per fold on a GPU and
several minutes per fold on a CPU.

Requires PyTorch (pip install torch). Not needed to run main.py.

Run:  python src/cnn.py                 (1 seed x 5 folds, 15 epochs)
      python src/cnn.py --seeds 3       (all 15 splits, matches classical runs)
      python src/cnn.py --folds 1 --epochs 2   (quick check)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import (
    DATA_DIR, RunCollector, iter_folds, save_detailed, save_table, to_row,
)


class SmallCNN(nn.Module):
    """Three conv blocks (32-64-128 filters) -> global pooling -> linear head."""

    def __init__(self, n_classes: int = 6):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_fold(X_tr, y_tr, X_te, epochs, batch_size, lr, device, seed):
    """Train on one training fold, return predictions for the test fold."""
    set_seed(seed)
    # Normalisation statistics come from the training fold only (no leakage).
    mean = X_tr.mean(axis=(0, 2, 3), keepdims=True)
    std = X_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xtr = torch.tensor((X_tr - mean) / std, dtype=torch.float32)
    Xte = torch.tensor((X_te - mean) / std, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)

    model = SmallCNN(int(y_tr.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=int(np.ceil(len(Xtr) / batch_size))
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            # Light augmentation: random horizontal flip.
            flip = torch.rand(len(xb), device=device) < 0.5
            xb[flip] = xb[flip].flip(-1)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"      epoch {epoch + 1:2d}/{epochs}  train loss {total / len(Xtr):.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            preds.append(model(Xte[i:i + 256].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def run_cnn(n_seeds=1, max_folds=5, epochs=15, batch_size=64, lr=3e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training SmallCNN on {device}")

    data = np.load(DATA_DIR / "intel_subset.npz")
    X = data["X_img"].astype(np.float32).transpose(0, 3, 1, 2) / 255.0  # (N, 3, 64, 64)
    y = data["y"]
    runs = RunCollector(len(data["class_names"]))

    for s, f, tr, te in iter_folds(n_seeds):
        if f >= max_folds:
            continue
        print(f"  seed {s} fold {f}")
        start = time.time()
        pred = train_one_fold(X[tr], y[tr], X[te], epochs, batch_size, lr, device, seed=100 * s + f)
        runs.add(y[te], pred, time.time() - start)
        print(f"    fold accuracy {runs.accs[-1]:.3f}  ({runs.fit_times[-1]:.0f}s)")

    res = runs.summary()
    res["label"] = f"SmallCNN ({epochs} epochs, raw 64x64 RGB)"
    print(f"\n  {res['label']}: acc {res['acc_mean']:.3f} ± {res['acc_std']:.3f}, "
          f"macro-F1 {res['f1_mean']:.3f} ± {res['f1_std']:.3f} over {res['n_runs']} runs")
    save_table([to_row(res, method=res["label"], device=device)], "cnn_results.csv")
    save_detailed(res, "cnn")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    run_cnn(a.seeds, a.folds, a.epochs, a.batch_size, a.lr)
