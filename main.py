"""
Multi-class Scene Classification (Intel Image Classification dataset)
Full pipeline entry point.

    python main.py          fast mode (default)
    python main.py --full   full mode: 3 seeds x 5 folds, regenerate all figures,
                            retrain the CNN (requires PyTorch)

Fast mode
  - features: loaded from features/ (extracted only if missing)
  - figures : regenerated only if missing
  - classical experiments: run in full, 1 seed x 5 stratified folds
    (mean +/- std over 5 runs)
  - CNN: results loaded from results/tables/cnn_results.csv. The CNN was
    trained once on Google Colab (T4 GPU) with `python src/cnn.py --seeds 3`
    and is not retrained here.

Full mode reproduces the numbers reported in the report.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)                      # relative paths in src/ assume this
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluate import FIGURES_DIR, TABLES_DIR


def banner(text: str):
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


def step_features():
    banner("STEP 1/5  Feature extraction")
    from src.features import extract_all_and_cache
    extract_all_and_cache(data_path=str(PROJECT_ROOT / "data" / "intel_subset.npz"),
                          output_dir=str(PROJECT_ROOT / "features"))


def step_visualizations(full: bool):
    banner("STEP 2/5  Feature visualisations")
    expected = ["01_sample_grid.png", "02_hog_visualizations.png", "03_lbp_maps.png",
                "04_gabor_responses.png", "05_hsv_histograms.png", "06_pca_comparison.png",
                "07_tsne_comparison.png", "08_fisher_top_features.png"]
    missing = [f for f in expected if not (FIGURES_DIR / f).exists()]
    if full or missing:
        from src.visualize import run_all_visualizations
        run_all_visualizations()
    else:
        print("  All figures already present in results/figures/ (use --full to regenerate).")


def step_cnn(full: bool):
    banner("STEP 3/5  Deep learning comparison (small CNN)")
    cnn_csv = TABLES_DIR / "cnn_results.csv"
    if cnn_csv.exists() and not full:
        print(f"  Loading saved CNN results from {cnn_csv.relative_to(PROJECT_ROOT)}")
        print("  (trained once with `python src/cnn.py`; use --full to retrain)")
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  PyTorch not installed and no saved CNN results - skipping CNN.")
        return
    from src.cnn import run_cnn
    run_cnn(n_seeds=3 if full else 1)


def step_classical(full: bool):
    n_seeds = 3 if full else 1
    banner(f"STEP 4/5  Classical experiments ({n_seeds} seed(s) x 5 folds)")
    from src.classifiers import run_all
    run_all(n_seeds)


def step_summary():
    banner("STEP 5/5  Results summary")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    feat = pd.read_csv(TABLES_DIR / "feature_comparison.csv")
    best = feat.loc[feat.groupby("features", sort=False)["acc_mean"].idxmax()]
    print("\n  Best classifier per feature set:")
    print(best[["features", "dims", "reduction", "classifier", "accuracy", "macro_f1"]]
          .to_string(index=False))

    red = pd.read_csv(TABLES_DIR / "reduction_comparison.csv")
    print("\n  Dimensionality reduction on handcrafted_all:")
    print(red[["reduction", "classifier", "accuracy", "macro_f1"]].to_string(index=False))

    fus = pd.read_csv(TABLES_DIR / "fusion_comparison.csv")
    rows = fus[["method", "accuracy", "macro_f1"]]
    cnn_csv = TABLES_DIR / "cnn_results.csv"
    if cnn_csv.exists():
        cnn = pd.read_csv(cnn_csv)
        rows = pd.concat([rows, cnn[["method", "accuracy", "macro_f1"]]], ignore_index=True)
    print("\n  Fusion methods vs deep learning:")
    print(rows.to_string(index=False))

    print(f"\n  Tables : {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="3 seeds, regenerate figures, retrain CNN")
    args = ap.parse_args()

    t0 = time.time()
    step_features()
    step_visualizations(args.full)
    step_cnn(args.full)          # before classical, so the comparison chart includes the CNN
    step_classical(args.full)
    step_summary()
    print(f"\n[✓] Pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
