"""
Compare several ML model families on the SAME calibrated features and the
SAME leave-one-session-out methodology used everywhere else in this
project, so the comparison is apples-to-apples with the Random Forest
numbers you already have.

Usage:
    python compare_models.py
    python compare_models.py --sessions part_1_data ... --window-seconds 0.75
"""

import argparse

import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from csi_common import CALIB_SECONDS_DEFAULT
from train_model import build_dataset

# Models that are sensitive to feature scale get a StandardScaler in front.
# Tree-based models (RF, GB) don't need it - scaling doesn't change their
# splits, it'd just be wasted computation.
MODELS = {
    "Random Forest (current)": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    "SVM (RBF kernel)": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
    "SVM (linear kernel)": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=42)),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+",
                     default=["part_1_data", "part_2_data", "part_3_data", "part_4_data",
                              "part_5_data", "part_6_data", "part_7_data"])
    ap.add_argument("--window-seconds", type=float, default=0.75)
    ap.add_argument("--calib-seconds", type=float, default=CALIB_SECONDS_DEFAULT)
    args = ap.parse_args()

    print(f"Sessions: {args.sessions}")
    print(f"Window size: {args.window_seconds}s\n")
    X, y, groups, amp_cols = build_dataset(args.sessions, args.window_seconds, args.calib_seconds)
    print(f"\nTotal windows: {len(X)}  |  features: {X.shape[1]}  |  "
          f"class counts: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    logo = LeaveOneGroupOut()
    results = []

    for name, model in MODELS.items():
        fold_accs = []
        for train_idx, test_idx in logo.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            fold_accs.append(accuracy_score(y[test_idx], pred))
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        results.append((name, mean_acc, std_acc, fold_accs))
        print(f"{name:26s}  LOSO mean: {mean_acc:.4f}  (std: {std_acc:.4f})  "
              f"per-fold: {[round(a, 3) for a in fold_accs]}")

    print("\n" + "=" * 70)
    print("Ranked by mean LOSO accuracy:")
    print("=" * 70)
    for name, mean_acc, std_acc, _ in sorted(results, key=lambda r: -r[1]):
        print(f"  {mean_acc:.4f} (±{std_acc:.4f})  {name}")


if __name__ == "__main__":
    main()
