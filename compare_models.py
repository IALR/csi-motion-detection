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
from sklearn.base import clone
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
from train_model import DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS, build_dataset

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
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
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
        fold_accs, fold_sizes = [], []
        for train_idx, test_idx in logo.split(X, y, groups):
            # clone() so each fold trains a genuinely fresh estimator. Reusing
            # one instance happens to work (fit() refits from scratch), but it
            # leaves the last fold's fitted state on a module-level object,
            # which is a trap for anyone who later reads MODELS after this runs.
            est = clone(model)
            est.fit(X[train_idx], y[train_idx])
            pred = est.predict(X[test_idx])
            fold_accs.append(accuracy_score(y[test_idx], pred))
            fold_sizes.append(len(test_idx))
        # Weighted = the real per-window accuracy. The unweighted mean counts a
        # 150-window session the same as a 600-window one, which flatters every
        # model here equally but by different amounts, so it can even change the
        # ranking. See train_model.py for the same correction.
        weighted = float(np.average(fold_accs, weights=fold_sizes))
        unweighted = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        results.append((name, weighted, unweighted, std_acc, fold_accs))
        print(f"{name:26s}  LOSO: {weighted:.4f} weighted / {unweighted:.4f} unweighted  "
              f"(std: {std_acc:.4f})  per-fold: {[round(a, 3) for a in fold_accs]}")

    print("\n" + "=" * 78)
    print("Ranked by WEIGHTED (per-window) LOSO accuracy:")
    print("=" * 78)
    for name, weighted, unweighted, std_acc, _ in sorted(results, key=lambda r: -r[1]):
        print(f"  {weighted:.4f} weighted  {unweighted:.4f} unweighted  "
              f"(±{std_acc:.4f} across sessions)  {name}")


if __name__ == "__main__":
    main()
