"""
True held-out evaluation: train on sessions 1-3, test on session 4 only.

Unlike train_model.py's leave-one-session-out (which rotates which session
is held out, so every session gets used for both training and testing across
folds), this fits ONE model on sessions 1-3 and scores it on session 4, which
never appears in training at all. This is the honest "does it generalize to
a new recording session" number.

Usage:
    python evaluate_holdout.py
    python evaluate_holdout.py --train part_1_data part_2_data part_3_data --test part_4_data
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_model import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["part_1_data", "part_2_data", "part_3_data"])
    ap.add_argument("--test", nargs="+", default=["part_4_data"])
    ap.add_argument("--window-seconds", type=float, default=2.0)
    ap.add_argument("--n-estimators", type=int, default=200)
    args = ap.parse_args()

    print(f"Train sessions: {args.train}")
    X_train, y_train, _, amp_cols_train = build_dataset(args.train, args.window_seconds)

    print(f"\nTest session(s): {args.test}")
    X_test, y_test, _, amp_cols_test = build_dataset(args.test, args.window_seconds)

    if amp_cols_train != amp_cols_test:
        raise ValueError("Train/test subcarrier columns differ - can't compare directly")

    print(f"\nTrain windows: {len(X_train)}  |  class counts: "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test windows:  {len(X_test)}  |  class counts: "
          f"{dict(zip(*np.unique(y_test, return_counts=True)))}\n")

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("=" * 55)
    print(f"Held-out accuracy on {args.test}: {acc:.4f}")
    print("=" * 55)
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (rows=true [0,2], cols=pred [0,2]):")
    print(confusion_matrix(y_test, pred, labels=[0, 2]))


if __name__ == "__main__":
    main()
