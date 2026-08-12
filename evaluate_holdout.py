"""
True held-out evaluation: train on sessions 1-3, test on session 4 only.

Unlike train_model.py's leave-one-session-out (which rotates which session
is held out, so every session gets used for both training and testing across
folds), this fits ONE model on sessions 1-3 and scores it on session 4, which
never appears in training at all. This is the honest "does it generalize to
a new recording session" number.

Usage:
    python evaluate_holdout.py --test room2_part_1_data
    python evaluate_holdout.py --train part_1_data part_2_data --test part_4_data

If --train is omitted it defaults to every known session EXCEPT the ones being
tested, which is what "does it generalize to this new recording" actually
means. It used to default to a fixed three-session list that never grew as
sessions were added, so the answer silently stopped reflecting the real
training set.
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_model import (DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS,
                          build_dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=None,
                    help="defaults to every known session except those in --test")
    # Required, not defaulted: this tool's whole answer depends on WHICH
    # session is held out, and a default quietly gives you an answer about
    # some arbitrary older session instead of the one you meant to check.
    ap.add_argument("--test", nargs="+", required=True,
                    help="session folder(s) to hold out and score on")
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    ap.add_argument("--n-estimators", type=int, default=200)
    args = ap.parse_args()

    if args.train is None:
        args.train = [s for s in DEFAULT_SESSIONS if s not in args.test]
        print(f"(--train not given: using all known sessions except {args.test})")
    overlap = set(args.train) & set(args.test)
    if overlap:
        # Otherwise the "held-out" score is measured on data the model trained
        # on, which silently turns this into a meaningless self-test.
        ap.error(f"--train and --test overlap on {sorted(overlap)} - a held-out "
                 f"evaluation requires them to be disjoint")

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
