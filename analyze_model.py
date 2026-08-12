"""
Produce the numbers behind train_model.py as JSON, for charting.

Re-runs the same leave-one-session-out pipeline as train_model.py and dumps:
  - per-fold accuracy + confusion matrix
  - feature importances from the final model
  - class balance per session
  - motion-energy distributions per class (the feature that separates them)

Usage:
    python analyze_model.py
    python analyze_model.py --sessions part_1_data part_2_data --window-seconds 2.0
"""

import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from csi_common import feature_names
from train_model import (DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS,
                          build_dataset)

# Sessions/window come from train_model.py so this reports on the DEPLOYED
# configuration. They used to be hardcoded here as 4 sessions at a 2.0s window
# with no way to override them, so this script could only ever describe a model
# that stopped being deployed in July.
LABEL_NAMES = {0: "Empty", 2: "Moving"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    ap.add_argument("--out", default="model_analysis.json")
    args = ap.parse_args()

    X, y, groups, amp_cols = build_dataset(args.sessions, args.window_seconds)
    names = feature_names(amp_cols)

    # index of the two motion-energy columns (mean, std) for the distribution chart
    me_mean_idx = names.index("motion_energy_mean_ratio")

    logo = LeaveOneGroupOut()
    folds = []
    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = groups[test_idx][0]
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], pred)
        cm = confusion_matrix(y[test_idx], pred, labels=[0, 2])
        folds.append({
            "session": held_out,
            "accuracy": round(float(acc), 4),
            "n_windows": int(len(test_idx)),
            "confusion_matrix": cm.tolist(),  # rows=true [0,2], cols=pred [0,2]
        })

    final_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    final_clf.fit(X, y)
    importances = final_clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    top_features = [{"name": names[i], "importance": round(float(importances[i]), 5)}
                     for i in top_idx]

    class_balance = {}
    for session in args.sessions:
        mask = groups == session
        counts = {LABEL_NAMES[int(lbl)]: int((y[mask] == lbl).sum()) for lbl in [0, 2]}
        class_balance[session] = counts

    motion_energy = {
        "Empty": [round(float(v), 3) for v in X[y == 0, me_mean_idx]],
        "Moving": [round(float(v), 3) for v in X[y == 2, me_mean_idx]],
    }

    # Weighted = the real per-window accuracy; the unweighted mean over-counts
    # small sessions. Both are emitted so a chart built off this JSON can say
    # which one it is showing. See train_model.py for the full reasoning.
    accs = [f["accuracy"] for f in folds]
    sizes = [f["n_windows"] for f in folds]

    out = {
        "sessions": args.sessions,
        "window_seconds": args.window_seconds,
        "total_windows": int(len(X)),
        "loso_weighted_accuracy": round(float(np.average(accs, weights=sizes)), 4),
        "loso_mean_accuracy": round(float(np.mean(accs)), 4),
        "folds": folds,
        "top_features": top_features,
        "class_balance": class_balance,
        "motion_energy": motion_energy,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print(json.dumps({k: v for k, v in out.items() if k != "motion_energy"}, indent=2))


if __name__ == "__main__":
    main()
