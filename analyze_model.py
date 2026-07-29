"""
Produce the numbers behind train_model.py as JSON, for charting.

Re-runs the same leave-one-session-out pipeline as train_model.py and dumps:
  - per-fold accuracy + confusion matrix
  - feature importances from the final model
  - class balance per session
  - motion-energy distributions per class (the feature that separates them)

Usage:
    python analyze_model.py
"""

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from csi_common import feature_names
from train_model import build_dataset

SESSIONS = ["part_1_data", "part_2_data", "part_3_data", "part_4_data"]
WINDOW_SECONDS = 2.0
LABEL_NAMES = {0: "Empty", 2: "Moving"}


def main():
    X, y, groups, amp_cols = build_dataset(SESSIONS, WINDOW_SECONDS)
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
    for session in SESSIONS:
        mask = groups == session
        counts = {LABEL_NAMES[int(lbl)]: int((y[mask] == lbl).sum()) for lbl in [0, 2]}
        class_balance[session] = counts

    motion_energy = {
        "Empty": [round(float(v), 3) for v in X[y == 0, me_mean_idx]],
        "Moving": [round(float(v), 3) for v in X[y == 2, me_mean_idx]],
    }

    out = {
        "sessions": SESSIONS,
        "window_seconds": WINDOW_SECONDS,
        "total_windows": int(len(X)),
        "loso_mean_accuracy": round(float(np.mean([f["accuracy"] for f in folds])), 4),
        "folds": folds,
        "top_features": top_features,
        "class_balance": class_balance,
        "motion_energy": motion_energy,
    }

    with open("model_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote model_analysis.json")
    print(json.dumps({k: v for k, v in out.items() if k != "motion_energy"}, indent=2))


if __name__ == "__main__":
    main()
