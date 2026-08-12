"""
Full model evaluation report: confusion matrix, feature correlation matrix,
two overfitting diagnostics, and a red/amber/green health summary - all in
one multi-page PDF.

Everything here is computed fresh from the real session data using the same
build_dataset() pipeline as train_model.py, so these numbers are always
consistent with whatever csi_model.joblib was actually trained on - nothing
here is hand-typed or copied from a prior run.

Usage:
    python model_evaluation.py
    python model_evaluation.py --sessions part_1_data ... --out my_report.pdf

Output:
    model_evaluation_report.pdf (or --out), plus a text summary on stdout.

Deps beyond requirements.txt: none (matplotlib is already listed).
"""

import argparse

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

import matplotlib.pyplot as plt

from csi_common import LABEL_NAMES, feature_names
from train_model import (DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS,
                          build_dataset)

N_ESTIMATORS = 200
TOP_K_CORRELATION = 25   # how many features to show in the correlation heatmap
DEPTH_SWEEP = [2, 3, 5, 8, 12, 20, None]  # for the overfitting validation curve


# ---------------------------------------------------------------------------
# Data + core LOSO run (shared by every plot below)
# ---------------------------------------------------------------------------

def run_loso(X, y, groups, max_depth=None):
    """One leave-one-session-out pass. Returns per-fold train/test accuracy,
    and the out-of-fold predictions stitched back into full-length arrays
    (each window's prediction from the one fold where its session was held
    out - this is the honest, no-leakage view of 'what would a user see')."""
    logo = LeaveOneGroupOut()
    y_pred_oof = np.empty_like(y)
    fold_names, fold_train_acc, fold_test_acc = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=max_depth, random_state=42)
        clf.fit(X[train_idx], y[train_idx])

        train_pred = clf.predict(X[train_idx])
        test_pred = clf.predict(X[test_idx])
        y_pred_oof[test_idx] = test_pred

        fold_names.append(groups[test_idx][0])
        fold_train_acc.append(accuracy_score(y[train_idx], train_pred))
        fold_test_acc.append(accuracy_score(y[test_idx], test_pred))

    return {
        "fold_names": fold_names,
        "fold_train_acc": np.array(fold_train_acc),
        "fold_test_acc": np.array(fold_test_acc),
        "y_pred_oof": y_pred_oof,
    }


# ---------------------------------------------------------------------------
# Page 1: Confusion matrix (out-of-fold - every prediction came from a model
# that never trained on that window's session)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(pdf, y, y_pred_oof):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [0, 2]
    display_labels = [LABEL_NAMES[l] for l in labels]

    cm = confusion_matrix(y, y_pred_oof, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Confusion matrix (counts)\nout-of-fold, leave-one-session-out")

    cm_norm = confusion_matrix(y, y_pred_oof, labels=labels, normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=display_labels).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".1%")
    axes[1].set_title("Confusion matrix (row-normalized)\nrecall per true class")

    fig.suptitle("Model Evaluation — Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    acc = accuracy_score(y, y_pred_oof)
    print(f"[confusion matrix] out-of-fold accuracy: {acc:.4f}")
    print(f"[confusion matrix] counts:\n{cm}")
    return acc, cm


# ---------------------------------------------------------------------------
# Page 2: Feature correlation matrix (top-K most important features)
# ---------------------------------------------------------------------------

def plot_correlation_matrix(pdf, X, names, importances):
    top_idx = np.argsort(importances)[::-1][:TOP_K_CORRELATION]
    top_names = [names[i] for i in top_idx]
    sub = X[:, top_idx]

    corr = np.corrcoef(sub, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_names)))
    ax.set_yticks(range(len(top_names)))
    ax.set_xticklabels(top_names, rotation=90, fontsize=7)
    ax.set_yticklabels(top_names, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    ax.set_title(f"Feature Correlation Matrix\ntop {TOP_K_CORRELATION} features by importance",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Flag strongly redundant pairs (|corr| > 0.9, excluding the diagonal)
    redundant = []
    for i in range(len(top_names)):
        for j in range(i + 1, len(top_names)):
            if abs(corr[i, j]) > 0.9:
                redundant.append((top_names[i], top_names[j], corr[i, j]))
    print(f"[correlation] {len(redundant)} feature pairs with |corr| > 0.9 among top {TOP_K_CORRELATION}")
    for a, b, c in redundant[:10]:
        print(f"    {a}  <->  {b}   corr={c:.2f}")
    return redundant


# ---------------------------------------------------------------------------
# Page 3: Feature importance (companion to the correlation matrix, makes it
# readable - which of those top-25 names actually matter most)
# ---------------------------------------------------------------------------

def plot_feature_importance(pdf, names, importances, top_n=25):
    top_idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[top_idx][::-1], color="#2a78d6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in top_idx][::-1], fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} Feature Importances\n(final model, trained on all sessions)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: Overfitting check #1 - train accuracy vs held-out accuracy, per
# session. A big, consistent gap here is the classic overfitting signature.
# ---------------------------------------------------------------------------

def plot_train_vs_test(pdf, loso_result):
    names = loso_result["fold_names"]
    train_acc = loso_result["fold_train_acc"]
    test_acc = loso_result["fold_test_acc"]
    gap = train_acc - test_acc

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, train_acc * 100, width=w, label="Train accuracy (in-sample)", color="#b7d3f6")
    ax.bar(x + w / 2, test_acc * 100, width=w, label="Held-out accuracy (out-of-sample)", color="#2a78d6")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", lw=0.5, ls="--")
    ax.legend()
    ax.set_title("Overfitting Check #1: Train vs. Held-out Accuracy per Session\n"
                 f"mean gap = {gap.mean()*100:.1f} points (bigger gap = more overfitting)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    print(f"[overfit check 1] mean train acc={train_acc.mean():.4f}  "
          f"mean held-out acc={test_acc.mean():.4f}  mean gap={gap.mean():.4f}")
    return gap


# ---------------------------------------------------------------------------
# Page 5: Overfitting check #2 - validation curve over tree depth. Train
# accuracy should keep climbing with depth; if held-out accuracy climbs with
# it, deeper is fine. If held-out accuracy plateaus/drops while train keeps
# climbing, that gap IS overfitting, visually.
# ---------------------------------------------------------------------------

def plot_depth_validation_curve(pdf, X, y, groups):
    train_means, test_means, test_stds = [], [], []
    for depth in DEPTH_SWEEP:
        result = run_loso(X, y, groups, max_depth=depth)
        train_means.append(result["fold_train_acc"].mean())
        test_means.append(result["fold_test_acc"].mean())
        test_stds.append(result["fold_test_acc"].std())
        print(f"[overfit check 2] max_depth={depth!s:>5}  "
              f"train={result['fold_train_acc'].mean():.4f}  "
              f"held-out={result['fold_test_acc'].mean():.4f}")

    x_labels = [str(d) if d is not None else "None\n(unlimited)" for d in DEPTH_SWEEP]
    x = np.arange(len(DEPTH_SWEEP))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, np.array(train_means) * 100, "o-", color="#b7d3f6", label="Train accuracy", lw=2)
    ax.errorbar(x, np.array(test_means) * 100, yerr=np.array(test_stds) * 100,
                fmt="o-", color="#2a78d6", label="Held-out (LOSO) accuracy", lw=2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("max_depth (tree complexity)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 102)
    ax.legend()
    ax.set_title("Overfitting Check #2: Accuracy vs. Tree Depth\n"
                 "train keeps climbing + held-out plateaus/drops = overfitting",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return train_means, test_means


# ---------------------------------------------------------------------------
# Page 6: Red/Amber/Green health summary
# ---------------------------------------------------------------------------

def rag(value, green_if, amber_if):
    """green_if/amber_if are (lo, hi) inclusive ranges."""
    if green_if[0] <= value <= green_if[1]:
        return "GREEN", "#0ca30c"
    if amber_if[0] <= value <= amber_if[1]:
        return "AMBER", "#c98500"
    return "RED", "#d03b3b"


def plot_rag_summary(pdf, overall_acc, loso_result, overfit_gap, class_counts, redundant_pairs):
    test_acc = loso_result["fold_test_acc"]
    worst_idx = int(np.argmin(test_acc))
    worst_name = loso_result["fold_names"][worst_idx]
    worst_acc = test_acc[worst_idx]
    std_acc = test_acc.std()

    total = sum(class_counts.values())
    empty_frac = class_counts.get(0, 0) / total * 100

    checks = [
        ("Overall LOSO accuracy", f"{overall_acc*100:.1f}%",
         rag(overall_acc * 100, (93, 100), (85, 93))),
        ("Overfit gap (train - held-out)", f"{overfit_gap.mean()*100:.1f} pts",
         rag(overfit_gap.mean() * 100, (0, 5), (5, 10))),
        ("Session-to-session variance (std)", f"{std_acc*100:.1f} pts",
         rag(std_acc * 100, (0, 3), (3, 6))),
        ("Worst single session", f"{worst_name}: {worst_acc*100:.1f}%",
         rag(worst_acc * 100, (90, 100), (80, 90))),
        ("Class balance (empty vs moving)", f"{empty_frac:.1f}% / {100-empty_frac:.1f}%",
         rag(empty_frac, (45, 55), (40, 60))),
        ("Redundant feature pairs (|corr|>0.9)", f"{len(redundant_pairs)} pairs",
         rag(len(redundant_pairs), (0, 3), (4, 8))),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    ax.set_title("Model Health Summary", fontsize=15, fontweight="bold", loc="left", pad=20)

    row_h = 1.0 / (len(checks) + 1)
    for i, (label, value, (status, color)) in enumerate(checks):
        y = 1 - row_h * (i + 1)
        ax.add_patch(plt.Circle((0.03, y + row_h * 0.3), 0.018, color=color, transform=ax.transAxes, clip_on=False))
        ax.text(0.08, y + row_h * 0.3, label, fontsize=11, va="center", transform=ax.transAxes)
        ax.text(0.62, y + row_h * 0.3, value, fontsize=11, va="center", transform=ax.transAxes, fontweight="bold")
        ax.text(0.86, y + row_h * 0.3, status, fontsize=10, va="center", transform=ax.transAxes,
                color=color, fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    print("\n=== RAG SUMMARY ===")
    for label, value, (status, _) in checks:
        print(f"  [{status:5s}] {label}: {value}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    ap.add_argument("--out", default="model_evaluation_report.pdf")
    args = ap.parse_args()

    print(f"Building dataset from {len(args.sessions)} sessions...")
    X, y, groups, amp_cols = build_dataset(args.sessions, args.window_seconds)
    names = feature_names(amp_cols)
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"Total windows: {len(X)}  |  features: {X.shape[1]}  |  class counts: {class_counts}\n")

    with PdfPages(args.out) as pdf:
        print("--- Running main LOSO pass (max_depth=None, matches deployed model) ---")
        main_result = run_loso(X, y, groups, max_depth=None)
        overall_acc, cm = plot_confusion_matrix(pdf, y, main_result["y_pred_oof"])

        print("\n--- Final model (trained on everything) for importance/correlation ---")
        final_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
        final_clf.fit(X, y)
        importances = final_clf.feature_importances_

        print("\n--- Feature correlation matrix ---")
        redundant_pairs = plot_correlation_matrix(pdf, X, names, importances)
        plot_feature_importance(pdf, names, importances)

        print("\n--- Overfitting check 1: train vs held-out per session ---")
        overfit_gap = plot_train_vs_test(pdf, main_result)

        print("\n--- Overfitting check 2: accuracy vs tree depth ---")
        plot_depth_validation_curve(pdf, X, y, groups)

        print("\n--- RAG health summary ---")
        plot_rag_summary(pdf, overall_acc, main_result, overfit_gap, class_counts, redundant_pairs)

    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()