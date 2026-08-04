"""
Full model evaluation report generator.

Evaluates a trained scikit-learn classifier with Leave-One-Session-Out (LOSO)
cross-validation and produces a multi-page, publication-quality PDF report
(title page, table of contents, executive summary, dataset/model description,
overall + per-fold + per-class metrics tables, and ~20 diagnostic figures),
plus the underlying numbers as CSVs.

Designed to be generic: any scikit-learn-compatible classifier and any
tabular dataset with a target column and a "session"/group column will work
(binary classification unlocks a few extra metrics/plots - ROC, PR curve,
calibration curve, log loss - multiclass degrades gracefully by skipping
those and using macro/weighted averages everywhere else).

Two ways to supply the dataset:
    1. --csv path/to/data.csv --target-col label --session-col session
       A plain CSV: every column is a feature except target-col/session-col.
    2. --sessions part_1_data part_2_data ... (this project's layout only)
       Builds the feature matrix directly from session folders using
       train_model.build_dataset(), so no intermediate CSV is needed.

Usage:
    python full_model_report.py --model csi_model.joblib \
        --sessions part_1_data part_2_data part_3_data part_4_data \
        part_5_data part_6_data part_7_data part_8_data \
        room2_part_1_data room2_part_2_data \
        --target-col label --session-col session \
        --output-dir report_output --report-name report.pdf

    python full_model_report.py --model model.joblib --csv data.csv \
        --target-col label --session-col session

Outputs (all inside --output-dir):
    report.pdf                 The full report (or whatever --report-name is)
    metrics.csv                Overall (out-of-fold) metrics, one row
    per_fold_results.csv       One row per LOSO fold
    feature_importance.csv     Feature importances of the all-data refit model
    figures/*.png               Every generated figure, 300 dpi
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut, learning_curve, validation_curve

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

FIGURE_DPI = 300
PAGE_W, PAGE_H = LETTER
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# =====================================================================
# 1. Data loading
# =====================================================================

def load_dataset_from_csv(csv_path: Path, target_col: str, session_col: str) -> pd.DataFrame:
    """Load a flat CSV where every column except target/session is a feature."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in (target_col, session_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")
    return df


def load_dataset_from_sessions(sessions: list[str], window_seconds: float,
                                target_col: str, session_col: str) -> pd.DataFrame:
    """Build the feature matrix directly from this project's session folders.

    Only works when run from the CSI project root (needs train_model.py and
    csi_common.py on the path) - this is the bridge between the generic
    CSV-based interface this script exposes and the project's native
    session-folder data layout.
    """
    try:
        from train_model import build_dataset
        from csi_common import feature_names
    except ImportError as exc:
        raise ImportError(
            "--sessions requires train_model.py and csi_common.py to be "
            "importable (run this script from the CSI project root)."
        ) from exc

    X, y, groups, amp_cols = build_dataset(sessions, window_seconds)
    df = pd.DataFrame(X, columns=feature_names(amp_cols))
    df[target_col] = y
    df[session_col] = groups
    return df


def load_model(model_path: Path):
    """Load a fitted estimator from a .joblib file.

    Accepts either a bare estimator, or (as this project's csi_model.joblib
    does) a dict with the estimator under one of a few common keys, in which
    case the rest of the dict is returned as metadata for the report.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        for key in ("model", "estimator", "classifier", "clf"):
            if key in obj:
                estimator = obj[key]
                metadata = {k: v for k, v in obj.items() if k != key}
                return estimator, metadata
        raise ValueError(
            f"{model_path} is a dict but none of the expected keys "
            "('model', 'estimator', 'classifier', 'clf') were found."
        )
    return obj, {}


# =====================================================================
# 2. Dataset summary
# =====================================================================

def dataset_summary(df: pd.DataFrame, target_col: str, session_col: str,
                     feature_cols: list[str]) -> dict:
    """Basic dataset statistics for the report's dataset-description section."""
    return {
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "n_classes": int(df[target_col].nunique()),
        "n_sessions": int(df[session_col].nunique()),
        "missing_values": int(df[feature_cols + [target_col]].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# =====================================================================
# 3. LOSO evaluation
# =====================================================================

def run_loso(estimator_template, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Leave-one-session-out evaluation.

    Refits a fresh clone of estimator_template on every fold (train on all
    sessions but one, test on the held-out session) so "training accuracy"
    is meaningful per fold, then assembles out-of-fold predictions across
    every fold for the overall metrics - every prediction comes from a
    model that never trained on that sample's session.
    """
    classes = np.unique(y)
    has_proba = hasattr(estimator_template, "predict_proba")

    oof_pred = np.empty_like(y)
    oof_proba = np.zeros((len(y), len(classes))) if has_proba else None

    logo = LeaveOneGroupOut()
    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        held_out = groups[test_idx][0]
        model = clone(estimator_template)
        model.fit(X[train_idx], y[train_idx])

        train_pred = model.predict(X[train_idx])
        test_pred = model.predict(X[test_idx])
        oof_pred[test_idx] = test_pred

        if has_proba:
            proba = model.predict_proba(X[test_idx])
            for local_col, cls in enumerate(model.classes_):
                global_col = int(np.where(classes == cls)[0][0])
                oof_proba[test_idx, global_col] = proba[:, local_col]

        train_acc = accuracy_score(y[train_idx], train_pred)
        test_acc = accuracy_score(y[test_idx], test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[test_idx], test_pred, average="weighted", zero_division=0
        )
        records.append({
            "fold": fold_idx,
            "held_out_session": held_out,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        })

    fold_df = pd.DataFrame(records)
    return fold_df, oof_pred, oof_proba, classes


def fold_stat_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    """mean / median / min / max / std / 95% CI (normal approximation) for
    every numeric per-fold metric."""
    numeric_cols = ["train_accuracy", "test_accuracy", "precision_weighted",
                     "recall_weighted", "f1_weighted"]
    rows = []
    n = len(fold_df)
    for col in numeric_cols:
        vals = fold_df[col].to_numpy(dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        ci = 1.96 * sem  # normal approximation (scipy is not a listed dependency)
        rows.append({
            "metric": col, "mean": mean, "median": np.median(vals),
            "min": vals.min(), "max": vals.max(), "std": std,
            "ci95_low": mean - ci, "ci95_high": mean + ci,
        })
    return pd.DataFrame(rows)


# =====================================================================
# 4. Overall / per-class metrics
# =====================================================================

def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: np.ndarray | None, classes: np.ndarray) -> dict:
    """All the whole-dataset (out-of-fold) metrics requested, degrading
    gracefully to macro/weighted-only when the problem isn't binary."""
    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"] = p_macro, r_macro, f_macro

    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    metrics["weighted_precision"], metrics["weighted_recall"], metrics["weighted_f1"] = p_w, r_w, f_w

    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    is_binary = len(classes) == 2
    if is_binary:
        pos_label = classes[-1]
        p_bin, r_bin, f_bin, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        metrics["precision"], metrics["recall"], metrics["f1"] = p_bin, r_bin, f_bin

        if y_proba is not None:
            y_true_bin = (y_true == pos_label).astype(int)
            pos_col = int(np.where(classes == pos_label)[0][0])
            try:
                metrics["roc_auc"] = roc_auc_score(y_true_bin, y_proba[:, pos_col])
                metrics["average_precision"] = average_precision_score(y_true_bin, y_proba[:, pos_col])
            except ValueError as exc:
                warnings.warn(f"Skipping ROC AUC / average precision: {exc}")

    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba, labels=classes)
        except ValueError as exc:
            warnings.warn(f"Skipping log loss: {exc}")

    return metrics


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df = df.drop(index=[i for i in ("accuracy",) if i in df.index], errors="ignore")
    return df


def feature_importance_table(estimator_template, X: np.ndarray, y: np.ndarray,
                              feature_cols: list[str]) -> pd.DataFrame:
    """Refits on the full dataset (the model you'd actually deploy) and pulls
    feature_importances_ / |coef_| depending on what the estimator exposes."""
    final_model = clone(estimator_template)
    final_model.fit(X, y)

    if hasattr(final_model, "feature_importances_"):
        importances = np.asarray(final_model.feature_importances_)
    elif hasattr(final_model, "coef_"):
        importances = np.abs(np.asarray(final_model.coef_)).mean(axis=0)
    else:
        warnings.warn("Estimator exposes neither feature_importances_ nor coef_; "
                       "feature importance table/figures will be empty.")
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df, final_model


# =====================================================================
# 5. Figures (all matplotlib, 300 dpi, saved as PNG)
# =====================================================================

class FigureStore:
    """Collects generated figures (path + the figsize they were saved at, so
    the PDF builder can scale them to the page width while keeping aspect
    ratio) and skips gracefully when a figure can't be produced."""

    def __init__(self, figures_dir: Path):
        self.dir = figures_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}

    def save(self, name: str, fig, figsize: tuple[float, float]):
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        self.entries[name] = {"path": path, "width": figsize[0], "height": figsize[1]}

    def skip(self, name: str, reason: str):
        print(f"  [skipped figure] {name}: {reason}")


def fig_confusion_matrix(store: FigureStore, cm: np.ndarray, classes: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (counts)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)), labels=[str(c) for c in classes])
    ax.set_yticks(range(len(classes)), labels=[str(c) for c in classes])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="count")
    store.save("01_confusion_matrix", fig, (6, 5))


def fig_confusion_matrix_normalized(store: FigureStore, cm: np.ndarray, classes: np.ndarray):
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (row-normalized)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)), labels=[str(c) for c in classes])
    ax.set_yticks(range(len(classes)), labels=[str(c) for c in classes])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="fraction of true class")
    store.save("02_confusion_matrix_normalized", fig, (6, 5))


def fig_train_vs_test_by_fold(store: FigureStore, fold_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fold_df))
    w = 0.35
    ax.bar(x - w / 2, fold_df["train_accuracy"], w, label="Train accuracy")
    ax.bar(x + w / 2, fold_df["test_accuracy"], w, label="Test (held-out) accuracy")
    ax.set_xticks(x, labels=fold_df["held_out_session"], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs. Held-Out Accuracy by LOSO Fold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    store.save("03_train_vs_test_by_fold", fig, (8, 5))


def fig_test_accuracy_by_fold(store: FigureStore, fold_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fold_df))
    ax.bar(x, fold_df["test_accuracy"], color="#4C72B0")
    mean_acc = fold_df["test_accuracy"].mean()
    ax.axhline(mean_acc, color="red", linestyle="--", label=f"Mean = {mean_acc:.3f}")
    ax.set_xticks(x, labels=fold_df["held_out_session"], rotation=45, ha="right")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Held-Out Accuracy by Fold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    store.save("04_test_accuracy_by_fold", fig, (8, 5))


def fig_boxplot_fold_accuracies(store: FigureStore, fold_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(fold_df["test_accuracy"], tick_labels=["Held-out accuracy"])
    ax.set_title("Distribution of Fold Accuracies")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3)
    store.save("05_boxplot_fold_accuracies", fig, (5, 5))


def fig_metric_by_class(store: FigureStore, per_class_df: pd.DataFrame, metric: str,
                         name: str, title: str):
    classes_df = per_class_df.drop(index=[i for i in ("macro avg", "weighted avg") if i in per_class_df.index])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(classes_df.index.astype(str), classes_df[metric], color="#55A868")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel("Class")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    store.save(name, fig, (6, 5))


def fig_roc_curve(store: FigureStore, y_true_bin: np.ndarray, y_score: np.ndarray, auc: float):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("09_roc_curve", fig, (6, 5))


def fig_pr_curve(store: FigureStore, y_true_bin: np.ndarray, y_score: np.ndarray, ap: float):
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("10_pr_curve", fig, (6, 5))


def fig_feature_importance_top20(store: FigureStore, fi_df: pd.DataFrame):
    top = fi_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["importance"], color="#C44E52")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top 20)")
    ax.grid(axis="x", alpha=0.3)
    store.save("11_feature_importance_top20", fig, (7, 8))


def fig_cumulative_feature_importance(store: FigureStore, fi_df: pd.DataFrame):
    cum = fi_df["importance"].cumsum() / fi_df["importance"].sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker=".", markersize=3)
    ax.axhline(0.8, color="orange", linestyle="--", label="80%")
    ax.axhline(0.95, color="red", linestyle="--", label="95%")
    ax.set_xlabel("Number of features (ranked by importance)")
    ax.set_ylabel("Cumulative importance")
    ax.set_title("Cumulative Feature Importance")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("12_cumulative_feature_importance", fig, (7, 5))


def fig_class_distribution(store: FigureStore, y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([str(v) for v in values], counts, color="#8172B2")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)
    store.save("13_class_distribution", fig, (6, 5))


def fig_learning_curve(store: FigureStore, train_sizes, train_scores, test_scores):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train accuracy")
    ax.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Held-out accuracy (LOSO)")
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                     test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("14_learning_curve", fig, (7, 5))


def fig_validation_curve(store: FigureStore, param_range, train_scores, test_scores, param_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(param_range))
    ax.plot(x, train_scores.mean(axis=1), "o-", label="Train accuracy")
    ax.plot(x, test_scores.mean(axis=1), "o-", label="Held-out accuracy (LOSO)")
    ax.set_xticks(x, labels=[str(p) for p in param_range])
    ax.set_xlabel(param_name)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Validation Curve ({param_name})")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("15_validation_curve", fig, (7, 5))


def fig_calibration_curve(store: FigureStore, prob_true, prob_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("16_calibration_curve", fig, (6, 5))


def fig_prediction_probability_histogram(store: FigureStore, confidences: np.ndarray,
                                          correct_mask: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(confidences[correct_mask], bins=20, alpha=0.6, label="Correct predictions")
    ax.hist(confidences[~correct_mask], bins=20, alpha=0.6, label="Incorrect predictions")
    ax.set_xlabel("Predicted-class probability (confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Histogram")
    ax.legend()
    ax.grid(alpha=0.3)
    store.save("17_prediction_probability_histogram", fig, (7, 5))


def fig_misclassification_distribution(store: FigureStore, fold_df: pd.DataFrame):
    errors = ((1 - fold_df["test_accuracy"]) * fold_df["n_test"]).round().astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(fold_df["held_out_session"], errors, color="#DD8452")
    ax.set_xticks(range(len(fold_df)), labels=fold_df["held_out_session"], rotation=45, ha="right")
    ax.set_ylabel("Misclassified windows")
    ax.set_title("Misclassification Distribution by Session")
    ax.grid(axis="y", alpha=0.3)
    store.save("18_misclassification_distribution", fig, (8, 5))


def fig_error_distribution(store: FigureStore, cm: np.ndarray, classes: np.ndarray):
    labels, counts = [], []
    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            if i != j and cm[i, j] > 0:
                labels.append(f"true={true_c}\npred={pred_c}")
                counts.append(cm[i, j])
    fig, ax = plt.subplots(figsize=(7, 5))
    if labels:
        ax.bar(labels, counts, color="#937860")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution (off-diagonal confusion cells)")
    ax.grid(axis="y", alpha=0.3)
    store.save("19_error_distribution", fig, (7, 5))


def fig_fold_performance_summary(store: FigureStore, fold_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fold_df))
    w = 0.2
    ax.bar(x - 1.5 * w, fold_df["test_accuracy"], w, label="Accuracy")
    ax.bar(x - 0.5 * w, fold_df["precision_weighted"], w, label="Precision (weighted)")
    ax.bar(x + 0.5 * w, fold_df["recall_weighted"], w, label="Recall (weighted)")
    ax.bar(x + 1.5 * w, fold_df["f1_weighted"], w, label="F1 (weighted)")
    ax.set_xticks(x, labels=fold_df["held_out_session"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fold Performance Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    store.save("20_fold_performance_summary", fig, (9, 5))


def generate_all_figures(store: FigureStore, fold_df: pd.DataFrame, y_true: np.ndarray,
                          y_pred: np.ndarray, y_proba: np.ndarray | None, classes: np.ndarray,
                          fi_df: pd.DataFrame, per_class_df: pd.DataFrame,
                          estimator_template, X: np.ndarray, groups: np.ndarray,
                          overall_metrics: dict):
    """Runs every figure function, skipping (with a printed reason) whatever
    doesn't apply to this dataset/model (e.g. ROC curve on a >2-class
    problem, or feature importance on a model with neither
    feature_importances_ nor coef_)."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig_confusion_matrix(store, cm, classes)
    fig_confusion_matrix_normalized(store, cm, classes)

    fig_train_vs_test_by_fold(store, fold_df)
    fig_test_accuracy_by_fold(store, fold_df)
    fig_boxplot_fold_accuracies(store, fold_df)

    fig_metric_by_class(store, per_class_df, "precision", "06_precision_by_class", "Precision by Class")
    fig_metric_by_class(store, per_class_df, "recall", "07_recall_by_class", "Recall by Class")
    fig_metric_by_class(store, per_class_df, "f1-score", "08_f1_by_class", "F1-Score by Class")

    is_binary = len(classes) == 2
    if is_binary and y_proba is not None and "roc_auc" in overall_metrics:
        pos_label = classes[-1]
        pos_col = int(np.where(classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)
        fig_roc_curve(store, y_true_bin, y_proba[:, pos_col], overall_metrics["roc_auc"])
        fig_pr_curve(store, y_true_bin, y_proba[:, pos_col], overall_metrics["average_precision"])
    else:
        store.skip("09_roc_curve", "requires exactly 2 classes and predict_proba")
        store.skip("10_pr_curve", "requires exactly 2 classes and predict_proba")

    if not fi_df.empty:
        fig_feature_importance_top20(store, fi_df)
        fig_cumulative_feature_importance(store, fi_df)
    else:
        store.skip("11_feature_importance_top20", "estimator has no feature_importances_/coef_")
        store.skip("12_cumulative_feature_importance", "estimator has no feature_importances_/coef_")

    fig_class_distribution(store, y_true)

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            clone(estimator_template), X, y_true, groups=groups,
            cv=LeaveOneGroupOut(), train_sizes=np.linspace(0.2, 1.0, 6),
            scoring="accuracy",
        )
        fig_learning_curve(store, train_sizes, train_scores, test_scores)
    except Exception as exc:  # learning_curve can fail on tiny/odd datasets
        store.skip("14_learning_curve", str(exc))

    param_grid = {"max_depth": [2, 3, 5, 8, 12, 20, None], "C": [0.01, 0.1, 1, 10, 100],
                  "n_neighbors": [1, 3, 5, 9, 15, 25]}
    param_name = next((p for p in param_grid if p in estimator_template.get_params()), None)
    if param_name:
        try:
            train_scores, test_scores = validation_curve(
                clone(estimator_template), X, y_true, param_name=param_name,
                param_range=param_grid[param_name], groups=groups,
                cv=LeaveOneGroupOut(), scoring="accuracy",
            )
            fig_validation_curve(store, param_grid[param_name], train_scores, test_scores, param_name)
        except Exception as exc:
            store.skip("15_validation_curve", str(exc))
    else:
        store.skip("15_validation_curve", "no recognized hyperparameter (max_depth/C/n_neighbors) to sweep")

    if is_binary and y_proba is not None:
        pos_label = classes[-1]
        pos_col = int(np.where(classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)
        try:
            prob_true, prob_pred = calibration_curve(y_true_bin, y_proba[:, pos_col], n_bins=10)
            fig_calibration_curve(store, prob_true, prob_pred)
        except ValueError as exc:
            store.skip("16_calibration_curve", str(exc))
    else:
        store.skip("16_calibration_curve", "requires exactly 2 classes and predict_proba")

    if y_proba is not None:
        pred_col = np.array([int(np.where(classes == p)[0][0]) for p in y_pred])
        confidences = y_proba[np.arange(len(y_pred)), pred_col]
        correct_mask = y_pred == y_true
        fig_prediction_probability_histogram(store, confidences, correct_mask)
    else:
        store.skip("17_prediction_probability_histogram", "estimator has no predict_proba")

    fig_misclassification_distribution(store, fold_df)
    fig_error_distribution(store, cm, classes)
    fig_fold_performance_summary(store, fold_df)

    return cm


# =====================================================================
# 6. PDF report assembly (ReportLab)
# =====================================================================

class NumberedCanvas(pdfcanvas.Canvas):
    """Defers footer drawing until save(), so it can print 'Page X of Y'
    (the total page count isn't known until every page has been laid out)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total_pages)
            super().showPage()
        super().save()

    def _draw_footer(self, total_pages: int):
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.grey)
        self.drawString(MARGIN, 0.5 * inch, "CSI Motion Detection - Model Evaluation Report")
        self.drawRightString(PAGE_W - MARGIN, 0.5 * inch,
                              f"Page {self._pageNumber} of {total_pages}")


class ReportDocTemplate(BaseDocTemplate):
    """Adds a running header and auto-populates the table of contents from
    every Heading1/Heading2 paragraph in the story (two-pass build via
    multiBuild resolves the page numbers correctly)."""

    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=LETTER, **kwargs)
        frame = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="normal")
        self.addPageTemplates([PageTemplate(id="normal", frames=[frame], onPage=self._header)])

    @staticmethod
    def _header(canvas_obj, _doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica-Bold", 9)
        canvas_obj.setFillColor(colors.HexColor("#2A2A2A"))
        canvas_obj.drawString(MARGIN, PAGE_H - 0.5 * inch, "Model Evaluation Report")
        canvas_obj.line(MARGIN, PAGE_H - 0.55 * inch, PAGE_W - MARGIN, PAGE_H - 0.55 * inch)
        canvas_obj.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name == "Heading1":
                text = flowable.getPlainText()
                self.notify("TOCEntry", (0, text, self.page))
                key = f"h1-{self.page}-{text}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=0)
            elif style_name == "Heading2":
                text = flowable.getPlainText()
                self.notify("TOCEntry", (1, text, self.page))
                key = f"h2-{self.page}-{text}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=1)


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=26, leading=32,
                               alignment=TA_CENTER, spaceAfter=12, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=13, leading=18,
                               alignment=TA_CENTER, textColor=colors.grey, spaceAfter=6))
    styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=11,
                               alignment=TA_CENTER, textColor=colors.grey, spaceBefore=2, spaceAfter=12))
    styles.add(ParagraphStyle(name="BodyJustify", parent=styles["BodyText"],
                               alignment=TA_LEFT, spaceAfter=8, leading=14))
    return styles


def _df_to_table(df: pd.DataFrame, styles, col_widths=None, float_fmt="{:.4f}") -> Table:
    """Renders a small DataFrame as a styled ReportLab Table (header row +
    alternating row shading + grid)."""
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
    header = [Paragraph(f"<b>{c}</b>", styles["BodyText"]) for c in
              (["index"] + list(display_df.columns) if display_df.index.name or not
               isinstance(display_df.index, pd.RangeIndex) else list(display_df.columns))]
    include_index = not isinstance(display_df.index, pd.RangeIndex)
    data = [header]
    for idx, row in display_df.iterrows():
        cells = ([str(idx)] if include_index else []) + [str(v) for v in row]
        data.append(cells)

    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2A2A2A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
    ]
    table.setStyle(TableStyle(style))
    return table


def _rag(value: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    """Small red/amber/green helper for the executive summary / strengths
    section, consistent with this project's existing model_evaluation.py."""
    if higher_is_better:
        if value >= good:
            return "GREEN"
        if value >= ok:
            return "AMBER"
        return "RED"
    if value <= good:
        return "GREEN"
    if value <= ok:
        return "AMBER"
    return "RED"


def build_pdf_report(output_path: Path, store: FigureStore, dataset_stats: dict,
                     model, model_metadata: dict, overall_metrics: dict,
                     fold_df: pd.DataFrame, fold_stats: pd.DataFrame,
                     per_class_df: pd.DataFrame, fi_df: pd.DataFrame,
                     samples_per_class: pd.Series, samples_per_session: pd.Series,
                     target_col: str, session_col: str):
    styles = _build_styles()
    story = []

    # ---- Title page ----
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph("Model Evaluation Report", styles["ReportTitle"]))
    story.append(Paragraph("Leave-One-Session-Out Cross-Validation Analysis", styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(f"Model: {type(model).__name__}", styles["ReportSubtitle"]))
    story.append(Paragraph(f"Sessions evaluated: {dataset_stats['n_sessions']}  |  "
                            f"Samples: {dataset_stats['n_samples']}  |  "
                            f"Features: {dataset_stats['n_features']}", styles["ReportSubtitle"]))
    story.append(PageBreak())

    # ---- Table of contents ----
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOCHeading1", fontSize=12, leading=16, spaceBefore=6, fontName="Helvetica-Bold"),
        ParagraphStyle(name="TOCHeading2", fontSize=10, leading=13, leftIndent=20),
    ]
    story.append(toc)
    story.append(PageBreak())

    # ---- Executive summary ----
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    acc = overall_metrics.get("accuracy", float("nan"))
    gap = fold_df["train_accuracy"].mean() - fold_df["test_accuracy"].mean()
    rag_acc = _rag(acc, 0.90, 0.80)
    rag_gap = _rag(gap, 0.05, 0.10, higher_is_better=False)
    summary_text = (
        f"This report evaluates a {type(model).__name__} across "
        f"{dataset_stats['n_sessions']} recording sessions using Leave-One-Session-Out "
        f"(LOSO) cross-validation - every prediction used in the metrics below came from "
        f"a model that never trained on that sample's session. Out-of-fold accuracy is "
        f"<b>{acc:.2%}</b> ({rag_acc}), and the mean gap between training and held-out "
        f"accuracy across folds is <b>{gap:.2%}</b> ({rag_gap}), the standard signature "
        f"of overfitting when large and healthy generalization when small."
    )
    story.append(Paragraph(summary_text, styles["BodyJustify"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_df_to_table(
        pd.DataFrame([{"metric": k, "value": v} for k, v in overall_metrics.items()]),
        styles))
    story.append(PageBreak())

    # ---- Dataset description ----
    story.append(Paragraph("Dataset Description", styles["Heading1"]))
    story.append(Paragraph(
        "Summary statistics of the dataset used for evaluation, before any "
        "train/test splitting.", styles["BodyJustify"]))
    story.append(_df_to_table(
        pd.DataFrame([{"stat": k, "value": v} for k, v in dataset_stats.items()]), styles))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Samples per Class", styles["Heading2"]))
    spc = samples_per_class.rename("count").to_frame()
    spc.index.name = target_col
    story.append(_df_to_table(spc, styles))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Samples per Session", styles["Heading2"]))
    sps = samples_per_session.rename("count").to_frame()
    sps.index.name = session_col
    story.append(_df_to_table(sps, styles))
    story.append(PageBreak())

    # ---- Methodology ----
    story.append(Paragraph("Evaluation Methodology", styles["Heading1"]))
    story.append(Paragraph(
        "Leave-One-Session-Out (LOSO) cross-validation treats each recording session as "
        "a held-out fold: a fresh copy of the model is trained on every session except "
        "one, then scored on the excluded session, which it never saw during training. "
        "This is repeated once per session. Compared to a random train/test split, LOSO "
        "avoids leaking session-specific characteristics (same environment, same "
        "recording conditions) between train and test, which would otherwise inflate "
        "accuracy in a way that would not hold up on genuinely new data.", styles["BodyJustify"]))
    story.append(PageBreak())

    # ---- Model description ----
    story.append(Paragraph("Model Description", styles["Heading1"]))
    story.append(Paragraph(f"Estimator class: <b>{type(model).__module__}.{type(model).__name__}</b>",
                            styles["BodyJustify"]))
    if model_metadata:
        story.append(Paragraph("Metadata", styles["Heading2"]))
        meta_rows = [{"key": k, "value": str(v)} for k, v in model_metadata.items()
                     if not isinstance(v, (list, np.ndarray)) or len(v) < 10]
        if meta_rows:
            story.append(_df_to_table(pd.DataFrame(meta_rows), styles))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Hyperparameters", styles["Heading2"]))
    try:
        params = model.get_params()
        params_df = pd.DataFrame([{"parameter": k, "value": str(v)} for k, v in sorted(params.items())])
        story.append(_df_to_table(params_df, styles))
    except AttributeError:
        story.append(Paragraph("This estimator does not expose get_params().", styles["BodyJustify"]))
    story.append(PageBreak())

    # ---- Overall results ----
    story.append(Paragraph("Overall Results", styles["Heading1"]))
    story.append(Paragraph(
        "Metrics computed on the full set of out-of-fold predictions (every sample "
        "scored by a model that never trained on its session).", styles["BodyJustify"]))
    story.append(_df_to_table(
        pd.DataFrame([{"metric": k, "value": v} for k, v in overall_metrics.items()]), styles))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Confusion Matrices", styles["Heading2"]))
    _embed_figure(story, styles, store, "01_confusion_matrix", "Raw confusion matrix counts.")
    _embed_figure(story, styles, store, "02_confusion_matrix_normalized", "Row-normalized (recall per class).")
    story.append(PageBreak())

    story.append(Paragraph("Per-Class Metrics", styles["Heading2"]))
    story.append(_df_to_table(per_class_df.reset_index().rename(columns={"index": "class"}), styles))
    _embed_figure(story, styles, store, "06_precision_by_class", "Precision by class.")
    _embed_figure(story, styles, store, "07_recall_by_class", "Recall by class.")
    _embed_figure(story, styles, store, "08_f1_by_class", "F1-score by class.")
    story.append(PageBreak())

    if "09_roc_curve" in store.entries:
        story.append(Paragraph("ROC and Precision-Recall Curves", styles["Heading2"]))
        _embed_figure(story, styles, store, "09_roc_curve", "Receiver operating characteristic.")
        _embed_figure(story, styles, store, "10_pr_curve", "Precision-recall curve.")
        story.append(PageBreak())

    if "16_calibration_curve" in store.entries:
        story.append(Paragraph("Calibration and Confidence", styles["Heading2"]))
        _embed_figure(store=store, story=story, styles=styles, name="16_calibration_curve",
                      caption="Calibration curve: predicted probability vs. observed frequency.")
    if "17_prediction_probability_histogram" in store.entries:
        _embed_figure(story, styles, store, "17_prediction_probability_histogram",
                      "Confidence distribution, correct vs. incorrect predictions.")
    if "16_calibration_curve" in store.entries or "17_prediction_probability_histogram" in store.entries:
        story.append(PageBreak())

    # ---- Per-fold analysis ----
    story.append(Paragraph("Per-Fold Analysis", styles["Heading1"]))
    story.append(_df_to_table(fold_df, styles))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Fold Statistics (mean / median / min / max / std / 95% CI)", styles["Heading2"]))
    story.append(_df_to_table(fold_stats, styles))
    story.append(PageBreak())

    _embed_figure(story, styles, store, "03_train_vs_test_by_fold", "Train vs. held-out accuracy, per fold.")
    _embed_figure(story, styles, store, "04_test_accuracy_by_fold", "Held-out accuracy, per fold.")
    story.append(PageBreak())
    _embed_figure(story, styles, store, "05_boxplot_fold_accuracies", "Spread of held-out accuracy across folds.")
    _embed_figure(story, styles, store, "20_fold_performance_summary", "Accuracy/precision/recall/F1, per fold.")
    story.append(PageBreak())
    _embed_figure(story, styles, store, "18_misclassification_distribution", "Misclassified windows, per session.")
    _embed_figure(story, styles, store, "19_error_distribution", "Which classes get confused with which.")
    story.append(PageBreak())

    # ---- Overfitting diagnostics ----
    if "14_learning_curve" in store.entries or "15_validation_curve" in store.entries:
        story.append(Paragraph("Overfitting Diagnostics", styles["Heading1"]))
        if "14_learning_curve" in store.entries:
            _embed_figure(story, styles, store, "14_learning_curve",
                          "Accuracy vs. training-set size - a shrinking gap as size grows means more data helps.")
        if "15_validation_curve" in store.entries:
            _embed_figure(story, styles, store, "15_validation_curve",
                          "Accuracy vs. model complexity - held-out accuracy falling while train accuracy "
                          "keeps rising would indicate overfitting.")
        story.append(PageBreak())

    story.append(Paragraph("Class Distribution", styles["Heading2"]))
    _embed_figure(story, styles, store, "13_class_distribution", "Sample count per class across the whole dataset.")
    story.append(PageBreak())

    # ---- Feature importance ----
    if not fi_df.empty:
        story.append(Paragraph("Feature Importance Analysis", styles["Heading1"]))
        story.append(_df_to_table(fi_df.head(25), styles))
        story.append(Spacer(1, 0.1 * inch))
        _embed_figure(story, styles, store, "11_feature_importance_top20", "Top 20 features by importance.")
        _embed_figure(story, styles, store, "12_cumulative_feature_importance",
                      "How many top features are needed to explain most of the model's decisions.")
        story.append(PageBreak())

    # ---- Strengths / limitations / conclusion ----
    story.append(Paragraph("Strengths", styles["Heading1"]))
    strengths = []
    if rag_acc == "GREEN":
        strengths.append(f"Out-of-fold accuracy ({acc:.2%}) is strong given the LOSO methodology, "
                          "which does not allow session-specific leakage between train and test.")
    if rag_gap == "GREEN":
        strengths.append(f"Small train/held-out gap ({gap:.2%}) indicates the model generalizes to "
                          "unseen sessions rather than memorizing training sessions.")
    if fold_df["test_accuracy"].std() < 0.05:
        strengths.append("Low variance across folds - performance is consistent across sessions, "
                          "not driven by one or two easy folds.")
    if not strengths:
        strengths.append("See per-fold and per-class tables above for a detailed breakdown.")
    for s in strengths:
        story.append(Paragraph(f"- {s}", styles["BodyJustify"]))

    story.append(Paragraph("Limitations", styles["Heading1"]))
    limitations = []
    worst_fold = fold_df.loc[fold_df["test_accuracy"].idxmin()]
    limitations.append(f"Weakest session: {worst_fold['held_out_session']} "
                        f"({worst_fold['test_accuracy']:.2%} held-out accuracy) - worth a targeted "
                        "diagnostic rather than averaging it away.")
    if rag_gap != "GREEN":
        limitations.append(f"Train/held-out gap of {gap:.2%} is larger than ideal - "
                            "check the validation curve for overfitting at high model complexity.")
    if fold_df["test_accuracy"].std() >= 0.05:
        limitations.append("Session-to-session variance is non-trivial - accuracy depends "
                            "meaningfully on which session is evaluated.")
    for l in limitations:
        story.append(Paragraph(f"- {l}", styles["BodyJustify"]))

    story.append(Paragraph("Final Conclusion", styles["Heading1"]))
    story.append(Paragraph(
        f"Across {len(fold_df)} leave-one-session-out folds, the model achieved "
        f"{acc:.2%} out-of-fold accuracy with a {gap:.2%} train/held-out gap. "
        "This report's figures and tables (confusion matrices, per-class metrics, "
        "learning/validation curves, and feature importance) provide the evidence "
        "behind that number; see the Limitations section above for what to "
        "investigate next.", styles["BodyJustify"]))

    doc = ReportDocTemplate(str(output_path))
    doc.multiBuild(story, canvasmaker=NumberedCanvas)


def _embed_figure(story, styles, store: FigureStore, name: str, caption: str):
    if name not in store.entries:
        return
    entry = store.entries[name]
    target_w = min(CONTENT_W, 6.5 * inch)
    scale = target_w / (entry["width"] * inch)
    img = RLImage(str(entry["path"]), width=entry["width"] * inch * scale,
                  height=entry["height"] * inch * scale)
    story.append(img)
    story.append(Paragraph(caption, styles["Caption"]))


# =====================================================================
# 7. Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a full LOSO evaluation report (PDF + CSVs) for a "
                    "trained scikit-learn classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to the .joblib model file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV with feature columns + target-col + session-col.")
    source.add_argument("--sessions", nargs="+", help="Session folder names (this project's layout).")
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--session-col", default="session")
    parser.add_argument("--window-seconds", type=float, default=0.75, help="Only used with --sessions.")
    parser.add_argument("--output-dir", type=Path, default=Path("report_output"))
    parser.add_argument("--report-name", default="report.pdf")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading model...")
        model, model_metadata = load_model(args.model)

        print("Loading dataset...")
        if args.csv is not None:
            df = load_dataset_from_csv(args.csv, args.target_col, args.session_col)
        else:
            df = load_dataset_from_sessions(args.sessions, args.window_seconds,
                                              args.target_col, args.session_col)

        feature_cols = [c for c in df.columns if c not in (args.target_col, args.session_col)]
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            warnings.warn(f"Dropping non-numeric feature columns: {non_numeric}")
            feature_cols = [c for c in feature_cols if c not in non_numeric]
        if not feature_cols:
            raise ValueError("No usable numeric feature columns found.")

        stats = dataset_summary(df, args.target_col, args.session_col, feature_cols)
        samples_per_class = df[args.target_col].value_counts().sort_index()
        samples_per_session = df[args.session_col].value_counts().sort_index()
        print(f"  {stats['n_samples']} samples, {stats['n_features']} features, "
              f"{stats['n_classes']} classes, {stats['n_sessions']} sessions")

        X = df[feature_cols].to_numpy(dtype=float)
        y = df[args.target_col].to_numpy()
        groups = df[args.session_col].to_numpy()

        print("Running Leave-One-Session-Out evaluation...")
        fold_df, oof_pred, oof_proba, classes = run_loso(model, X, y, groups)
        fold_stats = fold_stat_summary(fold_df)

        overall_metrics = compute_overall_metrics(y, oof_pred, oof_proba, classes)
        per_class_df = per_class_metrics(y, oof_pred)

        print("Computing feature importance (refit on full dataset)...")
        fi_result = feature_importance_table(model, X, y, feature_cols)
        if isinstance(fi_result, tuple):
            fi_df, _final_model = fi_result
        else:
            fi_df = fi_result

        print("Generating figures...")
        figures_dir = args.output_dir / "figures"
        store = FigureStore(figures_dir)
        generate_all_figures(store, fold_df, y, oof_pred, oof_proba, classes, fi_df,
                              per_class_df, model, X, groups, overall_metrics)

        print("Writing CSVs...")
        pd.DataFrame([overall_metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
        fold_df.to_csv(args.output_dir / "per_fold_results.csv", index=False)
        fi_df.to_csv(args.output_dir / "feature_importance.csv", index=False)

        print("Building PDF report...")
        report_path = args.output_dir / args.report_name
        build_pdf_report(report_path, store, stats, model, model_metadata, overall_metrics,
                          fold_df, fold_stats, per_class_df, fi_df,
                          samples_per_class, samples_per_session,
                          args.target_col, args.session_col)

        print(f"\nDone. Report saved to {report_path}")
        print(f"Out-of-fold accuracy: {overall_metrics['accuracy']:.4f}")

    except Exception as exc:  # top-level guard: always fail with a clear message
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
