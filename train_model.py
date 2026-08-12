"""
Train a Random Forest to classify empty (0) vs moving (2) from CSI amplitude data.

Data layout expected (one folder per recording session):
    part_1_data/all_csi_data.csv
    part_2_data/all_csi_data.csv
    part_3_data/all_csi_data.csv
    part_4_data/all_csi_data.csv

Each CSV comes from csi_label_collector.py and has columns:
    session_id, host_unix_us, timestamp_us, label, esp_label, settling,
    rssi, num_subcarriers, amp_0 ... amp_{N-1}

Pipeline:
    1. Drop settling==1 rows (the 5s "leave the room" window).
    2. Per session, calibrate PER EMPTY BLOCK: every time the recording
       re-enters an empty block (label 0 starting right after a moving
       block, or the very first block), the first CALIB_SECONDS of it
       become a fresh baseline, used for every window until the next
       empty block starts. A single baseline for the whole session was
       tried first and found to go stale within a few minutes - real
       sessions showed both sudden shifts (e.g. turning the AC on between
       two empty blocks) and slow continuous drift (thermal/environmental)
       that a start-of-session-only baseline can't track. See
       csi_common.py for the calibration math itself.
    3. Split into fixed-size, non-overlapping time windows. A window is kept
       only if every frame in it has the same label (windows straddling a
       label change are ambiguous and dropped).
    4. Turn each window into one baseline-relative feature vector
       (csi_common.calibrate_features), scored against whichever empty
       block's baseline most recently preceded it: per-subcarrier
       amplitude mean DELTA and std RATIO vs baseline, motion-energy
       mean/std RATIO, RSSI mean DELTA / std RATIO.
    5. Evaluate with leave-one-session-out (LeaveOneGroupOut), then refit on
       all sessions and save the model.

Usage:
    python train_model.py
    python train_model.py --sessions part_1_data part_2_data part_3_data --window-seconds 2.0
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from csi_common import (CALIB_SECONDS_DEFAULT, FRAME_HZ, assess_noise_floor,
                         calibrate_features, compute_baseline, feature_names,
                         raw_window_stats)


def load_session(folder):
    path = os.path.join(folder, "all_csi_data.csv")
    df = pd.read_csv(path)
    df = df[df["settling"] == 0].reset_index(drop=True)
    df = df.sort_values("timestamp_us").reset_index(drop=True)
    return df


def amp_columns(df):
    return [c for c in df.columns if c.startswith("amp_")]


def find_empty_block_starts(labels):
    """Indices where a new empty block begins: label 0 immediately after
    a different label (or index 0, if the recording starts empty - which
    it always does under the LEAVE->EMPTY collector protocol)."""
    starts = []
    prev = None
    for idx, lbl in enumerate(labels):
        if lbl == 0 and prev != 0:
            starts.append(idx)
        prev = lbl
    return starts


def compute_block_baselines(amps, rssi, labels, block_starts, calib_seconds):
    """One baseline per empty-block start, from that block's own first
    calib_seconds. Returns {start_index: baseline}."""
    frames_needed = max(1, round(calib_seconds * FRAME_HZ))
    baselines = {}
    n = len(labels)
    for start in block_starts:
        end = start
        while end < n and labels[end] == 0 and (end - start) < frames_needed:
            end += 1
        if end - start < frames_needed:
            print(f"  WARNING: only {end - start} frames available for the "
                  f"{calib_seconds}s calibration window starting at row {start} "
                  f"(wanted {frames_needed})")
        baselines[start] = compute_baseline(amps[start:end], rssi[start:end])
    return baselines


def windowize(df, amp_cols, window_seconds, calib_seconds):
    """Yield (feature_vector, label) for each pure, fixed-size window,
    each scored against whichever empty block's baseline most recently
    preceded it (recalibrating at every LEAVE->EMPTY transition, not
    just once at the start of the session)."""
    frames_per_window = max(1, round(window_seconds * FRAME_HZ))
    amps = df[amp_cols].to_numpy(dtype=float)
    labels = df["label"].to_numpy()
    rssi = df["rssi"].to_numpy(dtype=float)

    block_starts = find_empty_block_starts(labels)
    if not block_starts:
        raise ValueError("No empty-labeled frames available to calibrate from")
    baselines = compute_block_baselines(amps, rssi, labels, block_starts, calib_seconds)

    n = len(df)
    for start in range(0, n - frames_per_window + 1, frames_per_window):
        end = start + frames_per_window
        window_labels = labels[start:end]
        if not np.all(window_labels == window_labels[0]):
            continue  # spans a label transition, skip

        governing_start = max(s for s in block_starts if s <= start)
        baseline = baselines[governing_start]

        stats = raw_window_stats(amps[start:end], rssi[start:end])
        feat = calibrate_features(stats, baseline)
        yield feat, window_labels[0]


def build_dataset(session_folders, window_seconds, calib_seconds=CALIB_SECONDS_DEFAULT):
    X, y, groups = [], [], []
    amp_cols = None
    for folder in session_folders:
        df = load_session(folder)
        cols = amp_columns(df)
        if amp_cols is None:
            amp_cols = cols
        elif cols != amp_cols:
            raise ValueError(f"{folder}: subcarrier columns differ from other sessions")

        n_before = len(df)
        labels_arr = df["label"].to_numpy()
        n_blocks = len(find_empty_block_starts(labels_arr))
        for feat, label in windowize(df, amp_cols, window_seconds, calib_seconds):
            X.append(feat)
            y.append(label)
            groups.append(folder)

        # How disturbed this session's room was while EMPTY, and how far apart
        # the two classes actually are in raw signal. A session where empty and
        # moving energy are within ~10% of each other is one the model can only
        # partly separate no matter how it's trained - worth seeing at load
        # time rather than discovering it as a bad fold much later.
        amps_arr = df[amp_cols].to_numpy(dtype=float)
        frame_energy = np.abs(np.diff(amps_arr, axis=0)).mean(axis=1)
        e_lab = labels_arr[1:]
        empty_e = frame_energy[e_lab == 0].mean() if (e_lab == 0).any() else float("nan")
        move_e = frame_energy[e_lab == 2].mean() if (e_lab == 2).any() else float("nan")
        sep = move_e / empty_e if empty_e else float("nan")
        # Scale-free, matching what the live system reports: jitter as a
        # fraction of received amplitude, averaged over active subcarriers
        # only (dead guard bands sit at zero and would deflate the scale).
        empty_cols = amps_arr[labels_arr == 0].mean(axis=0)
        amp_scale = empty_cols[empty_cols > 0].mean() if (empty_cols > 0).any() else float("nan")
        rel = empty_e / amp_scale if amp_scale else float("nan")
        level, _, _ = assess_noise_floor(rel)
        flag = ""
        if sep < 1.15:
            flag = "  *** empty/moving barely differ - expect this fold to score poorly"
        elif level == "loud":
            flag = "  ** noisy empty room"

        print(f"{folder}: {n_before} frames -> "
              f"{sum(1 for g in groups if g == folder)} windows "
              f"({n_blocks} empty block(s), each recalibrated on its own "
              f"first {calib_seconds}s)")
        print(f"    noise: rel={rel:.4f} [{level}]  (empty={empty_e:.2f} "
              f"moving={move_e:.2f} sep={sep:.2f}x amp={amp_scale:.1f}){flag}")

    return np.array(X), np.array(y), np.array(groups), amp_cols


# The sessions and window size the DEPLOYED model was actually trained with.
# EVERY script that builds a dataset imports these rather than keeping its own
# copy - analyze_model.py, compare_models.py, evaluate_holdout.py and
# model_evaluation.py each used to hardcode their own list, and each had
# drifted to a different stale subset (4, 7, 3 and 8 sessions, two of them
# still at the old 2.0s window). Every one of those scripts was therefore
# quietly reporting numbers about a configuration that is not deployed, and
# two claims in README/PROJECT_HISTORY were derived from them. One definition,
# imported everywhere, is the only thing that stops that recurring.
#
# These used to default to the first 4 sessions at a 2.0s window - the values
# from early development - which meant a bare `python train_model.py` silently
# retrained on less than half the data at the wrong window size and overwrote
# csi_model.joblib with a much weaker model. The defaults now match reality,
# so the no-argument run reproduces the deployed model instead of degrading it.
DEFAULT_SESSIONS = ["part_1_data", "part_2_data", "part_3_data", "part_4_data",
                    "part_5_data", "part_6_data", "part_7_data", "part_8_data",
                    "room2_part_1_data", "room2_part_2_data"]
DEFAULT_WINDOW_SECONDS = 0.75


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    ap.add_argument("--calib-seconds", type=float, default=CALIB_SECONDS_DEFAULT)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--model-out", default="csi_model.joblib")
    args = ap.parse_args()

    print(f"Sessions: {args.sessions}")
    print(f"Window size: {args.window_seconds}s (~{round(args.window_seconds * FRAME_HZ)} frames)")
    print(f"Calibration: first {args.calib_seconds}s of EACH empty block "
          f"(recalibrates every time an empty block starts)\n")

    X, y, groups, amp_cols = build_dataset(args.sessions, args.window_seconds, args.calib_seconds)
    print(f"\nTotal windows: {len(X)}  |  class counts: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}\n")

    logo = LeaveOneGroupOut()
    fold_accs, fold_sizes, fold_names = [], [], []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        held_out = groups[test_idx][0]
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], pred)
        fold_accs.append(acc)
        fold_sizes.append(len(test_idx))
        fold_names.append(held_out)

        print(f"--- Fold {fold + 1}: held out {held_out} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y[test_idx], pred, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y[test_idx], pred))
        print()

    # Report the WEIGHTED mean as the headline. The unweighted mean of folds
    # treats a 150-window session as equal to a 600-window one, which flatters
    # the result whenever the hardest session is also the largest - exactly the
    # case here (room2_part_1_data is the worst fold AND 18% of all windows).
    # The weighted figure is the real per-window out-of-fold accuracy: the
    # fraction of all windows that were classified correctly by a model that
    # had never seen their session. Both are printed so the gap stays visible.
    weighted = float(np.average(fold_accs, weights=fold_sizes))
    unweighted = float(np.mean(fold_accs))
    worst_i = int(np.argmin(fold_accs))

    note = "  <- optimistic: small easy sessions count as much as big hard ones" \
        if unweighted > weighted + 0.002 else ""

    print("=" * 70)
    print(f"Leave-one-session-out accuracy (per-window, weighted): {weighted:.4f}")
    print(f"  unweighted mean of folds:  {unweighted:.4f}{note}")
    print(f"  spread across sessions:    std={np.std(fold_accs):.4f}  "
          f"worst={fold_accs[worst_i]:.4f} ({fold_names[worst_i]})")
    print(f"  per-fold: {[round(a, 4) for a in fold_accs]}")
    print("=" * 70)

    # Refit on everything for the model you'll actually deploy.
    final_clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    final_clf.fit(X, y)

    joblib.dump({
        "model": final_clf,
        "amp_columns": amp_cols,
        "feature_names": feature_names(amp_cols),
        "window_seconds": args.window_seconds,
        "calib_seconds": args.calib_seconds,
        "frame_hz": FRAME_HZ,
    }, args.model_out)
    print(f"\nSaved final model (trained on all sessions) to {args.model_out}")


if __name__ == "__main__":
    main()
