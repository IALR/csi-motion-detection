"""
Train and validate a 2-zone position classifier, as an ADD-ON to the binary
motion model - not a replacement.

The motion model (csi_model.joblib) answers "is someone moving?" and stays
authoritative. This one answers "which half of the room?" and is only ever
consulted once motion is already confirmed, so if it turns out to be poor the
motion detector is unaffected.

What it does
------------
Reads sessions recorded by csi_label_collector.py with zone tags (press '1' or
'2' instead of 'm'). Keeps only the MOVING windows - a zone is meaningless
when the room is empty - and learns to tell zone 1 from zone 2 using the same
266 calibrated features the motion model uses, imported from csi_common so the
two can never diverge.

Because zone detection benefits from seeing the room from two angles, sessions
recorded with --port-b contain node_a/ and node_b/ subfolders. Three variants
are compared under identical validation:

    node A alone (266 features)
    node B alone (266 features)
    both concatenated (532 features)

Per-node models are much simpler to run live (each pump already owns its own
window and baseline); the combined model needs the two live pumps to share a
time-aligned buffer, which is real extra machinery. So the combined variant has
to earn its place by being clearly better, not merely equal.

Time alignment
--------------
Combined features need both nodes' windows to cover the same wall-clock
period. Alignment uses `host_unix_us` - the COLLECTOR's clock, stamped as each
row is written - because the two ESP32s free-run on independent clocks with an
unknown offset, so their own timestamp_us values are not comparable. The
collector drains both ports on a ~5ms loop at 10Hz, so host_unix_us is
accurate to far better than the 0.75s window.

Validation
----------
Leave-one-SESSION-out, always. Fingerprinting's classic failure is a model
that scores brilliantly within one recording (it memorises that session's
exact multipath) and collapses on the next one. A within-session split would
hide that completely.

The script refuses to report a number if zone and session are confounded -
i.e. if any session contains only one zone. In that case a model can score
~100% by learning "which session" while knowing nothing about position.

Usage:
    python train_zone_model.py --sessions zone_room1_part_1_data zone_room1_part_2_data ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

import joblib

from csi_common import (CALIB_SECONDS_DEFAULT, FRAME_HZ, calibrate_features,
                        compute_baseline, feature_names, raw_window_stats)
from train_model import (DEFAULT_WINDOW_SECONDS, amp_columns,
                         compute_block_baselines, find_empty_block_starts)

ZONE_NAMES = {1: "Zone 1", 2: "Zone 2"}

# Chance is 50% for two zones, so these are the thresholds that decide whether
# this is worth building a UI for. Set before seeing any result, deliberately.
GATE_ABANDON = 0.60
GATE_MARGINAL = 0.75
GATE_USEFUL = 0.85


def load_node(folder):
    """Read one node's CSV, keeping the columns the zone pipeline needs."""
    path = os.path.join(folder, "all_csi_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if "zone" not in df.columns:
        raise ValueError(
            f"{path} has no 'zone' column - it predates zone recording.\n"
            f"Re-record it with csi_label_collector.py using the '1'/'2' keys.")
    df = df[df["settling"] == 0].reset_index(drop=True)
    return df.sort_values("timestamp_us").reset_index(drop=True)


def session_nodes(folder):
    """Return {node_id: dataframe}. Supports both the dual-node layout
    (node_a/ + node_b/) and a plain single-node session folder."""
    node_a = os.path.join(folder, "node_a")
    if os.path.isdir(node_a):
        nodes = {"A": load_node(node_a)}
        node_b = os.path.join(folder, "node_b")
        if os.path.isdir(node_b):
            nodes["B"] = load_node(node_b)
        return nodes
    return {"A": load_node(folder)}


def windowize_zones(df, amp_cols, window_seconds, calib_seconds):
    """Yield (window_index, feature_vector, zone) for MOVING windows only.

    Windowing mirrors train_model.windowize exactly - same 0.75s size, same
    "score against the most recent empty block's baseline" rule - so the
    features here are identical in construction to the motion model's, and to
    what make_features() produces live.

    window_index is derived from the COLLECTOR's clock so two nodes recorded
    together land in the same bucket.
    """
    frames_per_window = max(1, round(window_seconds * FRAME_HZ))
    amps = df[amp_cols].to_numpy(dtype=float)
    labels = df["label"].to_numpy()
    zones = df["zone"].to_numpy()
    rssi = df["rssi"].to_numpy(dtype=float)
    host = df["host_unix_us"].to_numpy(dtype=np.int64)

    block_starts = find_empty_block_starts(labels)
    if not block_starts:
        raise ValueError("No empty-labeled frames to calibrate from")
    baselines = compute_block_baselines(amps, rssi, labels, block_starts, calib_seconds)

    t0 = host[0]
    window_us = int(window_seconds * 1_000_000)
    n = len(df)
    for start in range(0, n - frames_per_window + 1, frames_per_window):
        end = start + frames_per_window
        win_labels = labels[start:end]
        win_zones = zones[start:end]
        # Only pure MOVING windows that stayed inside a single zone.
        if not np.all(win_labels == 2):
            continue
        if not np.all(win_zones == win_zones[0]) or win_zones[0] == 0:
            continue
        governing = [s for s in block_starts if s <= start]
        if not governing:
            continue  # window precedes any empty block: no baseline to score against
        baseline = baselines[max(governing)]
        stats = raw_window_stats(amps[start:end], rssi[start:end])
        feat = calibrate_features(stats, baseline)
        widx = int((host[start] - t0) // window_us)
        yield widx, feat, int(win_zones[0])


def build_zone_dataset(session_folders, window_seconds, calib_seconds):
    """Returns {variant: (X, y, groups)} for 'A', 'B' and 'AB'."""
    per_variant = {v: {"X": [], "y": [], "g": []} for v in ("A", "B", "AB")}
    amp_cols_ref = None

    for folder in session_folders:
        nodes = session_nodes(folder)
        windows = {}
        for node_id, df in nodes.items():
            cols = amp_columns(df)
            if amp_cols_ref is None:
                amp_cols_ref = cols
            elif cols != amp_cols_ref:
                raise ValueError(f"{folder} node {node_id}: subcarrier columns differ "
                                 f"from earlier sessions")
            windows[node_id] = {
                widx: (feat, zone)
                for widx, feat, zone in windowize_zones(df, cols, window_seconds,
                                                         calib_seconds)
            }

        for node_id, w in windows.items():
            if node_id not in per_variant:
                continue
            for widx, (feat, zone) in sorted(w.items()):
                per_variant[node_id]["X"].append(feat)
                per_variant[node_id]["y"].append(zone)
                per_variant[node_id]["g"].append(folder)

        # Combined: only window indices present in BOTH nodes, so each row is
        # genuinely the same moment seen from two places.
        if "A" in windows and "B" in windows:
            shared = sorted(set(windows["A"]) & set(windows["B"]))
            for widx in shared:
                fa, za = windows["A"][widx]
                fb, zb = windows["B"][widx]
                if za != zb:
                    continue  # nodes disagree on the label: skip, don't guess
                per_variant["AB"]["X"].append(np.concatenate([fa, fb]))
                per_variant["AB"]["y"].append(za)
                per_variant["AB"]["g"].append(folder)

        counts = {k: sum(1 for g in per_variant[k]["g"] if g == folder)
                  for k in ("A", "B", "AB")}
        zone_mix = {}
        for node_id, w in windows.items():
            for _, (_, z) in w.items():
                zone_mix[z] = zone_mix.get(z, 0) + 1
        print(f"{folder}: windows A={counts['A']} B={counts['B']} combined={counts['AB']}"
              f"  zone rows {zone_mix}")

    out = {}
    for v, d in per_variant.items():
        if d["X"]:
            out[v] = (np.array(d["X"]), np.array(d["y"]), np.array(d["g"]))
    return out, amp_cols_ref


def check_not_confounded(y, groups):
    """Refuse to report a number if any session holds only one zone.

    With one zone per session, 'which zone' and 'which session' are the same
    question: a model can score ~100% under leave-one-session-out while having
    learned nothing about position. This is the single easiest way to fool
    yourself here, so it is a hard error rather than a warning."""
    problems = []
    for g in sorted(set(groups)):
        zs = sorted(set(y[groups == g].tolist()))
        if len(zs) < 2:
            problems.append(f"  {g}: only zone(s) {zs}")
    if problems:
        raise SystemExit(
            "\nCONFOUNDED DATA - refusing to report an accuracy.\n"
            "Every session must contain BOTH zones, otherwise 'which zone' and\n"
            "'which session' are the same question and the score is meaningless.\n"
            + "\n".join(problems) +
            "\n\nRe-record alternating the zones within each session:\n"
            "  LEAVE -> EMPTY -> zone 1 -> LEAVE -> EMPTY -> zone 2 -> ...\n")


def evaluate(X, y, groups, n_estimators=200):
    logo = LeaveOneGroupOut()
    accs, sizes, names, oof = [], [], [], []
    for tr, te in logo.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                      n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        sizes.append(len(te))
        names.append(groups[te][0])
        oof.append((y[te], pred))
    weighted = float(np.average(accs, weights=sizes))
    return {
        "weighted": weighted,
        "unweighted": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "per_fold": list(zip(names, accs)),
        "oof": oof,
    }


def verdict(acc):
    if acc < GATE_ABANDON:
        return ("ABANDON", "barely above the 50% coin-flip - do not build a UI for this")
    if acc < GATE_MARGINAL:
        return ("MARGINAL", "works in principle but too unreliable to act on; try "
                            "moving the nodes further apart before investing more")
    if acc < GATE_USEFUL:
        return ("USEFUL", "good enough to show live")
    return ("STRONG", "re-confirm the sessions really were on different days and "
                      "every session contained both zones")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True,
                    help="zone session folders (need >=3 for leave-one-session-out)")
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    ap.add_argument("--calib-seconds", type=float, default=CALIB_SECONDS_DEFAULT)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--model-out", default="csi_zone_model.joblib")
    args = ap.parse_args()

    if len(args.sessions) < 3:
        print(f"WARNING: only {len(args.sessions)} session(s). Leave-one-session-out "
              f"needs at least 3 to mean anything; 2 gives a single noisy fold.\n")

    print(f"Window: {args.window_seconds}s  |  sessions: {len(args.sessions)}\n")
    datasets, amp_cols = build_zone_dataset(args.sessions, args.window_seconds,
                                            args.calib_seconds)
    if not datasets:
        raise SystemExit("No zone-tagged MOVING windows found. Did you record with "
                         "the '1'/'2' keys?")

    results = {}
    print("\n" + "=" * 78)
    print("LEAVE-ONE-SESSION-OUT  (chance = 50%)")
    print("=" * 78)
    for variant in ("A", "B", "AB"):
        if variant not in datasets:
            continue
        X, y, groups = datasets[variant]
        if len(set(groups)) < 2:
            print(f"{variant}: only one session, cannot hold one out - skipped")
            continue
        check_not_confounded(y, groups)
        r = evaluate(X, y, groups, args.n_estimators)
        results[variant] = (r, X, y, groups)
        label = {"A": "node A alone", "B": "node B alone",
                 "AB": "both nodes combined"}[variant]
        print(f"\n{label} ({X.shape[1]} features, {len(X)} windows)")
        print(f"  weighted {r['weighted']:.4f}   unweighted {r['unweighted']:.4f}   "
              f"std {r['std']:.4f}")
        for name, a in r["per_fold"]:
            print(f"    {name:32} {a:.4f}")

    if not results:
        raise SystemExit("\nNothing could be evaluated.")

    best = max(results, key=lambda v: results[v][0]["weighted"])
    r, X, y, groups = results[best]
    acc = r["weighted"]
    tag, advice = verdict(acc)

    print("\n" + "=" * 78)
    print(f"BEST VARIANT: {best}   weighted accuracy {acc:.4f}")
    print("=" * 78)
    y_true = np.concatenate([a for a, _ in r["oof"]])
    y_pred = np.concatenate([b for _, b in r["oof"]])
    labels = sorted(set(y_true.tolist()))
    print("\nOut-of-fold confusion matrix (rows=true, cols=pred), "
          f"labels={[ZONE_NAMES.get(l, l) for l in labels]}:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print()
    print(classification_report(y_true, y_pred, zero_division=0,
                                 target_names=[ZONE_NAMES.get(l, str(l)) for l in labels]))

    spread = max(a for _, a in r["per_fold"]) - min(a for _, a in r["per_fold"])
    print(f"GATE: {tag} - {advice}")
    if spread > 0.20:
        print(f"WARNING: per-fold spread is {spread:.2f}. A model that scores well on "
              f"some sessions and near chance on others has memorised sessions, not "
              f"zones - treat the mean as unreliable regardless of its value.")

    if tag == "ABANDON":
        print("\nNot saving a model. See docs/PROJECT_HISTORY.md for why this was "
              "expected to be uncertain.")
        return 1

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42,
                                 n_jobs=-1)
    clf.fit(X, y)
    names = feature_names(amp_cols)
    joblib.dump({
        "model": clf,
        "variant": best,                 # 'A'/'B' = per-node, 'AB' = needs both
        "feature_names": names if best != "AB" else names + names,
        "amp_columns": amp_cols,
        "window_seconds": args.window_seconds,
        "calib_seconds": args.calib_seconds,
        "frame_hz": FRAME_HZ,
        # Carried in the bundle so the live server reads zone names from the
        # model itself and cannot drift from whatever it was trained on.
        "zone_names": {int(k): v for k, v in ZONE_NAMES.items()},
        "holdout_accuracy": acc,
    }, args.model_out)
    print(f"\nSaved {args.model_out} (variant {best}).")
    if best == "AB":
        print("NOTE: this variant needs BOTH nodes time-aligned live, which the "
              "server does not do yet. If per-node accuracy is close, prefer it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
