"""
Export a trained Random Forest to a C header the ESP32 can run directly.

The board runs the SAME model the PC validated - not a retrained or
approximated one - so any accuracy number measured offline still describes
what the hardware does. With 16MB of flash the full 200-tree forest costs
about 430KB (~2.7%), so there is no reason to shrink it.

Layout
------
Four parallel arrays instead of an array of structs: a struct of
{int16, float, int16, int16} is 10 bytes of data that the compiler pads to 12
for float alignment, wasting ~85KB across 44k nodes. Parallel arrays keep it
at exactly 10 bytes/node.

    feature[i]    >= 0 : split on this feature index
                    -1 : LEAF
    threshold[i]  split point, or for a leaf the probability of the SECOND
                  class (see below)
    left[i]/right[i]  child indices, RELATIVE to the tree's own first node

Matching sklearn exactly
------------------------
RandomForestClassifier.predict_proba averages each tree's *probability*
distribution (normalised leaf class counts), then takes argmax - it does NOT
take a majority vote of hard per-tree labels. Those differ whenever trees are
impure, so leaves store a probability, not a class.

With two classes only p1 needs storing, and p0 = 1 - p1. sklearn's argmax
returns the FIRST maximum, so a dead-on 0.5 tie resolves to class 0; the C
side therefore uses `p1 > 0.5`, not `>=`, to reproduce that exactly.

Verification is not optional here: --verify replays every window of the real
recorded sessions through a Python simulation of the exact C algorithm and
compares against sklearn. It must be a perfect match before the header is
trusted.

Usage:
    python export_model_c.py                       # writes firmware/main/csi_model_data.h
    python export_model_c.py --verify              # ...and checks it against sklearn
"""

import argparse
import os
import sys

import joblib
import numpy as np

LEAF = -1


def c_float(v):
    """Format a float as a valid C float literal.

    "%.9g" renders whole numbers without a decimal point (20.0 -> "20"), and
    "20f" is an integer constant with a float suffix, which is a compile error.
    CSI amplitudes are integers and leaf probabilities are frequently exactly
    0 or 1, so this is hit constantly rather than being a corner case.
    """
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite value {v} cannot be exported")
    s = f"{v:.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def collect_trees(clf):
    """Flatten every tree into the parallel-array form the C code reads."""
    feature, threshold, left, right, offsets = [], [], [], [], [0]
    classes = clf.classes_
    if len(classes) != 2:
        raise SystemExit(f"This exporter assumes 2 classes, got {list(classes)}. "
                         f"The leaf encoding stores a single probability.")

    for est in clf.estimators_:
        t = est.tree_
        base = len(feature)
        for i in range(t.node_count):
            if t.children_left[i] == -1:          # leaf
                counts = t.value[i][0]
                total = counts.sum()
                p1 = float(counts[1] / total) if total > 0 else 0.0
                feature.append(LEAF)
                threshold.append(p1)
                left.append(0)
                right.append(0)
            else:
                feature.append(int(t.feature[i]))
                threshold.append(float(t.threshold[i]))
                # store child indices relative to this tree's first node
                left.append(int(t.children_left[i]))
                right.append(int(t.children_right[i]))
        offsets.append(len(feature))
        del base

    return (np.array(feature, dtype=np.int32),
            np.array(threshold, dtype=np.float32),
            np.array(left, dtype=np.int32),
            np.array(right, dtype=np.int32),
            np.array(offsets, dtype=np.int64),
            classes)


def predict_like_c(X, feature, threshold, left, right, offsets):
    """Python simulation of exactly what the C will do.

    Deliberately written the slow, literal way - one node hop at a time in
    float32 - so that it exercises the same arithmetic and the same tie
    handling as the generated C, rather than quietly falling back on numpy
    doing something cleverer in float64.
    """
    n_trees = len(offsets) - 1
    X32 = X.astype(np.float32)
    out = np.empty(len(X32), dtype=np.int32)
    for r in range(len(X32)):
        row = X32[r]
        acc = np.float32(0.0)
        for t in range(n_trees):
            base = offsets[t]
            node = 0
            while feature[base + node] != LEAF:
                f = feature[base + node]
                if row[f] <= threshold[base + node]:
                    node = left[base + node]
                else:
                    node = right[base + node]
            acc = np.float32(acc + threshold[base + node])
        p1 = np.float32(acc / np.float32(n_trees))
        out[r] = 1 if p1 > np.float32(0.5) else 0
    return out


def emit_header(path, bundle, feature, threshold, left, right, offsets, classes):
    n_nodes = len(feature)
    n_trees = len(offsets) - 1
    amp_cols = bundle["amp_columns"]
    n_feat = len(bundle["feature_names"])

    def rows(arr, per_line, fmt):
        out = []
        for i in range(0, len(arr), per_line):
            out.append("    " + " ".join(fmt(v) for v in arr[i:i + per_line]))
        return "\n".join(out)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"""// GENERATED by export_model_c.py - do not edit by hand.
//
// Random Forest exported from {os.path.basename(bundle.get("_src", "csi_model.joblib"))}.
// Trees: {n_trees}   nodes: {n_nodes}   features: {n_feat}   subcarriers: {len(amp_cols)}
// Approx flash: {n_nodes * 10 / 1024:.0f} KB
//
// feature[i] == CSI_LEAF marks a leaf; its threshold[i] then holds the
// probability of class {int(classes[1])}. Child indices are relative to the
// tree's first node (see csi_tree_offset).
#pragma once
#include <stdint.h>

#define CSI_LEAF        (-1)
#define CSI_N_TREES     {n_trees}
#define CSI_N_NODES     {n_nodes}
#define CSI_N_FEATURES  {n_feat}
#define CSI_N_SUBCARRIERS {len(amp_cols)}
#define CSI_WINDOW_SECONDS {bundle["window_seconds"]}f
#define CSI_CALIB_SECONDS  {bundle["calib_seconds"]}f
#define CSI_FRAME_HZ       {bundle["frame_hz"]}f

// Frame counts are computed HERE, in Python, with the same
// `max(1, round(seconds * FRAME_HZ))` the training pipeline uses - including
// Python's round-half-to-even, which turns 0.75*10 = 7.5 into 8, not 7.
// Deriving them in C from the float seconds above would both risk a different
// rounding rule and stop them being integer constant expressions, which C
// requires for array sizes at file scope.
#define CSI_WINDOW_FRAMES  {max(1, round(bundle["window_seconds"] * bundle["frame_hz"]))}
#define CSI_CALIB_FRAMES   {max(1, round(bundle["calib_seconds"] * bundle["frame_hz"]))}

// Class labels, in the order the probabilities refer to.
static const int8_t csi_classes[2] = {{ {int(classes[0])}, {int(classes[1])} }};

static const int16_t csi_feature[CSI_N_NODES] = {{
{rows(feature, 20, lambda v: f"{int(v)},")}
}};

static const float csi_threshold[CSI_N_NODES] = {{
{rows(threshold, 8, lambda v: c_float(v) + ",")}
}};

static const int16_t csi_left[CSI_N_NODES] = {{
{rows(left, 20, lambda v: f"{int(v)},")}
}};

static const int16_t csi_right[CSI_N_NODES] = {{
{rows(right, 20, lambda v: f"{int(v)},")}
}};

static const uint32_t csi_tree_offset[CSI_N_TREES + 1] = {{
{rows(offsets, 12, lambda v: f"{int(v)},")}
}};
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="csi_model.joblib")
    ap.add_argument("--out", default=os.path.join("firmware", "main", "csi_model_data.h"))
    ap.add_argument("--verify", action="store_true",
                    help="replay the real recorded windows through a simulation of "
                         "the generated C and require an exact match with sklearn")
    ap.add_argument("--verify-limit", type=int, default=0,
                    help="check only the first N windows (0 = all; the full run is slow)")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    bundle["_src"] = args.model
    clf = bundle["model"]
    feature, threshold, left, right, offsets, classes = collect_trees(clf)

    print(f"{args.model}: {len(offsets)-1} trees, {len(feature)} nodes, "
          f"{len(bundle['feature_names'])} features")
    print(f"packed size: {len(feature)*10/1024:.0f} KB "
          f"({len(feature)*10/(16*1024*1024)*100:.2f}% of 16MB flash)")

    if args.verify:
        from train_model import (DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS,
                                 build_dataset)
        print("\nRebuilding the real windows to verify against...")
        X, y, groups, _ = build_dataset(DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS)
        if args.verify_limit:
            X, y = X[:args.verify_limit], y[:args.verify_limit]
        print(f"replaying {len(X)} windows through the C algorithm simulation "
              f"(this is deliberately slow)...")
        sk = clf.predict(X)
        mine = classes[predict_like_c(X, feature, threshold, left, right, offsets)]
        bad = int((sk != mine).sum())
        print(f"\n  sklearn vs generated-C simulation: {len(X)-bad}/{len(X)} identical")
        if bad:
            idx = np.where(sk != mine)[0][:5]
            print(f"  MISMATCHES: {bad}")
            for i in idx:
                print(f"    window {i}: sklearn={sk[i]} c={mine[i]}")
            raise SystemExit("\nRefusing to write the header - the C algorithm does "
                             "not reproduce sklearn.")
        print("  EXACT MATCH - the exported model is faithful.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    emit_header(args.out, bundle, feature, threshold, left, right, offsets, classes)
    size = os.path.getsize(args.out)
    print(f"\nWrote {args.out} ({size/1024/1024:.1f} MB of source, "
          f"{len(feature)*10/1024:.0f} KB once compiled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
