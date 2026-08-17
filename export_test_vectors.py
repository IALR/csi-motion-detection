"""
Embed real recorded CSI windows, plus what the PC pipeline produced for them,
into a C header so the ESP32 can verify itself at boot.

There is no host C compiler on this machine, so the C feature extraction and
inference cannot be tested on the PC. Testing on the device is arguably better
anyway: it exercises the real single-precision FPU and the real xtensa
compiler, which is exactly where a port like this goes wrong.

Each vector carries RAW inputs only - the amplitude frames of a calibration
block and of a scoring window - and the EXPECTED outputs computed by
csi_common.py. The C side must recompute everything from the raw frames:
baseline, 266 features, forest prediction. Nothing is pre-chewed, so a bug
anywhere in the chain shows up.

Tolerances: features are compared with a relative tolerance because float32
arithmetic cannot be expected to match float64 bit-for-bit. The PREDICTION,
however, must match exactly - a differing label means the port is not
trustworthy no matter how close the numbers look. (Measured on the PC:
float32 vs float64 gave 0 disagreements across all 3302 windows, so an exact
label match is a fair bar.)

Usage:
    python export_test_vectors.py                 # 12 vectors, mixed classes
    python export_test_vectors.py --count 24
"""

import argparse
import os
import sys

import joblib
import numpy as np

from csi_common import (FRAME_HZ, calibrate_features, compute_baseline,
                        raw_window_stats)
from export_model_c import c_float  # one definition of a valid C float literal
from train_model import (DEFAULT_SESSIONS, DEFAULT_WINDOW_SECONDS, amp_columns,
                         find_empty_block_starts, load_session)


def build_vectors(session, window_seconds, calib_seconds, want):
    """Pull (calib frames, window frames, expected features, expected label)
    tuples from one real session, balanced across the two classes."""
    df = load_session(session)
    cols = amp_columns(df)
    amps = df[cols].to_numpy(dtype=float)
    labels = df["label"].to_numpy()
    rssi = df["rssi"].to_numpy(dtype=float)

    fpw = max(1, round(window_seconds * FRAME_HZ))
    cframes = max(1, round(calib_seconds * FRAME_HZ))

    starts = find_empty_block_starts(labels)
    if not starts:
        return []
    # Use the first empty block as the calibration block, exactly as the live
    # system does at startup.
    cstart = starts[0]
    cend = cstart
    while cend < len(labels) and labels[cend] == 0 and (cend - cstart) < cframes:
        cend += 1
    if cend - cstart < cframes:
        return []
    calib_amps = amps[cstart:cend]
    calib_rssi = rssi[cstart:cend]
    baseline = compute_baseline(calib_amps, calib_rssi)

    out = []
    per_class = {0: 0, 2: 0}
    limit = max(1, want // 2)
    for start in range(cend, len(df) - fpw + 1, fpw):
        end = start + fpw
        win = labels[start:end]
        if not np.all(win == win[0]):
            continue
        lbl = int(win[0])
        if per_class.get(lbl, 0) >= limit:
            continue
        stats = raw_window_stats(amps[start:end], rssi[start:end])
        feat = calibrate_features(stats, baseline)
        out.append({
            "calib_amps": calib_amps,
            "calib_rssi": calib_rssi,
            "win_amps": amps[start:end],
            "win_rssi": rssi[start:end],
            "features": feat,
            "true_label": lbl,
        })
        per_class[lbl] = per_class.get(lbl, 0) + 1
        if sum(per_class.values()) >= want:
            break
    return out


def fmt_floats(arr, per_line=8, indent="        "):
    flat = np.asarray(arr).ravel()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = " ".join(c_float(v) + "," for v in flat[i:i + per_line])
        lines.append(indent + chunk)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="csi_model.joblib")
    ap.add_argument("--session", default=None,
                    help="session to draw vectors from (default: first of DEFAULT_SESSIONS)")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", default=os.path.join("firmware", "main", "csi_testvectors.h"))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    session = args.session or DEFAULT_SESSIONS[0]
    ws = bundle["window_seconds"]
    cs = bundle["calib_seconds"]

    vecs = build_vectors(session, ws, cs, args.count)
    if not vecs:
        raise SystemExit(f"Could not build vectors from {session}")

    feats = np.array([v["features"] for v in vecs])
    preds = clf.predict(feats)
    for v, p in zip(vecs, preds):
        v["expected_pred"] = int(p)

    n_sub = feats.shape[1] // 2  # 2*NS + 10 features
    n_sub = len(bundle["amp_columns"])
    fpw = vecs[0]["win_amps"].shape[0]
    cfr = vecs[0]["calib_amps"].shape[0]

    print(f"session      : {session}")
    print(f"vectors      : {len(vecs)}  (calib {cfr} frames, window {fpw} frames)")
    print(f"expected pred: {dict(zip(*np.unique(preds, return_counts=True)))}")
    print(f"true labels  : {dict(zip(*np.unique([v['true_label'] for v in vecs], return_counts=True)))}")

    # One shared calibration block across vectors from the same session keeps
    # the header small - it is identical for every vector here.
    calib_amps = vecs[0]["calib_amps"]
    calib_rssi = vecs[0]["calib_rssi"]
    same = all(np.array_equal(v["calib_amps"], calib_amps) for v in vecs)
    if not same:
        raise SystemExit("vectors span different calibration blocks; not supported")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"""// GENERATED by export_test_vectors.py - do not edit by hand.
//
// Real recorded CSI from {os.path.basename(session)}, with the features and
// predictions the PC pipeline (csi_common.py + scikit-learn) produced for it.
// The device recomputes everything from the RAW frames below - baseline,
// features, forest - so a bug anywhere in the port is caught.
#pragma once
#include <stdint.h>

#define CSI_TV_COUNT        {len(vecs)}
#define CSI_TV_SUBCARRIERS  {n_sub}
#define CSI_TV_CALIB_FRAMES {cfr}
#define CSI_TV_WINDOW_FRAMES {fpw}
#define CSI_TV_N_FEATURES   {feats.shape[1]}

// Shared calibration block: the first {cfr} confirmed-empty frames.
static const float csi_tv_calib_amps[CSI_TV_CALIB_FRAMES * CSI_TV_SUBCARRIERS] = {{
{fmt_floats(calib_amps)}
}};

static const float csi_tv_calib_rssi[CSI_TV_CALIB_FRAMES] = {{
{fmt_floats(calib_rssi)}
}};

// Scoring windows.
static const float csi_tv_win_amps[CSI_TV_COUNT][CSI_TV_WINDOW_FRAMES * CSI_TV_SUBCARRIERS] = {{
""")
        for v in vecs:
            f.write("    {\n" + fmt_floats(v["win_amps"]) + "\n    },\n")
        f.write("};\n\nstatic const float csi_tv_win_rssi[CSI_TV_COUNT][CSI_TV_WINDOW_FRAMES] = {\n")
        for v in vecs:
            f.write("    {\n" + fmt_floats(v["win_rssi"]) + "\n    },\n")
        f.write("};\n\n// What csi_common.py computed for each window.\n")
        f.write("static const float csi_tv_expect_features[CSI_TV_COUNT][CSI_TV_N_FEATURES] = {\n")
        for v in vecs:
            f.write("    {\n" + fmt_floats(v["features"]) + "\n    },\n")
        f.write("};\n\n// What scikit-learn predicted. This must match EXACTLY.\n")
        f.write("static const int8_t csi_tv_expect_pred[CSI_TV_COUNT] = { "
                + " ".join(f"{v['expected_pred']}," for v in vecs) + " };\n")
        f.write("\n// The recorded ground truth, for context in the log.\n")
        f.write("static const int8_t csi_tv_true_label[CSI_TV_COUNT] = { "
                + " ".join(f"{v['true_label']}," for v in vecs) + " };\n")

    print(f"\nWrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
