# CSI Motion Detection — quick orientation

Wi-Fi CSI-based device-free motion detection on an ESP32-S3 (empty vs.
moving). **Before making any non-trivial change, read
`docs/PROJECT_HISTORY.md`** — it's the full development history: every
design decision, every bug found and fixed, and why, in chronological order.
This file is just a fast-start summary.

## Current state
- Model: Random Forest, 8 sessions (`part_1_data`..`part_8_data`), 0.75s
  windows, **95.54% leave-one-session-out accuracy**. Deployed as
  `csi_model.joblib`.
- Live system (Matplotlib `csi_live_predict.py` + web `csi_live_server.py`/
  `csi_dashboard.html`) works, including per-block + rolling recalibration
  and a manual "force recalibration" escape hatch. Two serious live-only
  bugs were found and fixed here — see PROJECT_HISTORY.md §15 before
  touching `csi_common.py`'s calibration code.
- Public repo: **https://github.com/IALR/csi-motion-detection** — commit +
  push after any change worth keeping, nothing happens automatically.
- Top open priority: a true cross-room held-out test has never been run
  (see PROJECT_HISTORY.md §18, item 1) — the single biggest gap.

## Operational notes
- Only one program can hold the ESP32's COM port at a time (collector,
  IDF monitor, and both live-inference tools all need exclusive access).
- `firmware/main/wifi_secrets.h` holds real Wi-Fi credentials and is
  gitignored — never remove it from `.gitignore`, never commit it. The
  committed template is `wifi_secrets.h.example`.
- `csi_common.py` is the single source of truth for parsing/features/
  calibration — training and both live tools import from it. Never
  duplicate that logic; edit it there so offline and online can't diverge.
- Every script recomputes real numbers when run — don't hand-write
  accuracy/result numbers into docs without actually running the script.

## The user
Not an ML/RF expert by background. Wants concrete, verified explanations
grounded in this project's actual data, not textbook generalities — and
wants to be told the truth about what's uncertain, not reassured. Confirm
before big changes (new features, discarding data, architecture changes)
rather than doing them silently.