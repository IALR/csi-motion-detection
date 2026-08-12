# CSI Motion Detection — quick orientation

Wi-Fi CSI-based device-free motion detection on an ESP32-S3 (empty vs.
moving). **Before making any non-trivial change, read
`docs/PROJECT_HISTORY.md`** — it's the full development history: every
design decision, every bug found and fixed, and why, in chronological order.
This file is just a fast-start summary.

## Current state
- Model: Random Forest, 10 sessions (8 room-1 `part_1_data`..`part_8_data`
  + 2 room-2 `room2_part_1_data`/`room2_part_2_data`), 0.75s windows,
  **94.25% leave-one-session-out accuracy** (per-window/weighted; the
  unweighted mean of folds is 95.07% and was the figure quoted before §23).
  Deployed as `csi_model.joblib`. A bare `python train_model.py` reproduces
  exactly that — its defaults are the deployed config (they used to be 4
  sessions at 2.0s, which silently degraded the model; fixed, §22).
- **Accuracy is environment-dependent, and this is the main open weakness**:
  the empty-room RF noise floor varies 7× across sessions; quiet rooms score
  95-99%, noisy ones 85-95% (§23). `assess_noise_floor()` in `csi_common.py`
  grades it and the live system reports it. Don't treat a single accuracy
  number as the system's behaviour — ask which regime it was measured in.
- **Tests: `python -m pytest tests/`** (25 tests, all passing). They pin the
  live-system failure modes — malformed serial lines, a node going offline,
  the AP changing channel width mid-run — because every bug this project
  has hit twice was a failure-mode bug, not a math bug. Run them after
  touching `csi_common.py` or `csi_live_server.py`.
- Live system now supports **1 or 2 ESP32 nodes** (`--port-b` on
  `csi_live_server.py`), OR-combined, dashboard-visible per node. Matplotlib
  `csi_live_predict.py` remains single-node. Includes per-block + rolling
  recalibration, a manual "force recalibration" escape hatch, and graceful
  per-node degradation on subcarrier mismatch (doesn't crash the other
  node). See PROJECT_HISTORY.md §15 and §21 before touching
  `csi_common.py`'s calibration code or `csi_live_server.py`.
- Public repo: **https://github.com/IALR/csi-motion-detection** — commit +
  push after any change worth keeping, nothing happens automatically.
- Top open priority: diverse "moving" data — all training data is one
  person's movement style; a second person has never been tested (see
  PROJECT_HISTORY.md §18, item 1). Cross-room and the second node are both
  done now (§21), though cross-room is only validated across 2 rooms so
  far.

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