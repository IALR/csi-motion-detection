# CSI Motion Detection — Full Project History & Context

This file exists so a **new chat session with no memory of prior conversations**
can pick this project up cold. It documents not just *what* was built, but *why*
each step happened — what was asked, what problem it solved, and what was
learned. Read top to bottom in order; later sections assume earlier ones.

If you are a fresh Claude session reading this: the user (goes by "Ilyas" based
on the Windows account) has been building this iteratively over several long
conversations. They are not an ML/RF expert by background — explanations have
been kept concrete and grounded in this project's actual data, not textbook
theory. Match that style. They value being told the truth about what's
uncertain or unproven, not reassurance.

---

## 0. What this project is

A device-free motion detection system: a single ESP32-S3 microcontroller
extracts Wi-Fi Channel State Information (CSI) from its link to a home
router/AP, and a trained classifier tells "empty room" apart from "someone is
moving" — no camera, no wearable. The physical idea: a person moving disturbs
the multipath propagation between the ESP32 and the router in a way an empty
room doesn't, and CSI (which every Wi-Fi receiver already computes internally
to decode data) captures that disturbance as a side effect.

**Stated goals, in the user's own words, roughly in this order of priority:**
1. Reliably detect movement vs. empty in the room the ESP32 currently sits in.
2. Eventually generalize to *any* room, not just this one (stated but not yet
   seriously attempted — see §16).

---

## 1. Starting state (before this conversation began)

The user had already, in a *different* prior session (using a tool called
Codex, not Claude), built:
- **Firmware** (`main/csi_node.c`, ESP-IDF, ESP32-S3): connects to Wi-Fi,
  pings the router at 100ms intervals (~10Hz) to trigger a steady stream of
  CSI updates, and streams parsed CSI frames over serial as ASCII lines:
  `CSI_AMP,<timestamp_us>,<label>,<num_subcarriers>,<rssi>,<amp_0>,...,<amp_N-1>`.
  Firmware history (already fixed before this conversation): CSI was
  originally formatted/printed *inside* the Wi-Fi callback, which stalled the
  radio and dropped the frame rate from 10Hz to 5Hz; fixed by moving that work
  to a separate FreeRTOS task via a queue, with the callback doing only a
  fast memcpy. Also filters CSI to the connected AP's BSSID only, and emits
  RSSI per frame.
- **`csi_label_collector.py`** (lives at
  `C:\Users\ilyas\Documents\Codex\2026-07-21\wh\outputs\`, NOT in this
  folder): an auto-cycling data collector. Protocol: 5s "leave the room"
  countdown (frames tagged `settling=1`, excluded from training) → records
  EMPTY (label 0) automatically for a block → waits for the operator to press
  `m` → records MOVING (label 2) for a block → repeats. This solves the
  labeling problem that a person must be *absent* to record empty and
  *present* to record moving.
- **`csi_live_monitor.py`** (same folder as above): a live waterfall +
  motion-energy verification tool, used to visually confirm the channel
  reacts to a person before trusting a recording.
- Already-collected datasets, `part_1_data`..`part_4_data`, in this folder.
- A **prior baseline result** (from before this conversation, using the
  above tools): Random Forest, 2-second windows, leave-one-session-out
  (LOSO) accuracy 96.6%, on the original 3 sessions.

The Python training/model scripts themselves did **not** exist on disk
anywhere — only the firmware, collector, monitor, and raw CSVs were found.
This conversation's first job was rebuilding the training pipeline from
scratch to reproduce and then extend that baseline.

---

## 2. Building the training pipeline (`train_model.py`)

**Why:** no training script existed; needed one to turn CSVs into a validated
model.

Design decisions made and why:
- **Windowing**: CSI frames are grouped into fixed, non-overlapping time
  windows (originally 2.0s = ~20 frames at 10Hz; later tuned down, see §10).
  A window is dropped if it spans a label transition (ambiguous).
- **Features (original/absolute version)**: per-subcarrier amplitude mean +
  std across the window (256 features for 128 subcarriers), frame-to-frame
  "motion energy" (mean/std of |amp[t] - amp[t-1]|, averaged across
  subcarriers) — 2 features, and RSSI mean/std — 2 features. 260 total.
- **Validation methodology: leave-one-session-out (LOSO)**, via
  `sklearn.model_selection.LeaveOneGroupOut`, treating each *recording
  session* as a group. Chosen over a random train/test split because windows
  within one session are highly correlated (same room, same day, same
  hardware position) — a random split would leak session-specific
  information into the "test" set and inflate accuracy dishonestly.
- **Model**: `RandomForestClassifier` (scikit-learn), n_estimators=200.

**Result**: reproduced the prior 96.6% baseline almost exactly — 96.96% LOSO
on the original 3 sessions — confirming the rebuilt pipeline was correct.

---

## 3. First visualization (an Artifact, not a file in this repo)

**Why asked:** the user wanted to *see* the training results, not just read
numbers, and to understand the model rather than trust it blindly.

Built an HTML/JS Artifact (a Claude "Artifact", not saved in this folder —
lives on claude.ai) showing: LOSO accuracy per session, confusion matrices,
feature importance (top 15), and a motion-energy separability chart.
**Finding surfaced**: `rssi_std` was the single most important feature,
followed by per-subcarrier std on a specific cluster of subcarrier indices
(~40-48, ~92-108). This turned out to be important later (see §13) — a
feature that dependent on absolute RSSI variability is fragile to whatever
else changes RSSI, motion or not.

---

## 4. The "session 4" saga — proving true generalization

**Why:** LOSO across 3 sessions collected in quick succession isn't a strong
enough test of "does this generalize to a session it's never seen under
different conditions." A 4th session was recorded specifically to be used
*only* as a true external test: train on sessions 1-3, score once on session
4, never fold it into training until validated.

This produced `evaluate_holdout.py` (train on named sessions, test on a
different named session — still used constantly, see §9).

**Three attempts at session 4, each diagnosed in detail:**

- **Attempt 1**: held-out accuracy **65.55%**. Diagnosis: motion energy
  during the "empty" blocks was elevated (mean 2.98) compared to sessions
  1-3's empty baseline (0.8-1.3), consistently across the *entire* recording,
  not just a leave-transient. Root cause (confirmed by the user): a real
  environmental disturbance was present during the empty blocks that wasn't
  present in earlier sessions. **This was a data-collection failure, not a
  model failure** — the "empty" label wasn't actually clean.
- **Attempt 2**: held-out accuracy **51.26%** (near chance) — worse. Empty
  and moving motion energy were now nearly identical *within the same
  session* (5.19 vs 5.06), and RSSI had dropped 5-7dB versus every prior
  session. Diagnosis: the ESP32 or router had likely been physically moved
  or newly obstructed between recordings — a real hardware/setup change, not
  just an ambient disturbance.
- **Attempt 3**: held-out accuracy **98.33%**. Motion energy and RSSI back in
  line with sessions 1-3. Success — folded into training.

**Lesson learned and repeated many times since**: when a new session scores
badly, the fix is almost never "the model is wrong" — it's diagnosing *why*
via a motion-energy / RSSI comparison across sessions, which has reliably
found a real, explainable physical cause every single time it's been done in
this project (see §9 for four more instances of this exact pattern).

**Final model on 4 sessions (absolute features)**: 97.72% LOSO.

---

## 5. Live inference — two front ends

**Why:** a validated offline model isn't useful without a way to run it
against the live serial stream in real time.

### 5a. `csi_live_predict.py` — Matplotlib desktop tool
Extends the pre-existing `csi_live_monitor.py` waterfall plot with real-time
prediction: buffers incoming frames into a rolling window, recomputes
features every frame using the same math as training, and shows the
prediction as a colored status line (green EMPTY / red MOVING) above the
waterfall.

### 5b. `csi_live_server.py` + `csi_dashboard.html` — Web dashboard
**Why asked:** the user wanted "better visualization" than Matplotlib.
Architecture decision made and explained: a browser cannot read a Windows COM
port directly (short of the Chrome-only Web Serial API), so a Python backend
remains necessary. Considered and rejected re-implementing the trained
Random Forest in JavaScript to go fully browser-native — judged too risky
(two implementations of the same logic can silently diverge).

- `csi_live_server.py`: opens the serial port in a background thread, parses
  frames, runs the model, and broadcasts JSON over a local WebSocket
  (`ws://localhost:8765`) to any connected browser tab.
- `csi_dashboard.html`: a self-contained static page — canvas waterfall
  heatmap, SVG motion-energy sparkline, big color-coded status badge,
  auto-reconnect, light/dark theme. No server framework needed to host it.

**`csi_common.py` was created at this point** specifically so the offline
training path and both live-inference paths share *exactly* the same
parsing and feature-extraction code — never two implementations of the same
math that could quietly diverge.

---

## 6. Debugging the live dashboard (first round)

**Symptom reported**: dashboard connected (green "Connected") but showed no
data, forever.

**Two real bugs found and fixed:**
1. **Lost status messages**: if the serial port failed to open (or any
   error/warning occurred) *before* a browser tab connected, the message was
   broadcast to zero clients and silently discarded — so a real error
   produced no visible symptom at all except "nothing happens." Fixed by
   caching the most recent message of each status type and replaying it to
   any client that connects afterward.
2. **No visibility into the raw serial link**: the backend gave no signal
   about whether *any* bytes were arriving, only whether they parsed as
   valid CSI. Added periodic `[serial] N bytes, N lines, N parsed` logging
   and echoing of the first few non-matching lines, so a wrong port, dead
   device, or format mismatch is immediately visible in the terminal.

A later, separate freeze was diagnosed as a **third bug**: `onFrame()` in the
dashboard JS would throw if an incoming frame's row length didn't match the
waterfall's allocated width, and since that line ran before every other
on-screen update, one bad frame silently broke *all future* rendering while
the WebSocket itself stayed open and green ("Connected"). A JS exception
inside `onmessage` does not close a WebSocket — it just fails silently.
Fixed by validating row length before touching the canvas buffer, wrapping
the rest of the frame-handling in try/catch, and adding a live
"last frame Xs ago" indicator so a real freeze is now instantly
distinguishable from a rendering bug (the indicator would show 0.1s if data
is flowing but rendering is broken; it climbs if data has genuinely stopped).

---

## 7. Model robustness: calibration and smoothing (first pass)

**Why asked:** the user reported the model "gets affected easily" and
sometimes classifies empty as moving. Recommended and then implemented, in
priority order:

### 7a. Baseline calibration
**Diagnosis**: the original absolute features (§2) encode "what does this
room's baseline look like right now" as much as "is someone moving." A
shifted baseline between sessions/deployments (repositioned hardware,
different day, temperature) could get misread as motion — this is *exactly*
what caused session-4 attempts 1 and 2 (§4).

**Fix**: every window's features are now expressed **relative to a short
calibration baseline** (the first ~10s of confirmed-empty data at the start
of a session/deployment) instead of as absolute values:
- means become **deltas** (`window_mean - baseline_mean`) — cancels offset
  drift.
- spreads become **ratios** (`window_std / baseline_std`) — an
  already-noisy baseline (fan, interference) raises the bar for what counts
  as "more disturbed than that," instead of a fixed threshold learned on a
  different day.

This logic lives in `csi_common.py`: `raw_window_stats()`,
`compute_baseline()`, `calibrate_features()`.

**Explicitly does NOT fix**: a room that is never quiet *during calibration
itself* — a continuous disturbance present throughout calibration becomes
the new "normal."

### 7b. Prediction smoothing (`PredictionSmoother` in `csi_common.py`)
A single noisy 2-second window could flip the displayed state. Fixed with
hysteresis: the *displayed* (confirmed) state only changes once a strict
majority of the last 5 raw window predictions agree. Both live front ends
show the confirmed/smoothed state as primary, with the raw single-window
prediction shown as secondary/debug text.

**Result after 7a+7b, retrained on 4 sessions**: 97.30% LOSO (vs 97.72%
absolute-feature version) — a small, expected cost for the robustness gain.
Held-out session 4 (train 1-3, test 4): 95.83% with calibration (vs 98.33%
absolute) — same trade-off, on data that happened to already match training
conditions well.

---

## 8. Per-block recalibration (a second, more important calibration fix)

**Why:** sessions 5 and 6 (see §9) revealed that a *single* calibration
baseline computed once at the start of a session goes stale within minutes —
both from a discrete event (session 5: someone turned the AC on partway
through) and from continuous drift with no obvious single cause (session 6).

**Fix, in `train_model.py`**: instead of calibrating once from the first
empty block of a session, the pipeline now **recalibrates at every
LEAVE→EMPTY transition** within a session — each empty block gets its own
fresh baseline from its own first ~10s. Windows are scored against whichever
empty block's baseline most recently preceded them.
(`find_empty_block_starts()`, `compute_block_baselines()`,
rewritten `windowize()`/`build_dataset()`.)

**Result**: this alone fixed session 5 (72.65% → 99.17% held-out) and
substantially improved session 6 (72.65% → 91.45% — not perfect, because
session 6's drift was *continuous*, not a single discrete step; one snapshot
per block can't fully track something that keeps moving within a block).
4-session LOSO also improved slightly (97.30% → 98.34%) — a pure win, no
regression anywhere.

---

## 9. Sessions 5-8: real diagnostic case studies (data collection phase)

The user's stated plan was to add sessions `part_5_data` through
`part_8_data`, each deliberately varying one condition to teach the model
that non-motion disturbance ≠ presence. Every single new session hit a real,
explainable problem before becoming usable — this is the most repeated
pattern in the whole project and is worth understanding as a template for
future sessions:

- **part_5, attempt 1**: recorded at 192 subcarriers instead of 128 (router
  auto-selected 40MHz channel width, likely due to less nighttime Wi-Fi
  congestion at ~1am). Consistent throughout (not the old mid-recording-
  switch bug), but incompatible feature space with the 128-subcarrier model.
  **Fix applied by the user**: forced the router to 20MHz-only channel width
  in its settings. Never happened again.
- **part_5, attempt 2**: 128 subcarriers, but held-out accuracy only 68.5%
  (later 75% on a 3rd attempt). Diagnosis: the user had moved to a
  **different, bigger room** *and* moved less vigorously — two confounded
  variables at once. This data was set aside (not deleted) rather than used,
  since it couldn't cleanly teach anything with two things changed at once.
- **part_5, final (accepted)**: clean, 128 subcarriers, but held-out accuracy
  only 75% even though the *within-session* separation looked perfect
  (0.89 vs 5.28 motion-energy ratio). Root cause diagnosis (per-window
  inspection): the *second* empty block within the session read completely
  differently from the *first* — because the user turned the AC on between
  them. This directly motivated §8 (per-block recalibration). After that fix
  was implemented, this exact session re-scored at **99.17%**.
- **part_6**: held-out 72.65% (raw features) / 91.45% (after §8's fix).
  Diagnosis: unlike part_5's AC (a step change), this was a **continuous,
  slow drift** in specific subcarriers' mean amplitude across the whole
  session, cause unconfirmed (possibly thermal). Motion energy stayed
  correctly flat/low throughout — it was the static per-subcarrier
  fingerprint that drifted, not anything genuinely motion-like.
- **part_7**: held-out 86% at the (already-deployed) 0.75s window, but 95%
  at the original 2.0s window. Diagnosis: not bad data — the *shorter*
  window is simply noisier for this particular session because its
  movement happened to be less vigorous than others, so the accuracy hit
  from shorter windows (§10) landed harder here than on average. Folded in
  as-is; genuinely useful data, not an artifact.
- **part_8** (deliberately: "opened the window"): held-out 95.33%, clean and
  one-directional errors (missed motion, no false alarms) — but a striking
  finding on closer inspection: the *aggregate* motion-energy feature was
  completely flat between empty and moving (0.98 vs 0.97), yet per-subcarrier
  features still separated the classes cleanly, concentrated in the *same*
  subcarrier band (39-49) that's shown up as important since the very first
  feature-importance analysis (§3). This directly motivated §13's
  order-invariant features.

**8-session final model (0.75s window, absolute-per-index + calibration)**:
95.87% LOSO. Confirmed via `compare_models.py` (§11) that Random Forest is
genuinely the best of 6 tested model families on this data, not an arbitrary
first guess.

---

## 10. Live reaction-speed tuning

**Why:** the user reported the live dashboard felt slow to react — up to
~2.5s total lag (2s window "ramp-in" + up to 0.5s of smoothing).

Tested window sizes empirically against real 8-session data via
`train_model.py --window-seconds`:

| Window | LOSO accuracy |
|---|---|
| 2.0s (original) | 96.49% (6-session), 97.72% (4-session) |
| 1.0s | 94.50% |
| 0.5s | 94.69% |
| **0.75s** | **95.14%** (7-session) / **95.87%** (8-session) — best tradeoff |

0.75s won outright — matching or beating 1.0s while being meaningfully
faster, and clearly better than 0.5s. **This is the currently deployed
window size.**

---

## 11. Model comparison (`compare_models.py`)

**Why asked:** "we've only tested Random Forest — is that actually the best
choice?" Built a script that runs the *same* LOSO methodology across 6
model families on identical calibrated features: Random Forest, Gradient
Boosting, Logistic Regression, SVM (RBF), SVM (linear), KNN.

**Result**: Random Forest won outright (95.14% mean, 0.025 std across
folds) — not just highest, but also low variance (not winning by luck on a
couple of easy sessions). Gradient Boosting was close second (94.71%).
Genuinely useful negative result: confirms RF wasn't an arbitrary first
choice.

---

## 12. LaTeX report (`report/csi_report.tex`)

**Why asked:** the user wanted a detailed report for Overleaf covering
everything from the beginning. A ~13-section report was written covering
sections 1-11 of *this* document in academic style, with a TikZ architecture
diagram and tables built from the real numbers above (no fabricated data).
Statically checked (brace/environment balance, no stray unescaped LaTeX
special characters) since no local LaTeX engine was available to compile it
directly. **This report predates §13 onward below** — it does not cover the
order-invariant features, the two live-bug fixes, or the dashboard overhaul.
If asked to update it, it needs a new section (or several) for everything
from §13 onward.

---

## 13. Order-invariant features — the cross-room-generalization attempt

**Why:** the user tested the live model in 2 different, untrained-on rooms
and got "6/10" detection. Diagnosis: absolute per-subcarrier features
(`amp_43_std_ratio`, etc.) are tied to *specific subcarrier indices* — the
Random Forest learned that indices ~40-48 matter because that's where *this
room's* multipath happened to put the sensitive spot (confirmed
independently in §9's part_8 finding). A different room's geometry could put
the sensitive band at entirely different indices, and index-specific
features have no way to transfer that.

**Fix, in `csi_common.py`'s `calibrate_features()`**: added 6 new
**order-invariant** features that summarize the whole subcarrier band
without caring which specific index is disturbed: `std_ratio_max`,
`std_ratio_topK_mean` (K=10), `std_ratio_p90`, `std_ratio_frac_elevated`,
`mean_delta_absmax`, `mean_delta_absmax_topK_mean`. These are *added*
alongside the original 260 index-specific features, not a replacement.

**Retrained, no regression** (95.54% LOSO vs 95.87% before — within normal
fold-to-fold noise). The user reported live improvement afterward
("it's better now"), but this was never rigorously measured with a true
held-out-room test (planned — record a session in the 2nd room, run
`evaluate_holdout.py` against it — but not yet executed as of this writing;
see §16/§18).

**This feature addition later caused a real, serious bug — see §15.**

---

## 14. Dashboard: live recalibration + full visual overhaul

### 14a. Live periodic recalibration
**Why:** §8 fixed recalibration *offline* (training), but the live
dashboard/predictor still only calibrated once at startup and never again —
so the exact AC-turning-on scenario that broke part_5 could still happen
live with no way to recover except restarting the server.

Added `RollingCalibrator` in `csi_common.py`: once the confirmed prediction
has been EMPTY for a sustained stretch (~10s, matching the model's
`calib_seconds`), it recalibrates. **This went through two versions** — see
§15 for why the first version was itself buggy.

Surfaced on the dashboard as a new **Calibration** card: last-recalibrated
time (live-ticking), a running count, and a small log of recent events.

### 14b. Full visual overhaul (requested: "add more details" + "more color")
- New **Session summary** card: uptime, time-empty vs. time-moving,
  occupancy %, state-change count — tracked client-side in the browser from
  the confirmed prediction stream.
- Manual **light/dark/auto theme toggle** (persisted in localStorage),
  overriding the OS-level `prefers-color-scheme` in either direction.
- **Hover tooltips** on the motion-energy chart — implemented as a plain
  absolutely-positioned HTML div, *not* an SVG overlay, specifically because
  the chart's SVG content gets fully replaced on every incoming frame
  (~10Hz) and an SVG-based crosshair would get wiped out mid-hover.
- New **"About this system"** explanatory card (3 short paragraphs: what CSI
  is, how this specific system works, what each dashboard section means) —
  requested so a first-time viewer has context, not just numbers.
- **Colorful gradient page background** (blue → violet → aqua → green wash,
  separate light/dark variants) — deliberately applied only to the page
  background, not the cards themselves, so the waterfall/energy/status
  colors stay exactly as legible as before; only the empty space got
  livelier.

---

## 15. Two serious live-only bugs (found during a "check everything" audit)

**Trigger**: after §13's order-invariant features and §14a's rolling
recalibration were both live, the user reported the model would predict
"moving" almost constantly, even when they were out of the room — a
regression from how it behaved right after session 8 was first trained.
Asked explicitly: "did you change the code... make sure to check
[everything]."

### Bug 1: dead-subcarrier ratio blowup
**Root cause** (confirmed with real data from `part_8_data`): 20 of the 128
subcarriers are structurally dead (OFDM guard bands / DC nulls) — always
exactly zero amplitude, so their calibration-baseline std is exactly 0.0.
Dividing by ~zero means even trivial live noise (std=0.01) on one of these
subcarriers produced a ratio of ~10,000 (a healthy ratio is ~1.0). §13's new
`std_ratio_max` / `std_ratio_topK_mean` features explicitly hunt for the
*largest* ratio across all 128 subcarriers each window — so they were almost
guaranteed to get hijacked by one of these 20 dead subcarriers' noise on
nearly every window, unrelated to real motion. The old per-index features
had the same theoretical flaw but a Random Forest could learn to simply
ignore a handful of consistently-garbage individual features; a max-based
aggregate cannot ignore its own defining operation.

**Fix**: the ratio denominator is now floored relative to *that baseline's
own median spread* (`amp_std_floor` = 10% of median baseline std, computed
in `compute_baseline()`), not a fixed tiny epsilon. Verified: the same
0.01-std noise now produces a ratio of 0.13, not 10,000. Retrained — 95.54%
LOSO, unchanged, confirming this was a pure live-inference bug, invisible in
offline validation because the short (~2 min) recorded sessions rarely
triggered it by chance.

### Bug 2: rolling recalibration could drift toward hypersensitivity
**Root cause**: the first version of `RollingCalibrator` (§14a) *fully
replaced* the baseline every ~10s using only that 10-second sample. A short
sample's noise estimate is statistically noisy — it can land smaller than
the room's true typical noise purely by chance. With no anchor pulling it
back, repeated full-replacement recalibration over a longer session had no
protection against drifting toward an unrealistically tight baseline, which
makes ordinary conditions look like motion. This plausibly explains "worked
right after setup, degraded over the course of a longer session."

**Fix, `RollingCalibrator` redesigned with two safety mechanisms**:
1. **Blend, don't replace** — each recalibration nudges the baseline
   (`blend_alpha=0.3`) toward the fresh sample rather than overwriting it,
   so one noisy sample can't take over on its own.
2. **Floor against the original startup baseline** — `amp_std`/`rssi_std`/
   `energy_std` can never be blended down below 60% of what the *original,
   deliberate* "leave the room" startup calibration measured. That startup
   calibration is a trustworthy fixed reference; nothing live is allowed to
   make the system *more* sensitive than that was, only less (to still
   absorb something genuinely louder, like an AC).

Verified with a 50-cycle simulation under constant synthetic noise: the
blended+floored version stayed essentially flat (0.986 → 0.993, never below
0.979); confirmed this really was a meaningful risk worth fixing.

### Bug 3 (architectural gap, not a code defect): no self-recovery from a stuck state
If the confirmed prediction ever locks onto MOVING for *any* reason,
recalibration is gated behind being confirmed EMPTY — so a stuck bad state
had no way to self-correct short of restarting the whole server. Rather
than auto-detecting "stuck" (risky — could suppress a real, long motion
event), added a **manual escape hatch**: a "Force recalibration now" button
on the dashboard (sends `{"type": "recalibrate"}` over the WebSocket to a
newly-added bidirectional control channel in `csi_live_server.py`), and
pressing **'r'** in the Matplotlib window for `csi_live_predict.py`. Both
immediately reset and restart the leave-the-room calibration on demand.

**All three fixes retrained/reconfirmed at 95.54% LOSO — no regression.**
As of this writing, the user has confirmed "it's better now" after these
fixes but has not yet done an extended (multi-hour) unattended test, which
is the real test of whether bug 2 in particular is fully resolved (see §18).

---

## 16. Current state of the system (as of this writing)

- **Model**: Random Forest, trained on 10 sessions — the original 8
  single-room sessions (`part_1_data`..`part_8_data`) plus two second-room
  sessions (`room2_part_1_data`, `room2_part_2_data`, see §21) — 0.75s
  windows, per-block + rolling calibration, 266 features (260
  index-specific + 6 order-invariant). **95.07% LOSO accuracy.** Saved to
  `csi_model.joblib`. Reproducible via `python train_model.py --sessions
  part_1_data part_2_data part_3_data part_4_data part_5_data part_6_data
  part_7_data part_8_data room2_part_1_data room2_part_2_data
  --window-seconds 0.75` — note the script's CLI defaults are only the
  first 4 sessions at a 2.0s window (a leftover from early development, see
  §21), so always pass the full session list and `--window-seconds 0.75`
  explicitly; running the script bare will silently retrain and overwrite
  `csi_model.joblib` with a much weaker, wrong-config model.
- **Validated single-room performance**: strong and repeatedly confirmed via
  LOSO, true held-out-session tests, and cross-algorithm comparison.
- **Cross-room performance**: tested for the first time (§21). A held-out
  second room scored 82.5% when trained only on room 1; folding two room-2
  sessions into training brought overall LOSO to 95.07% with room2_part_2
  as its own held-out fold scoring 98.44%. Real evidence it generalizes
  across rooms, though still only two rooms total — not yet the "genuinely
  unproven" state this section used to describe, but not exhaustively
  proven either.
- **Live system**: both front ends (Matplotlib + web dashboard) working.
  The web backend/dashboard were rewritten for **two simultaneous ESP32
  nodes** (§21) combined via OR logic, plus two further live-only bugs
  found and fixed since §15 (a WebSocket keepalive-timeout crash from
  double model inference, and a subcarrier-count-mismatch crash that used
  to take down the whole server instead of just disabling that one node's
  predictions). Confirmed working live with both nodes physically separated
  — not yet burn-tested over a long unattended period (§18 item 3).
- **Off-axis blind spot**: previously an open, unfixed limitation (physics
  of single-link CSI, not a bug — see prior wording below). **Now
  mitigated**: a second ESP32 node was added, physically separated from the
  first and combined via OR logic (§21), and the user has confirmed live
  that this measurably improves room coverage. Not a complete fix (each
  node still individually has its own blind spot, and the two nodes'
  results aren't cross-checked against each other beyond OR-combination),
  but the single biggest practical gap here is closed.

---

## 17. File inventory (this folder)

| File | Role |
|---|---|
| `firmware/main/csi_node.c` | ESP32-S3 firmware (CSI capture, ping trigger, AP filtering, UART output) |
| `csi_common.py` | **Single source of truth** for parsing, feature extraction, calibration (`compute_baseline`, `calibrate_features`, `RollingCalibrator`), and prediction smoothing (`PredictionSmoother`). Every other script imports from here — this is why offline and online math can't silently diverge. |
| `train_model.py` | Loads sessions, per-block-calibrates, windows, trains + LOSO-validates, saves `csi_model.joblib` |
| `evaluate_holdout.py` | True held-out evaluation: train on named sessions, test on a different named session (the "is this new session usable" check, used for every part_5-8) |
| `analyze_model.py` | Produces the numbers behind a trained model as JSON (fold accuracy, feature importance, class balance, motion-energy distributions) — built for an early Artifact visualization, now somewhat superseded by later analysis but still functional |
| `compare_models.py` | Compares 6 ML model families (RF, GB, LR, SVM×2, KNN) under identical LOSO methodology |
| `model_evaluation.py` | Produces `model_evaluation_report.pdf` — confusion matrix, feature correlation matrix, two overfitting diagnostics, RAG health summary (§20) |
| `full_model_report.py` | Generic (not CSI-specific) LOSO evaluation report generator — 16 metrics, ~20 figures, paginated PDF with a real TOC via reportlab. Accepts `--csv` or `--sessions` so it can evaluate any tabular classification model, not just this project's (§21) |
| `csi_label_collector.py` | Auto-cycling data collector (leave/empty/moving protocol) — copied into the repo root as part of §19's GitHub prep, originally lived outside this folder |
| `csi_live_monitor.py` | Pre-model live waterfall + motion-energy verification tool — same origin note as above |
| `csi_live_predict.py` | Matplotlib-based live inference: waterfall + real-time prediction, with the same calibration/smoothing/rolling-recal logic as the web version. Press 'r' to force recalibration. |
| `csi_live_server.py` | Python WebSocket backend for the web dashboard: serial → calibrated prediction → browser, with a bidirectional control channel (client can request `recalibrate`) |
| `csi_dashboard.html` | Browser dashboard: canvas waterfall, energy sparkline (with hover), prediction badge, session summary, calibration status + manual recalibrate button, theme toggle, about-this-system explainer |
| `csi_model.joblib` | The deployed model + metadata (`amp_columns`, `feature_names`, `window_seconds`, `calib_seconds`, `frame_hz`) |
| `report/csi_report.tex` | LaTeX writeup for Overleaf, covering roughly §1-§11 of this document (predates order-invariant features and the live-bug fixes) |
| `report/csi_file_inventory.tex` | A separate LaTeX file-inventory doc found already present when the repo was prepared (§19) — origin predates the active context at that point, likely from an earlier, since-summarized part of this same ongoing project |
| `README.md` | GitHub-facing overview: architecture diagram, results, setup instructions, limitations |
| `LICENSE` | MIT |
| `.gitignore` | Excludes `build/`, the legacy contaminated dataset, secrets, regenerable output |
| `requirements.txt` | Python dependencies for all the tool scripts |
| `docs/PROJECT_HISTORY.md` | This file |
| `CLAUDE.md` (repo root) | Short pointer file, auto-loaded by Claude Code at the start of every session in this folder — summarizes current state and directs to this file for full detail |

### Recording sessions in detail

Each `part_N_data/` folder contains exactly three files:
`all_csi_data.csv` (every frame, both labels), `label_0.csv` (empty frames
only), `label_2.csv` (moving frames only). All currently 128 subcarriers,
recorded at 10Hz. `session_id` below is the value in that folder's
`session_id` column (also embeds the recording date/time).

| Folder | session_id | What it is / what it tested | Held-out result | Status |
|---|---|---|---|---|
| `part_1_data/` | `20260726_153123_c36ca3` | First of the 3 original sessions | — (part of original LOSO baseline) | in training |
| `part_2_data/` | `20260726_164847_905931` | Second original session | — | in training |
| `part_3_data/` | `20260726_165845_e1ec34` | Third original session | — | in training |
| `part_4_data/` | `20260726_184128_ec02d1` | The "true held-out" session (§4) — 3rd recording attempt, first two attempts at this slot failed (65.55%, 51.26% — contaminated empty, then hardware moved) and were overwritten, not kept | 98.33% (train 1-3, test this) | in training |
| `part_5_data/` | `20260727_141022_05cbed` | Final accepted attempt (3rd try — 1st was 192 subcarriers/router auto-40MHz, 2nd was confounded room-change+weak-movement and was set aside, not deleted). This one: AC turned on partway through, motivated per-block recalibration (§8) | 75% before §8's fix → 99.17% after | in training |
| `part_6_data/` | `20260727_142713_ba7d42` | Continuous/slow drift within the session, cause unconfirmed (possibly thermal) — motion energy stayed correctly flat throughout, only the static per-subcarrier fingerprint drifted | 72.65% before §8's fix → 91.45% after (best achievable — drift is continuous, not a single step, so per-block recalibration can't fully track it) | in training |
| `part_7_data/` | `20260729_140007_dba60b` | Normal recording; the "86% vs 95%" case that turned out to be a window-size artifact (§10), not bad data — less vigorous movement made it more sensitive to the shorter 0.75s window than average | 95% at 2.0s window / 86% at 0.75s window | in training |
| `part_8_data/` | `20260729_152303_3a4d7f` | Deliberately "opened the window." Clean, one-directional errors only. Revealed the aggregate motion-energy feature can go flat while per-subcarrier features still separate the classes — directly motivated §13's order-invariant features | 95.33% | in training |
| `room2_part_1_data/` | — | First session recorded in a genuinely different room (§21) — the cross-room held-out test | 82.5% (trained on all 8 room-1 sessions only) | in training |
| `room2_part_2_data/` | — | Second room-2 session, recorded to fold room-2 data into training rather than leave it as a single held-out sample (§21) | 98.44% (as its own held-out fold, 10-session LOSO) | in training |

**Sessions that were recorded but discarded — confirmed NOT present on disk
anymore** (each re-recording used the same folder name, so the collector
overwrote the previous attempt rather than preserving it):
session-4's first two attempts (contaminated empty; then apparent
hardware/router repositioning) and session-5's first two attempts
(192-subcarrier recording from an auto-40MHz router event; a
room-change+weak-movement recording that was technically valid but too
confounded to use). Only the final, accepted version of each survives in
`part_4_data/` and `part_5_data/` respectively. If any of these ever need to
be reproduced for comparison, they'd have to be re-recorded from scratch —
the diagnosis details (motion-energy numbers, RSSI values) are preserved
narratively in §4 and §9 above, but not the raw CSVs.

**Update (GitHub repo prep)**: `csi_label_collector.py` and
`csi_live_monitor.py` were originally only at
`C:\Users\ilyas\Documents\Codex\2026-07-21\wh\outputs\`, outside this folder
— they've since been copied into this repo's root so the repo is
self-contained. `csi_live_monitor.py` is superseded by `csi_live_predict.py`
/ `csi_dashboard.html` for anything beyond raw signal sanity-checking, but
kept since it's still a smaller, more minimal verification tool.

Also as part of GitHub repo prep: the ESP-IDF project (`main/`,
`CMakeLists.txt`, `sdkconfig`) moved into `firmware/` (now
`firmware/main/csi_node.c` etc.), and this file moved into `docs/`. Every
other path mentioned elsewhere in this document (before this note) reflects
where things were *at the time each section was written*, not necessarily
today's layout — check `README.md`'s project structure section for the
current, authoritative layout.

---

## 18. Open items / recommended next steps

**Current #1 priority, as of this writing**: diverse "moving" data / a
second person (item 1 below) — the cross-room test and the second node are
both done now (§21), so the highest-value remaining untested axis is
whether the model generalizes beyond the one person it's ever been trained
on.

1. **Diverse "moving" data / a second person** — all moving data so far is
   one person (the user) with one general movement style. Untested whether
   the model detects "a person" specifically vs. "this person." Now that
   room coverage (cross-room + 2-node, §21) is validated, this is the
   single biggest remaining generalization gap.
2. **A genuine third room** — cross-room has been tested in exactly 2 rooms
   (§21). Two rooms shows "it transfers, roughly"; a third would show
   whether the generalization gain is a real trend or a fluke (this was the
   user's own reasoning on 2026-08-04, see memory).
3. **Extended live soak test** — leave the dashboard running for hours,
   unattended, in normal use, to confirm §15's bug 2 fix (rolling-
   recalibration drift) holds over a long real session, not just a short
   test. Lower urgency now that §15's fixes have been running without
   complaint, but never formally soak-tested. Now doubly relevant with two
   nodes running concurrently (§21) rather than one.
4. **`validate_session.py`** (proposed multiple times, never built): every
   new session (§9) has been manually diagnosed by hand — subcarrier
   consistency check, motion-energy/RSSI separability comparison against
   existing sessions, per-block breakdown. Should be a real script, still
   isn't one. `model_evaluation.py` (§20) covers *model* health, not new
   *session* validation before folding it into training — this is a
   different, still-missing tool.
5. **Feature redundancy** — §20's health check flagged 68 highly-correlated
   feature pairs among the top 25 by importance (expected: neighboring
   subcarriers move together, and the order-invariant features from §13 are
   near-duplicates of each other by construction). Not urgent (Random
   Forest tolerates correlated features fine), but a real candidate for
   future feature trimming if a leaner model is ever wanted.
6. ~~Second ESP32 node~~ — **done**, see §21. Two nodes, physically
   separated, combined via OR logic; user has confirmed live that this
   measurably improves room coverage over a single node.
7. ~~The cross-room held-out test~~ — **done**, see §21. Still only 2 rooms
   total (item 2 above carries the "is this really a trend" follow-up).

**Explicitly deferred, by the user's own choice, not because they're bad
ideas**: true position/coordinate tracking (not feasible with current
single-link hardware — this was explained in detail and the user agreed to
drop it); zone/room classification via fingerprinting (same reasoning:
deferred until the 2-class detector is solid).

---

## 19. GitHub repository published

**Why asked:** the user wanted the project on GitHub, "organized well" and
"well presented."

**Repo**: **https://github.com/IALR/csi-motion-detection** (public).

What was done:
- **Security first**: found real, hardcoded Wi-Fi credentials
  (`WIFI_SSID`/`WIFI_PASS`/`ROUTER_IP`) in `main/csi_node.c` in plain text.
  Moved them into `firmware/main/wifi_secrets.h` (gitignored, real values,
  never committed) with `firmware/main/wifi_secrets.h.example` (placeholder
  values, committed) as the template. Verified with `git grep` against the
  actual committed HEAD tree (not just the working directory) that neither
  the real password, SSID, nor router IP appear anywhere in git history
  before pushing.
- **Reorganized** from a flat folder into: `firmware/` (the ESP-IDF project
  — `main/`, `CMakeLists.txt`, `sdkconfig` all moved there), `docs/`
  (`PROJECT_HISTORY.md` moved there), `report/` (already existed). Tool
  scripts, data folders (`part_1_data`..`part_8_data`), and `csi_model.joblib`
  were deliberately **kept at the repo root**, not nested further, because
  every script's default relative paths (`part_1_data`, etc.) assume the
  working directory is the repo root — moving them would have required
  updating path logic in every script for no real benefit. Verified nothing
  broke by re-running `train_model.py` after the move (95.54% LOSO,
  unchanged).
- **Copied in** `csi_label_collector.py` and `csi_live_monitor.py` from
  their original location outside this folder
  (`C:\Users\ilyas\Documents\Codex\2026-07-21\wh\outputs\`) so the repo is
  self-contained — a clone of the GitHub repo now has everything needed,
  nothing lives outside it anymore.
- **New files created**: `README.md` (architecture diagram in Mermaid,
  results table, project structure, setup instructions, known limitations),
  `LICENSE` (MIT), `.gitignore`, `requirements.txt`.
- **Excluded via `.gitignore`**: `build/` (158MB of regenerable ESP-IDF
  build output), `csi_dataset_20260723_184612/` (the legacy pre-fix dataset
  with mixed subcarrier counts, explicitly known-contaminated — see §1),
  `model_analysis.json` (regenerable), `__pycache__/`.
- **`gh` CLI wasn't installed initially** — found later at
  `C:\Program Files\GitHub CLI\gh.exe` but not on the shell session's PATH
  (a stale-environment issue, not a missing install); worked around by
  calling the full path directly rather than relying on PATH resolution.
- Git identity had never been configured on this machine at all (`git
  config user.name`/`user.email` were unset) — set locally
  (`--local`, scoped to just this repo, not globally) using the user's
  known email, since a first commit is impossible without it.

**Result**: 47 files, single initial commit, pushed and tracking
`origin/master`. Future updates: commit + push as normal, nothing automatic.

---

## 20. Model evaluation report (`model_evaluation.py`)

**Why asked:** the user wanted a dedicated script producing a confusion
matrix, a feature correlation matrix, overfitting diagnostics, and an
overall red/amber/green ("RAG") health summary — "a file with all graphs to
evaluate the model."

Produces `model_evaluation_report.pdf` (6 pages, regenerated fresh every run
from whatever sessions/window size are passed in — nothing hardcoded from a
prior run) plus a console log:

1. **Confusion matrix** (out-of-fold LOSO predictions — every prediction
   came from a fold that never trained on that window's session): counts +
   row-normalized versions.
2. **Feature correlation matrix** — top 25 features by importance, Pearson
   correlation heatmap.
3. **Feature importance bar chart** (companion to #2).
4. **Overfitting check #1**: train accuracy vs. held-out accuracy, per
   session, as a grouped bar chart.
5. **Overfitting check #2**: a validation curve sweeping `max_depth` from 2
   to unlimited, plotting mean train accuracy and mean held-out (LOSO)
   accuracy at each depth.
6. **RAG health summary**: six checks (overall accuracy, overfit gap,
   session-to-session variance, worst single session, class balance,
   feature redundancy), each colored green/amber/red against explicit
   thresholds defined in the script's `rag()` function.

**Results from the run in this conversation** (8 sessions, 0.75s window):
out-of-fold accuracy 95.52% (2,252 windows: 44 false alarms, 57 missed
detections — roughly balanced, not skewed toward one error type); mean
train-vs-held-out gap 4.5 points (small, healthy); held-out accuracy
*improves* with tree depth up to ~8 then plateaus around 95.5% rather than
dropping (the signature of a model that is **not** overfitting — if it
were, held-out accuracy would fall as depth increased while train accuracy
kept climbing); worst single session part_7 at 91.7%. Five of six RAG
checks GREEN; the sixth (feature redundancy, 68 correlated pairs among the
top 25) is RED but not urgent — see §18 item 5.

**A follow-up question worth remembering the answer to**: the user asked
whether it's OK that train accuracy hits exactly 100% on every fold. Yes —
this is structurally expected for an unconstrained Random Forest (each tree
grows until leaves are pure) and is not evidence of a problem *on its own*;
what actually matters is the *held-out* accuracy and the *gap* between the
two, both of which are healthy here. The depth-sweep chart (page 5) is the
independent evidence for this: if 100% train accuracy were genuinely
hurting generalization, restricting depth should have improved held-out
accuracy, and it didn't.

---

## 21. Cross-room retrain, dual-node hardware, and live-server hardening

**Why asked:** §18's #1 priority (cross-room test) plus the user acquiring
a second ESP32-S3 board, leading naturally into "how do I use 2 nodes
properly."

**Cross-room test.** Recorded `room2_part_1_data` in a genuinely different
room (same router link, 128 subcarriers, clean protocol). Trained on all 8
room-1 sessions, scored once on it via `evaluate_holdout.py`: **82.5%**
(vs. 95.5% same-room LOSO at the time). Errors were lopsided — 26%
false-alarm rate on true-empty windows, only 9% miss rate on true-moving —
the model leaned toward over-alerting in the new room. Diagnosed the same
way as every prior anomaly in this project (§9's pattern): room2's
`rssi_std_ratio` empty/moving separation (0.37 vs 1.65) was real but much
weaker than typical room-1 sessions (e.g. part_1: 0.24 vs 3.7), so
borderline windows tipped toward "moving" more easily. Rather than
immediately fold this one session into training, the user chose to record
a second room-2 session first (`room2_part_2_data`), reasoning that two
single-room-2 data points can't distinguish "generalizes" from "got
lucky/unlucky once."

**10-session retrain.** Trained on all 8 room-1 sessions plus both room-2
sessions, 0.75s window: **95.07% LOSO accuracy**, with `room2_part_2_data`
scoring 98.44% as its own held-out fold — a real, repeatable (confirmed on
a second run, same `random_state=42`) result, not the informal "6/10" this
project's docs used to cite. Full historical accuracy timeline: 96.96% →
97.30%/95.83% → 98.34% → 95.87% → 95.54% → cross-room 82.5% → 10-session
retrain 95.07%. **Gotcha discovered while re-verifying this for the docs**:
`train_model.py`'s CLI defaults are only `part_1_data`..`part_4_data` at a
2.0s window (leftover from early development, §2) — running the script
with no arguments silently retrains and overwrites `csi_model.joblib` with
a much weaker, wrong-config model. Always pass the full 10-session list and
`--window-seconds 0.75` explicitly (see the exact command in §16).

**`full_model_report.py`** was built alongside this: a generic (not
CSI-specific) reportlab-based LOSO evaluation report generator — 16
metrics including ROC AUC/MCC/Cohen kappa, ~20 figures, a real paginated
TOC via `BaseDocTemplate`. Verified end-to-end against this project's own
data (94.2% out-of-fold, matching `model_evaluation.py`'s independent
number).

**Dual-node architecture.** With a second ESP32-S3 available, built a real
2-node system rather than just wiring up a second serial port:
- Each node runs its **own independent pipeline** end-to-end — own serial
  connection, own calibration baseline, own rolling window, own model
  inference. No timestamp synchronization or shared feature vector between
  nodes; the only combination point is the very last step.
- **Combination logic** (`compute_combined()` in `csi_live_server.py`): OR
  across nodes — MOVING if *any* node says MOVING, EMPTY only if *all*
  configured nodes say EMPTY. Chosen because the two nodes' value is
  covering *different* physical geometry (different blind spots), not
  cross-checking each other — requiring agreement would only reintroduce
  the single-node blind spot problem this was built to fix. Unit-tested
  with 9 explicit cases before deployment.
- `csi_live_server.py` gained a `--port-b` CLI arg, per-node message
  tagging (`"node": "A"|"B"` on every broadcast), a `combined` message type
  for the OR-ed overall state, and a per-node control channel (client can
  request `{"type":"recalibrate","node":"A"|"B"|"all"}`).
- `csi_dashboard.html` gained a "Nodes" card (per-node tiles: status,
  RSSI, energy, a "View"/"Viewing" switch), a node-switch control on the
  Signal (waterfall/energy) card, per-node calibration sub-sections, and a
  `describeCombined()` helper that annotates the hero prediction badge with
  *which* node(s) triggered it (e.g. "confirmed by node B").

**Two live-server bugs found and fixed in the same pass:**
1. **Keepalive-timeout crash / reported prediction lag** — root cause was
   calling `.predict()` and `.predict_proba()` separately (two full forest
   walks) directly on the asyncio event loop, blocking WebSocket ping/pong
   handling long enough to trip the default 20s `ping_timeout`. Fixed by
   collapsing to a single `predict_proba()` call run via
   `loop.run_in_executor()` (off the event loop entirely) plus raising
   `ping_timeout` to 60s as a safety margin.
2. **Subcarrier-mismatch crash** — a stale AP config (see below) caused one
   node's stream to have 192 subcarriers instead of the model's expected
   128. The existing mismatch check only *warned once* but never actually
   prevented the doomed `predict_proba()` call on malformed data, which
   raised inside `asyncio.gather()` and crashed the **entire** server —
   taking down both nodes over one node's problem. Fixed by recomputing the
   mismatch check fresh every loop iteration and gating the whole
   prediction block on it, so a mismatched node degrades gracefully (raw
   data keeps streaming, predictions disabled, persistent warning shown)
   instead of killing the shared process.

**Two intervening hardware/network issues hit during bring-up, both
config-level, not logic bugs:**
- Node A produced zero `CSI_AMP` output after a rebuild. Root cause:
  `firmware/main/wifi_secrets.h`'s `ROUTER_IP` was stale (pointed at an
  old gateway on a different subnet than the boot log's actual `gw:`).
  Since CSI is triggered by pinging `ROUTER_IP` at 100ms intervals, every
  ping silently failed and no CSI-triggering traffic was ever generated.
  Fixed by updating `ROUTER_IP` to match the real gateway.
- The `ValueError: X has 394 features, but ... expecting 266` crash above
  — root cause was the AP (a phone hotspot in this case) auto-negotiating a
  40MHz channel width instead of 20MHz, doubling subcarriers 128→192, an
  exact repeat of the failure mode documented in §9. No router admin panel
  is available to force 20MHz on a phone hotspot, so the durable fix is an
  actual router; the software-side crash (see bug 2 above) is fixed
  regardless of which AP is used.

**Result, confirmed live by the user**: both nodes running simultaneously,
physically separated (not co-located — co-located nodes see nearly
identical multipath geometry and won't demonstrate the coverage benefit),
dashboard showing both node tiles live and correctly combining via OR. The
user confirmed this measurably improves room coverage over the single-node
setup — the off-axis blind spot (§16) is no longer a fully open problem.

---

## Working style notes for whoever picks this up

- The user wants concrete numbers and real diagnosis, not reassurance. When
  something breaks, they've responded well to "here's exactly why, here's
  the evidence, here's the fix" — not "should be fine now."
- Every "is this real" question in this project has been answered by
  actually running the script live and showing raw output, not a summary.
  Keep doing that.
- Big changes (new features, architecture changes, discarding a data
  session) have generally been proposed with a clear recommendation and
  confirmed with the user before executing, rather than done silently.
- The user runs Windows, uses PowerShell/Git Bash, and the collector/live
  scripts require exclusive access to the COM port (only one program can
  hold it at a time) — this has caused confusion before ("is it stuck"
  turned out to be "another program still has the port open").
