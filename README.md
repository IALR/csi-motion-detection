# CSI Motion Detection

Device-free motion detection using Wi-Fi Channel State Information (CSI) on
one or two ESP32-S3 boards — no camera, no wearable. A person moving through
a room disturbs the multipath propagation between the ESP32 and the router
in a way an empty room doesn't, and CSI (which every Wi-Fi receiver already
computes internally to decode data) captures that disturbance as a side
effect.

**Current state**: Random Forest classifier, 10 labeled recording sessions
across 2 rooms, **94.25% leave-one-session-out accuracy**, with a live
desktop tool and a browser dashboard supporting one or two ESP32 nodes
(combined via OR logic) for real-time monitoring. Accuracy depends strongly
on the room's RF noise floor — 95-99% in quiet rooms, 85-95% in noisy ones —
and the dashboard now measures and reports which regime you're in.

---

## How it works

```mermaid
flowchart TD
    A[ESP32-S3 node A] -->|CSI over serial, ~10Hz| B[csi_label_collector.py]
    A -->|CSI over serial| E[csi_live_server.py / csi_live_predict.py]
    A2[ESP32-S3 node B, optional] -->|CSI over serial| E
    B -->|labeled CSVs| C[part_1_data ... part_8_data, room2_part_1_data, room2_part_2_data]
    C --> D[train_model.py]
    D -->|per-session-out validated| F[csi_model.joblib]
    F --> E
    E -->|WebSocket, per-node + OR-combined| G[csi_dashboard.html]
    E --> H[Matplotlib live view]
```

1. **Firmware** (`firmware/`) pings the router ~10 times/second to keep a
   steady stream of CSI updates, and streams parsed frames over serial.
2. **`csi_label_collector.py`** auto-cycles a labeling protocol: countdown to
   leave the room → record EMPTY → wait for you to return → record MOVING →
   repeat. Solves the problem that you must be absent to record "empty" and
   present to record "moving."
3. **`train_model.py`** turns labeled CSVs into a validated Random Forest,
   using **leave-one-session-out** cross-validation — every session is held
   out and scored on a model that never trained on it, which is what makes
   the accuracy numbers below honest rather than optimistic.
4. **`csi_live_server.py`** (+ `csi_dashboard.html`) or **`csi_live_predict.py`**
   run the trained model against the live serial stream in real time.

---

## Results

| Check | Result |
|---|---|
| Leave-one-session-out accuracy (10 sessions, 2 rooms), per-window | **94.25%** |
| …unweighted mean of the 10 folds (the figure previously quoted) | 95.07% |
| Spread across sessions | std 3.9pp, best 98.7%, worst 84.7% |
| Best of 6 compared model families (RF, Gradient Boosting, Logistic Regression, SVM×2, KNN) | Random Forest wins outright, lowest variance too |
| True held-out session tests (trained on all others, scored once on a session never touched) | 91-99% across 4 independently tested sessions |
| Cross-room held-out test (trained on room 1 only, scored on an untouched room 2 session) | 82.5% |
| Live reaction window | 0.75s (tuned — the best accuracy/speed tradeoff of 4 tested window sizes) |

Every one of those numbers came from actually running the scripts in this
repo, not from hand-picking a good run — see
[`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) for the full,
warts-and-all account of what was tried, what failed, and why.

---

## What this project actually learned (the interesting part)

This project surfaced several real, diagnosable failures along the way —
documented in detail in [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md)
because the diagnostic process is the most reusable part of the work:

- **A single bad "empty" recording can silently wreck a session** — CSI is
  sensitive enough to register a person lingering near a doorway as
  disturbance. Fixed with an explicit leave-the-room countdown protocol.
- **A single calibration snapshot goes stale within minutes** — turning on
  an AC mid-session, or slow thermal drift, can make a genuinely empty room
  start reading as "moving." Fixed by recalibrating at every empty block
  during training, and periodically during live deployment (see
  `RollingCalibrator` in `csi_common.py`).
- **Dead subcarriers can silently dominate aggregate features** — ~20 of 128
  subcarriers are structurally zero (OFDM guard bands); a feature that hunts
  for "the most disturbed subcarrier" can get hijacked by noise on a dead
  one, producing ratios of 10,000+ from nothing. Fixed with a variance floor.
- **Absolute features don't transfer across rooms** — a model can end up
  keying on "subcarrier index 43" only because that's where *this room's*
  multipath happened to be sensitive. Added order-invariant features
  (max/percentile across the whole band) that don't care which index fires.

---

## Project structure

```
firmware/                  ESP-IDF project for the ESP32-S3 sensor node
  main/csi_node.c             CSI capture, Wi-Fi ping trigger, UART output
  main/wifi_secrets.h.example  Copy to wifi_secrets.h and fill in your network

csi_label_collector.py     Auto-cycling data collector (leave/empty/moving protocol)
csi_live_monitor.py        Live waterfall + motion-energy verification tool

csi_common.py              Shared parsing, feature extraction, calibration, smoothing
                            (single source of truth - training and live inference
                            both import from here, so they can't silently diverge)
train_model.py              Trains + leave-one-session-out validates the model
evaluate_holdout.py         True held-out evaluation on a named session
analyze_model.py            Dumps fold accuracy / feature importance / etc. as JSON
compare_models.py           Compares 6 ML model families under identical validation
full_model_report.py        Generic LOSO evaluation report generator (16 metrics, ~20 figures, PDF)
diagnose_nodes.py           Hardware diagnosis: compares two nodes' signal strength and
                             noise floor to separate a noisy room from a bad board/antenna

csi_live_predict.py         Matplotlib desktop live inference tool
csi_live_server.py          WebSocket backend for the browser dashboard, supports 1 or 2 ESP32 nodes (--port-b)
csi_dashboard.html          Browser dashboard: waterfall, motion energy, live status, per-node view

part_1_data/ .. part_8_data/       Labeled recording sessions, room 1 (see docs/PROJECT_HISTORY.md
                                    section "Recording sessions in detail" for what each one is)
room2_part_1_data/, room2_part_2_data/  Labeled recording sessions, room 2 (cross-room test)
csi_model.joblib             The trained, deployed model

report/csi_report.tex        LaTeX writeup (for Overleaf)
docs/PROJECT_HISTORY.md      Full project history - every decision and why
```

---

## Getting started

### 1. Firmware

Requires [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/).

```
cd firmware
cp main/wifi_secrets.h.example main/wifi_secrets.h
# edit main/wifi_secrets.h with your Wi-Fi SSID/password and router IP
idf.py set-target esp32s3
idf.py -p COM9 flash monitor
```

### 2. Python environment

```
python -m pip install -r requirements.txt
```

### 3. Record data

```
python csi_label_collector.py -p COM9 -b 115200 --subcarriers 128 --rssi-field --output-dir my_session_data
```
Leave the room when it counts down, press `m` when you're back and ready to move.

### 4. Train

```
python train_model.py                                    # reproduces the deployed model
python train_model.py --sessions part_1_data ... my_session_data --window-seconds 0.75
```

### 4b. Run the tests

```
python -m pytest tests/
```
25 tests covering parsing, calibration, prediction smoothing, and the live
system's failure modes (node offline, AP channel-width change mid-run).

### 5. Run it live

Desktop view:
```
python csi_live_predict.py -p COM9 -b 115200
```

Or the browser dashboard:
```
python csi_live_server.py -p COM9 -b 115200
# then open csi_dashboard.html in any browser
```

With a second ESP32 node (recommended — see "Off-axis blind spot" below),
add `--port-b`:
```
python csi_live_server.py -p COM9 --port-b COM6 -b 115200
```
The dashboard's "Nodes" card shows both live; the hero prediction is the two
nodes' states combined with OR (MOVING if either node says MOVING). For the
coverage benefit to actually matter, place the two boards apart from each
other (e.g. diagonally opposite corners of the room), not side by side.

Because the nodes are OR'd, one node throwing false alarms makes the whole
system throw them. Each node tile has a **Mute** button that withdraws that
node's vote while leaving it streaming and visible — useful when one board is
in a bad spot or a noisy patch. It's manual on purpose: measured on this
project's own recordings, the noise reading can't reliably predict which node
will misbehave (two sessions with near-identical noise floors had 0.7% and
26% false-alarm rates), so an automatic version would silence good nodes too.

Only one program can hold a given serial port at a time — close the IDF
monitor or the collector before running a live inference tool.

---

## Known limitations

- **Only 2 rooms tested.** Cross-room generalization has been tested once
  (82.5% held out, 94.25% after folding both room-2 sessions into
  training) — real evidence it transfers, but only across 2 rooms so far. A
  third, different room is the natural next test — see
  `docs/PROJECT_HISTORY.md` for the current status.
- **Off-axis blind spot — mitigated, not eliminated.** Movement more than
  ~1m off the direct line between an ESP32 and the router is detected much
  more weakly on that node — this is physics (single-link CSI sensitivity
  is strongest near the direct path), not a bug. Running two physically
  separated nodes (see `--port-b` below), combined via OR logic, covers
  each other's blind spots and has been confirmed live to improve coverage;
  each individual node still has its own.
- **Accuracy depends on the room's RF noise floor.** In this project's own
  recordings the empty-room noise floor varied ~9× between sessions (0.025 to
  0.221, measured as jitter relative to received signal). Where it is high,
  an empty room is nearly as disturbed as an occupied one and results get
  variable (85-95% held out, versus 95-99% in quiet rooms), with false
  "MOVING" readings the likelier error. The system measures this during
  calibration and reports it on the dashboard rather than degrading silently
  — see `assess_noise_floor()` in `csi_common.py`. Use `diagnose_nodes.py` to
  tell a noisy *room* apart from a badly-*receiving* board.
- **One person, one movement style.** All training data so far is one
  person's walking. Not yet tested against a different person or movement
  type — the single biggest remaining generalization gap.
- **Not true position/localization.** This detects *whether* someone is
  moving, not *where*. Real position tracking needs phase data and multiple
  synchronized antennas, which this amplitude-only, unsynchronized setup
  doesn't have — even with 2 nodes.

---

## License

MIT — see [LICENSE](LICENSE).