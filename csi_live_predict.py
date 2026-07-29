"""
Live CSI waterfall + motion energy + REAL-TIME MODEL PREDICTION.

This is csi_live_monitor.py's plot, extended to run csi_model.joblib on a
rolling window of incoming frames and show "EMPTY" / "MOVING" on screen
instead of (or alongside) the old fixed-threshold heuristic.

How prediction works:
    - CALIBRATION FIRST: leave the room for the model's calib_seconds
      (normally 10s) after data starts flowing. Those frames become the
      baseline every later window is compared AGAINST (see csi_common.py) -
      this is what makes it robust to the room's baseline having drifted
      since training, instead of comparing to a fixed absolute threshold.
    - After that, frames are buffered into a rolling window of
      WINDOW_FRAMES (normally 20 frames = 2s at 10Hz). Every new frame
      recomputes the baseline-relative features and the model predicts.
    - The raw per-window prediction is smoothed: the displayed state only
      changes once a majority of the last 5 predictions agree, so one
      noisy window can't flip "EMPTY" to "MOVING" by itself.

IMPORTANT: only one program can open the COM port at a time. Close the
ESP-IDF monitor and the collector first.

Usage:
    python csi_live_predict.py -p COM9 -b 115200

Deps:
    python -m pip install pyserial matplotlib numpy joblib
"""

import argparse
import sys
import time
from collections import deque

import joblib
import numpy as np

try:
    import serial
except ImportError:
    print("Missing dependency: pyserial  ->  python -m pip install pyserial")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    print("Missing dependency: matplotlib  ->  python -m pip install matplotlib")
    sys.exit(1)


from csi_common import (CALIB_SECONDS_DEFAULT, LABEL_NAMES, PredictionSmoother,
                         RollingCalibrator, compute_baseline, make_features, parse_line)


def main():
    ap = argparse.ArgumentParser(description="Live CSI waterfall with real-time empty/moving prediction.")
    ap.add_argument("-p", "--port", default="COM9")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--model", default="csi_model.joblib")
    ap.add_argument("--history", type=int, default=150)
    ap.add_argument("--energy-history", type=int, default=250)
    ap.add_argument("--interval", type=int, default=40)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    n_subcarriers_expected = len(bundle["amp_columns"])
    frame_hz = bundle["frame_hz"]
    window_frames = max(1, round(bundle["window_seconds"] * frame_hz))
    calib_seconds = bundle.get("calib_seconds", CALIB_SECONDS_DEFAULT)
    calib_frames = max(1, round(calib_seconds * frame_hz))
    print(f"Loaded {args.model}: {n_subcarriers_expected} subcarriers, "
          f"{window_frames}-frame window ({bundle['window_seconds']}s @ {frame_hz} Hz), "
          f"{calib_seconds}s calibration")
    print(f"IMPORTANT: leave the room for the first {calib_seconds}s once data starts flowing.")

    print(f"Opening {args.port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        print("Is another program (IDF monitor / collector) using the port?")
        sys.exit(1)
    ser.reset_input_buffer()

    state = {
        "n": None,
        "active": None,
        "waterfall": None,
        "energy": deque(maxlen=args.energy_history),
        "amp_buf": deque(maxlen=window_frames),
        "rssi_buf": deque(maxlen=window_frames),
        "calib_amp": [],
        "calib_rssi": [],
        "baseline": None,
        "calib": [],
        "clim": None,
    }
    smoother = PredictionSmoother(size=5)
    roller = RollingCalibrator(calib_frames)
    last_recalibrated_at = [None]  # mutable box so update() can write to it

    plt.rcParams["figure.dpi"] = 90
    fig, (ax_wf, ax_en) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Live CSI — real-time empty/moving prediction", fontsize=12)

    im = None
    (energy_line,) = ax_en.plot([], [], lw=1.5, color="#D85A30")
    ax_en.set_ylabel("motion energy")
    ax_en.set_xlabel("recent frames")
    ax_en.set_xlim(0, args.energy_history)
    ax_en.set_ylim(0, 6)
    ax_en.grid(True, alpha=0.3)
    status_text = ax_wf.text(
        0.01, 1.04, "waiting for data...", transform=ax_wf.transAxes,
        fontsize=14, fontweight="bold"
    )
    hint_text = ax_wf.text(
        0.99, 1.04, "press 'r' to force recalibration", transform=ax_wf.transAxes,
        fontsize=9, color="gray", ha="right"
    )

    def on_key(event):
        # Manual escape hatch: if it ever gets stuck reading MOVING in an
        # empty room, this is faster than restarting the whole script.
        if event.key == "r":
            state["baseline"] = None
            state["calib_amp"] = []
            state["calib_rssi"] = []
            smoother.reset()
            roller.reset()
            print("\nForced recalibration - leave the room...\n")
    fig.canvas.mpl_connect("key_press_event", on_key)

    rx = {"buf": ""}

    def drain_frames():
        try:
            n = ser.in_waiting
            if n:
                rx["buf"] += ser.read(n).decode("utf-8", errors="ignore")
        except Exception:
            return []
        frames = []
        if "\n" in rx["buf"]:
            *lines, rx["buf"] = rx["buf"].split("\n")
            for ln in lines:
                r = parse_line(ln)
                if r is not None:
                    frames.append(r)
        return frames

    def init_state(amps):
        n = len(amps)
        if n != n_subcarriers_expected:
            print(f"WARNING: stream has {n} subcarriers, model expects "
                  f"{n_subcarriers_expected}. Predictions will be wrong.")
        active = np.where(np.abs(amps) > 0)[0]
        if len(active) == 0:
            active = np.arange(n)
        state["n"] = n
        state["active"] = active
        state["waterfall"] = np.zeros((args.history, len(active)))
        print(f"Locked to {n} subcarriers, {len(active)} informative.")
        print(f"CALIBRATING - leave the room for {calib_seconds:.0f}s...\n")

    def update(_):
        nonlocal im
        frames = drain_frames()
        if not frames:
            return (im, energy_line, status_text) if im else ()

        if state["n"] is None:
            init_state(frames[0][1])

        wf = state["waterfall"]
        active = state["active"]
        prediction_display = None

        for rssi, amps in frames:
            if len(amps) != state["n"]:
                continue
            row = amps[active]
            wf[:-1] = wf[1:]
            wf[-1] = row

            if state["clim"] is None:
                state["calib"].append(row)
                if len(state["calib"]) >= 30:
                    stacked = np.concatenate(state["calib"])
                    state["clim"] = (
                        float(np.percentile(stacked, 2)),
                        float(np.percentile(stacked, 98)),
                    )

            # ---- Calibration phase: buffer confirmed-empty frames first ----
            if state["baseline"] is None:
                state["calib_amp"].append(amps)
                state["calib_rssi"].append(rssi)
                if len(state["calib_amp"]) >= calib_frames:
                    state["baseline"] = compute_baseline(state["calib_amp"], state["calib_rssi"])
                    roller.set_floor_reference(state["baseline"])
                    print("Calibrated. Predicting now.\n")
                continue  # calibration frames don't get scored or feed motion energy

            state["amp_buf"].append(amps)
            state["rssi_buf"].append(rssi)

            if len(state["amp_buf"]) >= 2:
                energy = float(np.mean(np.abs(state["amp_buf"][-1] - state["amp_buf"][-2])))
                state["energy"].append(energy)

            if len(state["amp_buf"]) == window_frames:
                feat = make_features(list(state["amp_buf"]), list(state["rssi_buf"]), state["baseline"])
                pred = clf.predict(feat)[0]
                confirmed_label, vote_fraction = smoother.update(int(pred))
                if confirmed_label is not None:
                    prediction_display = (LABEL_NAMES.get(confirmed_label, str(confirmed_label)), vote_fraction)

                # Mirrors train_model.py's per-empty-block recalibration:
                # once confirmed empty for a full calib_frames stretch,
                # quietly refresh the baseline instead of comparing against
                # the startup snapshot forever.
                new_baseline = roller.observe(confirmed_label, state["amp_buf"][-1], state["rssi_buf"][-1],
                                               state["baseline"])
                if new_baseline is not None:
                    state["baseline"] = new_baseline
                    last_recalibrated_at[0] = time.monotonic()
                    print(f"Recalibrated (rolling, {calib_frames} confirmed-empty frames).")

        if im is None:
            im = ax_wf.imshow(
                wf, aspect="auto", origin="lower", cmap="viridis",
                interpolation="nearest", animated=True,
            )
            ax_wf.set_ylabel("time (older -> newer)")
            ax_wf.set_xlabel("informative subcarrier")
            fig.colorbar(im, ax=ax_wf, label="amplitude")
        else:
            im.set_data(wf)
            if state["clim"] is not None:
                im.set_clim(*state["clim"])

        e = state["energy"]
        if e:
            energy_line.set_data(range(len(e)), e)
            top = max(e)
            cur_top = ax_en.get_ylim()[1]
            if top > cur_top or top < cur_top * 0.4:
                ax_en.set_ylim(0, max(top * 1.3, 1.0))

        if state["baseline"] is None:
            remaining = max(0, calib_frames - len(state["calib_amp"]))
            status_text.set_text(f"CALIBRATING - leave the room... {remaining / frame_hz:.0f}s")
            status_text.set_color("darkorange")
        elif prediction_display is not None:
            label, conf = prediction_display
            recal_note = ""
            if last_recalibrated_at[0] is not None and time.monotonic() - last_recalibrated_at[0] < 4:
                recal_note = "  [recalibrated]"
            status_text.set_text(f"{label}  ({conf * 100:.0f}%){recal_note}")
            status_text.set_color("crimson" if label == "MOVING" else "green")
        elif len(state["amp_buf"]) < window_frames:
            status_text.set_text(f"warming up... {len(state['amp_buf'])}/{window_frames}")
            status_text.set_color("gray")

        return im, energy_line, status_text

    ani = FuncAnimation(
        fig, update, interval=args.interval, blit=True, cache_frame_data=False
    )

    try:
        plt.tight_layout()
        plt.show()
    finally:
        ser.close()
        print("\nPort closed.")


if __name__ == "__main__":
    main()
