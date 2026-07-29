"""
Real-time CSI monitor (fast version) for the ESP32-S3 CSI node.

Same idea as before -- waterfall + motion energy so you can confirm the channel
reacts to a person -- but tuned to run smoothly:

  * blitting: only the image and the energy line are redrawn, not the whole canvas
  * fixed color scale: calibrated once from the first frames, then locked
    (recomputing percentiles every frame was the main source of lag AND flicker)
  * buffer draining: every tick reads ALL waiting frames so the plot never lags
  * lightweight per-frame work

IMPORTANT: only one program can open COM9 at a time. Close the ESP-IDF monitor
and the collector first. Use this to VERIFY, then close it and run the collector.

Usage:
    python csi_live_monitor_fast.py -p COM9 -b 115200

Deps:
    python -m pip install pyserial matplotlib numpy

If it is STILL not smooth enough, the real upgrade is pyqtgraph (built for
real-time). See the note at the bottom of this file.
"""

import argparse
import sys
from collections import deque

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


MARKER = "CSI_AMP,"


def parse_line(line):
    i = line.find(MARKER)
    if i == -1:
        return None
    parts = line[i:].strip().split(",")
    if len(parts) < 5 or parts[0] != "CSI_AMP":
        return None
    try:
        return np.array([float(x) for x in parts[4:]])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Fast live CSI waterfall + motion energy.")
    ap.add_argument("-p", "--port", default="COM9")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--history", type=int, default=150,
                    help="Frames shown in the waterfall (smaller = lighter)")
    ap.add_argument("--energy-history", type=int, default=250)
    ap.add_argument("--interval", type=int, default=40,
                    help="Redraw interval in ms (lower = smoother, more CPU)")
    args = ap.parse_args()

    print(f"Opening {args.port} @ {args.baud} baud...")
    print("Close the IDF monitor / collector first if this fails to open.\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0)  # non-blocking
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        print("Is another program (IDF monitor / collector) using the port?")
        sys.exit(1)
    ser.reset_input_buffer()

    state = {
        "n": None,
        "active": None,
        "waterfall": None,
        "prev": None,
        "energy": deque(maxlen=args.energy_history),
        "calib": [],          # collect early frames to fix the color scale
        "clim": None,         # (vmin, vmax) once calibrated
    }

    plt.rcParams["figure.dpi"] = 90  # lower dpi = faster
    fig, (ax_wf, ax_en) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Live CSI (fast)  -  walk in and out to check it reacts", fontsize=12)

    im = None
    (energy_line,) = ax_en.plot([], [], lw=1.5, color="#D85A30")
    ax_en.set_ylabel("motion energy")
    ax_en.set_xlabel("recent frames")
    ax_en.set_xlim(0, args.energy_history)
    ax_en.set_ylim(0, 6)
    ax_en.grid(True, alpha=0.3)
    status_text = ax_wf.text(
        0.01, 1.04, "", transform=ax_wf.transAxes, fontsize=12, fontweight="bold"
    )

    # read a text buffer incrementally so we split on real line boundaries
    rx = {"buf": ""}

    def drain_frames():
        """Read everything waiting and return a list of amplitude arrays."""
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
                amps = parse_line(ln)
                if amps is not None:
                    frames.append(amps)
        return frames

    def init_state(amps):
        n = len(amps)
        active = np.where(np.abs(amps) > 0)[0]
        if len(active) == 0:
            active = np.arange(n)
        state["n"] = n
        state["active"] = active
        state["waterfall"] = np.zeros((args.history, len(active)))
        print(f"Locked to {n} subcarriers, {len(active)} informative.\n")

    def update(_):
        nonlocal im
        frames = drain_frames()
        if not frames:
            return (im, energy_line, status_text) if im else ()

        if state["n"] is None:
            init_state(frames[0])

        wf = state["waterfall"]
        active = state["active"]
        for amps in frames:
            if len(amps) != state["n"]:
                continue
            row = amps[active]
            wf[:-1] = wf[1:]      # scroll up (cheaper than np.roll for this)
            wf[-1] = row
            if state["prev"] is not None:
                state["energy"].append(float(np.mean(np.abs(row - state["prev"]))))
            state["prev"] = row

            if state["clim"] is None:
                state["calib"].append(row)
                if len(state["calib"]) >= 30:  # lock scale after ~3 s
                    stacked = np.concatenate(state["calib"])
                    state["clim"] = (
                        float(np.percentile(stacked, 2)),
                        float(np.percentile(stacked, 98)),
                    )

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
                im.set_clim(*state["clim"])   # fixed scale: no per-frame percentile

        e = state["energy"]
        if e:
            energy_line.set_data(range(len(e)), e)
            top = max(e)
            cur_top = ax_en.get_ylim()[1]
            if top > cur_top or top < cur_top * 0.4:  # rescale rarely, not every frame
                ax_en.set_ylim(0, max(top * 1.3, 1.0))

            recent = np.mean(list(e)[-8:])
            base = np.percentile(e, 20) if len(e) > 30 else recent
            if recent > base * 2.2 + 0.3:
                status_text.set_text("MOTION DETECTED")
                status_text.set_color("crimson")
            else:
                status_text.set_text("quiet / empty")
                status_text.set_color("green")

        return im, energy_line, status_text

    # blit=True is the big win: only the returned artists get redrawn
    ani = FuncAnimation(
        fig, update, interval=args.interval, blit=True, cache_frame_data=False
    )

    try:
        plt.tight_layout()
        plt.show()
    finally:
        ser.close()
        print("\nPort closed. You can now run the collector.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# If this is still not fluid enough, matplotlib is the ceiling, not your code.
# pyqtgraph renders real-time plots an order of magnitude faster. Install with:
#     python -m pip install pyqtgraph pyqt5
# and ask for the pyqtgraph version -- it uses the same parser, just a faster
# rendering backend. For a 10 Hz stream the version above should already feel
# smooth; pyqtgraph matters more if you push the sample rate up later.
# ---------------------------------------------------------------------------