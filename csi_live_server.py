"""
Serial -> WebSocket bridge for the JS/HTML live dashboard.

Reads the ESP32 CSI stream, runs csi_model.joblib on a rolling window (same
parsing and feature math as csi_live_predict.py, imported not duplicated),
and pushes JSON frames to any browser tab connected on ws://localhost:8765.

Run this first, then open csi_dashboard.html in a browser.

Usage:
    python csi_live_server.py -p COM9 -b 115200

Deps:
    python -m pip install pyserial joblib scikit-learn websockets numpy
"""

import argparse
import asyncio
import json
import queue
import sys
import threading
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
    import websockets
except ImportError:
    print("Missing dependency: websockets  ->  python -m pip install websockets")
    sys.exit(1)

from csi_common import (CALIB_SECONDS_DEFAULT, LABEL_NAMES, PredictionSmoother,
                         RollingCalibrator, compute_baseline, make_features, parse_line)


def serial_reader_thread(port, baud, out_queue, stop_event):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"[serial] Could not open {port}: {e}", flush=True)
        out_queue.put({"type": "error", "message": f"Could not open {port}: {e}"})
        return
    print(f"[serial] {port} opened @ {baud}. Waiting for data "
          f"(ESP32 usually resets on port-open, so this can take ~10-15s: "
          f"boot + Wi-Fi connect + the firmware's built-in delay before CSI starts)...",
          flush=True)
    ser.reset_input_buffer()
    buf = ""
    total_bytes = 0
    total_lines = 0
    parsed_lines = 0
    last_report = time.monotonic()
    try:
        while not stop_event.is_set():
            try:
                n = ser.in_waiting
                data = ser.read(n if n else 1)
            except Exception as e:
                print(f"[serial] read error: {e}", flush=True)
                continue
            if data:
                total_bytes += len(data)
                buf += data.decode("utf-8", errors="ignore")
            if "\n" in buf:
                *lines, buf = buf.split("\n")
                for ln in lines:
                    total_lines += 1
                    r = parse_line(ln)
                    if r is not None:
                        parsed_lines += 1
                        out_queue.put({"type": "frame", "rssi": r[0], "amps": r[1]})
                    elif total_lines <= 5 or parsed_lines == 0:
                        # Show what the device IS sending (boot logs, Wi-Fi
                        # status, etc.) until we've seen at least one good frame.
                        print(f"[serial] non-CSI line: {ln.strip()[:120]}", flush=True)

            now = time.monotonic()
            if now - last_report > 3:
                last_report = now
                print(f"[serial] {total_bytes} bytes, {total_lines} lines, "
                      f"{parsed_lines} parsed as CSI so far", flush=True)
                if total_bytes == 0:
                    print("[serial] Zero bytes received - wrong COM port, "
                          "device not powered, or nothing else transmitting.", flush=True)
                    out_queue.put({"type": "warning",
                        "message": "No serial data received at all - check the port/device."})
    finally:
        ser.close()


async def broadcast(clients, msg, last_status=None):
    # Remember non-frame status messages so a client that connects AFTER
    # this was sent still finds out (frames are too frequent to replay).
    if last_status is not None and msg.get("type") != "frame":
        last_status[msg["type"]] = msg
    if not clients:
        return
    data = json.dumps(msg)
    dead = []
    # Snapshot before iterating: `await ws.send()` yields control back to the
    # event loop, and a client disconnecting mid-broadcast (handler()'s
    # `clients.discard(ws)`) would otherwise mutate `clients` while this
    # loop is still walking it.
    for ws in list(clients):
        try:
            await ws.send(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def pump(out_queue, clients, model_bundle, last_status, control):
    clf = model_bundle["model"]
    n_expected = len(model_bundle["amp_columns"])
    frame_hz = model_bundle["frame_hz"]
    window_frames = max(1, round(model_bundle["window_seconds"] * frame_hz))
    calib_seconds = model_bundle.get("calib_seconds", CALIB_SECONDS_DEFAULT)
    calib_frames = max(1, round(calib_seconds * frame_hz))

    amp_buf = deque(maxlen=window_frames)
    rssi_buf = deque(maxlen=window_frames)
    calib_amp_buf = []
    calib_rssi_buf = []
    baseline = None
    smoother = PredictionSmoother(size=5)
    roller = RollingCalibrator(calib_frames)

    n_subcarriers = None
    active_idx = None
    warned_mismatch = False

    while True:
        if control.get("recalibrate"):
            control["recalibrate"] = False
            baseline = None
            calib_amp_buf, calib_rssi_buf = [], []
            amp_buf.clear()
            rssi_buf.clear()
            smoother.reset()
            roller.reset()
            print("[predict] Forced recalibration requested - restarting calibration phase.", flush=True)

        try:
            item = out_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        if item["type"] in ("error", "warning"):
            await broadcast(clients, item, last_status)
            continue

        rssi, amps = item["rssi"], item["amps"]

        if n_subcarriers is None:
            n_subcarriers = len(amps)
            active_idx = np.where(np.abs(amps) > 0)[0]
            if len(active_idx) == 0:
                active_idx = np.arange(n_subcarriers)
            await broadcast(clients, {"type": "init", "n_subcarriers": int(n_subcarriers),
                                       "n_active": int(len(active_idx))}, last_status)
        if len(amps) != n_subcarriers:
            continue
        if n_subcarriers != n_expected and not warned_mismatch:
            warned_mismatch = True
            await broadcast(clients, {"type": "warning",
                "message": f"Stream has {n_subcarriers} subcarriers, model expects {n_expected}."}, last_status)

        # ---- Calibration phase: leave the room, buffer confirmed-empty
        # frames, compute the baseline every window is scored against. ----
        if baseline is None:
            calib_amp_buf.append(amps)
            calib_rssi_buf.append(rssi)
            remaining = max(0, calib_frames - len(calib_amp_buf))
            await broadcast(clients, {
                "type": "calibrating",
                "buffered": len(calib_amp_buf),
                "calib_frames": calib_frames,
                "remaining_seconds": remaining / frame_hz,
            })
            if len(calib_amp_buf) >= calib_frames:
                baseline = compute_baseline(calib_amp_buf, calib_rssi_buf)
                roller.set_floor_reference(baseline)
                await broadcast(clients, {"type": "calibrated",
                    "at_unix_ms": int(time.time() * 1000)}, last_status)
                print(f"[predict] Calibrated on {len(calib_amp_buf)} frames.", flush=True)
            continue  # calibration frames don't get scored

        amp_buf.append(amps)
        rssi_buf.append(rssi)

        energy = None
        if len(amp_buf) >= 2:
            energy = float(np.mean(np.abs(amp_buf[-1] - amp_buf[-2])))

        prediction, confidence = None, None
        raw_prediction, raw_confidence = None, None
        if len(amp_buf) == window_frames:
            feat = make_features(list(amp_buf), list(rssi_buf), baseline)
            pred = clf.predict(feat)[0]
            proba = clf.predict_proba(feat)[0]
            raw_confidence = float(proba[list(clf.classes_).index(pred)])
            raw_prediction = LABEL_NAMES.get(int(pred), str(pred))

            confirmed_label, vote_fraction = smoother.update(int(pred))
            if confirmed_label is not None:
                prediction = LABEL_NAMES.get(confirmed_label, str(confirmed_label))
                confidence = vote_fraction

            # Mirrors train_model.py's per-empty-block recalibration: once
            # confirmed empty for a full calib_frames stretch, quietly
            # refresh the baseline instead of comparing against startup
            # forever. Fed the LATEST raw frame each call, so the buffer
            # it accumulates is the most recent calib_frames of clean data.
            new_baseline = roller.observe(confirmed_label, amp_buf[-1], rssi_buf[-1], baseline)
            if new_baseline is not None:
                baseline = new_baseline
                await broadcast(clients, {"type": "recalibrated",
                    "at_unix_ms": int(time.time() * 1000)}, last_status)
                print(f"[predict] Recalibrated (rolling, {calib_frames} confirmed-empty frames).",
                      flush=True)

        await broadcast(clients, {
            "type": "frame",
            "row": amps[active_idx].tolist(),
            "rssi": rssi,
            "energy": energy,
            "prediction": prediction,          # smoothed/confirmed (majority of last 5)
            "confidence": confidence,           # vote fraction behind that
            "raw_prediction": raw_prediction,   # this window alone, unsmoothed
            "raw_confidence": raw_confidence,   # model's own probability for it
            "buffered": len(amp_buf),
            "window_frames": window_frames,
        })  # frames aren't stored in last_status - too frequent to replay


async def main_async(args, model_bundle):
    out_queue = queue.Queue()
    stop_event = threading.Event()
    clients = set()
    last_status = {}  # type -> most recent message of that type (error/warning/init)
    control = {"recalibrate": False}  # dashboard -> pump() one-way signal

    reader = threading.Thread(
        target=serial_reader_thread,
        args=(args.port, args.baud, out_queue, stop_event),
        daemon=True,
    )
    reader.start()

    async def handler(ws):
        clients.add(ws)
        print(f"Client connected ({len(clients)} total)")
        # Replay whatever status we already know, in case it happened
        # before this client connected (frames are live-only, not replayed).
        for msg in last_status.values():
            try:
                await ws.send(json.dumps(msg))
            except Exception:
                pass
        try:
            # async for (instead of wait_closed) so the dashboard can send
            # commands back - currently just a manual "recalibrate" request,
            # for when the system gets stuck reading MOVING in an empty
            # room and needs a way out that doesn't require restarting the
            # whole server.
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("type") == "recalibrate":
                    control["recalibrate"] = True
                    print("[server] Recalibration requested by a client.", flush=True)
        finally:
            clients.discard(ws)
            print(f"Client disconnected ({len(clients)} total)")

    async with websockets.serve(handler, "localhost", args.ws_port):
        print(f"WebSocket server on ws://localhost:{args.ws_port}")
        print("Open csi_dashboard.html in a browser to view.")
        await pump(out_queue, clients, model_bundle, last_status, control)


def main():
    ap = argparse.ArgumentParser(description="Serial -> WebSocket bridge with live model prediction.")
    ap.add_argument("-p", "--port", default="COM9")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--model", default="csi_model.joblib")
    ap.add_argument("--ws-port", type=int, default=8765)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    calib_seconds = bundle.get("calib_seconds", CALIB_SECONDS_DEFAULT)
    print(f"Loaded {args.model}: {len(bundle['amp_columns'])} subcarriers, "
          f"{bundle['window_seconds']}s window @ {bundle['frame_hz']} Hz, "
          f"{calib_seconds}s calibration")
    print(f"IMPORTANT: leave the room for the first {calib_seconds}s after "
          f"data starts flowing - that's the calibration window.")

    try:
        asyncio.run(main_async(args, bundle))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
