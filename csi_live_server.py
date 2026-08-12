"""
Serial -> WebSocket bridge for the JS/HTML live dashboard.

Reads the ESP32 CSI stream, runs csi_model.joblib on a rolling window (same
parsing and feature math as csi_live_predict.py, imported not duplicated),
and pushes JSON frames to any browser tab connected on ws://localhost:8765.

Supports one or two independent nodes. Each node is a completely separate
ESP32 with its own serial port, its own calibration baseline, its own
rolling window, and its own confirmed prediction - the two never share
state. With two nodes, their confirmed predictions are combined with OR
logic ("either node confirms MOVING -> the room is MOVING") and broadcast
as a third "combined" message, alongside each node's own status. This is
deliberately NOT a shared/synchronized feature vector across nodes: two
independent single-node pipelines, OR'd together, directly attacks the
off-axis blind spot (a person off-axis for one node's ESP32<->router line
can still be on-axis for the other's) without needing any new training
data or any timestamp synchronization between the two boards.

Run this first, then open csi_dashboard.html in a browser.

Usage:
    python csi_live_server.py -p COM9 -b 115200
    python csi_live_server.py -p COM9 --port-b COM10 -b 115200   # two nodes

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


def serial_reader_thread(port, baud, out_queue, stop_event, node_id):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"[serial:{node_id}] Could not open {port}: {e}", flush=True)
        out_queue.put({"type": "error", "node": node_id, "message": f"Could not open {port}: {e}"})
        return
    print(f"[serial:{node_id}] {port} opened @ {baud}. Waiting for data "
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
                print(f"[serial:{node_id}] read error: {e}", flush=True)
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
                        print(f"[serial:{node_id}] non-CSI line: {ln.strip()[:120]}", flush=True)

            now = time.monotonic()
            if now - last_report > 3:
                last_report = now
                print(f"[serial:{node_id}] {total_bytes} bytes, {total_lines} lines, "
                      f"{parsed_lines} parsed as CSI so far", flush=True)
                if total_bytes == 0:
                    print(f"[serial:{node_id}] Zero bytes received - wrong COM port, "
                          "device not powered, or nothing else transmitting.", flush=True)
                    out_queue.put({"type": "warning", "node": node_id,
                        "message": "No serial data received at all - check the port/device."})
    finally:
        ser.close()


async def broadcast(clients, msg, last_status=None):
    # Remember non-frame status messages so a client that connects AFTER
    # this was sent still finds out (frames are too frequent to replay).
    # Keyed by type+node so node A's status can't overwrite node B's.
    if last_status is not None and msg.get("type") != "frame":
        last_status[f"{msg['type']}:{msg.get('node', '-')}"] = msg
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


def compute_combined(combined_state):
    """OR logic: any node confirming MOVING is immediate MOVING (don't wait
    for a still-calibrating second node to catch what the first already
    caught). Only reports EMPTY once every configured node has confirmed
    something and none of them is MOVING. None means "still warming up" -
    genuinely unknown yet, not "probably empty"."""
    values = list(combined_state.values())
    if "MOVING" in values:
        return "MOVING"
    if values and all(v == "EMPTY" for v in values):
        return "EMPTY"
    return None


async def pump(out_queue, clients, model_bundle, last_status, control, node_id,
                combined_state, last_combined_broadcast):
    loop = asyncio.get_running_loop()
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

    async def broadcast_combined_if_changed():
        overall = compute_combined(combined_state)
        if overall != last_combined_broadcast[0]:
            last_combined_broadcast[0] = overall
            await broadcast(clients, {"type": "combined", "prediction": overall,
                                       "nodes": dict(combined_state)}, last_status)

    while True:
        if control.get("recalibrate"):
            control["recalibrate"] = False
            baseline = None
            calib_amp_buf, calib_rssi_buf = [], []
            amp_buf.clear()
            rssi_buf.clear()
            smoother.reset()
            roller.reset()
            combined_state[node_id] = None
            await broadcast_combined_if_changed()
            print(f"[predict:{node_id}] Forced recalibration requested - restarting calibration phase.", flush=True)

        try:
            item = out_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        if item["type"] in ("error", "warning"):
            await broadcast(clients, {**item, "node": node_id}, last_status)
            continue

        rssi, amps = item["rssi"], item["amps"]

        if n_subcarriers is None:
            n_subcarriers = len(amps)
            active_idx = np.where(np.abs(amps) > 0)[0]
            if len(active_idx) == 0:
                active_idx = np.arange(n_subcarriers)
            await broadcast(clients, {"type": "init", "node": node_id,
                                       "n_subcarriers": int(n_subcarriers),
                                       "n_active": int(len(active_idx))}, last_status)
        if len(amps) != n_subcarriers:
            continue
        subcarrier_mismatch = n_subcarriers != n_expected
        if subcarrier_mismatch and not warned_mismatch:
            warned_mismatch = True
            # A mismatch here means the feature vector this node would build
            # can never match what the model was trained on (e.g. the AP
            # auto-negotiated a 40MHz channel, doubling subcarriers from 128
            # to 192) - calling predict_proba() on it doesn't just give a bad
            # answer, it raises inside sklearn (wrong feature count) and,
            # left unguarded, that exception used to propagate out of this
            # task and crash the ENTIRE server via asyncio.gather - taking
            # every other node down too. Below, prediction is skipped for
            # this node instead: it keeps streaming raw waterfall/energy
            # data and stays visibly in "warming up" rather than either
            # crashing or silently producing garbage predictions.
            await broadcast(clients, {"type": "warning", "node": node_id,
                "message": f"Stream has {n_subcarriers} subcarriers, model expects {n_expected} - "
                           f"predictions disabled for this node until it matches "
                           f"(likely the AP changed channel width; a 20MHz-only AP is required)."},
                last_status)

        # ---- Calibration phase: leave the room, buffer confirmed-empty
        # frames, compute the baseline every window is scored against. ----
        if baseline is None:
            calib_amp_buf.append(amps)
            calib_rssi_buf.append(rssi)
            remaining = max(0, calib_frames - len(calib_amp_buf))
            await broadcast(clients, {
                "type": "calibrating",
                "node": node_id,
                "buffered": len(calib_amp_buf),
                "calib_frames": calib_frames,
                "remaining_seconds": remaining / frame_hz,
            })
            if len(calib_amp_buf) >= calib_frames:
                baseline = compute_baseline(calib_amp_buf, calib_rssi_buf)
                roller.set_floor_reference(baseline)
                await broadcast(clients, {"type": "calibrated", "node": node_id,
                    "at_unix_ms": int(time.time() * 1000)}, last_status)
                print(f"[predict:{node_id}] Calibrated on {len(calib_amp_buf)} frames.", flush=True)
            continue  # calibration frames don't get scored

        amp_buf.append(amps)
        rssi_buf.append(rssi)

        energy = None
        if len(amp_buf) >= 2:
            energy = float(np.mean(np.abs(amp_buf[-1] - amp_buf[-2])))

        prediction, confidence = None, None
        raw_prediction, raw_confidence = None, None
        confirmed_label = None
        if len(amp_buf) == window_frames and not subcarrier_mismatch:
            feat = make_features(list(amp_buf), list(rssi_buf), baseline)
            # A single predict_proba() in a thread executor: one forest walk
            # instead of two (predict() + predict_proba() separately), and
            # off the event loop entirely so a slow prediction can never
            # delay WebSocket ping/pong keepalives (that stall is what was
            # producing "keepalive ping timeout" disconnects under load).
            # Sharing one loaded model across both nodes' executor calls is
            # safe: predict_proba() only reads the fitted trees, never
            # mutates them, so concurrent calls from two nodes can't race.
            proba = (await loop.run_in_executor(None, clf.predict_proba, feat))[0]
            best_idx = int(np.argmax(proba))
            pred = clf.classes_[best_idx]
            raw_confidence = float(proba[best_idx])
            raw_prediction = LABEL_NAMES.get(int(pred), str(pred))

            confirmed_label, vote_fraction = smoother.update(int(pred))
            if confirmed_label is not None:
                prediction = LABEL_NAMES.get(confirmed_label, str(confirmed_label))
                confidence = vote_fraction
                if combined_state.get(node_id) != prediction:
                    combined_state[node_id] = prediction
                    await broadcast_combined_if_changed()

            # Mirrors train_model.py's per-empty-block recalibration: once
            # confirmed empty for a full calib_frames stretch, quietly
            # refresh the baseline instead of comparing against startup
            # forever. Fed the LATEST raw frame each call, so the buffer
            # it accumulates is the most recent calib_frames of clean data.
            new_baseline = roller.observe(confirmed_label, amp_buf[-1], rssi_buf[-1], baseline)
            if new_baseline is not None:
                baseline = new_baseline
                await broadcast(clients, {"type": "recalibrated", "node": node_id,
                    "at_unix_ms": int(time.time() * 1000)}, last_status)
                print(f"[predict:{node_id}] Recalibrated (rolling, {calib_frames} confirmed-empty frames).",
                      flush=True)

        await broadcast(clients, {
            "type": "frame",
            "node": node_id,
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
    clients = set()
    last_status = {}  # "type:node" -> most recent message of that kind (error/warning/init/...)

    nodes = [("A", args.port)]
    if args.port_b:
        nodes.append(("B", args.port_b))

    combined_state = {node_id: None for node_id, _ in nodes}
    last_combined_broadcast = [None]  # mutable box, shared across pump() tasks

    stop_event = threading.Event()
    control = {node_id: {"recalibrate": False} for node_id, _ in nodes}
    tasks = []

    for node_id, port in nodes:
        out_queue = queue.Queue()
        reader = threading.Thread(
            target=serial_reader_thread,
            args=(port, args.baud, out_queue, stop_event, node_id),
            daemon=True,
        )
        reader.start()
        tasks.append(asyncio.create_task(
            pump(out_queue, clients, model_bundle, last_status, control[node_id], node_id,
                 combined_state, last_combined_broadcast)
        ))

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
            # for when a node gets stuck reading MOVING in an empty room and
            # needs a way out that doesn't require restarting the server.
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("type") == "recalibrate":
                    target = msg.get("node", "all")
                    targets = control.keys() if target in ("all", None) else [target]
                    for t in targets:
                        if t in control:
                            control[t]["recalibrate"] = True
                    print(f"[server] Recalibration requested by a client (node={target}).", flush=True)
        finally:
            clients.discard(ws)
            print(f"Client disconnected ({len(clients)} total)")

    # ping_timeout raised from the websockets default (20s) - a backgrounded
    # or minimized browser tab gets its JS timers throttled by the browser
    # and can miss a pong well within 20s without the connection actually
    # being dead; this was closing healthy-but-idle tabs with a scary
    # "keepalive ping timeout" error.
    async with websockets.serve(handler, "localhost", args.ws_port,
                                 ping_interval=20, ping_timeout=60):
        print(f"WebSocket server on ws://localhost:{args.ws_port}")
        print(f"Nodes: {', '.join(f'{n}={p}' for n, p in nodes)}")
        print("Open csi_dashboard.html in a browser to view.")
        await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser(description="Serial -> WebSocket bridge with live model prediction. "
                                              "Supports one node, or two independent nodes OR'd together.")
    ap.add_argument("-p", "--port", default="COM9", help="Serial port for node A.")
    ap.add_argument("--port-b", default=None, help="Serial port for a second, independent node (node B). "
                                                     "Omit to run single-node, exactly as before.")
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
          f"data starts flowing on EACH node - that's the calibration window.")

    try:
        asyncio.run(main_async(args, bundle))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
