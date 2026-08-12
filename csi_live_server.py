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
                         RollingCalibrator, assess_noise_floor, baseline_noise_stats,
                         compute_baseline, make_features, parse_line)


RECONNECT_SECONDS = 3.0   # wait between attempts to (re)open a serial port


def serial_reader_thread(port, baud, out_queue, stop_event, node_id):
    """Owns one node's serial port for the life of the process.

    Wrapped in a reconnect loop: a board that is unplugged, browns out, or
    resets mid-run makes every subsequent read raise, and an earlier version
    caught that exception and immediately `continue`d - a tight busy-loop
    that pinned a core and flooded the console with the same error thousands
    of times a second. Now a failed open or a failed read backs off for
    RECONNECT_SECONDS and retries the port, so unplugging a board degrades
    to "that node reconnects when you plug it back in" instead of taking the
    machine down. This matters most for exactly the unattended long-run case
    the single-node version was never tested under."""
    announced_open = False
    while not stop_event.is_set():
        try:
            ser = serial.Serial(port, baud, timeout=0.1)
        except serial.SerialException as e:
            if not announced_open:
                # Only report the first failure of a streak, otherwise a port
                # that is simply absent would emit a message every 3s forever.
                print(f"[serial:{node_id}] Could not open {port}: {e} "
                      f"(retrying every {RECONNECT_SECONDS:.0f}s)", flush=True)
                out_queue.put({"type": "error", "node": node_id,
                               "message": f"Could not open {port}: {e}"})
                announced_open = True
            stop_event.wait(RECONNECT_SECONDS)
            continue

        announced_open = False
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
                    print(f"[serial:{node_id}] read error: {e} - reconnecting in "
                          f"{RECONNECT_SECONDS:.0f}s", flush=True)
                    out_queue.put({"type": "warning", "node": node_id,
                        "message": f"Lost the serial connection ({e}). Reconnecting..."})
                    break  # drop out to the reconnect loop, don't spin here
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
            try:
                ser.close()
            except Exception:
                pass
        stop_event.wait(RECONNECT_SECONDS)


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


OFFLINE = "OFFLINE"       # a node that has failed or gone silent
NODE_STALE_SECONDS = 20.0  # no frames for this long -> treat the node as offline
SUBCARRIER_RELATCH_FRAMES = 20  # sustained wrong-length frames before re-latching


def compute_combined(combined_state):
    """OR logic: any node confirming MOVING is immediate MOVING (don't wait
    for a still-calibrating second node to catch what the first already
    caught). Only reports EMPTY once every participating node has confirmed
    something and none of them is MOVING. None means "still warming up" -
    genuinely unknown yet, not "probably empty".

    OFFLINE nodes are excluded from the vote entirely. Without that, a node
    that never came up (wrong COM port, board unplugged, or - the common one
    in this project - the IDF monitor still holding the port) sits at None
    forever, and since None is not "EMPTY", the `all(...)` test below can
    never pass: the room would show MOVING correctly but could NEVER return
    to EMPTY, leaving the dashboard stuck on "warming up" while the healthy
    node worked perfectly. A node that isn't reporting must not get a vote;
    it must not silently veto the nodes that are.

    If EVERY node is offline the result is None, not EMPTY - no data means
    unknown, and an empty room must never be *inferred* from the absence of
    working sensors."""
    values = [v for v in combined_state.values() if v != OFFLINE]
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
    wrong_length_streak = 0
    last_frame_at = None
    marked_offline = False

    async def broadcast_combined_if_changed():
        overall = compute_combined(combined_state)
        if overall != last_combined_broadcast[0]:
            last_combined_broadcast[0] = overall
            await broadcast(clients, {"type": "combined", "prediction": overall,
                                       "nodes": dict(combined_state)}, last_status)

    async def set_offline(reason):
        """Take this node out of the combined vote (see compute_combined)."""
        nonlocal marked_offline
        if marked_offline:
            return
        marked_offline = True
        combined_state[node_id] = OFFLINE
        await broadcast_combined_if_changed()
        print(f"[predict:{node_id}] Marked OFFLINE: {reason}", flush=True)

    async def clear_offline():
        nonlocal marked_offline
        if not marked_offline:
            return
        marked_offline = False
        combined_state[node_id] = None  # back to "warming up", not to a stale label
        await broadcast_combined_if_changed()
        print(f"[predict:{node_id}] Back online - recalibrating.", flush=True)

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
            marked_offline = False
            await broadcast_combined_if_changed()
            print(f"[predict:{node_id}] Forced recalibration requested - restarting calibration phase.", flush=True)

        try:
            item = out_queue.get_nowait()
        except queue.Empty:
            # A node that has stopped producing frames (board unplugged, reset,
            # Wi-Fi dropped so the ping trigger stopped) must stop voting, or
            # its last known state would freeze into the combined result
            # forever. This branch runs continuously even with no data, so it
            # is where staleness gets noticed.
            if (last_frame_at is not None and not marked_offline
                    and time.monotonic() - last_frame_at > NODE_STALE_SECONDS):
                await set_offline(f"no frames for {NODE_STALE_SECONDS:.0f}s")
                await broadcast(clients, {"type": "warning", "node": node_id,
                    "message": f"No data for {NODE_STALE_SECONDS:.0f}s - this node is "
                               f"no longer counted until it starts streaming again."},
                    last_status)
            await asyncio.sleep(0.01)
            continue

        if item["type"] in ("error", "warning"):
            await broadcast(clients, {**item, "node": node_id}, last_status)
            if item["type"] == "error":
                # Port never opened: this node cannot vote at all.
                await set_offline(item.get("message", "serial error"))
            continue

        rssi, amps = item["rssi"], item["amps"]
        last_frame_at = time.monotonic()
        if marked_offline:
            # Data is flowing again after a reconnect. Everything downstream
            # (baseline, window, smoother) refers to a stream that has since
            # been interrupted, so start this node over from calibration
            # rather than resuming against a stale baseline.
            baseline = None
            calib_amp_buf, calib_rssi_buf = [], []
            amp_buf.clear()
            rssi_buf.clear()
            smoother.reset()
            roller.reset()
            await clear_offline()

        if n_subcarriers is None:
            n_subcarriers = len(amps)
            active_idx = np.where(np.abs(amps) > 0)[0]
            if len(active_idx) == 0:
                active_idx = np.arange(n_subcarriers)
            await broadcast(clients, {"type": "init", "node": node_id,
                                       "n_subcarriers": int(n_subcarriers),
                                       "n_active": int(len(active_idx))}, last_status)
        if len(amps) != n_subcarriers:
            # One-off wrong-length frames are line corruption - drop them. But
            # a SUSTAINED change is the AP genuinely renegotiating its channel
            # width mid-session (20MHz/128 <-> 40MHz/192), and the count latched
            # from the first frame is now simply wrong. Previously every frame
            # after such a switch failed this check and was dropped forever:
            # the node went permanently, silently dead with no warning at all.
            # Re-latch instead, and restart calibration since the whole feature
            # layout just changed underneath us.
            wrong_length_streak += 1
            if wrong_length_streak < SUBCARRIER_RELATCH_FRAMES:
                continue
            print(f"[predict:{node_id}] Subcarrier count changed {n_subcarriers} -> "
                  f"{len(amps)} for {wrong_length_streak} frames; re-latching.", flush=True)
            n_subcarriers = len(amps)
            active_idx = np.where(np.abs(amps) > 0)[0]
            if len(active_idx) == 0:
                active_idx = np.arange(n_subcarriers)
            baseline = None
            calib_amp_buf, calib_rssi_buf = [], []
            amp_buf.clear()
            rssi_buf.clear()
            smoother.reset()
            roller.reset()
            warned_mismatch = False  # let the new count warn on its own merits
            combined_state[node_id] = None
            await broadcast_combined_if_changed()
            await broadcast(clients, {"type": "init", "node": node_id,
                                       "n_subcarriers": int(n_subcarriers),
                                       "n_active": int(len(active_idx))}, last_status)
        wrong_length_streak = 0
        subcarrier_mismatch = n_subcarriers != n_expected
        if not subcarrier_mismatch and warned_mismatch:
            # Resolved (e.g. the AP went back to 20MHz). Clear the banner
            # instead of leaving a stale warning on screen forever.
            warned_mismatch = False
            await broadcast(clients, {"type": "warning", "node": node_id,
                "message": ""}, last_status)
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
                # The calibration block IS a measurement of the empty room's
                # noise floor, and how well this system can possibly work today
                # depends on it far more than on the model. Report it instead of
                # discarding it: a loud room degrades accuracy from ~96% to ~90%
                # with 4x the false alarms, and previously that happened
                # silently - the operator just saw a flakier system.
                absolute, relative, amp_scale = baseline_noise_stats(baseline)
                level, headline, detail = assess_noise_floor(relative)
                await broadcast(clients, {"type": "calibrated", "node": node_id,
                    "at_unix_ms": int(time.time() * 1000),
                    # relative is the comparable-between-nodes figure; the raw
                    # one and the amplitude scale ride along so the dashboard
                    # can explain WHY two nodes differ (weak reception vs noise)
                    "noise_floor": float(relative),
                    "noise_raw": float(absolute),
                    "amp_scale": float(amp_scale),
                    "noise_level": level,
                    "noise_headline": headline,
                    "noise_detail": detail}, last_status)
                print(f"[predict:{node_id}] Calibrated on {len(calib_amp_buf)} frames. "
                      f"{headline}", flush=True)
                if level == "loud":
                    print(f"[predict:{node_id}] WARNING: {detail}", flush=True)
                    await broadcast(clients, {"type": "warning", "node": node_id,
                        "message": f"{headline} - {detail}"}, last_status)
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
                absolute, relative, amp_scale = baseline_noise_stats(baseline)
                level, headline, _ = assess_noise_floor(relative)
                await broadcast(clients, {"type": "recalibrated", "node": node_id,
                    "at_unix_ms": int(time.time() * 1000),
                    "noise_floor": float(relative),
                    "noise_raw": float(absolute),
                    "amp_scale": float(amp_scale),
                    "noise_level": level,
                    "noise_headline": headline}, last_status)
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
    except OSError as e:
        # Almost always an already-running copy of this server still holding
        # the port - the same "something else already has it" confusion the
        # COM ports cause, so say so plainly instead of printing a raw
        # socket traceback.
        if getattr(e, "errno", None) in (48, 98, 10048):
            print(f"\nPort {args.ws_port} is already in use - another copy of this "
                  f"server is probably still running.\nClose it, or start this one "
                  f"with a different port: --ws-port {args.ws_port + 1}\n"
                  f"(the dashboard's WebSocket URL must match).")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
