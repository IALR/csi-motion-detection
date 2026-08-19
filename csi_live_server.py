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
import os
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
            # ESP32 dev boards wire the USB bridge's RTS to EN (reset) and DTR
            # to GPIO0 (boot select). Whatever state the previous program left
            # those lines in is inherited here, and an asserted RTS holds the
            # chip in RESET - so the port opens perfectly, the board runs
            # nothing, and it looks identical to "wrong COM port / device not
            # powered". Exiting `idf.py monitor` is enough to leave it that
            # way. Deasserting both, then pulsing EN, guarantees the board is
            # actually running rather than silently held down.
            ser.dtr = False
            ser.rts = True          # assert EN low = hold in reset
            time.sleep(0.05)
            ser.rts = False         # release: the board boots from here
            ser.reset_input_buffer()
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


def compute_combined(combined_state, muted=()):
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

    MUTED nodes are excluded the same way. Muting is manual and deliberate:
    because the two nodes are OR'd, one node throwing false alarms makes the
    WHOLE system throw them, and there is no reliable way to detect that
    automatically - measured on this project's own sessions, two recordings
    with near-identical noise floors (0.216 and 0.221) had false-alarm rates
    of 0.7% and 26.3%, so the noise reading cannot be used to gate a node
    without also gating good ones. The operator can see which node is
    misbehaving; this lets them act on that without restarting anything. A
    muted node keeps streaming and stays fully visible - it just stops
    voting.

    If EVERY node is offline or muted the result is None, not EMPTY - no
    data means unknown, and an empty room must never be *inferred* from the
    absence of participating sensors."""
    values = [v for k, v in combined_state.items()
              if v != OFFLINE and k not in muted]
    if "MOVING" in values:
        return "MOVING"
    if values and all(v == "EMPTY" for v in values):
        return "EMPTY"
    return None


async def push_combined(clients, combined_state, muted, last_combined_broadcast,
                         last_status, force=False):
    """Broadcast the OR-combined result when it changes.

    `force` is for changes that alter the MEANING of the result without
    necessarily changing its value - muting a node while both read EMPTY
    still leaves EMPTY, but the dashboard has to learn that one node stopped
    counting, otherwise the mute button appears to do nothing."""
    overall = compute_combined(combined_state, muted)
    if overall != last_combined_broadcast[0] or force:
        last_combined_broadcast[0] = overall
        await broadcast(clients, {"type": "combined", "prediction": overall,
                                   "nodes": dict(combined_state),
                                   "muted": sorted(muted)}, last_status)


async def pump(out_queue, clients, model_bundle, last_status, control, node_id,
                combined_state, last_combined_broadcast, muted):
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

    async def broadcast_combined_if_changed(force=False):
        await push_combined(clients, combined_state, muted,
                            last_combined_broadcast, last_status, force)

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


class AlertManager:
    """Fires an action once the room has been continuously occupied.

    Why a sustained hold rather than alerting on the first MOVING window:
    measured on this project's own out-of-fold predictions, per-window false
    alarms run about 11% in a noisy room, but requiring several CONSECUTIVE
    seconds of confirmed MOVING collapses that to roughly one spurious alert
    per recording session - and none at all across the quiet sessions. The
    hold is doing most of the work of making an alert trustworthy.

    Be honest about the limits, though: that was measured over only ~21
    minutes of empty-room recording, so "no false alerts in the quiet
    sessions" bounds the rate loosely rather than proving it is zero. In a
    loud room (noise floor above ~0.15) expect this to fire spuriously
    several times an hour.

    The cooldown exists because one person walking around otherwise produces
    a continuous stream of alerts; after firing, the manager stays quiet
    until either the cooldown elapses or the room goes EMPTY and is occupied
    again.
    """

    def __init__(self, hold_seconds, cooldown_seconds):
        self.hold = hold_seconds
        self.cooldown = cooldown_seconds
        self.moving_since = None
        self.last_fired_at = None
        self.fired_this_occupancy = False
        self.count = 0

    def update(self, combined, now):
        """Feed the combined room state. Returns the sustained duration in
        seconds when an alert should fire, else None."""
        if combined != "MOVING":
            # Any return to EMPTY (or to unknown) ends this occupancy, so the
            # next person triggers a fresh alert rather than being suppressed
            # by a cooldown started minutes ago.
            self.moving_since = None
            self.fired_this_occupancy = False
            return None

        if self.moving_since is None:
            self.moving_since = now
        held = now - self.moving_since
        if held < self.hold or self.fired_this_occupancy:
            return None
        if self.last_fired_at is not None and (now - self.last_fired_at) < self.cooldown:
            return None

        self.last_fired_at = now
        self.fired_this_occupancy = True
        self.count += 1
        return held


def _send_email(cfg, message, payload):
    """Send the alert by SMTP. Blocking - always called from an executor.

    The password is read from the environment ONLY, never from a command-line
    flag: anything passed as an argument is visible to every other process on
    the machine (Task Manager, `ps`) and gets written into shell history. An
    app password for a mail account is worth protecting from both.
    """
    import smtplib
    from email.message import EmailMessage

    password = os.environ.get("CSI_SMTP_PASS")
    if not password:
        print("[alert] email skipped: CSI_SMTP_PASS is not set in the "
              "environment. See --alert-email in --help.", flush=True)
        return

    msg = EmailMessage()
    msg["Subject"] = "CSI motion detected"
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(
        f"{message}\n\n"
        f"Held for : {payload.get('held_seconds')}s\n"
        f"At       : {payload.get('at')}\n"
        f"Nodes    : {json.dumps(payload.get('nodes', {}))}\n"
        f"Alert #  : {payload.get('alert_number')}\n\n"
        f"Sent by csi_live_server.py")

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as smtp:
            smtp.starttls()
            smtp.login(cfg["user"], password)
            smtp.send_message(msg)
        print(f"[alert] email -> {cfg['to']}", flush=True)
    except smtplib.SMTPAuthenticationError as e:
        # Show what the server actually said as well as the likely causes: an
        # earlier version asserted "you used your account password", which is
        # only one of several possibilities and sends people down the wrong
        # path when the real problem is a different account or 2FA being off.
        detail = e.smtp_error.decode(errors="replace") if e.smtp_error else ""
        print(f"[alert] email FAILED: server rejected the login "
              f"(code {e.smtp_code}).\n"
              f"         server said: {detail.strip()}\n"
              f"         For Gmail, check in this order:\n"
              f"           1. The app password must belong to THIS account "
              f"({cfg['user']}) - one generated while signed in to a different\n"
              f"              Google account will be rejected exactly like this.\n"
              f"           2. 2-Step Verification must be ON for that account, "
              f"or app passwords do not work at all.\n"
              f"           3. It must be an APP PASSWORD, not the normal "
              f"account password.\n"
              f"           4. Spaces in the 16 characters are optional; both "
              f"forms were accepted historically.", flush=True)
    except Exception as e:
        print(f"[alert] email FAILED: {e}", flush=True)


def _deliver_alert(webhook, command, message, payload, email_cfg=None):
    """Blocking delivery, run in an executor so it can never stall the event
    loop - a webhook to an unreachable host would otherwise freeze prediction
    and every connected dashboard along with it."""
    import subprocess
    import urllib.request

    if webhook:
        try:
            # Plain text body: ntfy.sh takes the message as-is, and most other
            # endpoints (Discord, IFTTT, Home Assistant, a script of your own)
            # accept either. The JSON is sent as a header so one call suits
            # both styles without needing to know which service it is.
            req = urllib.request.Request(
                webhook, data=message.encode("utf-8"), method="POST",
                headers={"Content-Type": "text/plain; charset=utf-8",
                         "Title": "CSI motion detected",
                         "X-CSI-Payload": json.dumps(payload)})
            with urllib.request.urlopen(req, timeout=10) as r:
                print(f"[alert] webhook -> HTTP {r.status}", flush=True)
        except Exception as e:
            print(f"[alert] webhook FAILED: {e}", flush=True)

    if email_cfg:
        _send_email(email_cfg, message, payload)

    if command:
        try:
            subprocess.run(command, shell=True, timeout=30,
                           env={**os.environ,
                                "CSI_MESSAGE": message,
                                "CSI_HELD_SECONDS": str(payload.get("held_seconds", "")),
                                "CSI_NODES": json.dumps(payload.get("nodes", {}))})
            print(f"[alert] ran command", flush=True)
        except Exception as e:
            print(f"[alert] command FAILED: {e}", flush=True)


async def alert_task(alerts, clients, combined_state, muted, args, last_status):
    """Polls the combined room state and fires when occupancy is sustained."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(0.25)
        combined = compute_combined(combined_state, muted)
        held = alerts.update(combined, time.monotonic())
        if held is None:
            continue

        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = (f"Motion detected - someone has been in the room for "
                   f"{held:.0f}s (since {stamp}).")
        payload = {"event": "sustained_motion", "held_seconds": round(held, 1),
                   "at": stamp, "nodes": dict(combined_state),
                   "alert_number": alerts.count}
        print(f"\n*** ALERT #{alerts.count}: {message} ***\n", flush=True)

        # Tell any open dashboard too, so the alert is visible on screen and
        # not only wherever the webhook goes.
        await broadcast(clients, {"type": "warning", "node": "-",
                                   "message": message}, last_status)

        email_cfg = None
        if args.alert_email:
            email_cfg = {"to": args.alert_email, "host": args.smtp_host,
                         "port": args.smtp_port,
                         "user": args.smtp_user or os.environ.get("CSI_SMTP_USER", "")}
        if args.alert_webhook or args.alert_command or email_cfg:
            await loop.run_in_executor(None, _deliver_alert, args.alert_webhook,
                                       args.alert_command, message, payload,
                                       email_cfg)


async def main_async(args, model_bundle):
    clients = set()
    last_status = {}  # "type:node" -> most recent message of that kind (error/warning/init/...)

    nodes = [("A", args.port)]
    if args.port_b:
        nodes.append(("B", args.port_b))

    combined_state = {node_id: None for node_id, _ in nodes}
    last_combined_broadcast = [None]  # mutable box, shared across pump() tasks
    muted = set()                      # node ids the operator has silenced

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
                 combined_state, last_combined_broadcast, muted)
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
                elif msg.get("type") == "mute":
                    target = msg.get("node")
                    if target in control:
                        if msg.get("muted"):
                            muted.add(target)
                        else:
                            muted.discard(target)
                        print(f"[server] Node {target} "
                              f"{'MUTED' if target in muted else 'unmuted'} by a client "
                              f"(it keeps streaming; it just stops voting).", flush=True)
                        # force=True: muting while both nodes read EMPTY leaves
                        # the combined value unchanged, but the dashboard still
                        # has to learn that a node stopped counting.
                        await push_combined(clients, combined_state, muted,
                                            last_combined_broadcast, last_status,
                                            force=True)
        finally:
            clients.discard(ws)
            print(f"Client disconnected ({len(clients)} total)")

    # ping_timeout raised from the websockets default (20s) - a backgrounded
    # or minimized browser tab gets its JS timers throttled by the browser
    # and can miss a pong well within 20s without the connection actually
    # being dead; this was closing healthy-but-idle tabs with a scary
    # "keepalive ping timeout" error.
    alerts = AlertManager(args.alert_seconds, args.alert_cooldown)
    if args.alert_seconds > 0:
        tasks.append(asyncio.create_task(
            alert_task(alerts, clients, combined_state, muted, args, last_status)))
        print(f"Alerting after {args.alert_seconds:.0f}s of sustained MOVING "
              f"(cooldown {args.alert_cooldown:.0f}s)"
              + (f" -> POST {args.alert_webhook}" if args.alert_webhook else "")
              + (f" -> run: {args.alert_command}" if args.alert_command else "")
              + ("" if (args.alert_webhook or args.alert_command)
                 else " -> console + dashboard only (add --alert-webhook or "
                      "--alert-command to send it somewhere)"))

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
    ap.add_argument("--alert-seconds", type=float, default=5.0,
                    help="fire an alert after this many CONSECUTIVE seconds of "
                         "confirmed MOVING (0 disables alerting). The hold is what "
                         "makes an alert trustworthy: per-window false alarms run "
                         "~11%% in a noisy room, but sustained ones are rare.")
    ap.add_argument("--alert-cooldown", type=float, default=60.0,
                    help="minimum seconds between alerts, so one person moving "
                         "around does not send a stream of them")
    ap.add_argument("--alert-webhook", default=None,
                    help="POST the alert to this URL. Works with ntfy.sh (free "
                         "phone push, no account: https://ntfy.sh/your-topic), a "
                         "Discord webhook, IFTTT, Home Assistant, or your own "
                         "endpoint. The message is the body; a JSON payload is "
                         "also sent in the X-CSI-Payload header.")
    ap.add_argument("--alert-email", default=None,
                    help="email address to alert. The SMTP password comes from the "
                         "CSI_SMTP_PASS environment variable and is never accepted "
                         "as a flag, because command-line arguments are visible to "
                         "other processes and land in shell history. For Gmail use "
                         "an APP PASSWORD, not your account password.")
    ap.add_argument("--smtp-host", default="smtp.gmail.com")
    ap.add_argument("--smtp-port", type=int, default=587)
    ap.add_argument("--smtp-user", default=None,
                    help="the sending account (defaults to $CSI_SMTP_USER)")
    ap.add_argument("--test-alert", action="store_true",
                    help="fire one alert immediately and exit, without touching the "
                         "serial port. Checks the delivery setup on its own, instead "
                         "of standing in a room waving until something happens.")
    ap.add_argument("--alert-command", default=None,
                    help="shell command to run on alert. CSI_MESSAGE, "
                         "CSI_HELD_SECONDS and CSI_NODES are set in its "
                         "environment.")
    args = ap.parse_args()

    if args.test_alert:
        # Deliberately before the model loads and before any port is opened:
        # this is purely a delivery check and must work with no hardware
        # attached and nothing else running.
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = ("TEST alert from csi_live_server.py - if you are reading this, "
                   "delivery works. No motion was detected; this was sent by "
                   "--test-alert.")
        payload = {"event": "test", "held_seconds": 0, "at": stamp,
                   "nodes": {}, "alert_number": 0}
        email_cfg = None
        if args.alert_email:
            email_cfg = {"to": args.alert_email, "host": args.smtp_host,
                         "port": args.smtp_port,
                         "user": args.smtp_user or os.environ.get("CSI_SMTP_USER", "")}
        if not (args.alert_webhook or args.alert_command or email_cfg):
            print("Nothing to test: add --alert-webhook, --alert-email or "
                  "--alert-command.")
            return
        print(f"Sending a test alert at {stamp}...")
        _deliver_alert(args.alert_webhook, args.alert_command, message, payload,
                       email_cfg)
        print("Done. If nothing arrived, the messages above say why.")
        return

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
