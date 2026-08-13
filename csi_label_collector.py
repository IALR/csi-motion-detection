"""
Auto-cycling CSI collector for the empty / moving protocol.

The cycle you asked for:

  1. On start: 5 s countdown to LEAVE the room.
  2. Records EMPTY (label 0) for 60 s automatically.
  3. Stops and waits -- press 'm' when you are back in the room and ready.
  4. On 'm': records MOVING (label 2) for 60 s.
  5. Automatically gives you 5 s to LEAVE again, then records EMPTY 60 s.
  6. Waits for 'm' again. Repeat.

So EMPTY blocks are automatic (with a leave countdown) and MOVING blocks are
started by hand -- because you must be present to move, but absent to be "empty".

Frames during the 5 s leave window are written with settling=1 so you can drop
them at training time. Frames while waiting for 'm' are not recorded.

Requires the improved firmware (emits rssi). Run with --rssi-field.

ZONES (optional, for position detection)
----------------------------------------
Pressing '1' or '2' instead of 'm' tags that moving block with a zone, written
to a separate `zone` column. The `label` column stays 0/2 regardless, so a
zone recording is ALSO ordinary training data for the binary motion model -
train_model.py reads it unchanged and never looks at `zone`.

For zone work, record BOTH boards at once with --port-b: the two nodes must
cover the same events from different angles to be combinable. Each node then
writes into its own node_a/ and node_b/ subfolder. Alternate the zones WITHIN
each session (zone 1, then zone 2, then zone 1...) - recording one zone per
session would confound zone with session and produce a meaningless score.

Usage:
    python csi_label_collector.py -p COM9 -b 921600 --subcarriers 128 --rssi-field
    python csi_label_collector.py -p COM9 --port-b COM6 -b 921600 \
        --subcarriers 128 --rssi-field -o zone_room1_part_1_data

Controls while running:
    m  -> start a plain MOVING block (zone 0)
    1  -> start a MOVING block tagged ZONE 1
    2  -> start a MOVING block tagged ZONE 2
    q  -> stop and save
"""

import argparse
import csv
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime

try:
    import serial
except ImportError:
    print("Missing dependency: pyserial  ->  python -m pip install pyserial")
    sys.exit(1)


# ---- cross-platform non-blocking key reader ----
class KeyReader:
    def __init__(self):
        self.mode = None
        try:
            import msvcrt  # noqa: F401
            self.mode = "windows"
        except ImportError:
            try:
                import termios, tty  # noqa: F401
                if sys.stdin.isatty():
                    self.mode = "posix"
                    self._termios, self._tty = termios, tty
            except Exception:
                pass
        if self.mode is None:
            print("WARNING: no interactive keyboard; you can't press 'm' or 'q'.\n")

    def __enter__(self):
        if self.mode == "posix":
            self._old = self._termios.tcgetattr(sys.stdin)
            self._tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self.mode == "posix":
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old)

    def get(self):
        if self.mode == "windows":
            import msvcrt
            return msvcrt.getwch().lower() if msvcrt.kbhit() else None
        if self.mode == "posix":
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        return None


# `zone` sits alongside `label`, it does NOT replace it: label stays 0/2 exactly
# as before, so every recording made here remains valid training data for the
# binary motion model with no changes to train_model.py. zone is 0 for empty
# blocks and for plain 'm' moving blocks, and 1/2 when a zone key was used.
# Older CSVs predate this column; anything reading zone must cope with it being
# absent (train_model.py never reads it, so it is unaffected).
BASE_COLUMNS = ["session_id", "host_unix_us", "timestamp_us", "label", "zone",
                "esp_label", "settling", "rssi", "num_subcarriers"]


def open_csv(path, amplitude_count):
    is_new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if is_new:
        header = list(BASE_COLUMNS) + [f"amp_{i}" for i in range(amplitude_count)]
        writer.writerow(header)
        handle.flush()
    return handle, writer


MARKER = "CSI_AMP,"


def parse_line(line, rssi_field):
    """Firmware format: CSI_AMP,<ts_us>,<label>,<num_subcarriers>,<rssi>,<amp0>,...

    Values stay as STRINGS here (unlike csi_common.parse_line, which returns
    floats) because they are written to CSV verbatim - re-parsing them to float
    and back would lose the firmware's exact integer formatting.

    The declared subcarrier count is checked against how many amplitudes
    actually arrived. A line mangled in transit can still contain the marker
    and still split into enough fields to look valid, just short - and this is
    the DATA COLLECTOR, so an unchecked short row would be written into the
    training set permanently."""
    i = line.find(MARKER)
    if i == -1:
        return None
    p = line[i:].strip().split(",")
    if len(p) < 5 or p[0] != "CSI_AMP":
        return None
    ts, esp_label, num = p[1], p[2], p[3]
    if rssi_field:
        rssi, amps = p[4], p[5:]
    else:
        rssi, amps = "", p[4:]
    try:
        if int(num) != len(amps):
            return None
    except ValueError:
        return None
    return ts, esp_label, num, rssi, amps


class NodeRecorder:
    """One ESP32's serial connection plus its CSV writers.

    Exists so a single state machine can drive one OR two boards: zone
    detection benefits from both nodes' viewpoints, and the two recordings
    must cover the SAME events to be combinable, which means one shared
    protocol driving both ports rather than two separate runs.

    In single-node mode `out_dir` is the session folder itself, so the output
    is byte-identical to before this class existed. With --port-b each node
    gets its own subfolder.
    """

    def __init__(self, node_id, port, baud, out_dir, session_id,
                 amplitude_count, rssi_field):
        self.node_id = node_id
        self.out_dir = out_dir
        self.session_id = session_id
        self.rssi_field = rssi_field
        self.amplitude_count = amplitude_count
        os.makedirs(out_dir, exist_ok=True)
        self.ser = serial.Serial(port, baud, timeout=0)
        self.ser.reset_input_buffer()
        self._rx = ""
        self._all_path = os.path.join(out_dir, "all_csi_data.csv")
        self._all_file = self._all_writer = None
        self._handles, self._writers = {}, {}
        self.counts = defaultdict(int)
        self.rejected = defaultdict(int)

    def drain(self):
        try:
            k = self.ser.in_waiting
            if k:
                self._rx += self.ser.read(k).decode("utf-8", errors="ignore")
        except Exception:
            return []
        out = []
        if "\n" in self._rx:
            *lines, self._rx = self._rx.split("\n")
            for ln in lines:
                r = parse_line(ln, self.rssi_field)
                if r:
                    out.append(r)
        return out

    def tell_esp(self, ch):
        try:
            self.ser.write(ch.encode("ascii"))
            self.ser.flush()
        except Exception:
            pass

    def _writer_for(self, label):
        if self._all_writer is None:
            self._all_file, self._all_writer = open_csv(self._all_path,
                                                         self.amplitude_count)
        if label not in self._handles:
            h, w = open_csv(os.path.join(self.out_dir, f"label_{label}.csv"),
                            self.amplitude_count)
            self._handles[label], self._writers[label] = h, w
        return self._all_writer, self._writers[label]

    def record(self, frames, label, settling, zone=0):
        for ts, esp_label, num, rssi, amps in frames:
            if self.amplitude_count is None:
                self.amplitude_count = len(amps)
                print(f"[{self.node_id}] locked to {self.amplitude_count} subcarriers.")
            if len(amps) != self.amplitude_count:
                self.rejected[f"{len(amps)}sc"] += 1
                continue
            aw, lw = self._writer_for(label)
            row = [self.session_id, int(time.time() * 1e6), ts, label, zone,
                   esp_label, settling, rssi, num] + amps
            aw.writerow(row)
            lw.writerow(row)
            self.counts[label] += 1

    def flush(self):
        if self._all_file:
            self._all_file.flush()
        for h in self._handles.values():
            h.flush()

    def close(self):
        if self._all_file:
            self._all_file.close()
        for h in self._handles.values():
            h.close()
        try:
            self.ser.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Auto-cycling empty/moving CSI collector.")
    ap.add_argument("-p", "--port", default="COM9")
    ap.add_argument("--port-b", default=None,
                    help="Second ESP32's port. Both boards then record the SAME "
                         "protocol simultaneously, into node_a/ and node_b/ "
                         "subfolders - needed for zone detection, which relies on "
                         "the two nodes seeing the same events from different angles.")
    ap.add_argument("-b", "--baud", type=int, default=921600)
    ap.add_argument("-o", "--output-dir", default=None)
    ap.add_argument("--subcarriers", type=int, default=None,
                    help="Expected subcarrier count; mismatched frames are dropped.")
    ap.add_argument("--rssi-field", action="store_true",
                    help="Firmware emits rssi before the amplitudes (improved firmware does).")
    ap.add_argument("--leave-seconds", type=float, default=5.0)
    ap.add_argument("--block-seconds", type=float, default=60.0)
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = "csi_dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    # Single node keeps writing straight into the session folder, exactly as
    # before, so existing sessions and every path in train_model.py still work.
    specs = [("A", args.port, args.output_dir)]
    if args.port_b:
        specs = [("A", args.port, os.path.join(args.output_dir, "node_a")),
                 ("B", args.port_b, os.path.join(args.output_dir, "node_b"))]

    nodes = []
    try:
        for node_id, port, out_dir in specs:
            nodes.append(NodeRecorder(node_id, port, args.baud, out_dir, session_id,
                                       args.subcarriers, args.rssi_field))
    except serial.SerialException as e:
        for n in nodes:
            n.close()
        print(f"Could not open a serial port: {e}")
        sys.exit(1)

    def record_all(frames_by_node, label, settling, zone=0):
        for n, frames in zip(nodes, frames_by_node):
            n.record(frames, label, settling, zone)

    def tell_all(ch):
        for n in nodes:
            n.tell_esp(ch)

    def row_summary():
        return " | ".join(f"{n.node_id}:L0={n.counts['0']},L2={n.counts['2']}"
                          for n in nodes)

    # ---- state machine ----
    # states: LEAVE (5s->empty), EMPTY (60s), WAIT (idle), MOVING (60s)
    # One machine drives every node, so both boards record the same events.
    state = "LEAVE"
    state_end = time.time() + args.leave_seconds
    last_tick = 0
    zone = 0            # which zone the CURRENT moving block is recording
    last_flush = 0
    tell_all("e")

    print(f"Session {session_id}")
    print(f"Writing to {os.path.abspath(args.output_dir)}")
    print(f"Nodes: " + ", ".join(f"{nid}={port}" for nid, port, _ in specs) + "\n")
    print(">>> LEAVE THE ROOM NOW <<<")

    try:
        with KeyReader() as keys:
            while True:
                key = keys.get()
                if key == "q":
                    print("\nStopping...")
                    break

                now = time.time()
                remain = state_end - now
                frames_by_node = [n.drain() for n in nodes]

                if state == "LEAVE":
                    record_all(frames_by_node, "0", settling=1)   # kept but flagged
                    if int(remain) != last_tick:
                        last_tick = int(remain)
                        print(f"  leaving... {max(0,int(remain)+1)} s", end="\r", flush=True)
                    if remain <= 0:
                        state, state_end = "EMPTY", now + args.block_seconds
                        print("\n>>> RECORDING EMPTY (60 s) <<<          ")

                elif state == "EMPTY":
                    record_all(frames_by_node, "0", settling=0)
                    if int(remain) != last_tick:
                        last_tick = int(remain)
                        print(f"  empty  {max(0,int(remain))} s | {row_summary()}",
                              end="\r", flush=True)
                    if remain <= 0:
                        state = "WAIT"
                        print("\n>>> Come back in. Press '1' or '2' for a ZONE, "
                              "or 'm' for plain moving. <<<")

                elif state == "WAIT":
                    # not recording; person is returning to the room
                    if key in ("m", "1", "2"):
                        # '1'/'2' tag the block with a zone; 'm' behaves exactly
                        # as it always has (zone 0 = not applicable). The LABEL
                        # is "2" either way, so the binary motion pipeline reads
                        # zone recordings without knowing zones exist.
                        zone = 0 if key == "m" else int(key)
                        tell_all("m")
                        state, state_end = "MOVING", now + args.block_seconds
                        where = "MOVING" if zone == 0 else f"MOVING in ZONE {zone}"
                        print(f">>> RECORDING {where} (60 s) <<<")

                elif state == "MOVING":
                    record_all(frames_by_node, "2", settling=0, zone=zone)
                    if int(remain) != last_tick:
                        last_tick = int(remain)
                        tag = "moving" if zone == 0 else f"zone {zone}"
                        print(f"  {tag} {max(0,int(remain))} s | {row_summary()}",
                              end="\r", flush=True)
                    if remain <= 0:
                        tell_all("e")
                        zone = 0
                        state, state_end = "LEAVE", now + args.leave_seconds
                        print("\n>>> LEAVE THE ROOM NOW <<<")

                # flush ~1/s
                if int(now) != last_flush:
                    last_flush = int(now)
                    for n in nodes:
                        n.flush()

                time.sleep(0.005)
    finally:
        for n in nodes:
            n.close()

    print("\n--- summary ---")
    print(f"session {session_id}")
    for n in nodes:
        print(f"  node {n.node_id} -> {n.out_dir}")
        for lbl in sorted(n.counts):
            print(f"    label {lbl}: {n.counts[lbl]} rows")
        if n.rejected:
            print("    rejected:", dict(n.rejected))


if __name__ == "__main__":
    main()