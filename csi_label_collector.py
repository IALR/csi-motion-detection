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

Usage:
    python csi_auto_collector.py -p COM9 -b 921600 --subcarriers 128 --rssi-field

Controls while running:
    m  -> start a MOVING block (only does something while waiting)
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


BASE_COLUMNS = ["session_id", "host_unix_us", "timestamp_us", "label",
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
    return ts, esp_label, num, rssi, amps


def main():
    ap = argparse.ArgumentParser(description="Auto-cycling empty/moving CSI collector.")
    ap.add_argument("-p", "--port", default="COM9")
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

    all_path = os.path.join(args.output_dir, "all_csi_data.csv")
    all_file = all_writer = None
    handles, writers = {}, {}
    amplitude_count = args.subcarriers
    counts = defaultdict(int)
    rejected = defaultdict(int)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        sys.exit(1)
    ser.reset_input_buffer()
    rxbuf = {"s": ""}

    def drain():
        try:
            k = ser.in_waiting
            if k:
                rxbuf["s"] += ser.read(k).decode("utf-8", errors="ignore")
        except Exception:
            return []
        out = []
        if "\n" in rxbuf["s"]:
            *lines, rxbuf["s"] = rxbuf["s"].split("\n")
            for ln in lines:
                r = parse_line(ln, args.rssi_field)
                if r:
                    out.append(r)
        return out

    def tell_esp(ch):
        try:
            ser.write(ch.encode("ascii")); ser.flush()
        except Exception:
            pass

    def writer_for(label):
        nonlocal all_file, all_writer
        if all_writer is None:
            all_file, all_writer = open_csv(all_path, amplitude_count)
        if label not in handles:
            h, w = open_csv(os.path.join(args.output_dir, f"label_{label}.csv"),
                            amplitude_count)
            handles[label], writers[label] = h, w
        return all_writer, writers[label]

    def record(frames, label, settling):
        nonlocal amplitude_count
        for ts, esp_label, num, rssi, amps in frames:
            if amplitude_count is None:
                amplitude_count = len(amps)
                print(f"Locked to {amplitude_count} subcarriers.")
            if len(amps) != amplitude_count:
                rejected[f"{len(amps)}sc"] += 1
                continue
            aw, lw = writer_for(label)
            row = [session_id, int(time.time() * 1e6), ts, label,
                   esp_label, settling, rssi, num] + amps
            aw.writerow(row); lw.writerow(row)
            counts[label] += 1

    # ---- state machine ----
    # states: LEAVE (5s->empty), EMPTY (60s), WAIT (idle), MOVING (60s)
    state = "LEAVE"
    state_end = time.time() + args.leave_seconds
    last_tick = 0
    tell_esp("e")

    print(f"Session {session_id}")
    print(f"Writing to {os.path.abspath(args.output_dir)}\n")
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
                frames = drain()

                if state == "LEAVE":
                    record(frames, "0", settling=1)   # kept but flagged
                    if int(remain) != last_tick:
                        last_tick = int(remain)
                        print(f"  leaving... {max(0,int(remain)+1)} s", end="\r", flush=True)
                    if remain <= 0:
                        state, state_end = "EMPTY", now + args.block_seconds
                        print("\n>>> RECORDING EMPTY (60 s) <<<          ")

                elif state == "EMPTY":
                    record(frames, "0", settling=0)
                    if int(remain) != last_tick:
                        last_tick = int(remain)
                        print(f"  empty  {max(0,int(remain))} s | rows L0:{counts['0']} L2:{counts['2']}",
                              end="\r", flush=True)
                    if remain <= 0:
                        state = "WAIT"
                        print("\n>>> Come back in. Press 'm' when ready to MOVE. <<<")

                elif state == "WAIT":
                    # not recording; person is returning to the room
                    if key == "m":
                        tell_esp("m")
                        state, state_end = "MOVING", now + args.block_seconds
                        print(">>> RECORDING MOVING (60 s) <<<")

                elif state == "MOVING":
                    record(frames, "2", settling=0)
                    if int(remain) != last_tick:
                        last_tick = int(remain)
                        print(f"  moving {max(0,int(remain))} s | rows L0:{counts['0']} L2:{counts['2']}",
                              end="\r", flush=True)
                    if remain <= 0:
                        tell_esp("e")
                        state, state_end = "LEAVE", now + args.leave_seconds
                        print("\n>>> LEAVE THE ROOM NOW <<<")

                # flush ~1/s
                if all_file and int(now) != getattr(record, "_lf", 0):
                    record._lf = int(now)
                    all_file.flush()
                    for h in handles.values():
                        h.flush()

                time.sleep(0.005)
    finally:
        if all_file:
            all_file.close()
        for h in handles.values():
            h.close()
        ser.close()

    print("\n--- summary ---")
    print(f"session {session_id}")
    for lbl in sorted(counts):
        print(f"  label {lbl}: {counts[lbl]} rows")
    if rejected:
        print("  rejected:", dict(rejected))


if __name__ == "__main__":
    main()