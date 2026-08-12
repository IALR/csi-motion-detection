"""Side-by-side hardware diagnosis for one or two ESP32 CSI nodes.

Answers the question "why does one board read a much higher noise floor than
the other at the same spot?" - which is a hardware/link question, not a room
question, and needs different measurements than the model pipeline provides.

The key distinction it draws:

  * WEAK SIGNAL       -> low RSSI, low mean amplitude. The board is receiving
                         badly (antenna switched to an unconnected u.FL, a
                         damaged trace antenna, or shielding).
  * QUANTISATION NOISE -> low mean amplitude but HIGH energy relative to that
                         amplitude. CSI arrives as int8 I/Q pairs, so a weak
                         signal sits near +/-4 counts where one quantisation
                         step is a large fraction of the value: amplitude then
                         jitters frame-to-frame with nothing moving. This is
                         the usual reason a weak board looks 'noisy' rather
                         than merely quiet.
  * INTERFERENCE      -> normal amplitude and RSSI, but high energy spread
                         evenly across the band (USB 3.0 ports are a classic
                         2.4GHz offender; so is a congested channel).
  * HARDWARE FAULT    -> an odd subcarrier profile, e.g. far fewer active
                         subcarriers than the other board, or the energy
                         concentrated in a handful of them.

Both boards must be idle and the room EMPTY while this runs - it is measuring
the empty-room floor, so anyone moving invalidates it.

Usage:
    python diagnose_nodes.py -p COM9                 # one node
    python diagnose_nodes.py -p COM9 --port-b COM6   # compare two
    python diagnose_nodes.py -p COM9 --port-b COM6 --seconds 60

Nothing else may hold the serial ports - close the live server and the IDF
monitor first.
"""
import argparse
import sys
import threading
import time

import numpy as np

try:
    import serial
except ImportError:
    print("Missing dependency: pyserial  ->  python -m pip install pyserial")
    sys.exit(1)

from csi_common import assess_noise_floor, parse_line


def collect(port, baud, seconds, out, node_id, warmup):
    """Read one node and stash (rssi, amps) rows in `out`.

    Opening the port RESETS the ESP32, and it then needs to boot, join Wi-Fi
    and start its ping trigger before any CSI appears - typically 10-15s. So
    the sampling clock starts at the FIRST CSI FRAME, not at open; `warmup` is
    how long to wait for that frame before giving up. Counting boot time
    against the sample window is what made an otherwise fine board look like
    it had captured almost nothing.

    Every non-CSI line is also kept. When a board yields no CSI at all, what
    it IS printing (Wi-Fi errors, boot logs, a crash backtrace) is the entire
    diagnosis, and discarding it leaves nothing to go on."""
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        out["error"] = f"could not open {port}: {e}"
        out["hint"] = ("Another program is probably holding it - the live server, "
                       "the IDF monitor, or the data collector. Only one may have "
                       "a COM port at a time.")
        return
    ser.reset_input_buffer()
    rows, rssis, other_lines = [], [], []
    buf = ""
    started = time.monotonic()
    give_up_at = started + warmup
    deadline = None          # set once the first CSI frame arrives
    first_frame_at = None
    try:
        while True:
            now = time.monotonic()
            if deadline is None and now > give_up_at:
                break
            if deadline is not None and now > deadline:
                break
            try:
                n = ser.in_waiting
                data = ser.read(n if n else 1)
            except Exception as e:
                out["error"] = f"read error: {e}"
                break
            if data:
                buf += data.decode("utf-8", errors="ignore")
            if "\n" in buf:
                *lines, buf = buf.split("\n")
                for ln in lines:
                    r = parse_line(ln)
                    if r is None:
                        s = ln.strip()
                        if s and len(other_lines) < 40:
                            other_lines.append(s[:160])
                        continue
                    if first_frame_at is None:
                        first_frame_at = time.monotonic()
                        deadline = first_frame_at + seconds
                        waited = first_frame_at - started
                        print(f"  [{node_id}] streaming (first frame after "
                              f"{waited:.0f}s), sampling {seconds:.0f}s...", flush=True)
                    rssis.append(r[0])
                    rows.append(r[1])
    finally:
        ser.close()
    out["amps"] = rows
    out["rssi"] = rssis
    out["span"] = (time.monotonic() - first_frame_at) if first_frame_at else 0.0
    out["other_lines"] = other_lines


def analyse(node_id, data):
    if data.get("error"):
        return {"node": node_id, "error": data["error"],
                "hint": data.get("hint"), "other_lines": data.get("other_lines") or []}
    rows = data.get("amps") or []
    if len(rows) < 20:
        return {"node": node_id,
                "error": f"only {len(rows)} CSI frames captured",
                "other_lines": data.get("other_lines") or []}

    widths = {len(r) for r in rows}
    if len(widths) > 1:
        # Keep the dominant width so one corrupt frame can't abort the run.
        common = max(widths, key=lambda w: sum(1 for r in rows if len(r) == w))
        rows = [r for r in rows if len(r) == common]

    A = np.stack(rows)
    rssi = np.array(data["rssi"][-len(rows):], dtype=float)
    active = A[:, (np.abs(A) > 0).any(axis=0)]

    diffs = np.abs(np.diff(A, axis=0))
    energy = diffs.mean(axis=1)
    amp_mean = float(active.mean()) if active.size else 0.0
    floor = float(energy.mean())

    per_sc = diffs.mean(axis=0)
    tot = per_sc.sum()
    top5 = float(np.sort(per_sc)[::-1][:5].sum() / tot * 100) if tot else 0.0

    return {
        "node": node_id,
        "frames": len(rows),
        "rate": len(rows) / data["span"] if data["span"] else 0.0,
        "subcarriers": A.shape[1],
        "active": active.shape[1],
        "rssi_mean": float(rssi.mean()),
        "rssi_std": float(rssi.std()),
        "amp_mean": amp_mean,
        "floor": floor,
        "relative": floor / amp_mean if amp_mean else float("nan"),
        "top5_share": top5,
    }


def verdict(r, other=None):
    """Turn the numbers into the most likely cause, conservatively."""
    out = []
    # Grade the SCALE-FREE figure: a board receiving half the signal has half
    # the raw jitter and would otherwise be graded as the quieter of the two
    # while actually being noisier relative to what it receives.
    level, headline, _ = assess_noise_floor(r["relative"])
    out.append(f"noise {r['relative']:.3f} (raw {r['floor']:.2f}) -> {level}")

    if r["rssi_mean"] < -70:
        out.append("VERY weak signal (RSSI < -70): suspect the antenna - a u.FL/IPEX "
                   "selector set to external with nothing attached, or a damaged "
                   "trace antenna")
    elif r["rssi_mean"] < -60:
        out.append("weak signal (RSSI < -60): worth checking the antenna and how far "
                   "this board sits from the AP")

    if r["amp_mean"] < 12:
        out.append(f"low mean amplitude ({r['amp_mean']:.1f}): CSI is int8 I/Q, so at this "
                   "level one quantisation step is a large fraction of the value and the "
                   "amplitude jitters on its own - this alone raises the noise floor "
                   "without anything moving")

    if r["relative"] >= 0.15 and r["amp_mean"] >= 12:
        out.append(f"high relative jitter ({r['relative']:.3f}) at a normal amplitude: this "
                   "looks like interference rather than weak reception. USB 3.0 ports and "
                   "cables are a common 2.4GHz source - try a USB 2.0 port, a different "
                   "cable, or move the board away from the PC")

    if r["top5_share"] > 25:
        out.append(f"energy concentrated in a few subcarriers (top-5 = {r['top5_share']:.0f}%): "
                   "narrowband interference or a subcarrier fault, not broadband noise")

    if r["rate"] < 6:
        out.append(f"low frame rate ({r['rate']:.1f} Hz, expected ~10): the board is dropping "
                   "frames - pings failing, weak link, or UART saturated")

    if other and not other.get("error"):
        if r["amp_mean"] < other["amp_mean"] * 0.7:
            out.append(f"receives {100*(1-r['amp_mean']/other['amp_mean']):.0f}% weaker than "
                       f"node {other['node']} at the same spot -> a per-board reception "
                       "problem, not the room")
        if r["floor"] > other["floor"] * 1.8 and r["rssi_mean"] > other["rssi_mean"] - 4:
            out.append(f"much noisier than node {other['node']} WITHOUT a matching drop in "
                       "RSSI -> points at local interference or power quality on this "
                       "board, not signal strength")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", default="COM9", help="serial port for node A")
    ap.add_argument("--port-b", default=None, help="serial port for node B (optional)")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="how long to sample each node ONCE it is streaming (default 30)")
    ap.add_argument("--warmup", type=float, default=45.0,
                    help="how long to wait for a board's first CSI frame before giving "
                         "up (default 45; opening the port resets the ESP32, which then "
                         "needs to boot and join Wi-Fi)")
    args = ap.parse_args()

    nodes = [("A", args.port)] + ([("B", args.port_b)] if args.port_b else [])

    print("=" * 74)
    print("ESP32 CSI node diagnosis")
    print("=" * 74)
    print(f"Nodes: " + ", ".join(f"{n}={p}" for n, p in nodes))
    print(f"Waiting up to {args.warmup:.0f}s for each board to start streaming, then")
    print(f"sampling {args.seconds:.0f}s from it.")
    print("\nOpening the port resets the ESP32, so the first frames take ~10-15s")
    print("(boot + Wi-Fi join). The room must be EMPTY and still for this - it")
    print("measures the empty-room noise floor, so anyone moving invalidates it.\n")

    raw, threads = {}, []
    for node_id, port in nodes:
        raw[node_id] = {}
        t = threading.Thread(target=collect,
                             args=(port, args.baud, args.seconds, raw[node_id],
                                   node_id, args.warmup),
                             daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    results = [analyse(n, raw[n]) for n, _ in nodes]
    ok = [r for r in results if not r.get("error")]

    for r in results:
        if not r.get("error"):
            continue
        print("\n" + "!" * 74)
        print(f"node {r['node']}: {r['error']}")
        print("!" * 74)
        if r.get("hint"):
            print(r["hint"])
        lines = r.get("other_lines") or []
        if lines:
            # What the board IS printing is the whole diagnosis when no CSI
            # arrives - a Wi-Fi failure, a boot loop, or a crash backtrace all
            # look identical ("no frames") without it.
            print(f"\nWhat this board actually sent ({len(lines)} non-CSI lines):")
            for ln in lines[:20]:
                print(f"  | {ln}")
            joined = " ".join(lines).lower()
            print("\nReading:")
            if any(k in joined for k in ("disconnect", "auth", "assoc", "no ap", "beacon", "handshake")):
                print("  - Wi-Fi association/authentication trouble. This board is not")
                print("    staying joined to the AP, so its ping trigger never runs and no")
                print("    CSI is ever produced. Check the SSID/password in")
                print("    firmware/main/wifi_secrets.h, and that ROUTER_IP matches the")
                print("    CURRENT gateway - a stale ROUTER_IP has caused exactly this in")
                print("    this project before.")
            elif any(k in joined for k in ("panic", "backtrace", "guru", "rst:", "boot:")):
                print("  - The board looks like it is resetting or crashing rather than")
                print("    running. Reflash it and watch `idf.py monitor` directly.")
            elif any(k in joined for k in ("csi", "wifi", "sta")):
                print("  - It is booting and talking, but never reached the CSI-streaming")
                print("    stage within the warm-up window. Try --warmup 60.")
            else:
                print("  - Output does not look like this project's firmware. Is this board")
                print("    flashed with the CSI firmware, and is the baud rate right?")
        else:
            print("\nThis board sent NOTHING AT ALL - not even boot messages.")
            print("Reading:")
            print("  - The port exists but the device is silent. Most likely: it is not")
            print("    powered (a charge-only USB cable does this), the wrong COM port for")
            print("    this board, or it needs longer to boot - try --warmup 60.")
            print("  - Confirm independently with:  idf.py -p <PORT> monitor")

    if not ok:
        print("\nNo usable data captured.")
        return 1

    print()
    print(f"{'metric':26}" + "".join(f"{'node ' + r['node']:>16}" for r in ok))
    print("-" * (26 + 16 * len(ok)))
    rows = [
        ("frames captured",      lambda r: f"{r['frames']}"),
        ("frame rate (Hz)",      lambda r: f"{r['rate']:.1f}"),
        ("subcarriers (active)", lambda r: f"{r['subcarriers']} ({r['active']})"),
        ("RSSI mean",            lambda r: f"{r['rssi_mean']:.1f}"),
        ("RSSI std",             lambda r: f"{r['rssi_std']:.2f}"),
        ("mean amplitude",       lambda r: f"{r['amp_mean']:.2f}"),
        ("NOISE FLOOR",          lambda r: f"{r['floor']:.2f}"),
        ("relative (floor/amp)", lambda r: f"{r['relative']:.3f}"),
        ("top-5 subc share",     lambda r: f"{r['top5_share']:.1f}%"),
    ]
    for label, fn in rows:
        print(f"{label:26}" + "".join(f"{fn(r):>16}" for r in ok))

    print("\n" + "=" * 74)
    print("Reading")
    print("=" * 74)
    for i, r in enumerate(ok):
        other = ok[1 - i] if len(ok) == 2 else None
        print(f"\nnode {r['node']}:")
        for line in verdict(r, other):
            print(f"  - {line}")

    if len(ok) == 2:
        a, b = ok
        print("\n" + "-" * 74)
        # Check RECEPTION disparity before comparing noise floors. A board with
        # a broken antenna receives so little signal that its absolute floor can
        # look LOW while the board is badly broken - comparing floors alone
        # would call that "both fine, shared cause", which is backwards.
        weak, strong = (a, b) if a["amp_mean"] < b["amp_mean"] else (b, a)
        if weak["amp_mean"] < strong["amp_mean"] * 0.7:
            pct = 100 * (1 - weak["amp_mean"] / strong["amp_mean"])
            print(f"Node {weak['node']} receives {pct:.0f}% less signal than node "
                  f"{strong['node']} at the same spot")
            print(f"  (amplitude {weak['amp_mean']:.1f} vs {strong['amp_mean']:.1f}, "
                  f"RSSI {weak['rssi_mean']:.0f} vs {strong['rssi_mean']:.0f}).")
            print("That is a RECEPTION problem, not a room problem. Check that board's")
            print("antenna first - many ESP32-S3 boards have a u.FL/IPEX connector with a")
            print("solder-bridge selector, and one set to 'external' with no antenna")
            print("attached behaves exactly like this.")
            print("\nThen SWAP THE TWO BOARDS' POSITIONS and re-run: if the problem follows")
            print("the BOARD it is hardware; if it stays with the SPOT it is placement.")
        elif abs(a["floor"] - b["floor"]) < 1.0:
            print("Both boards read a similar floor and receive comparable signal ->")
            print("shared cause (the AP, the channel, or the room), not one bad board.")
            if max(a["floor"], b["floor"]) >= 3.5:
                print("Since both are loud, look at the ACCESS POINT: a phone hotspot's")
                print("power-saving and rate adaptation produce exactly this, and so does")
                print("a congested channel. A dedicated router on a quiet 20MHz channel is")
                print("the durable fix.")
        else:
            worse, better = (a, b) if a["floor"] > b["floor"] else (b, a)
            print(f"Node {worse['node']} is materially noisier than node {better['node']}")
            print(f"  (floor {worse['floor']:.2f} vs {better['floor']:.2f}) despite comparable")
            print("signal strength, so this is NOT weak reception -> suspect local")
            print("interference or power quality on that board: USB 3.0 ports and cables")
            print("are a well-known 2.4GHz noise source. Try a USB 2.0 port, a different")
            print("cable, or move the board away from the PC, then re-run.")
            print("\nAlso worth doing: SWAP THE TWO BOARDS' POSITIONS and re-run. If the")
            print("problem follows the BOARD it is hardware; if it stays with the SPOT it")
            print("is that location.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
