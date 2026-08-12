"""
Shared logic for the CSI empty/moving pipeline: parsing, feature extraction,
calibration, and prediction smoothing. Used by train_model.py,
evaluate_holdout.py, analyze_model.py, csi_live_predict.py and
csi_live_server.py so the offline (training) and online (live) code paths
can never quietly diverge.

Calibration
-----------
Earlier versions used raw, absolute features (mean amplitude per subcarrier,
raw RSSI). Those encode "what this room's baseline looks like right now" as
much as "is someone moving" - so a shifted RSSI baseline between sessions
(different day, router nudged, temperature) could get misread as motion.

Now every window is expressed RELATIVE to a short calibration baseline
(the first CALIB_SECONDS of confirmed-empty data at the start of a
session/deployment):
  - means become deltas (offset drift cancels out)
  - spreads (std) become ratios against the baseline's own spread (so an
    already-noisy baseline - a fan, interference - raises the bar for what
    counts as "more disturbed than that", instead of comparing against a
    fixed absolute threshold learned on a different day)

This does NOT fix a room that is never quiet during calibration itself -
if the disturbance is present throughout, it becomes the new "normal" and
motion on top of it may still be missed. It fixes the "everything looks
like motion because the baseline moved" failure mode specifically.
"""

import numpy as np

FRAME_HZ = 10.0  # ping interval is 100ms -> ~10 CSI frames/sec
CALIB_SECONDS_DEFAULT = 10.0
EPS = 1e-6

MARKER = "CSI_AMP,"
LABEL_NAMES = {0: "EMPTY", 2: "MOVING"}


def parse_line(line):
    """Return (rssi, amps) or None. Firmware format:
    CSI_AMP,<ts_us>,<label>,<num_subcarriers>,<rssi>,<amp0>,<amp1>,...

    The firmware states how many subcarriers it is about to send (field 3),
    so the count is CHECKED against how many actually arrived rather than
    trusting whatever made it down the wire. A line mangled in transit -
    dropped bytes on the UART, a decode("utf-8", errors="ignore") swallowing
    a chunk mid-line - can still contain the marker and still split into
    enough fields to look valid, just with some amplitudes missing. Without
    this check that silently becomes a short amplitude array; and because
    the live tools latch their subcarrier count from the FIRST frame they
    see, one mangled first line would mis-size the entire session."""
    i = line.find(MARKER)
    if i == -1:
        return None
    parts = line[i:].strip().split(",")
    if len(parts) < 6 or parts[0] != "CSI_AMP":
        return None
    try:
        declared_n = int(parts[3])
        rssi = float(parts[4])
        amps = np.array([float(x) for x in parts[5:]])
    except ValueError:
        return None
    if declared_n != len(amps):
        return None
    return rssi, amps


def raw_window_stats(amp_window, rssi_window):
    """amp_window: (frames, subcarriers). Returns the unnormalized stats
    for one window, before any calibration is applied."""
    amps = np.stack(amp_window)
    diffs = np.abs(np.diff(amps, axis=0)).mean(axis=1) if len(amps) > 1 else np.array([0.0])
    rssi = np.array(rssi_window)
    return {
        "amp_mean": amps.mean(axis=0),
        "amp_std": amps.std(axis=0),
        "energy_mean": float(diffs.mean()),
        "energy_std": float(diffs.std()),
        "rssi_mean": float(rssi.mean()),
        "rssi_std": float(rssi.std()),
    }


def compute_baseline(amp_frames, rssi_frames):
    """Build a calibration baseline from a block of confirmed-empty frames
    (amp_frames: (frames, subcarriers), rssi_frames: (frames,)).

    Some subcarriers (guard bands / DC nulls in the OFDM structure) are
    structurally dead - always exactly zero amplitude, so their baseline
    std is exactly 0. A ratio feature dividing by that (even with a tiny
    EPS floor) blows up to a huge number from the smallest live noise on
    that subcarrier - e.g. a live std of 0.001 against a baseline std of 0
    gives a ratio of ~1000, versus ~1.0 for a genuinely stable subcarrier.
    Aggregate features that hunt for the MAX ratio across all subcarriers
    (std_ratio_max, etc.) get hijacked by this noise on nearly every
    window, regardless of real motion. std_floor fixes it: no subcarrier's
    baseline std counts as smaller than 10% of that baseline's own median,
    so a truly dead subcarrier can't produce a runaway ratio."""
    stats = raw_window_stats(amp_frames, rssi_frames)
    stats["amp_std_floor"] = max(float(np.median(stats["amp_std"])) * 0.1, EPS)
    stats["rssi_std_floor"] = max(stats["rssi_std"] * 0.1, EPS)
    return stats


TOP_K = 10          # how many subcarriers count as "the disturbed ones"
ELEVATED_RATIO = 1.5  # a subcarrier counts as "elevated" above this std ratio
ENERGY_STD_FLOOR = 0.05  # same dead-denominator problem as amp_std, fixed scale is fine here


# --- Empty-room noise floor -------------------------------------------------
# How disturbed the room looks while EMPTY. The metric is the mean
# frame-to-frame amplitude change DIVIDED BY the mean amplitude - i.e. jitter
# as a FRACTION of signal, not in raw amplitude units.
#
# The division matters and was learned the hard way. The first version of this
# used raw amplitude change, which is fine for watching ONE node over time but
# is wrong for comparing two nodes: a board receiving a weaker signal has
# proportionally smaller amplitudes, so its raw frame-to-frame differences are
# smaller too. Measured on this project's two boards sitting at the SAME spot:
#
#     node A   RSSI -47.0   amplitude 32.1   raw 1.63   relative 0.051
#     node B   RSSI -59.3   amplitude 16.6   raw 1.17   relative 0.071
#
# On the raw scale node B looks CLEANER than node A. It is not - it receives
# 12dB less signal and is actually 39% noisier relative to what it receives.
# Dividing by amplitude makes the two nodes comparable.
#
# Thresholds are read off this project's OWN 10 recorded sessions, not picked
# by feel. Relative floor, then how far apart empty/moving energy ended up
# (the "separation"), then that session's held-out accuracy:
#
#     part_1        0.0248   4.67x   96.0%
#     part_4        0.0258   3.80x   95.7%
#     part_2        0.0312   3.13x   98.7%
#     part_3        0.0357   2.88x   98.7%
#     part_5        0.0445   5.96x   95.3%
#     part_7        0.0995   1.36x   94.0%
#     room2_part_2  0.1374   1.89x   98.4%
#     part_8        0.2162   0.99x   95.0%
#     part_6        0.2207   1.02x   94.3%
#     room2_part_1  0.2208   1.02x   84.7%
#
# Above ~0.15 the aggregate motion signal has essentially vanished (empty and
# moving energy within ~2% of each other) and only finer per-subcarrier
# structure still separates the classes.
#
# BE HONEST ABOUT WHAT THIS PREDICTS. A loud floor is a RISK INDICATOR, not a
# forecast: part_6 and part_8 sit at the very top of the scale and still
# scored 94-95%, while room2_part_1 at a near-identical floor scored 84.7%.
# So loud means "this is the regime where results get variable (85-95%)
# instead of consistent (95-99%)", NOT "it is about to fail". Grouped, the
# effect is real: LOSO within the loud sessions is 87.8% vs 94.1% in quiet.
#
# (Footnote for anyone re-deriving these: against SEPARATION alone the raw
# measure correlates marginally better, -0.85 vs -0.81, because all ten
# sessions came off boards of similar signal strength so the division adds a
# little variance without adding information. Across DIFFERENT boards the raw
# measure is simply wrong, as the node A/B table above shows, and the relative
# one also fixes a real misgrade: room2_part_2 was called 'loud' on the raw
# scale despite scoring 98.4%, and grades correctly as 'moderate' here.)
NOISE_REL_QUIET = 0.06   # at or below: matches the project's most reliable sessions
NOISE_REL_LOUD = 0.15    # at or above: matches its most variable ones


def baseline_noise_stats(baseline):
    """Pull the noise-floor numbers out of a calibration baseline.

    Returns (absolute, relative, amp_scale). `amp_scale` averages only the
    ACTIVE subcarriers - the ~20 structurally dead guard-band ones sit at
    exactly zero and would otherwise drag the scale down and inflate the
    ratio."""
    amp = np.asarray(baseline["amp_mean"], dtype=float)
    active = amp[amp > 0]
    scale = float(active.mean()) if active.size else 0.0
    absolute = float(baseline["energy_mean"])
    relative = absolute / scale if scale > 0 else float("nan")
    return absolute, relative, scale


def assess_noise_floor(relative):
    """Grade an empty-room noise floor given the RELATIVE (scale-free) value
    from baseline_noise_stats(). Returns (level, headline, detail) where level
    is 'quiet' | 'moderate' | 'loud'."""
    if not np.isfinite(relative):
        return ("quiet", "Noise floor unavailable",
                "No amplitude scale to compare against yet.")
    if relative <= NOISE_REL_QUIET:
        return ("quiet",
                f"Quiet environment (noise {relative:.3f})",
                "Matches this project's most reliable recording sessions, "
                "which scored 95-99% held out.")
    if relative < NOISE_REL_LOUD:
        return ("moderate",
                f"Moderately noisy environment (noise {relative:.3f})",
                "Above the quiet sessions but the classes still separate well. "
                "Detection should work normally.")
    return ("loud",
            f"Noisy environment (noise {relative:.3f})",
            "An empty room here is nearly as disturbed as an occupied one, so "
            "the aggregate motion signal is weak and detection leans on finer "
            "per-subcarrier structure. Sessions recorded in this regime scored "
            "between 85% and 95% held out, versus 95-99% in quiet rooms - "
            "results get more variable, with false 'MOVING' readings the more "
            "likely error. It does not mean detection will fail. Common "
            "causes: Wi-Fi congestion on this channel, a fan or AC running, or "
            "something physically moving in the room during calibration.")


def calibrate_features(stats, baseline):
    """Turn raw window stats into baseline-relative features. Order must
    match feature_names().

    The per-subcarrier amp_mean_delta/amp_std_ratio arrays are tied to
    SPECIFIC subcarrier INDICES - a Random Forest trained on one room can
    end up splitting on "index 43", which is only meaningful because
    index 43 happened to be where THIS room's multipath put the sensitive
    spot. A different room's geometry can put the sensitive band somewhere
    else entirely, and per-index features have no way to transfer that.

    The block below adds ORDER-INVARIANT summaries of the same per-subcarrier
    arrays - max, top-K mean, 90th percentile, fraction elevated - that
    answer "is *something* in the band disturbed" without caring *which*
    index it is. These should generalize across rooms better than the
    per-index features alone, at some cost in within-room precision.
    Both sets are kept; a Random Forest can lean on whichever fits."""
    amp_std_floor = baseline.get("amp_std_floor", EPS)
    rssi_std_floor = baseline.get("rssi_std_floor", EPS)

    amp_mean_delta = stats["amp_mean"] - baseline["amp_mean"]
    amp_std_ratio = stats["amp_std"] / np.maximum(baseline["amp_std"], amp_std_floor)
    energy_mean_ratio = stats["energy_mean"] / max(baseline["energy_mean"], ENERGY_STD_FLOOR)
    energy_std_ratio = stats["energy_std"] / max(baseline["energy_std"], ENERGY_STD_FLOOR)
    rssi_mean_delta = stats["rssi_mean"] - baseline["rssi_mean"]
    rssi_std_ratio = stats["rssi_std"] / max(baseline["rssi_std"], rssi_std_floor)

    abs_mean_delta = np.abs(amp_mean_delta)
    sorted_std_ratio = np.sort(amp_std_ratio)[::-1]
    sorted_abs_delta = np.sort(abs_mean_delta)[::-1]
    k = min(TOP_K, len(amp_std_ratio))

    order_invariant = np.array([
        amp_std_ratio.max(),                     # std_ratio_max
        sorted_std_ratio[:k].mean(),              # std_ratio_topK_mean
        float(np.percentile(amp_std_ratio, 90)),  # std_ratio_p90
        float(np.mean(amp_std_ratio > ELEVATED_RATIO)),  # std_ratio_frac_elevated
        abs_mean_delta.max(),                     # mean_delta_absmax
        sorted_abs_delta[:k].mean(),               # mean_delta_absmax_topK_mean
    ])

    return np.concatenate([
        amp_mean_delta,
        amp_std_ratio,
        [energy_mean_ratio, energy_std_ratio],
        [rssi_mean_delta, rssi_std_ratio],
        order_invariant,
    ])


def feature_names(amp_cols):
    names = [f"{c}_mean_delta" for c in amp_cols]
    names += [f"{c}_std_ratio" for c in amp_cols]
    names += ["motion_energy_mean_ratio", "motion_energy_std_ratio",
              "rssi_mean_delta", "rssi_std_ratio"]
    names += ["std_ratio_max", "std_ratio_topK_mean", "std_ratio_p90",
              "std_ratio_frac_elevated", "mean_delta_absmax", "mean_delta_absmax_topK_mean"]
    return names


def make_features(amp_window, rssi_window, baseline):
    """Online/streaming entry point: one window -> one calibrated feature row."""
    stats = raw_window_stats(amp_window, rssi_window)
    return calibrate_features(stats, baseline).reshape(1, -1)


class PredictionSmoother:
    """Hysteresis over raw per-window predictions so a single noisy window
    can't flip the displayed state. Requires a majority of the last `size`
    raw predictions to agree before changing the confirmed state."""

    def __init__(self, size=5):
        self.size = size
        self.history = []
        self.confirmed = None

    def reset(self):
        self.history = []
        self.confirmed = None

    def update(self, raw_label):
        self.history.append(raw_label)
        if len(self.history) > self.size:
            self.history.pop(0)

        counts = {}
        for lbl in self.history:
            counts[lbl] = counts.get(lbl, 0) + 1
        majority_label, majority_count = max(counts.items(), key=lambda kv: kv[1])
        vote_fraction = majority_count / len(self.history)

        if majority_count > len(self.history) / 2:
            self.confirmed = majority_label

        return self.confirmed, vote_fraction


class RollingCalibrator:
    """Keeps a LIVE deployment's baseline fresh, mirroring what
    train_model.py now does offline (recalibrate at every empty block)
    instead of calibrating once at startup and comparing against that
    forever. Without this, something like turning on the AC mid-session
    would trip a false alarm live even though training already handles
    that exact scenario correctly.

    Call observe() every frame once a CONFIRMED prediction exists (i.e.
    after PredictionSmoother has settled on a state). Whenever the system
    has been confirmed EMPTY for a sustained stretch (calib_frames worth,
    matching the model's own calib_seconds), it recalibrates. Never
    touches MOVING frames, so a burst of motion can't corrupt the
    baseline - the buffer resets the moment MOVING is confirmed.

    Two safety mechanisms, both fixing the same failure mode (a first
    version of this class did neither and would drift into predicting
    MOVING while the room sat empty, sometimes within one session):

    1. BLEND, don't replace. A ~10s sample's noise level is a noisy
       estimate of the room's true noise floor - it can land smaller than
       typical just by chance. Fully replacing the baseline with that one
       sample, over and over, lets a single unlucky quiet window ratchet
       sensitivity up with nothing pulling it back. Blending (a small
       weighted step toward the fresh sample) still tracks real drift
       over many cycles, but one noisy sample can't take over on its own.
    2. FLOOR against the startup baseline. amp_std/rssi_std/energy_std can
       never be blended down below a fraction of what the ORIGINAL,
       deliberate "leave the room" calibration measured. That startup
       calibration is a fixed, trustworthy reference point; nothing here
       is allowed to make the system more sensitive than that baseline
       ever was, only less (to absorb something genuinely louder, like an
       AC switching on).
    """

    def __init__(self, calib_frames, blend_alpha=0.3, min_std_fraction=0.6):
        self.calib_frames = calib_frames
        self.blend_alpha = blend_alpha       # weight given to each fresh sample
        self.min_std_fraction = min_std_fraction
        self._amp_buf = []
        self._rssi_buf = []
        self._floor_baseline = None          # the startup calibration, set once

    def set_floor_reference(self, startup_baseline):
        """Call once, right after startup calibration completes."""
        self._floor_baseline = startup_baseline

    def reset(self):
        """Call when a fresh startup calibration is about to happen (e.g.
        a forced/manual recalibration), so stale accumulated frames and
        the old floor reference don't leak into the new one."""
        self._amp_buf = []
        self._rssi_buf = []
        self._floor_baseline = None

    def observe(self, confirmed_label, amps, rssi, current_baseline):
        """Returns a new (blended, floored) baseline dict once enough
        confirmed-empty frames have accumulated, else None."""
        if confirmed_label != 0:
            self._amp_buf = []
            self._rssi_buf = []
            return None

        self._amp_buf.append(amps)
        self._rssi_buf.append(rssi)
        if len(self._amp_buf) < self.calib_frames:
            return None

        fresh = compute_baseline(self._amp_buf, self._rssi_buf)
        self._amp_buf = []
        self._rssi_buf = []

        a = self.blend_alpha
        blended = {
            "amp_mean": (1 - a) * current_baseline["amp_mean"] + a * fresh["amp_mean"],
            "amp_std": (1 - a) * current_baseline["amp_std"] + a * fresh["amp_std"],
            "energy_mean": (1 - a) * current_baseline["energy_mean"] + a * fresh["energy_mean"],
            "energy_std": (1 - a) * current_baseline["energy_std"] + a * fresh["energy_std"],
            "rssi_mean": (1 - a) * current_baseline["rssi_mean"] + a * fresh["rssi_mean"],
            "rssi_std": (1 - a) * current_baseline["rssi_std"] + a * fresh["rssi_std"],
        }

        if self._floor_baseline is not None:
            fb = self._floor_baseline
            blended["amp_std"] = np.maximum(blended["amp_std"], fb["amp_std"] * self.min_std_fraction)
            blended["rssi_std"] = max(blended["rssi_std"], fb["rssi_std"] * self.min_std_fraction)
            blended["energy_std"] = max(blended["energy_std"], fb["energy_std"] * self.min_std_fraction)

        blended["amp_std_floor"] = max(float(np.median(blended["amp_std"])) * 0.1, EPS)
        blended["rssi_std_floor"] = max(blended["rssi_std"] * 0.1, EPS)
        return blended
