"""Regression tests for the CSI pipeline's failure modes.

Run:  python -m pytest tests/ -v     (or just: python tests/test_pipeline.py)

These exist because every bug this project has hit twice was a FAILURE-MODE
bug, not a math bug: the model was fine, but something upstream broke in a
way that looked like the model was wrong. Each test below pins one such
failure so it can't come back silently:

  - parse_line: a mangled serial line must be rejected, not silently turned
    into a short amplitude array (which would mis-size the whole session,
    since the live tools latch their subcarrier count from the first frame)
  - compute_combined: a node that is offline must not be able to veto EMPTY
  - pump(): a node going silent, or the AP changing channel width mid-run,
    must degrade to "this node stops voting / recalibrates" rather than
    "this node is silently dead forever" or "the whole server crashes"

The pump tests drive the REAL coroutine with synthetic frames and a stub
model, so they cover the live control flow without needing an ESP32.
"""
import asyncio
import queue
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csi_live_server as S
import train_model
from csi_common import (NOISE_REL_LOUD, NOISE_REL_QUIET, PredictionSmoother,
                        RollingCalibrator, assess_noise_floor, baseline_noise_stats,
                        compute_baseline, parse_line)


# --------------------------------------------------------------------------
# parse_line: the firmware declares its subcarrier count; trust but verify
# --------------------------------------------------------------------------

def test_parse_line_valid():
    rssi, amps = parse_line("CSI_AMP,123,0,4,-40,1,2,3,4")
    assert rssi == -40.0
    assert list(amps) == [1.0, 2.0, 3.0, 4.0]


def test_parse_line_accepts_leading_junk():
    # Serial lines often arrive with log output prefixed on the same line.
    rssi, amps = parse_line("I (999) wifi: x CSI_AMP,1,0,3,-50,1,2,3")
    assert rssi == -50.0 and len(amps) == 3


@pytest.mark.parametrize("line", [
    "CSI_AMP,123,0,4,-40,1,2",          # truncated: declares 4, sent 2
    "CSI_AMP,123,0,4,-40,1,2,3,4,5,6",  # over-long: declares 4, sent 6
    "CSI_AMP,123,0,4,-40",              # no amplitudes at all
    "I (1234) wifi: connected",         # not a CSI line
    "",
])
def test_parse_line_rejects_malformed(line):
    assert parse_line(line) is None


# --------------------------------------------------------------------------
# compute_combined: OR logic across nodes, with offline nodes excluded
# --------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", [
    ({"A": "EMPTY"},                        "EMPTY"),
    ({"A": "MOVING"},                       "MOVING"),
    ({"A": None},                           None),
    ({"A": "EMPTY", "B": "EMPTY"},          "EMPTY"),
    ({"A": "MOVING", "B": "EMPTY"},         "MOVING"),
    ({"A": "EMPTY", "B": None},             None),      # B still warming up
    ({"A": "MOVING", "B": None},            "MOVING"),  # don't wait to alarm
    ({"A": "EMPTY", "B": S.OFFLINE},        "EMPTY"),   # offline must not veto
    ({"A": "MOVING", "B": S.OFFLINE},       "MOVING"),
    ({"A": S.OFFLINE, "B": S.OFFLINE},      None),      # never infer EMPTY
])
def test_compute_combined(state, expected):
    assert S.compute_combined(state) == expected


@pytest.mark.parametrize("state,muted,expected,why", [
    ({"A": "EMPTY", "B": "MOVING"}, {"B"},      "EMPTY",
     "the whole point: a muted node's false alarm must not reach the output"),
    ({"A": "MOVING", "B": "EMPTY"}, {"B"},      "MOVING",
     "muting B must never suppress A's genuine detection"),
    ({"A": "EMPTY", "B": "EMPTY"},  {"B"},      "EMPTY",
     "muting an agreeing node changes nothing"),
    ({"A": "EMPTY", "B": "MOVING"}, {"A", "B"}, None,
     "everything muted is UNKNOWN - never infer an empty room"),
    ({"A": "MOVING"},               {"A"},      None,
     "muting the only node is unknown, not empty"),
    ({"A": S.OFFLINE, "B": "MOVING"}, {"B"},    None,
     "offline plus muted leaves nobody voting"),
    ({"A": "EMPTY", "B": "MOVING"}, set(),      "MOVING",
     "unmuted again, B's vote counts once more"),
])
def test_compute_combined_with_muted_nodes(state, muted, expected, why):
    assert S.compute_combined(state, muted) == expected, why


# --------------------------------------------------------------------------
# csi_common calibration primitives
# --------------------------------------------------------------------------

def test_baseline_std_floor_blocks_dead_subcarrier_blowup():
    """Guard-band subcarriers sit at exactly 0, so their baseline std is 0.
    Without the floor, the tiniest live noise there produces a runaway ratio
    that hijacks every max/top-K feature."""
    frames = np.random.default_rng(0).normal(20, 2, (50, 16))
    frames[:, 3] = 0.0                       # a structurally dead subcarrier
    base = compute_baseline(frames, np.full(50, -40.0))
    assert base["amp_std"][3] == 0.0         # genuinely zero...
    assert base["amp_std_floor"] > 0         # ...but the floor is not
    ratio = 0.001 / max(base["amp_std"][3], base["amp_std_floor"])
    assert ratio < 1.0, "dead subcarrier still able to produce a huge ratio"


# --------------------------------------------------------------------------
# Every analysis script must describe the DEPLOYED configuration.
# Four of them used to hardcode their own session list and window size, each
# drifted to a different stale subset, and each therefore reported numbers
# about a model that is not deployed - two documented claims came from them.
# --------------------------------------------------------------------------

ANALYSIS_SCRIPTS = ["analyze_model", "compare_models", "evaluate_holdout",
                    "model_evaluation"]


@pytest.mark.parametrize("module_name", ANALYSIS_SCRIPTS)
def test_analysis_scripts_do_not_hardcode_sessions(module_name):
    """No script may carry its own copy of the session list - they import the
    one in train_model.py so it cannot drift."""
    src = (Path(__file__).resolve().parent.parent / f"{module_name}.py").read_text(encoding="utf-8")
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    # A literal list of session folders is the pattern that kept going stale.
    assert not re.search(r'default\s*=\s*\[\s*["\']part_\d_data', body), (
        f"{module_name}.py hardcodes a session list; import DEFAULT_SESSIONS "
        f"from train_model instead")
    assert "DEFAULT_SESSIONS" in src, f"{module_name}.py should use DEFAULT_SESSIONS"


@pytest.mark.parametrize("module_name", ANALYSIS_SCRIPTS)
def test_analysis_scripts_use_the_deployed_window(module_name):
    """Two scripts silently defaulted to a 2.0s window while the deployed
    model uses 0.75s, so their numbers described a different system."""
    src = (Path(__file__).resolve().parent.parent / f"{module_name}.py").read_text(encoding="utf-8")
    assert not re.search(r'"--window-seconds".*default\s*=\s*[\d.]+', src), (
        f"{module_name}.py sets a literal window default; use "
        f"DEFAULT_WINDOW_SECONDS from train_model")


def test_default_sessions_all_exist_on_disk():
    root = Path(__file__).resolve().parent.parent
    for s in train_model.DEFAULT_SESSIONS:
        assert (root / s / "all_csi_data.csv").exists(), f"{s} is listed but missing"


def test_default_window_matches_the_deployed_model():
    """If csi_model.joblib was trained at a different window than the default,
    a bare retrain would silently change the deployed behaviour."""
    bundle_path = Path(__file__).resolve().parent.parent / "csi_model.joblib"
    if not bundle_path.exists():
        pytest.skip("no trained model present")
    bundle = joblib.load(bundle_path)
    assert bundle["window_seconds"] == train_model.DEFAULT_WINDOW_SECONDS


# --------------------------------------------------------------------------
# Empty-room noise floor: the thresholds are calibrated against this
# project's real recorded sessions, so pin them to those measurements.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("session,relative,expected", [
    # measured scale-free empty-room noise from each session's own recording
    ("part_1_data",       0.0248, "quiet"),
    ("part_4_data",       0.0258, "quiet"),
    ("part_2_data",       0.0312, "quiet"),
    ("part_3_data",       0.0357, "quiet"),
    ("part_5_data",       0.0445, "quiet"),
    ("part_7_data",       0.0995, "moderate"),
    ("room2_part_2_data", 0.1374, "moderate"),  # raw scale wrongly called this loud
    ("part_8_data",       0.2162, "loud"),
    ("part_6_data",       0.2207, "loud"),
    ("room2_part_1_data", 0.2208, "loud"),
])
def test_noise_floor_grades_real_sessions(session, relative, expected):
    level, headline, detail = assess_noise_floor(relative)
    assert level == expected, f"{session} (relative {relative}) graded {level}"
    assert headline and detail


def test_noise_floor_thresholds_are_ordered():
    assert NOISE_REL_QUIET < NOISE_REL_LOUD
    assert assess_noise_floor(NOISE_REL_QUIET)[0] == "quiet"
    assert assess_noise_floor(NOISE_REL_LOUD)[0] == "loud"


def test_noise_floor_is_scale_free_across_nodes():
    """The defect this replaced: on a RAW scale a board receiving half the
    signal shows half the jitter and looks cleaner than a healthy one. These
    are this project's two real boards measured at the same spot."""
    node_a = baseline_noise_stats({"amp_mean": np.full(64, 32.09),
                                   "energy_mean": 1.63})
    node_b = baseline_noise_stats({"amp_mean": np.full(64, 16.59),
                                   "energy_mean": 1.17})
    abs_a, rel_a, _ = node_a
    abs_b, rel_b, _ = node_b
    assert abs_b < abs_a, "precondition: B's RAW jitter is lower than A's"
    assert rel_b > rel_a, "but relative to received signal, B must read NOISIER"


def test_baseline_noise_stats_ignores_dead_subcarriers():
    """Guard-band subcarriers sit at exactly 0; averaging them into the
    amplitude scale would deflate it and inflate the ratio."""
    amp = np.full(64, 20.0)
    amp[:16] = 0.0                     # a quarter of the band is dead
    _, relative, scale = baseline_noise_stats({"amp_mean": amp, "energy_mean": 2.0})
    assert scale == pytest.approx(20.0), "dead subcarriers must not drag the scale down"
    assert relative == pytest.approx(0.1)


def test_noise_floor_wording_does_not_overclaim():
    """A loud floor correlates with accuracy only -0.50, and the two loudest
    sessions still scored 94-95%. The warning must describe a risk regime,
    not predict failure."""
    _, _, detail = assess_noise_floor(6.0)
    assert "does not mean detection will fail" in detail.lower()


def test_real_baseline_flows_through_to_a_grade():
    """End-to-end: a baseline from compute_baseline must carry the keys
    baseline_noise_stats needs, or the live warning silently never fires."""
    frames = np.random.default_rng(3).normal(20, 2, (40, 16))
    base = compute_baseline(frames, np.full(40, -40.0))
    absolute, relative, scale = baseline_noise_stats(base)
    assert scale > 0 and np.isfinite(relative)
    assert relative == pytest.approx(absolute / scale)
    assert assess_noise_floor(relative)[0] in ("quiet", "moderate", "loud")


def test_smoother_requires_majority_to_flip():
    sm = PredictionSmoother(size=5)
    for _ in range(5):
        sm.update(0)
    assert sm.confirmed == 0
    sm.update(2)                             # a single noisy window
    assert sm.confirmed == 0, "one window must not flip the confirmed state"
    for _ in range(3):
        sm.update(2)
    assert sm.confirmed == 2, "a sustained majority must flip it"


def test_rolling_calibrator_ignores_moving_frames():
    roller = RollingCalibrator(calib_frames=5)
    amps = np.full(8, 20.0)
    for _ in range(10):
        assert roller.observe(2, amps, -40.0, None) is None, \
            "MOVING frames must never contribute to the baseline"


def test_rolling_calibrator_floor_prevents_sensitivity_ratchet():
    """A blended baseline must never become MORE sensitive than the original
    deliberate 'leave the room' calibration - that ratchet is what used to
    drift the live system into predicting MOVING in an empty room."""
    rng = np.random.default_rng(1)
    startup = compute_baseline(rng.normal(20, 2.0, (40, 8)), np.full(40, -40.0))
    roller = RollingCalibrator(calib_frames=5, blend_alpha=0.9)
    roller.set_floor_reference(startup)
    quiet = rng.normal(20, 0.01, (8,))       # an unusually quiet sample
    new = None
    for _ in range(5):
        new = roller.observe(0, quiet, -40.0, startup)
    assert new is not None
    assert np.all(new["amp_std"] >= startup["amp_std"] * roller.min_std_fraction)


# --------------------------------------------------------------------------
# pump(): live control flow under failure, driven with synthetic frames
# --------------------------------------------------------------------------

class _StubModel:
    """Always says EMPTY, and enforces the same feature-count contract a real
    RandomForest does - so a test that wrongly lets a mismatched window reach
    the model fails loudly instead of passing by luck."""
    classes_ = np.array([0, 2])

    def __init__(self, n_features):
        self._n = n_features

    def predict_proba(self, feat):
        if feat.shape[1] != self._n:
            raise ValueError(f"X has {feat.shape[1]} features, expected {self._n}")
        return np.array([[0.9, 0.1]])


def _bundle(n_sub):
    return {
        "model": _StubModel(2 * n_sub + 10),
        "amp_columns": [f"amp_{i}" for i in range(n_sub)],
        "frame_hz": 10.0,
        "window_seconds": 0.75,
        "calib_seconds": 1.0,
    }


def _frame(n_sub, seed):
    rng = np.random.default_rng(seed)
    return {"type": "frame", "rssi": -40.0, "amps": rng.normal(20, 1, n_sub)}


async def _drive(scenario, n_sub=8, extra_nodes=None):
    q = queue.Queue()
    combined_state = {"A": None}
    if extra_nodes:
        combined_state.update(extra_nodes)
    last_combined = [None]
    task = asyncio.create_task(
        S.pump(q, set(), _bundle(n_sub), {}, {"recalibrate": False}, "A",
               combined_state, last_combined, set()))
    try:
        await scenario(q, combined_state, last_combined)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_pump_reaches_empty():
    async def scenario(q, state, _last):
        for i in range(60):
            q.put(_frame(8, i))
        await asyncio.sleep(0.6)
        assert state["A"] == "EMPTY"
    await _drive(scenario)


@pytest.mark.asyncio
async def test_silent_node_goes_offline_and_stops_vetoing(monkeypatch):
    monkeypatch.setattr(S, "NODE_STALE_SECONDS", 0.3)

    async def scenario(q, state, last):
        for i in range(60):
            q.put(_frame(8, i))
        await asyncio.sleep(0.5)
        await asyncio.sleep(0.6)             # then go silent past the threshold
        assert state["A"] == S.OFFLINE
        assert last[0] == "EMPTY", "healthy node B must still be able to report EMPTY"
    await _drive(scenario, extra_nodes={"B": "EMPTY"})


@pytest.mark.asyncio
async def test_subcarrier_change_relatches_instead_of_dying(monkeypatch):
    """The AP renegotiating 20MHz<->40MHz mid-session used to make every
    subsequent frame fail a length check and get dropped - the node went
    permanently, silently dead."""
    monkeypatch.setattr(S, "SUBCARRIER_RELATCH_FRAMES", 5)

    async def scenario(q, state, _last):
        for i in range(40):
            q.put(_frame(8, i))
        await asyncio.sleep(0.5)
        assert state["A"] == "EMPTY"
        for i in range(60):
            q.put(_frame(12, 1000 + i))     # channel width changed
        await asyncio.sleep(0.8)
        assert state["A"] is None, "should re-latch and recalibrate, not freeze"
        for i in range(60):
            q.put(_frame(8, 2000 + i))      # and back again
        await asyncio.sleep(0.8)
        assert state["A"] == "EMPTY", "should recover once the count matches"
    await _drive(scenario)


@pytest.mark.asyncio
async def test_isolated_bad_frames_do_not_trigger_relatch():
    async def scenario(q, state, _last):
        for i in range(40):
            q.put(_frame(8, i))
            if i % 10 == 0:
                q.put(_frame(9, 500 + i))   # one-off corruption
        await asyncio.sleep(0.6)
        assert state["A"] == "EMPTY"
    await _drive(scenario)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
