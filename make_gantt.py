"""
Project timeline for the report, drawn as a bordered phase table.

EVERY DATE COMES FROM THE REPOSITORY OR THE PROJECT NOTES, not from memory:

  * recording dates      the `session_id` column of each session's
                         all_csi_data.csv (it embeds YYYYMMDD_HHMMSS)
  * development dates    `git log` author dates
  * everything else      the student's own project notes

TWO SOURCES, KEPT VISUALLY DISTINCT
-----------------------------------
Filled bars are dated by the repository. Outlined bars come from the project
notes, where the repository holds no record - presenting them identically would
overstate what the evidence supports. Each task carries source="repo" or
source="notes", and the key says which is which.

    python make_gantt.py                # bordered table (default)
    python make_gantt.py --style bars   # classic Gantt bars

Output: a 300 dpi PNG sized for a full page of an A4 report.
"""

import argparse
import math
import os
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

# --------------------------------------------------------------------------
# Palette, matching the report's other figures
# --------------------------------------------------------------------------
C_INIT   = "#5b6670"   # project initiation
C_DATA   = "#0f62d0"   # data collection
C_MODEL  = "#157f3d"   # modelling
C_LIVE   = "#6b46a8"   # live system and robustness
C_EMBED  = "#c2352f"   # embedded deployment
C_DOC    = "#0e7c86"   # documentation

C_HEADER = "#d6d6d6"   # header and phase-row fill
C_RULE   = "#8b8e95"   # cell borders
C_TEXT   = "#14161a"
C_ACT    = "#1f3a5f"   # activity labels
C_MUTED  = "#55585f"

PHASES = [
    ("PHASE 1: PROJECT INITIATION",          C_INIT),
    ("PHASE 2: DATA COLLECTION",             C_DATA),
    ("PHASE 3: MODELLING AND VALIDATION",    C_MODEL),
    ("PHASE 4: LIVE SYSTEM AND ROBUSTNESS",  C_LIVE),
    ("PHASE 5: EMBEDDED DEPLOYMENT",         C_EMBED),
    ("PHASE 6: DOCUMENTATION AND REPORTING", C_DOC),
]

P1, P2, P3 = PHASES[0][0], PHASES[1][0], PHASES[2][0]
P4, P5, P6 = PHASES[3][0], PHASES[4][0], PHASES[5][0]


def T(name, phase, start, end, source, evidence):
    return dict(name=name, phase=phase, start=start, end=end,
                source=source, evidence=evidence)


TASKS = [
    # --- Phase 1 ----------------------------------------------------------
    T("Literature review and article search", P1,
      date(2026, 7, 5), date(2026, 7, 21), "notes",
      "project notes: subject settled, searching for articles"),
    T("ESP-IDF toolchain, first CSI capture", P1,
      date(2026, 7, 21), date(2026, 7, 23), "repo",
      "earliest dated working folder"),
    T("Exploratory dataset (later discarded)", P1,
      date(2026, 7, 23), date(2026, 7, 24), "repo",
      "csi_dataset_20260723_184612"),

    # --- Phase 2 ----------------------------------------------------------
    T("Labelling protocol and collector", P2,
      date(2026, 7, 24), date(2026, 7, 26), "repo",
      "collector precedes first session"),
    T("Room 1 sessions 1-8 recorded", P2,
      date(2026, 7, 26), date(2026, 7, 30), "repo",
      "session_id 20260726_* to 20260729_*"),
    T("Room 2 sessions recorded", P2,
      date(2026, 8, 4), date(2026, 8, 5), "repo",
      "session_id 20260804_*"),
    T("Room 2 data collection and model improvements", P2,
      date(2026, 8, 5), date(2026, 8, 12), "notes",
      "project notes: continued Room 2 work, no commits in this window"),

    # --- Phase 3 ----------------------------------------------------------
    T("Feature extraction and calibration", P3,
      date(2026, 7, 24), date(2026, 7, 29), "repo",
      "csi_common.py in initial commit"),
    T("Random Forest, leave-one-session-out", P3,
      date(2026, 7, 26), date(2026, 7, 30), "repo",
      "train_model.py in initial commit"),
    T("Per-block recalibration", P3,
      date(2026, 7, 27), date(2026, 7, 29), "repo",
      "PROJECT_HISTORY section 8"),
    T("Model evaluation report", P3,
      date(2026, 8, 4), date(2026, 8, 5), "repo",
      "commit: add model_evaluation.py"),
    T("Retrain on ten sessions, cross-room test", P3,
      date(2026, 8, 4), date(2026, 8, 5), "repo",
      "commit: add cross-room data"),

    # --- Phase 4 ----------------------------------------------------------
    T("Live inference and browser dashboard", P4,
      date(2026, 7, 29), date(2026, 8, 5), "repo",
      "initial commit to 4 Aug commits"),
    T("Second node, OR fusion", P4,
      date(2026, 8, 12), date(2026, 8, 13), "repo", "commit: two esp"),
    T("Robustness audit and 72-test suite", P4,
      date(2026, 8, 12), date(2026, 8, 14), "repo", "commits 12-13 Aug"),
    T("Scale-free noise-floor metric", P4,
      date(2026, 8, 13), date(2026, 8, 14), "repo", "commit: analysis scripts"),
    T("Per-node mute (automatic version rejected)", P4,
      date(2026, 8, 13), date(2026, 8, 14), "repo", "commit: mute button"),
    T("Zone-detection scaffolding (unproven)", P4,
      date(2026, 8, 13), date(2026, 8, 14), "repo", "commit: 2-zone add-on"),
    T("Sustained-occupancy alerting", P4,
      date(2026, 8, 19), date(2026, 8, 20), "repo", "commits 19 Aug"),

    # --- Phase 5 ----------------------------------------------------------
    T("Embedded feasibility study", P5,
      date(2026, 8, 14), date(2026, 8, 17), "notes",
      "PROJECT_HISTORY section 27 records this preceding the C port, undated"),
    T("Model exported to C, on-device parity", P5,
      date(2026, 8, 17), date(2026, 8, 18), "repo", "commits 17 Aug"),
    T("Standalone detection and web server", P5,
      date(2026, 8, 17), date(2026, 8, 19), "repo", "commits 17-18 Aug"),
    T("Recurring failures fixed at source", P5,
      date(2026, 8, 18), date(2026, 8, 19), "repo",
      "commits: 20MHz pin, DHCP gateway"),

    # --- Phase 6 ----------------------------------------------------------
    T("Development history and README", P6,
      date(2026, 8, 20), date(2026, 8, 21), "repo",
      "commit: document sections 28-29"),
    T("Result figures and report", P6,
      date(2026, 8, 20), date(2026, 8, 24), "repo",
      "figure regeneration, report drafting"),
]

# --------------------------------------------------------------------------
# Simple style: the same fifty days, rolled up so the chart reads at a glance.
# Each row's span is COMPUTED from its member tasks - no dates are restated
# here, so the simple chart cannot drift away from the detailed one.
# --------------------------------------------------------------------------
SIMPLE_GROUPS = [
    ("Literature review and planning",       P1, ["Literature review"]),
    ("Setup and first CSI capture",          P1, ["ESP-IDF toolchain",
                                                  "Exploratory dataset"]),
    ("Data collection, Room 1",              P2, ["Labelling protocol",
                                                  "Room 1 sessions"]),
    ("Feature extraction and model training", P3, ["Feature extraction",
                                                   "Random Forest",
                                                   "Per-block recalibration"]),
    ("Live inference and dashboard",         P4, ["Live inference"]),
    ("Room 2 data and model improvements",   P2, ["Room 2 sessions",
                                                  "Room 2 data collection",
                                                  "Model evaluation report",
                                                  "Retrain on ten sessions"]),
    ("Second node, robustness and alerting", P4, ["Second node",
                                                  "Robustness audit",
                                                  "Scale-free",
                                                  "Per-node mute",
                                                  "Zone-detection",
                                                  "Sustained-occupancy"]),
    ("Embedded deployment on the ESP32",     P5, ["Embedded feasibility",
                                                  "Model exported to C",
                                                  "Standalone detection",
                                                  "Recurring failures"]),
    ("Documentation and report",             P6, ["Development history",
                                                  "Result figures"]),
]


def roll_up(tasks):
    """Collapse TASKS into the simple rows, checking none is lost or reused."""
    rows, claimed = [], []
    for label, phase, prefixes in SIMPLE_GROUPS:
        members = [t for t in tasks
                   if any(t["name"].startswith(p) for p in prefixes)]
        if not members:
            raise SystemExit(f"no task matches group {label!r}")
        claimed += members
        rows.append(dict(name=label, phase=phase,
                         start=min(t["start"] for t in members),
                         end=max(t["end"] for t in members),
                         members=len(members)))
    names = [t["name"] for t in claimed]
    missed = [t["name"] for t in tasks if t["name"] not in names]
    dupes = {n for n in names if names.count(n) > 1}
    if missed:
        print(f"  WARNING: {len(missed)} task(s) in no simple group: {missed}")
    if dupes:
        print(f"  WARNING: task(s) in more than one group: {sorted(dupes)}")
    return rows


MILESTONES = [
    ("First validated model;\nrepository published", date(2026, 7, 29)),
    ("Ten-session model,\n94.25 % held out",         date(2026, 8, 4)),
    ("On-device parity\n3302/3302",                  date(2026, 8, 17)),
    ("Alerting complete",                            date(2026, 8, 19)),
]

TITLE = "Wi-Fi CSI Motion Detection System — Project Timeline"


# ==========================================================================
# Simple style - one bar per rolled-up row, week columns, nothing else
# ==========================================================================
def build_simple(tasks, out_path, dpi=300):
    phase_colour = dict(PHASES)
    rows = roll_up(tasks)

    span_start = min(t["start"] for t in tasks)
    span_end = max(t["end"] for t in tasks)
    n_weeks = math.ceil((span_end - span_start).days / 7.0)
    n = len(rows)

    def wx(d):
        return (d - span_start).days / 7.0

    LEFT = 3.5                          # room for the row labels
    fig, ax = plt.subplots(figsize=(11.6, 0.60 * n + 2.4))
    ax.set_xlim(-LEFT, n_weeks + 0.08)
    ax.set_ylim(-0.55, n + 0.85)
    ax.axis("off")

    # Week columns: dashed rules, label centred over each column.
    for w in range(n_weeks + 1):
        if w:
            ax.plot([w, w], [-0.30, n + 0.10], color="#c2c6cc",
                    linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)
        if w < n_weeks:
            ax.text(w + 0.5, n + 0.34, f"Week {w + 1}", ha="center",
                    va="bottom", fontsize=11, color=C_TEXT)

    # The single solid axis, as in a hand-drawn tracker.
    ax.plot([0, 0], [-0.30, n + 0.22], color=C_TEXT, linewidth=1.8, zorder=3)
    ax.plot([0, n_weeks], [-0.30, -0.30], color=C_TEXT, linewidth=1.8, zorder=3)

    labels = []
    for i, r in enumerate(rows):
        y = n - i - 0.5
        labels.append(ax.text(-0.18, y, r["name"], ha="right", va="center",
                              fontsize=10.5, color=C_ACT))
        x0, x1 = wx(r["start"]), wx(r["end"])
        ax.add_patch(FancyBboxPatch(
            (x0, y - 0.21), max(x1 - x0, 0.10), 0.42,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=phase_colour[r["phase"]], edgecolor="none", zorder=4))

    weeks = (span_end - span_start).days / 7.0
    ax.text(-LEFT, n + 1.55, "Wi-Fi CSI Motion Detection", fontsize=21,
            fontweight="bold", color=C_TEXT, va="bottom")
    ax.text(-LEFT, n + 1.05,
            f"Project timeline  ·  {span_start:%d %B} to {span_end:%d %B %Y}  ·  "
            f"{(span_end - span_start).days} days, {weeks:.1f} weeks",
            fontsize=12, color=C_MUTED, va="bottom")
    ax.text(-LEFT, -1.05,
            "Dates come from the recording timestamps and the commit history, "
            "except the 5-21 July, 5-11 August and 14-17 August windows,\n"
            "which come from the project notes.",
            fontsize=8.8, color=C_MUTED, va="top", linespacing=1.5)

    # Same measured check as the table style: a label that crosses the axis
    # would sit on top of the bars.
    fig.canvas.draw()
    axis_x = ax.transData.transform((-0.02, 0))[0]
    over = [(t.get_text(), t.get_window_extent().x1 - axis_x)
            for t in labels if t.get_window_extent().x1 > axis_x]
    if over:
        worst = max(over, key=lambda o: o[1])
        print(f"  WARNING: {len(over)} row label(s) cross the axis, worst "
              f"{worst[1]:.0f} px: {worst[0]!r}")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ==========================================================================
# Table style
# ==========================================================================
def build_table(tasks, out_path, dpi=300):
    phase_colour = dict(PHASES)
    span_start = min(t["start"] for t in tasks)
    span_end = max(t["end"] for t in tasks)
    n_weeks = int(math.ceil((span_end - span_start).days / 7.0))

    # Rows: title, header, milestone strip, then a phase row + its activities
    ordered = []
    for phase, _c in PHASES:
        members = [t for t in tasks if t["phase"] == phase]
        if not members:
            continue
        ordered.append(("phase", phase))
        for t in sorted(members, key=lambda x: x["start"]):
            ordered.append(("task", t))

    # The milestone strip is three rows deep so its captions sit INSIDE the
    # table. Annotated upward from a single-height row they ran straight
    # through the column header and the title.
    MS_H = 3.0
    n_rows = len(ordered) + 2 + MS_H   # + title + header + milestone strip
    # The activity column has to hold the longest label without spilling over
    # the divider into Week 1 - "Room 2 data collection and model improvements"
    # is the one that sets the width.
    ACT_W, WEEK_W = 3.90, 1.0
    total_w = ACT_W + n_weeks * WEEK_W

    fig, ax = plt.subplots(figsize=(1.03 * total_w, 0.315 * n_rows + 1.35))
    ax.set_xlim(-0.02, total_w + 0.02)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    def cell(x, y, w, h, fill="white", lw=0.7, ec=C_RULE, z=1):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=ec,
                               linewidth=lw, zorder=z))

    def week_x(d):
        """Fractional horizontal position of a date within the week grid."""
        return ACT_W + ((d - span_start).days / 7.0) * WEEK_W

    row_top = n_rows

    # ---- title -----------------------------------------------------------
    row_top -= 1
    cell(0, row_top, total_w, 1, fill="white", lw=1.4, ec=C_TEXT)
    ax.text(total_w / 2, row_top + 0.5, TITLE, ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_TEXT, zorder=3)

    # ---- column header ---------------------------------------------------
    row_top -= 1
    cell(0, row_top, ACT_W, 1, fill=C_HEADER)
    ax.text(ACT_W / 2, row_top + 0.5, "Activity", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_TEXT, zorder=3)
    for w in range(n_weeks):
        x = ACT_W + w * WEEK_W
        cell(x, row_top, WEEK_W, 1, fill=C_HEADER)
        wk_from = span_start + timedelta(days=7 * w)
        wk_to = wk_from + timedelta(days=6)
        ax.text(x + WEEK_W / 2, row_top + 0.62, f"Week {w + 1}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=C_TEXT, zorder=3)
        ax.text(x + WEEK_W / 2, row_top + 0.27,
                f"{wk_from:%d %b} – {wk_to:%d %b}",
                ha="center", va="center", fontsize=6.8, color=C_MUTED, zorder=3)

    # ---- milestone strip -------------------------------------------------
    row_top -= MS_H
    cell(0, row_top, ACT_W, MS_H, fill="white")
    ax.text(ACT_W - 0.12, row_top + MS_H - 0.5, "Milestones", ha="right",
            va="center", fontsize=8.5, style="italic", color=C_MUTED, zorder=3)
    for w in range(n_weeks):
        cell(ACT_W + w * WEEK_W, row_top, WEEK_W, MS_H, fill="white")
    for i, (label, when) in enumerate(sorted(MILESTONES, key=lambda m: m[1])):
        x = week_x(when)
        y_d = row_top + MS_H - 0.45
        ax.plot([x], [y_d], marker="D", markersize=7, color=C_TEXT, zorder=5)
        # Two caption heights: the 17 and 19 August milestones fall in the same
        # week column and their captions are wider than it.
        y_t = y_d - (0.40 if i % 2 == 0 else 1.30)
        ax.plot([x, x], [y_d - 0.12, y_t + 0.04], color=C_TEXT,
                linewidth=0.7, alpha=0.5, zorder=4)
        ax.text(x, y_t, label, ha="center", va="top", fontsize=7.2,
                color=C_TEXT, linespacing=1.3, zorder=6)

    # ---- phase and activity rows ----------------------------------------
    act_texts = []                      # measured against the divider below
    for kind, item in ordered:
        row_top -= 1
        if kind == "phase":
            cell(0, row_top, total_w, 1, fill=C_HEADER)
            ax.text(0.16, row_top + 0.5, item, ha="left", va="center",
                    fontsize=9, fontweight="bold", color=C_TEXT, zorder=3)
            continue

        t = item
        cell(0, row_top, ACT_W, 1, fill="white")
        act_texts.append(ax.text(0.16, row_top + 0.5, t["name"], ha="left",
                                 va="center", fontsize=8.0, color=C_ACT,
                                 zorder=3))
        for w in range(n_weeks):
            cell(ACT_W + w * WEEK_W, row_top, WEEK_W, 1, fill="white")

        x0, x1 = week_x(t["start"]), week_x(t["end"])
        width = max(x1 - x0, 0.16)      # keep one-day tasks visible
        colour = phase_colour[t["phase"]]
        notes = (t["source"] == "notes")

        bar = FancyBboxPatch(
            (x0 + 0.05, row_top + 0.24), width - 0.10, 0.52,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor="white" if notes else colour,
            edgecolor=colour, linewidth=1.6 if notes else 0.9,
            linestyle=(0, (2.2, 1.4)) if notes else "solid", zorder=4)
        ax.add_patch(bar)

        # Dots inside the bar, echoing the reference layout. Count scales with
        # width so a short task does not end up with dots outside its bar.
        n_dots = max(1, min(5, int((width - 0.10) / 0.17)))
        cx = x0 + 0.05 + (width - 0.10) / 2
        step = 0.145
        for k in range(n_dots):
            ax.plot([cx + (k - (n_dots - 1) / 2) * step], [row_top + 0.50],
                    marker="o", markersize=2.6,
                    color=colour if notes else "white", zorder=5)

    # ---- key -------------------------------------------------------------
    handles = [Patch(facecolor=c, edgecolor=c,
                     label=p.split(": ")[1].capitalize()) for p, c in PHASES]
    handles.append(Patch(facecolor="white", edgecolor=C_MUTED, linewidth=1.6,
                         linestyle=(0, (2.2, 1.4)),
                         label="From project notes (not dated by the repository)"))
    handles.append(plt.Line2D([], [], marker="D", linestyle="none",
                              color=C_TEXT, markersize=7, label="Milestone"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.012), ncol=3, frameon=False, fontsize=8.6)

    # Footnote below the key, not level with it. In data coordinates one row is
    # one unit, so the three-row key clears roughly two units.
    weeks = (span_end - span_start).days / 7.0
    ax.text(0, -2.9,
            f"{span_start:%d %B} to {span_end:%d %B %Y}   "
            f"({(span_end - span_start).days} days, {weeks:.1f} weeks).   "
            "Filled bars are dated by session identifiers and commit history.",
            fontsize=8.2, color=C_MUTED, va="top", clip_on=False)

    # Measure rather than eyeball: an activity label that runs past the column
    # divider spills into Week 1, and that is only visible if you open the PNG.
    fig.canvas.draw()
    divider = ax.transData.transform((ACT_W - 0.10, 0))[0]
    overflow = [(t.get_text(), t.get_window_extent().x1 - divider)
                for t in act_texts if t.get_window_extent().x1 > divider]
    if overflow:
        worst = max(overflow, key=lambda o: o[1])
        print(f"  WARNING: {len(overflow)} activity label(s) overflow the "
              f"column, worst by {worst[1]:.0f} px: {worst[0]!r}")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ==========================================================================
# Classic bar style, kept available via --style bars
# ==========================================================================
def build_bars(tasks, out_path, dpi=300):
    phase_colour = dict(PHASES)
    rows = list(reversed(tasks))

    fig, ax = plt.subplots(figsize=(10.2, 3.0 + 0.235 * len(rows)))
    plt.rcParams["hatch.linewidth"] = 1.1
    for i, t in enumerate(rows):
        notes = (t["source"] == "notes")
        ax.barh(i, (t["end"] - t["start"]).days, left=t["start"], height=0.62,
                color=phase_colour[t["phase"]], alpha=0.45 if notes else 0.88,
                hatch="///" if notes else None,
                edgecolor=phase_colour[t["phase"]] if notes else "white",
                linewidth=1.0 if notes else 0.8, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([t["name"] for t in rows], fontsize=10)
    ax.set_ylim(-0.9, len(rows) - 0.1)

    span_start = min(t["start"] for t in tasks)
    span_end = max(t["end"] for t in tasks)
    ax.set_xlim(span_start - timedelta(days=1), span_end + timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="x", color="#d8d8d4", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    top = len(rows) - 0.15
    for rank, (label, when) in enumerate(sorted(MILESTONES, key=lambda m: m[1])):
        ax.axvline(when, color=C_TEXT, linestyle=":", linewidth=1.0, alpha=0.55)
        ax.plot([when], [top], marker="D", markersize=7, color=C_TEXT,
                zorder=6, clip_on=False)
        ax.annotate(label, xy=(when, top),
                    xytext=(0, 10 if rank % 2 == 0 else 30),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=C_TEXT, linespacing=1.25,
                    clip_on=False, annotation_clip=False, zorder=7)

    handles = [Patch(facecolor=c, alpha=0.88, label=p) for p, c in PHASES]
    handles.append(plt.Line2D([], [], marker="D", linestyle="none",
                              color=C_TEXT, markersize=7, label="Milestone"))
    handles.append(Patch(facecolor=C_MUTED, alpha=0.45, hatch="///",
                         edgecolor=C_MUTED, label="From project notes"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
              ncol=3, frameon=False, fontsize=9.5)

    weeks = (span_end - span_start).days / 7.0
    ax.text(0, 1.215, TITLE, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=C_TEXT, va="bottom")
    ax.text(0, 1.158,
            f"{span_start:%d %B} to {span_end:%d %B %Y}   "
            f"({(span_end - span_start).days} days, {weeks:.1f} weeks)\n"
            "Solid bars are dated by session identifiers and commit history; "
            "hatched bars come from the project notes.",
            transform=ax.transAxes, fontsize=9, color=C_MUTED, va="bottom",
            linespacing=1.5)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["simple", "table", "bars"],
                    default="simple")
    ap.add_argument("--out-dir", nargs="+",
                    default=[os.path.join("report", "figures"),
                             os.path.join("report_output", "figures")])
    ap.add_argument("--name", default="28_project_timeline.png")
    args = ap.parse_args()

    build = {"simple": build_simple, "table": build_table,
             "bars": build_bars}[args.style]
    for d in args.out_dir:
        os.makedirs(d, exist_ok=True)
        print("  wrote", build(TASKS, os.path.join(d, args.name)))

    span_start = min(t["start"] for t in TASKS)
    span_end = max(t["end"] for t in TASKS)
    n_notes = sum(1 for t in TASKS if t["source"] == "notes")
    detail = (f"{len(SIMPLE_GROUPS)} rows rolled up from {len(TASKS)} activities"
              if args.style == "simple"
              else f"{len(TASKS)} activities in {len(PHASES)} phases, "
                   f"{len(MILESTONES)} milestones")
    print(f"\n  style={args.style}  {detail}")
    print(f"  span {span_start} to {span_end} = {(span_end - span_start).days} days "
          f"({(span_end - span_start).days / 7.0:.1f} weeks)")
    print(f"  {len(TASKS) - n_notes} repository-dated, {n_notes} from project notes")


if __name__ == "__main__":
    main()
