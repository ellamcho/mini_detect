"""Summary plots of detection results: one dot per cell and a bar at the mean,
with conditions side by side. No GUI code here.

Two ways to split the cells into figures:

- by condition: every cell from every day, grouped by condition;
- by day: one figure per session folder (recording day), comparing the
  conditions recorded that day;
- before and after: cells recorded twice, such as 20261117-2 and
  20261117-2_picro, joined by a line and compared with a paired t test.
"""
from __future__ import annotations

import csv
import os
import re
import warnings
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy import stats

from . import detection as det

BY_CONDITION = "condition"
BY_DAY = "day"
BY_PAIR = "pair"
BASE_LABEL = "Base"

# (axis label, value for one CellResult)
METRICS = (
    ("Hz (Minis/s)", lambda r: r.frequency_hz),
    ("Amplitude (pA)", lambda r: r.mean_amplitude),
    ("Rs (MΩ)", lambda r: r.mean_rs),
    ("Rm (MΩ)", lambda r: r.mean_rm),
    ("Cm (pF)", lambda r: r.mean_cm),
)

COLORS = ("#E6C3B5", "#27557F", "#EFB265", "#A8BFDD",
          "#8DB596", "#B49CC8", "#D9837B", "#8C8C8C")

MARKER_PT = 7.5          # dot diameter
AXES_HEIGHT_IN = 2.3
GROUP_WIDTH_IN = 0.5     # room along the x axis for each condition
FONT_SIZE = 11
LINE_WIDTH = 1.5


def _pick_font() -> str:
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "sans-serif"


FONT = _pick_font()
# Keep text as text in PDF and SVG so labels can be edited in Illustrator.
SAVE_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


@dataclass
class FigureSpec:
    title: str
    filename: str   # without tag or extension
    mode: str       # event type, for the shared axis limits
    groups: list    # [(condition, colour, [CellResult, ...])] left to right
    pairs: list = None    # before-and-after figures: [(before, after), ...]
    treatment: str = ""   # ... and what was done between the two recordings


def condition_of(cell: str, conditions) -> str:
    if conditions is None:
        return "All"
    return conditions.get(cell, "Unknown")


def condition_order(results, conditions) -> list:
    """Conditions that have cells, in the order the metadata file lists them."""
    present = {condition_of(r.cell, conditions) for r in results}
    listed = list(dict.fromkeys(conditions.values())) if conditions else []
    order = [c for c in listed if c in present]
    return order + sorted(present.difference(order), key=det.natural_key)


def build_figures(results, conditions=None, by=BY_CONDITION) -> list:
    """Describe the figures to draw. One per event type, and with by=BY_DAY one
    per recording day as well."""
    if by == BY_PAIR:
        return _pair_figures(results)
    order = condition_order(results, conditions)
    colors = {c: COLORS[i % len(COLORS)] for i, c in enumerate(order)}
    specs = []
    for mode in dict.fromkeys(r.mode for r in results):
        of_mode = [r for r in results if r.mode == mode]
        if by == BY_DAY:
            days = sorted({r.session for r in of_mode}, key=det.natural_key)
            subsets = [(day or "Unnamed day", day or "day",
                        [r for r in of_mode if r.session == day]) for day in days]
        else:
            subsets = [("all days", "by_condition", of_mode)]
        for label, stem, subset in subsets:
            groups = []
            for condition in order:
                members = [r for r in subset if condition_of(r.cell, conditions) == condition]
                if members:
                    groups.append((condition, colors[condition], members))
            n = len(subset)
            specs.append(FigureSpec(
                title=f"{mode}, {label}  (n = {n} cell{'' if n == 1 else 's'})",
                filename=f"{stem}_{mode}", mode=mode, groups=groups))
    return specs


def find_pairs(results) -> dict:
    """Cells recorded twice: {(event type, treatment): [(before, after), ...]}.

    A cell whose name is another cell's name plus an underscore and a label is
    the "after" recording of that cell, and the label is the treatment:
    20261117-2_picro is 20261117-2 after picro.
    """
    pairs = {}
    for mode in dict.fromkeys(r.mode for r in results):
        by_name = {r.cell: r for r in results if r.mode == mode}
        for name in sorted(by_name, key=det.natural_key):
            bases = [base for base in by_name if name.startswith(base + "_")]
            if bases:
                base = min(bases, key=len)
                pairs.setdefault((mode, name[len(base) + 1:]), []).append(
                    (by_name[base], by_name[name]))
    return pairs


def _pair_figures(results) -> list:
    found = find_pairs(results)
    treatments = sorted({treatment for _, treatment in found}, key=det.natural_key)
    specs = []
    for (mode, treatment), pairs in found.items():
        color = COLORS[1 + treatments.index(treatment) % (len(COLORS) - 1)]
        groups = [(BASE_LABEL, COLORS[0], [before for before, _ in pairs]),
                  (treatment, color, [after for _, after in pairs])]
        n = len(pairs)
        specs.append(FigureSpec(
            title=f"{mode}, before and after {treatment}  (n = {n} pair{'' if n == 1 else 's'})",
            filename=f"paired_{treatment}_{mode}", mode=mode, groups=groups,
            pairs=pairs, treatment=treatment))
    return specs


def paired_stats(spec) -> list:
    """Paired t test (two-sided) of after against before, for each measurement."""
    rows = []
    for label, getter in METRICS:
        before = np.array([getter(b) for b, _ in spec.pairs], dtype=float)
        after = np.array([getter(a) for _, a in spec.pairs], dtype=float)
        both = ~(np.isnan(before) | np.isnan(after))
        before, after = before[both], after[both]
        t = p = np.nan
        if len(before) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # identical pairs: no test is possible
                test = stats.ttest_rel(after, before)
            t, p = float(test.statistic), float(test.pvalue)
        rows.append({
            "Event_Type": spec.mode,
            "Treatment": spec.treatment,
            "Measurement": label,
            "N_Pairs": int(len(before)),
            "Mean_Before": float(before.mean()) if len(before) else np.nan,
            "Mean_After": float(after.mean()) if len(after) else np.nan,
            "Mean_Difference": float((after - before).mean()) if len(before) else np.nan,
            "t": t,
            "p": p,
        })
    return rows


def stars(p) -> str:
    """Prism-style summary of a p value."""
    if p is None or np.isnan(p):
        return ""
    for limit, text in ((0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < limit:
            return text
    return "ns"


def save_paired_stats(specs, path) -> None:
    rows = [row for spec in specs if spec.pairs for row in paired_stats(spec)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["Event_Type"])
        writer.writeheader()
        writer.writerows(rows)


def _values(members, getter) -> np.ndarray:
    values = np.array([getter(r) for r in members], dtype=float)
    return values[~np.isnan(values)]


def axis_limits(results, headroom: float = 0.0) -> dict:
    """{(event type, metric index): tick positions}, shared by every figure so
    days can be compared at a glance. Each axis starts at zero. `headroom` adds
    that fraction of space above the data, for a significance bracket."""
    limits = {}
    for mode in dict.fromkeys(r.mode for r in results):
        of_mode = [r for r in results if r.mode == mode]
        for i, (_, getter) in enumerate(METRICS):
            values = _values(of_mode, getter)
            low = min(0.0, float(values.min()) * 1.05) if len(values) else 0.0
            high = max(0.0, float(values.max()) * (1.05 + headroom)) if len(values) else 0.0
            if low == high:
                high = 1.0
            limits[(mode, i)] = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(
                low, high)
    return limits


def _swarm(values, span, group_width_in) -> np.ndarray:
    """Sideways offsets (in x-axis units) so dots with similar values sit next
    to each other instead of on top of each other."""
    y_unit = span * MARKER_PT / (AXES_HEIGHT_IN * 72)      # one dot height, in data units
    x_unit = MARKER_PT / (group_width_in * 72)             # one dot width, in x units
    step = 0.9 * x_unit
    tries = [0.0]
    for k in range(1, 4):
        tries += [k * step, -k * step]
    tries = [t for t in tries if abs(t) <= 0.36] or [0.0]
    offsets = np.zeros(len(values))
    placed = []
    for i in np.argsort(values):
        for dx in tries:
            if all(((dx - px) / x_unit) ** 2 + ((values[i] - py) / y_unit) ** 2 >= 0.8
                   for px, py in placed):
                break
        else:
            dx = 0.0
        offsets[i] = dx
        placed.append((dx, values[i]))
    return offsets


def figure_size(spec) -> tuple:
    n_groups = max(len(spec.groups), 1)
    longest = max((len(str(label)) for label, _, _ in spec.groups), default=3)
    # Rotated labels reach down and to the left of their tick.
    panel = 0.9 + GROUP_WIDTH_IN * n_groups + max(0.0, 0.07 * longest - 0.35)
    height = 0.55 + AXES_HEIGHT_IN + 0.3 + 0.055 * longest
    return panel * len(METRICS), height


def make_figure(spec, limits) -> Figure:
    """Draw one figure: a panel per measurement, a dot per cell and a black bar
    at each condition's mean."""
    fig = Figure(figsize=figure_size(spec), dpi=100, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.08, wspace=0.04)
    axes = fig.subplots(1, len(METRICS))
    text = {"fontfamily": FONT, "fontsize": FONT_SIZE}
    labels = [str(label) for label, _, _ in spec.groups]
    significance = [stars(row["p"]) for row in paired_stats(spec)] if spec.pairs else []
    for i, (ax, (ylabel, getter)) in enumerate(zip(axes, METRICS)):
        ticks = limits[(spec.mode, i)]
        low, high = float(ticks[0]), float(ticks[-1])
        if spec.pairs:
            _draw_pairs(ax, spec, getter, high - low, significance[i])
        for x, (_, color, members) in enumerate(() if spec.pairs else spec.groups):
            values = _values(members, getter)
            if not len(values):
                continue
            ax.scatter(x + _swarm(values, high - low, GROUP_WIDTH_IN), values,
                       s=MARKER_PT ** 2, color=color, edgecolors="none",
                       zorder=3, clip_on=False)
            ax.hlines(float(values.mean()), x - 0.3, x + 0.3, color="black",
                      linewidth=2, zorder=4)
        ax.set_xlim(-0.6, max(len(labels), 1) - 0.4)
        ax.set_ylim(low, high)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor", **text)
        ax.set_ylabel(ylabel, fontfamily=FONT, fontsize=FONT_SIZE + 1)
        for tick in ax.get_yticklabels():
            tick.set(**text)
        ax.tick_params(direction="out", length=6, width=LINE_WIDTH, color="black")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(LINE_WIDTH)
    fig.suptitle(spec.title, fontfamily=FONT, fontsize=FONT_SIZE, x=0.01, ha="left")
    return fig


def _draw_pairs(ax, spec, getter, span, significance) -> None:
    """Before on the left, after on the right, a line joining each cell's two
    recordings, and a bracket with the paired t test result above."""
    before = np.array([getter(b) for b, _ in spec.pairs], dtype=float)
    after = np.array([getter(a) for _, a in spec.pairs], dtype=float)
    for b, a in zip(before, after):
        if not (np.isnan(b) or np.isnan(a)):
            ax.plot([0, 1], [b, a], color="black", linewidth=1, zorder=2, clip_on=False)
    for x, values, (_, color, _) in zip((0, 1), (before, after), spec.groups):
        values = values[~np.isnan(values)]
        ax.scatter(np.full(len(values), x), values, s=MARKER_PT ** 2, color=color,
                   edgecolors="none", zorder=3, clip_on=False)
    shown = np.concatenate([before, after])
    shown = shown[~np.isnan(shown)]
    if significance and len(shown):
        y = float(shown.max()) + 0.07 * span
        tip = 0.035 * span
        ax.plot([0, 0, 1, 1], [y, y + tip, y + tip, y], color="black",
                linewidth=1.2, zorder=4, clip_on=False)
        if significance == "ns":
            ax.text(0.5, y + tip + 0.02 * span, "ns", ha="center", va="bottom",
                    fontfamily=FONT, fontsize=FONT_SIZE + 1)
        else:  # asterisks sit high in their line of text, so tuck them down
            ax.text(0.5, y + tip - 0.035 * span, significance, ha="center", va="bottom",
                    fontfamily=FONT, fontsize=FONT_SIZE + 4, fontweight="bold")


def save_figures(specs, limits, folder, tag, formats=("pdf", "png")) -> list:
    """Write every figure to `folder` in each format. Returns the paths written."""
    os.makedirs(folder, exist_ok=True)
    written = []
    with matplotlib.rc_context(SAVE_RC):
        for spec in specs:
            fig = make_figure(spec, limits)
            stem = re.sub(r'[\\/:*?"<>|\s]+', "_", f"{tag}_{spec.filename}".strip("_"))
            for extension in formats:
                path = os.path.join(folder, f"{stem}.{extension}")
                fig.savefig(path, dpi=300)
                written.append(path)
    return written
