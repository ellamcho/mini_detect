"""Checks how results are split into figures and that the figures are written."""
import numpy as np

from mini_detect import plots
from mini_detect.analysis import CellResult, SweepResult


def cell(name, day, amplitudes, rs=15.0, mode="EPSC"):
    sweep = SweepResult("x.mat", list(amplitudes), list(range(len(amplitudes))), 100000, 10000.0)
    return CellResult(name, mode, 1, 1, {}, [sweep], [(rs, 300.0, 20.0, 1.0)], day)


RESULTS = [
    cell("d1-1", "d1", [-20, -22, -24]),
    cell("d1-2", "d1", [-30, -31]),
    cell("d1-3", "d1", [-15]),
    cell("d2-1", "d2", [-18, -19]),
    cell("d2-2", "d2", []),           # no events: no amplitude to plot
    cell("d10-1", "d10", [-40]),
    cell("d1-1", "d1", [12, 14], mode="IPSC"),
]
CONDITIONS = {"d1-2": "drug", "d1-1": "base", "d2-1": "base", "d2-2": "drug", "d1-3": "drug"}


def test_figures_by_condition():
    specs = plots.build_figures(RESULTS, CONDITIONS, plots.BY_CONDITION)
    assert [s.filename for s in specs] == ["by_condition_EPSC", "by_condition_IPSC"]
    # Conditions keep the order of the metadata file; unlisted cells come last.
    assert [(label, [r.cell for r in members]) for label, _, members in specs[0].groups] == [
        ("drug", ["d1-2", "d1-3", "d2-2"]), ("base", ["d1-1", "d2-1"]), ("Unknown", ["d10-1"])]
    assert "n = 6 cells" in specs[0].title
    assert [label for label, _, _ in specs[1].groups] == ["base"]


def test_figures_by_day():
    specs = plots.build_figures(RESULTS, CONDITIONS, plots.BY_DAY)
    assert [s.filename for s in specs] == ["d1_EPSC", "d2_EPSC", "d10_EPSC", "d1_IPSC"]
    day2 = specs[1]
    assert [(label, len(members)) for label, _, members in day2.groups] == [
        ("drug", 1), ("base", 1)]
    # A condition has the same colour in every figure.
    colour = {label: c for label, c, _ in specs[0].groups}
    assert all(colour[label] == c for label, c, _ in day2.groups)


def test_without_metadata_everything_is_one_group():
    specs = plots.build_figures(RESULTS, None, plots.BY_CONDITION)
    assert [label for label, _, _ in specs[0].groups] == ["All"]


def test_axes_start_at_zero_and_are_shared():
    limits = plots.axis_limits(RESULTS)
    hz, amplitude = limits[("EPSC", 0)], limits[("EPSC", 1)]
    assert hz[0] == 0 and hz[-1] >= 0.3
    assert amplitude[-1] == 0 and amplitude[0] <= -40
    assert limits[("IPSC", 1)][0] == 0


def test_dots_with_similar_values_are_spread_out():
    offsets = plots._swarm(np.array([10.0, 10.0, 10.0, 50.0]), 60.0, plots.GROUP_WIDTH_IN)
    assert len(set(np.round(offsets[:3], 6))) == 3
    assert offsets[3] == 0


def test_save_figures(tmp_path):
    specs = plots.build_figures(RESULTS, CONDITIONS, plots.BY_DAY)
    written = plots.save_figures(specs, plots.axis_limits(RESULTS), str(tmp_path), "tag")
    assert len(written) == 8
    assert (tmp_path / "tag_d2_EPSC.pdf").stat().st_size > 2000
    assert (tmp_path / "tag_d2_EPSC.png").stat().st_size > 2000


def recording(name, hz, amplitude=-20.0, mode="EPSC"):
    """A cell with a 10 s sweep holding `hz` events per second."""
    n = int(round(hz * 10))
    sweep = SweepResult("x.mat", [amplitude] * n, list(range(n)), 100000, 10000.0)
    return CellResult(name, mode, 1, 1, {}, [sweep], [(15.0, 300.0, 20.0, 1.0)], "d1")


PAIRED = [
    recording("d1-1", 2.0), recording("d1-1_picro", 4.0),
    recording("d1-2", 3.0), recording("d1-2_picro", 6.0),
    recording("d1-3", 5.0), recording("d1-3_picro", 7.0),
    recording("d1-10", 1.0), recording("d1-10_wash", 1.5),
    recording("d1-4", 9.0),                        # recorded once: not in any pair
    recording("d1-1", 1.0, 15.0, mode="IPSC"),     # other event type: paired separately
]


def test_cells_recorded_twice_are_paired():
    pairs = plots.find_pairs(PAIRED)
    assert sorted(pairs) == [("EPSC", "picro"), ("EPSC", "wash")]
    assert [(b.cell, a.cell) for b, a in pairs[("EPSC", "picro")]] == [
        ("d1-1", "d1-1_picro"), ("d1-2", "d1-2_picro"), ("d1-3", "d1-3_picro")]
    # d1-1 is not mistaken for the start of d1-10.
    assert [(b.cell, a.cell) for b, a in pairs[("EPSC", "wash")]] == [("d1-10", "d1-10_wash")]


def test_paired_t_test():
    specs = plots.build_figures(PAIRED, None, plots.BY_PAIR)
    assert [s.filename for s in specs] == ["paired_picro_EPSC", "paired_wash_EPSC"]
    assert [label for label, _, _ in specs[0].groups] == ["Base", "picro"]
    hz = plots.paired_stats(specs[0])[0]
    # Differences 2, 3, 2: mean 7/3, sd 0.577, so t = 7 and p = 0.0198 with 2 df.
    assert hz["N_Pairs"] == 3 and hz["Mean_Difference"] == 7 / 3
    assert abs(hz["t"] - 7.0) < 1e-9
    assert abs(hz["p"] - 0.019804) < 1e-5
    assert plots.stars(hz["p"]) == "*"
    # One pair is not enough for a test, and no bracket is drawn for it.
    assert np.isnan(plots.paired_stats(specs[1])[0]["p"])
    assert plots.stars(float("nan")) == ""
    assert [plots.stars(p) for p in (0.2, 0.049, 0.009, 0.0009, 0.00009)] == [
        "ns", "*", "**", "***", "****"]


def test_save_paired_figures_and_statistics(tmp_path):
    specs = plots.build_figures(PAIRED, None, plots.BY_PAIR)
    paired = [r for pairs in plots.find_pairs(PAIRED).values() for pair in pairs for r in pair]
    limits = plots.axis_limits(paired, headroom=0.25)
    assert limits[("EPSC", 0)][-1] >= 7 * 1.25   # room for the bracket above the data
    plots.save_figures(specs, limits, str(tmp_path), "tag")
    assert (tmp_path / "tag_paired_picro_EPSC.pdf").stat().st_size > 2000
    plots.save_paired_stats(specs, str(tmp_path / "stats.csv"))
    lines = (tmp_path / "stats.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Event_Type,Treatment,Measurement,N_Pairs")
    assert len(lines) == 1 + 2 * len(plots.METRICS)

