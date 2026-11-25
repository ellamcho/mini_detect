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
