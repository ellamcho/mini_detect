"""Checks that the fast detection gives exactly what the original notebook's
sample-by-sample scan gave, and that a folder of sweeps runs end to end."""
import os

import numpy as np
import pytest
from conftest import FS, synthetic_trace

from mini_detect import analysis
from mini_detect import detection as det

def reference_decay_windows(trace, window_size, sample_step, mode):
    """The notebook's original loop, kept here as the reference."""
    inward = det.is_inward(mode)
    found = []
    i = 0
    while i <= len(trace) - window_size:
        sampled = trace[i:i + window_size][::sample_step]
        if len(sampled) > 1:
            steps = np.diff(sampled)
            if np.all(steps > 0) if inward else np.all(steps < 0):
                found.append(i)
                i += window_size
                continue
        i += 1
    return found


@pytest.mark.parametrize("mode", det.MODES)
@pytest.mark.parametrize("params", [(60, 15, 5), (100, 25, 15), (130, 35, 20), (61, 7, 3)])
def test_decay_windows_match_reference(mode, params):
    sign = -1.0 if det.is_inward(mode) else 1.0
    trace = synthetic_trace(seed=params[0], sign=sign)
    window_size, sample_step, _ = params
    expected = reference_decay_windows(trace, window_size, sample_step, mode)
    assert expected, "the synthetic trace should contain detectable decays"
    assert det.decay_windows(trace, window_size, sample_step, mode) == expected


def test_decay_windows_edge_cases():
    trace = np.arange(50.0)
    assert det.decay_windows(trace, 100, 10, "EPSC") == []  # window longer than trace
    assert det.decay_windows(trace, 10, 10, "EPSC") == []   # only one sampled point
    assert det.decay_windows(trace, 20, 5, "EPSC") == [0, 20]
    assert det.decay_windows(trace, 20, 5, "IPSC") == []


def test_load_and_detect(data_folder):
    sweep = det.load_sweep(str(data_folder / "cellA" / "AD0_1.mat"))
    assert sweep.fs == FS
    assert sweep.control_window == (2000, 2500)
    values, indices = det.detect_events(sweep, 60, 15, 5, "EPSC")
    assert len(values) >= 15  # 25 events were added to the trace
    assert all(v < -det.MIN_AMPLITUDE_PA for v in values)
    assert not any(2000 <= i <= 2500 for i in indices)
    assert det.detect_events(sweep, 60, 15, 5, "IPSC")[0] == []


def test_run_and_export(data_folder, tmp_path):
    parameters = {
        "cellA": [analysis.make_param_set(60, 15, 5, "EPSC"),
                  analysis.make_param_set(60, 15, 5, "IPSC")],
        "cellB": [analysis.make_param_set(100, 25, 15, "EPSC")],
    }
    results = analysis.run_detection(str(data_folder), parameters, log=lambda _: None)
    assert [(r.cell, r.mode, len(r.sweeps)) for r in results] == [
        ("cellA", "EPSC", 2), ("cellA", "IPSC", 2), ("cellB", "EPSC", 1)]
    cell_a = results[0]
    assert cell_a.frequency_hz == pytest.approx(cell_a.n_events / 12.0)
    assert len(cell_a.amplitudes) == cell_a.n_events

    out = tmp_path / "out"
    written = analysis.export_results(
        results, str(out), "test", conditions={"cellA": "control"}, log=lambda _: None)
    names = {p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in written}
    assert {"Amplitude_combined_test.csv", "Frequency_combined_test.csv",
            "Rs_Rm_Cm_all_cells_test.csv", "control_IPI_by_cell_test.csv",
            "Unknown_ampls_by_cell_test.csv"} <= names

    import pandas as pd
    combined = pd.read_csv(out / "Amplitude_combined_test.csv")
    assert list(combined["Condition"]) == ["control", "control", "Unknown"]
    per_event = pd.read_csv(out / "control_ampls_by_cell_test.csv")
    assert list(per_event.columns) == ["Amplitude_Number", "cellA (EPSC)", "cellA (IPSC)"]


def test_parameter_files_round_trip(tmp_path):
    old_style = {"cell1": {"window_size": 60, "sample_step": 15, "interval": 5},
                 "cell2": [{"window_size": 130, "sample_step": 35, "interval": 20,
                            "mode": "ipsc_high_cl", "control_pulse_timing": "late"}]}
    normal = analysis.normalize_parameters(old_style)
    assert normal["cell1"][0]["mode"] == "EPSC"
    assert normal["cell2"][0]["mode"] == "IPSC_high_Cl"
    path = tmp_path / "p.json"
    analysis.save_parameters(str(path), normal)
    assert analysis.load_parameters(str(path)) == normal


def test_several_session_folders(session_folders):
    day1 = os.path.normpath(str(session_folders / "day1"))
    day2 = os.path.normpath(str(session_folders / "day2"))

    # A folder of session folders opens as each of them; a session folder as itself.
    sessions = det.find_sessions(str(session_folders))
    assert [folder for folder, _ in sessions] == [day1, day2]
    assert [folder for folder, _ in det.find_sessions(day2)] == [day2]
    assert det.find_sessions(str(session_folders / "figures")) == []

    # A cell name is used once: the first folder keeps it.
    cells, skipped = det.merge_sessions(sessions)
    assert [(c.name, c.folder) for c in cells] == [("cellC", day1), ("cellD", day2)]
    assert skipped == {day2: ["cellC"]}

    # Named folder-cell, the two cellC folders are different cells.
    cells, skipped = det.merge_sessions(sessions, name_with_folder=True)
    assert [c.name for c in cells] == ["day1-cellC", "day2-cellC", "day2-cellD"]
    assert skipped == {}
    one_set = [analysis.make_param_set(60, 15, 5, "EPSC")]
    results = analysis.run_detection(
        [day1, day2], {"day2-cellC": one_set}, log=lambda _: None, name_with_folder=True)
    assert [r.cell for r in results] == ["day2-cellC"]
    assert os.path.dirname(os.path.dirname(results[0].sweeps[0].path)) == day2

    parameters = {name: [analysis.make_param_set(60, 15, 5, "EPSC")]
                  for name in ("cellC", "cellD")}
    log = []
    results = analysis.run_detection([day1, day2], parameters, log=log.append)
    assert [r.cell for r in results] == ["cellC", "cellD"]
    assert os.path.dirname(os.path.dirname(results[0].sweeps[0].path)) == day1
    assert any("Left out of day2" in line and "cellC" in line for line in log)
