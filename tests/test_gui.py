"""Drives the real window through the whole workflow: open a data folder, tune,
save parameters, run detection and export.

Runs without a display by default. Set QT_QPA_PLATFORM to the platform's own
plugin (for example `windows`) to exercise real windows, and
MINI_DETECT_SCREENSHOTS to a folder to keep pictures of what was drawn.
"""
import json
import os
import time

import pandas as pd
import pytest
import shiboken6

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QMimeData, QPointF, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from mini_detect.gui import state  # noqa: E402
from mini_detect.gui.app import MainWindow, create_app  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Keep the test away from the real user's settings and autosaved parameters.
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    autosave = tmp_path / "autosave" / "session.json"
    monkeypatch.setattr(state.Session, "_autosave_path", lambda self: str(autosave))
    return create_app([])


def describe(window):
    session = window.session
    return (f"scans running: {window._scans}; folders: {session.folders}; "
            f"missing: {session.missing}; cells: {[c.name for c in session.cells]}; "
            f"status: {window.statusBar().currentMessage()!r}")


def wait_for(app, condition, timeout=60):
    deadline = time.time() + timeout
    while not condition():
        app.processEvents(QEventLoop.AllEvents, 50)
        if time.time() > deadline:
            windows = [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]
            pytest.fail("timed out waiting for the window\n"
                        + "\n".join(describe(w) for w in windows))
    for _ in range(5):
        app.processEvents(QEventLoop.AllEvents, 50)


def snapshot(app, window, name):
    folder = os.environ.get("MINI_DETECT_SCREENSHOTS")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    wait_for(app, lambda: True)
    window.grab().save(os.path.join(folder, f"{name}.png"))
    screen = app.primaryScreen()
    if screen is not None:  # what is actually on the desktop, where there is one
        screen.grabWindow(0).save(os.path.join(folder, f"{name}_desktop.png"))


def test_whole_workflow(app, data_folder, tmp_path):
    cell_a, cell_b = f"{data_folder.name}-cellA", f"{data_folder.name}-cellB"
    window = MainWindow()
    window.show()
    window.add_folders([str(data_folder)])
    wait_for(app, lambda: len(window.session.cells) == 2)
    # Cells are named after the session folder and their own folder.
    assert [c.name for c in window.session.cells] == [cell_a, cell_b]

    # Tune: sweeps are drawn and events are marked.
    tune = window.tune
    wait_for(app, lambda: len(tune._shown) == 2)
    assert tune.current_cell().name == cell_a
    assert sum(len(scatter.getData()[0]) for _, scatter in tune._plots) >= 15

    # Changing a parameter re-runs detection on screen.
    before = tune.stats.text()
    tune.preset.setCurrentIndex(5)
    tune._preset_chosen(5)
    wait_for(app, lambda: tune.stats.text() != before)
    assert tune.current_set()["window_size"] == 300
    tune.preset.setCurrentIndex(1)
    tune._preset_chosen(1)
    wait_for(app, lambda: tune.current_set()["window_size"] == 60)
    snapshot(app, window, "tune")

    # Save and move through both cells.
    tune.save_next.click()
    wait_for(app, lambda: tune.current_cell().name == cell_b and len(tune._shown) == 1)
    tune.save_next.click()
    wait_for(app, lambda: window.session.n_tuned() == 2)
    assert sorted(window.session.parameters) == [cell_a, cell_b]
    assert window.tree.topLevelItem(0).child(0).text(0) == cell_a
    assert window.tree.topLevelItem(0).child(0).text(2) == "EPSC 60/15/5"

    # Detect and export.
    run = window.run
    window.tabs.setCurrentIndex(1)
    out = tmp_path / "out"
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(f"Cell,Condition\n{cell_a},control\n{cell_b},drug\n")
    run._set_metadata(str(metadata))
    run.output.setText(str(out))
    run.tag.setText("test")
    run.rs_enabled.setChecked(False)
    assert run.run.isEnabled() and not run.export.isEnabled()
    run.run.click()
    wait_for(app, lambda: run.results is not None and run._job is None)
    assert run.table.rowCount() == 2
    assert int(run.table.item(0, 4).text()) >= 15
    assert run.export.isEnabled()
    run.export.click()
    snapshot(app, window, "export")

    combined = pd.read_csv(out / "Frequency_combined_test.csv")
    assert list(combined["Cell_Name"]) == [cell_a, cell_b]
    assert list(combined["Condition"]) == ["control", "drug"]
    assert (combined["Frequency_Value"] > 1).all()
    for name in ("control_Amplitude_test.csv", "drug_IPI_by_cell_test.csv",
                 "drug_ampls_by_cell_test.csv", "Rs_Rm_Cm_all_cells_test.csv",
                 "test_optimized_parameters.json", "test_optimized_results.pkl"):
        assert (out / name).is_file(), name

    # The folder and its parameters come back when the app is started again.
    window.close()
    reopened = MainWindow()
    wait_for(app, lambda: len(reopened.session.cells) == 2)
    assert reopened.session.n_tuned() == 2

    # Saved results can be loaded instead of running detection again, from the
    # results file or from the .pkl saved beside it.
    run = reopened.run
    assert run.results is None and not run.plot.isEnabled()
    run.load_results(str(out / "test_optimized_results.pkl"))
    assert [r.cell for r in run.results] == [cell_a, cell_b]
    assert run.table.rowCount() == 2 and int(run.table.item(0, 4).text()) >= 15
    assert run.plot.isEnabled() and run.export.isEnabled()
    assert "loaded from test_detection_results.json" in run.note.text()
    # They stay when a parameter is changed, and export the same files again.
    reopened.session.save_set(cell_a, reopened.session.parameters[cell_a][0])
    assert run.export.isEnabled()
    run.output.setText(str(tmp_path / "out_again"))
    run.tag.setText("test")
    run.rs_enabled.setChecked(False)
    run.export.click()
    for name in ("Frequency_combined_test.csv", "control_Amplitude_test.csv",
                 "test_optimized_parameters.json"):
        assert (tmp_path / "out_again" / name).read_text() == (out / name).read_text(), name
    reopened.close()


def test_several_session_folders(app, session_folders, tmp_path):
    day2 = os.path.normpath(str(session_folders / "day2"))
    window = MainWindow()
    window.show()
    names = lambda: [c.name for c in window.session.cells]  # noqa: E731

    # Adding the folder that holds the session folders opens both of them. Both
    # have a folder called cellC; naming cells folder-cell keeps them apart.
    window.add_folders([str(session_folders)])
    wait_for(app, lambda: len(window.session.cells) == 3)
    assert [os.path.basename(f) for f in window.session.folders] == ["day1", "day2"]
    assert names() == ["day1-cellC", "day2-cellC", "day2-cellD"]
    assert window.name_check.isChecked()
    assert window.tree.topLevelItemCount() == 2
    assert "left out" not in window.tree.topLevelItem(1).text(0)

    # Tune every cell across both folders, then save them to one file.
    tune = window.tune
    for name in names():
        wait_for(app, lambda: tune.current_cell() is not None
                 and tune.current_cell().name == name and len(tune._shown) == 1)
        tune.save_next.click()
    wait_for(app, lambda: window.session.n_tuned() == 3)
    saved = tmp_path / "all_sessions.json"
    window.save_parameters_to(str(saved))
    assert sorted(json.loads(saved.read_text())) == names()
    snapshot(app, window, "two_folders")

    # Detection covers the cells of every open folder.
    run = window.run
    run.run.click()
    wait_for(app, lambda: run.results is not None and run._job is None)
    assert [r.cell for r in run.results] == names()

    # Plot the results: one figure for all days, or one per day; then save them.
    run.output.setText(str(tmp_path / "out"))
    run.rs_enabled.setChecked(False)
    assert run.plot.isEnabled()
    run.plot.click()
    plot_window = run.plot_window
    wait_for(app, plot_window.isVisible)
    assert plot_window.which.count() == 1 and "all days" in plot_window.which.itemText(0)
    plot_window.by.setCurrentIndex(1)
    assert [plot_window.which.itemText(i).split()[1] for i in range(2)] == ["day1", "day2"]
    plot_window.which.setCurrentIndex(1)
    snapshot(app, plot_window, "plots")
    plot_window.save_btn.click()
    saved_plots = sorted(os.listdir(tmp_path / "out" / "plots"))
    assert [n for n in saved_plots if n.endswith(".pdf")] == [
        f"{run.tag.text()}_day1_EPSC.pdf", f"{run.tag.text()}_day2_EPSC.pdf"]
    assert len(saved_plots) == 4
    plot_window.close()

    # With the naming turned off a cell goes by its own folder name. The second
    # cellC then clashes and is left out; saved parameters follow the new names.
    window.name_check.setChecked(False)
    wait_for(app, lambda: len(window.session.cells) == 2)
    assert names() == ["cellC", "cellD"]
    assert "1 left out" in window.tree.topLevelItem(1).text(0)
    assert window.session.n_tuned() == 2
    assert tune.current_cell().name in names()
    window.name_check.setChecked(True)
    wait_for(app, lambda: len(window.session.cells) == 3)
    assert window.session.n_tuned() == 3
    assert sorted(window.session.parameters) == names()

    # Removing a folder keeps its parameters; adding it back restores its cells.
    window.tree.setCurrentItem(window.tree.topLevelItem(1))
    assert window.remove_btn.isEnabled()
    window.remove_btn.click()
    wait_for(app, lambda: len(window.session.cells) == 1)
    assert "day2-cellD" in window.session.parameters
    drop = QMimeData()
    drop.setUrls([QUrl.fromLocalFile(day2)])  # as if dragged onto the window
    window.dropEvent(QDropEvent(QPointF(20, 20), Qt.CopyAction, drop,
                                Qt.LeftButton, Qt.NoModifier))
    wait_for(app, lambda: len(window.session.cells) == 3)
    assert window.session.n_tuned() == 3
    window.close()


def test_saved_state_from_before_folder_names(app, session_folders):
    """State saved by a version that named cells by their own folder alone is
    opened as it was, then renamed to folder-cell without losing parameters."""
    one_set = [{"window_size": 60, "sample_step": 15, "interval": 5, "mode": "EPSC"}]
    autosave = state.Session._autosave_path(None)
    os.makedirs(os.path.dirname(autosave), exist_ok=True)
    with open(autosave, "w") as f:
        json.dump({"folders": [str(session_folders / "day1"), str(session_folders / "day2")],
                   "parameters": {"cellC": one_set, "cellD": one_set}}, f)

    window = MainWindow()
    wait_for(app, lambda: len(window.session.cells) == 3)
    assert window.name_check.isChecked()
    assert sorted(window.session.parameters) == ["day1-cellC", "day2-cellD"]
    assert window.session.n_tuned() == 2
    window.close()

    # The choice is remembered: turned off, it stays off after a restart.
    window = MainWindow()
    wait_for(app, lambda: len(window.session.cells) == 3)
    window.name_check.setChecked(False)
    window.close()
    window = MainWindow()
    wait_for(app, lambda: len(window.session.cells) == 2)
    assert not window.name_check.isChecked()
    assert sorted(window.session.parameters) == ["cellC", "cellD"]
    window.close()


def test_background_results_always_arrive(app):
    """Every background job must report back. Jobs used to delete themselves
    while their result was still being delivered, which lost it now and then."""
    from PySide6.QtCore import QObject

    class Receiver(QObject):
        def __init__(self):
            super().__init__()
            self.got = set()

        def on_done(self, token, _result):
            self.got.add(token)

    receiver = Receiver()
    for i in range(500):
        job = state.Job(receiver, lambda job: None, token=i)
        job.done.connect(receiver.on_done)
        job.start()
        deadline = time.time() + 5
        while i not in receiver.got and time.time() < deadline:
            app.processEvents(QEventLoop.AllEvents, 50)
        assert i in receiver.got, f"the result of job {i} never arrived"


def test_garbage_collector_is_paused_while_a_job_runs(app):
    """Python's garbage collector must not run on a background thread, where it
    could destroy window objects and crash Qt. Jobs pause it until they report back."""
    import gc

    from PySide6.QtCore import QObject

    class Receiver(QObject):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_done(self, _token, result):
            self.seen.append(result)

    def pump_until(condition):
        deadline = time.time() + 10
        while not condition() and time.time() < deadline:
            app.processEvents(QEventLoop.AllEvents, 50)

    receiver = Receiver()
    pump_until(gc.isenabled)   # jobs left over from earlier tests finish first
    assert gc.isenabled()
    first = state.Job(receiver, lambda job: gc.isenabled())
    second = state.Job(receiver, lambda job: gc.isenabled())
    for job in (first, second):
        job.done.connect(receiver.on_done)
        job.start()
    assert not gc.isenabled()
    pump_until(lambda: len(receiver.seen) == 2)
    assert receiver.seen == [False, False]   # it was off inside both jobs
    pump_until(gc.isenabled)
    assert gc.isenabled()                     # and is back on once both are done


def test_before_and_after_plots(app, tmp_path):
    from test_plots import PAIRED

    from mini_detect.gui.plot_window import PlotWindow

    # As in the app, the plot window belongs to another window. One left on its
    # own is destroyed in the wrong order when Python exits.
    host = QWidget()
    window = PlotWindow(PAIRED, None, "tag", str(tmp_path), parent=host)
    window.show()
    window.by.setCurrentIndex(window.by.findData("pair"))
    assert [window.which.itemText(i) for i in range(window.which.count())] == [
        "EPSC, before and after picro  (n = 3 pairs)",
        "EPSC, before and after wash  (n = 1 pair)"]
    assert "Hz p = 0.0198 (*)" in window.info.text()
    snapshot(app, window, "pairs")
    window.save_btn.click()
    saved = sorted(os.listdir(tmp_path / "plots"))
    assert "tag_paired_picro_EPSC.pdf" in saved and "tag_paired_t_tests.csv" in saved
    window.close()

    # Nothing recorded twice: the window says so instead of showing a figure.
    window = PlotWindow([r for r in PAIRED if "_" not in r.cell], None, "tag", str(tmp_path),
                        parent=host)
    window.by.setCurrentIndex(window.by.findData("pair"))
    assert window.which.count() == 0 and not window.save_btn.isEnabled()
    window.close()
    shiboken6.delete(host)   # destroys the host and both plot windows now

