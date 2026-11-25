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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QMimeData, QPointF, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402

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
    window = MainWindow()
    window.show()
    window.add_folders([str(data_folder)])
    wait_for(app, lambda: len(window.session.cells) == 2)
    assert [c[0] for c in window.session.cells] == ["cellA", "cellB"]

    # Tune: sweeps are drawn and events are marked.
    tune = window.tune
    wait_for(app, lambda: len(tune._shown) == 2)
    assert tune._cell[0] == "cellA"
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
    wait_for(app, lambda: tune._cell[0] == "cellB" and len(tune._shown) == 1)
    tune.save_next.click()
    wait_for(app, lambda: window.session.n_tuned() == 2)
    assert window.tree.topLevelItem(0).child(0).text(2) == "EPSC 60/15/5"

    # Detect and export.
    run = window.run
    window.tabs.setCurrentIndex(1)
    out = tmp_path / "out"
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("Cell,Condition\ncellA,control\ncellB,drug\n")
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
    assert list(combined["Cell_Name"]) == ["cellA", "cellB"]
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
    reopened.close()


def test_several_session_folders(app, session_folders, tmp_path):
    day2 = os.path.normpath(str(session_folders / "day2"))
    window = MainWindow()
    window.show()

    # Adding the folder that holds the session folders opens both of them.
    window.add_folders([str(session_folders)])
    wait_for(app, lambda: len(window.session.cells) == 2)
    assert [os.path.basename(f) for f in window.session.folders] == ["day1", "day2"]
    assert [c.name for c in window.session.cells] == ["cellC", "cellD"]
    assert window.tree.topLevelItemCount() == 2
    assert "1 left out" in window.tree.topLevelItem(1).text(0)  # the second cellC

    # Tune every cell across both folders, then save them to one file.
    tune = window.tune
    wait_for(app, lambda: tune.current_cell() is not None and len(tune._shown) == 1)
    assert tune.current_cell().name == "cellC"
    tune.save_next.click()
    wait_for(app, lambda: tune.current_cell().name == "cellD" and len(tune._shown) == 1)
    tune.save_next.click()
    wait_for(app, lambda: window.session.n_tuned() == 2)
    saved = tmp_path / "all_sessions.json"
    window.save_parameters_to(str(saved))
    assert sorted(json.loads(saved.read_text())) == ["cellC", "cellD"]
    snapshot(app, window, "two_folders")

    # Detection covers the cells of every open folder.
    run = window.run
    run.run.click()
    wait_for(app, lambda: run.results is not None and run._job is None)
    assert [r.cell for r in run.results] == ["cellC", "cellD"]

    # Removing a folder keeps its parameters; adding it back restores its cells.
    window.tree.setCurrentItem(window.tree.topLevelItem(1))
    assert window.remove_btn.isEnabled()
    window.remove_btn.click()
    wait_for(app, lambda: len(window.session.cells) == 1)
    assert "cellD" in window.session.parameters
    drop = QMimeData()
    drop.setUrls([QUrl.fromLocalFile(day2)])  # as if dragged onto the window
    window.dropEvent(QDropEvent(QPointF(20, 20), Qt.CopyAction, drop,
                                Qt.LeftButton, Qt.NoModifier))
    wait_for(app, lambda: len(window.session.cells) == 2)
    assert window.session.n_tuned() == 2
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
