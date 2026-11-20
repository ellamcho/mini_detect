"""Main window and entry point."""
from __future__ import annotations

import os
import sys

import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .. import analysis
from .. import detection as det
from .run import RunPanel
from .state import Job, Session
from .tune import TunePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Detect")
        self.resize(1400, 860)
        self.session = Session()
        self._scan_token = 0

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row = QHBoxLayout()
        row.addWidget(QLabel("Data folder"))
        self.folder_label = QLabel("None chosen. Pick the folder that holds one "
                                   "sub-folder of .mat sweeps per recording.")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.folder_label, 1)
        choose = QPushButton("Choose folder…")
        choose.clicked.connect(self._choose_folder)
        row.addWidget(choose)
        layout.addLayout(row)

        splitter = QSplitter()
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.count = QLabel("")
        left_layout.addWidget(self.count)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Cell", "Sweeps", "Saved parameters"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setColumnWidth(0, 110)
        self.tree.setColumnWidth(1, 60)
        left_layout.addWidget(self.tree, 1)
        buttons = QHBoxLayout()
        self.load_btn = QPushButton("Load…")
        self.load_btn.setToolTip("Load parameters from a JSON file saved earlier.")
        self.save_btn = QPushButton("Save as…")
        self.save_btn.setToolTip("Save every cell's parameters to a JSON file.")
        self.clear_btn = QPushButton("Clear all")
        for b in (self.load_btn, self.save_btn, self.clear_btn):
            buttons.addWidget(b)
        left_layout.addLayout(buttons)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.tune = TunePanel(self.session)
        self.run = RunPanel(self.session)
        self.tabs.addTab(self.tune, "1  Tune")
        self.tabs.addTab(self.run, "2  Detect && export")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1070])

        self.tree.currentItemChanged.connect(self._cell_selected)
        self.load_btn.clicked.connect(self._load_parameters)
        self.save_btn.clicked.connect(self._save_parameters)
        self.clear_btn.clicked.connect(self._clear_parameters)
        self.session.cells_changed.connect(self._fill_tree)
        self.session.parameters_changed.connect(self._refresh_tree)
        self.tune.next_cell_requested.connect(self._next_untuned)
        self.tune.status.connect(self._show_status)
        self.run.status.connect(self._show_status)
        self._refresh_tree()

        settings = QSettings()
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        last = settings.value("data_folder", "", str)
        if last and os.path.isdir(last):
            self.open_folder(last)

    def _show_status(self, text):
        self.statusBar().showMessage(text, 8000)

    # -------------------------------------------------------------- data folder

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the data folder", self.session.data_folder)
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder):
        self._scan_token += 1
        self.folder_label.setText(f"{folder}   (scanning…)")
        job = Job(self, lambda job: (folder, det.scan_cells(folder)), token=self._scan_token)
        job.done.connect(self._scanned)
        job.failed.connect(self._scan_failed)
        job.start()

    def _scanned(self, token, result):
        if token != self._scan_token:
            return
        folder, cells = result
        self.folder_label.setText(folder)
        restored = self.session.open_folder(folder, cells)
        if not cells:
            QMessageBox.information(
                self, "No recordings found",
                "No sub-folders containing .mat files were found here.\n\n"
                "Choose the folder that holds one sub-folder per recording, "
                "each with that recording's sweeps.")
            return
        QSettings().setValue("data_folder", folder)
        if restored:
            self._show_status(
                f"Restored saved parameters for {restored} cells from your last session.")
        else:
            self._show_status(f"Found {len(cells)} cells.")

    def _scan_failed(self, token, text):
        if token == self._scan_token:
            self.folder_label.setText("Could not open that folder.")
            QMessageBox.warning(self, "Data folder", text.strip().splitlines()[-1])

    # ---------------------------------------------------------------- cell list

    def _fill_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        for name, _, n_sweeps in self.session.cells:
            item = QTreeWidgetItem([name, str(n_sweeps), ""])
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            self.tree.addTopLevelItem(item)
        self.tree.blockSignals(False)
        self._refresh_tree()
        if self.session.cells:
            self.tree.setCurrentItem(self.tree.topLevelItem(self._first_untuned(0) or 0))
        else:
            self.tune.set_cell(None)

    def _refresh_tree(self):
        for i, (name, _, _) in enumerate(self.session.cells):
            sets = self.session.parameters.get(name, [])
            text = ";  ".join(analysis.describe_param_set(p) for p in sets)
            self.tree.topLevelItem(i).setText(2, text)
            self.tree.topLevelItem(i).setToolTip(2, text)
        n = len(self.session.cells)
        self.count.setText(
            f"{self.session.n_tuned()} of {n} cells have saved parameters" if n else "Cells")
        has_any = bool(self.session.parameters)
        self.save_btn.setEnabled(has_any)
        self.clear_btn.setEnabled(has_any)
        self.load_btn.setEnabled(bool(self.session.data_folder))

    def _cell_selected(self, item, _previous=None):
        index = self.tree.indexOfTopLevelItem(item) if item else -1
        self.tune.set_cell(self.session.cells[index] if index >= 0 else None)

    def _first_untuned(self, start):
        """Index of the first cell without parameters at or after `start`, wrapping."""
        n = len(self.session.cells)
        for step in range(n):
            i = (start + step) % n
            if not self.session.parameters.get(self.session.cells[i][0]):
                return i
        return None

    def _next_untuned(self):
        current = self.tree.indexOfTopLevelItem(self.tree.currentItem())
        target = self._first_untuned(current + 1)
        if target is None:
            self._show_status("Every cell has saved parameters. "
                              "Go to the Detect & export tab when you are ready.")
        else:
            self.tree.setCurrentItem(self.tree.topLevelItem(target))

    # ----------------------------------------------------------- parameter files

    def _parameter_dir(self):
        return QSettings().value("parameter_dir", self.session.data_folder, str)

    def _load_parameters(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", self._parameter_dir(), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            parameters = analysis.load_parameters(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load parameters", f"Could not read this file:\n{exc}")
            return
        if not parameters:
            QMessageBox.warning(self, "Load parameters",
                                "No detection parameters were found in this file.")
            return
        merge = False
        if self.session.parameters:
            box = QMessageBox(self)
            box.setWindowTitle("Load parameters")
            box.setText("You already have saved parameters.")
            box.setInformativeText(
                "Replace them all with the file, or add the file's cells to what you have?")
            replace = box.addButton("Replace", QMessageBox.DestructiveRole)
            add = box.addButton("Add to current", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() not in (replace, add):
                return
            merge = box.clickedButton() is add
        self.session.replace_parameters(parameters, merge=merge)
        QSettings().setValue("parameter_dir", os.path.dirname(path))
        names = {name for name, _, _ in self.session.cells}
        here = sum(1 for cell in parameters if cell in names)
        self._show_status(f"Loaded parameters for {len(parameters)} cells "
                          f"({here} of them are in this data folder).")

    def _save_parameters(self):
        from datetime import date
        default = os.path.join(self._parameter_dir(),
                               f"{date.today():%Y%m%d}_optimized_parameters.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", default, "JSON files (*.json)")
        if not path:
            return
        try:
            analysis.save_parameters(path, self.session.parameters)
        except OSError as exc:
            QMessageBox.warning(self, "Save parameters", f"Could not save:\n{exc}")
            return
        QSettings().setValue("parameter_dir", os.path.dirname(path))
        self._show_status(f"Saved parameters for {len(self.session.parameters)} cells to {path}")

    def _clear_parameters(self):
        answer = QMessageBox.question(
            self, "Clear all parameters",
            f"Remove the saved parameters for all {len(self.session.parameters)} cells?")
        if answer == QMessageBox.Yes:
            self.session.replace_parameters({})

    def closeEvent(self, event):
        QSettings().setValue("geometry", self.saveGeometry())
        self.run.remember()
        super().closeEvent(event)


def create_app(argv=None):
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setOrganizationName("mini_detect")
    app.setApplicationName("Mini Detect")
    palette = app.palette()
    pg.setConfigOptions(
        background=palette.color(QPalette.Base),
        foreground=palette.color(QPalette.Text),
        antialias=False,
    )
    return app


def main():
    app = create_app()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
