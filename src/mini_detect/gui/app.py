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

FOLDER_ROLE = Qt.UserRole       # on a session-folder row: the folder's path
CELL_ROLE = Qt.UserRole + 1     # on a cell row: the cell's path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Detect")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self.session = Session()
        self._scans = 0          # folder scans still running
        self._cell_items = {}    # cell path -> its row in the tree
        self._folder_items = {}  # session folder -> its row in the tree

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row = QHBoxLayout()
        self.folder_label = QLabel("")
        self.folder_label.setWordWrap(True)
        row.addWidget(self.folder_label, 1)
        self.add_btn = QPushButton("Add folder…")
        self.add_btn.setToolTip(
            "Add a session folder (one sub-folder of .mat sweeps per cell), or a "
            "folder that holds several session folders. You can add as many as "
            "you like, and you can also drag folders onto this window.")
        self.remove_btn = QPushButton("Remove folder")
        self.remove_btn.setToolTip(
            "Take the selected session folder off the list. Nothing is deleted "
            "and its saved parameters are kept.")
        row.addWidget(self.add_btn)
        row.addWidget(self.remove_btn)
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
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(14)
        self.tree.setColumnWidth(0, 125)
        self.tree.setColumnWidth(1, 60)
        left_layout.addWidget(self.tree, 1)
        buttons = QHBoxLayout()
        self.load_btn = QPushButton("Load…")
        self.load_btn.setToolTip("Load parameters from a JSON file saved earlier.")
        self.save_btn = QPushButton("Save as…")
        self.save_btn.setToolTip(
            "Save the parameters of every cell, from all session folders, to one JSON file.")
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
        splitter.setSizes([340, 1060])

        self.add_btn.clicked.connect(self._choose_folder)
        self.remove_btn.clicked.connect(self._remove_folder)
        self.tree.currentItemChanged.connect(self._row_selected)
        self.load_btn.clicked.connect(self._load_parameters)
        self.save_btn.clicked.connect(self._save_parameters)
        self.clear_btn.clicked.connect(self._clear_parameters)
        self.session.cells_changed.connect(self._fill_tree)
        self.session.parameters_changed.connect(self._refresh_tree)
        self.tune.next_cell_requested.connect(self._next_untuned)
        self.tune.status.connect(self._show_status)
        self.run.status.connect(self._show_status)
        self._fill_tree()

        settings = QSettings()
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        folders = self.session.restore()
        if not folders:  # version 0.1 remembered a single data folder here
            last = settings.value("data_folder", "", str)
            folders = [last] if last else []
            settings.remove("data_folder")
        if folders:
            self.add_folders(folders, restoring=True)

    def _show_status(self, text):
        self.statusBar().showMessage(text, 10000)

    # ----------------------------------------------------------- session folders

    def _choose_folder(self):
        start = QSettings().value("folder_dir", self.session.start_dir, str)
        folder = QFileDialog.getExistingDirectory(
            self, "Add a session folder, or a folder of session folders", start)
        if folder:
            QSettings().setValue("folder_dir", os.path.dirname(folder))
            self.add_folders([folder])

    def add_folders(self, folders, restoring=False):
        """Scan folders in the background and add the session folders in them."""
        folders = list(folders)

        def work(job):
            found, empty = [], []
            for folder in folders:
                try:
                    sessions = det.find_sessions(folder)
                except OSError:
                    sessions = []
                found.extend(sessions)
                if not sessions:
                    empty.append(folder)
            return found, empty, restoring

        self._scans += 1
        self._refresh_summary()
        job = Job(self, work)
        job.done.connect(self._scanned)
        job.failed.connect(self._scan_failed)
        job.start()

    def _scanned(self, _token, result):
        found, empty, restoring = result
        self._scans -= 1
        before = len(self.session.cells)
        added = self.session.add_sessions(found, missing=empty if restoring else ())
        self._refresh_summary()
        if restoring:
            if self.session.parameters:
                self._show_status(
                    f"Restored your {_plural(len(self.session.folders), 'session folder')} "
                    f"and the saved parameters for {len(self.session.parameters)} cells.")
            return
        if empty:
            QMessageBox.information(
                self, "No recordings found",
                "No recordings were found in:\n\n" + "\n".join(empty) + "\n\n"
                "Add a session folder, which holds one sub-folder of .mat sweeps "
                "per cell, or a folder that holds several session folders.")
        if not found:
            return
        message = (f"Added {_plural(len(added), 'session folder')} with "
                   f"{_plural(len(self.session.cells) - before, 'cell')}."
                   if added else "Those folders are already open; refreshed them.")
        left_out = sum(len(self.session.skipped.get(folder, [])) for folder, _ in found)
        if left_out:
            message += (f" {_plural(left_out, 'cell')} left out because another open "
                        "folder has a cell with the same name.")
        self._show_status(message)

    def _scan_failed(self, _token, text):
        self._scans -= 1
        self._refresh_summary()
        QMessageBox.warning(self, "Add folder", text.strip().splitlines()[-1])

    def _remove_folder(self):
        item = self.tree.currentItem()
        folder = item.data(0, FOLDER_ROLE) if item else None
        if folder:
            self.session.remove_folder(folder)
            self._show_status(f"Removed {os.path.basename(folder)} from the list. "
                              "Its saved parameters are kept.")

    def _refresh_summary(self):
        n = len(self.session.folders)
        if self._scans:
            text = "Looking for recordings…"
        elif n:
            text = (f"{_plural(n, 'session folder')} open. Add more with the button "
                    "or by dragging folders onto this window.")
        else:
            text = ("No folders open. Add a session folder (one sub-folder of .mat "
                    "sweeps per cell), or a folder that holds several session folders.")
        self.folder_label.setText(text)

    def dragEnterEvent(self, event):
        if any(os.path.isdir(url.toLocalFile()) for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        folders = [url.toLocalFile() for url in event.mimeData().urls()]
        folders = [f for f in folders if os.path.isdir(f)]
        if folders:
            event.acceptProposedAction()
            self.add_folders(folders)

    # ---------------------------------------------------------------- cell list

    def _fill_tree(self):
        """Rebuild the list: one row per session folder, with its cells below."""
        current = self.tune.current_cell()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._cell_items = {}
        self._folder_items = {}
        loaded = {cell.path for cell in self.session.cells}
        for folder, cells in self.session.sessions:
            top = self._add_folder_row(folder)
            for cell in cells:
                if cell.path not in loaded:  # left out: duplicate name
                    continue
                item = QTreeWidgetItem(top, [cell.name, str(cell.n_sweeps), ""])
                item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                item.setData(0, CELL_ROLE, cell.path)
                self._cell_items[cell.path] = item
        for folder in self.session.missing:
            top = self._add_folder_row(folder)
            top.setText(0, f"{os.path.basename(folder)}  —  not found")
            top.setToolTip(0, f"{folder}\nThis folder could not be opened. If it is on a "
                              "drive that is not connected, connect it and add the folder "
                              "again.")
            top.setForeground(0, self.palette().brush(QPalette.Disabled, QPalette.Text))
        self.tree.expandAll()

        keep = self._cell_items.get(current.path) if current else None
        if keep is not None:
            self.tree.setCurrentItem(keep)
        self.tree.blockSignals(False)
        self._refresh_tree()
        self._refresh_summary()
        if keep is None:
            first = self._first_untuned(0)
            if first is None and self.session.cells:
                first = 0
            if first is None:
                self.tune.set_cell(None)
            else:
                self.tree.setCurrentItem(self._cell_items[self.session.cells[first].path])
        self._row_selected(self.tree.currentItem())

    def _add_folder_row(self, folder):
        top = QTreeWidgetItem(self.tree, [os.path.basename(folder)])
        top.setFirstColumnSpanned(True)
        top.setData(0, FOLDER_ROLE, folder)
        font = top.font(0)
        font.setBold(True)
        top.setFont(0, font)
        self._folder_items[folder] = top
        return top

    def _refresh_tree(self):
        tuned = {}
        for cell in self.session.cells:
            sets = self.session.parameters.get(cell.name, [])
            text = ";  ".join(analysis.describe_param_set(p) for p in sets)
            item = self._cell_items[cell.path]
            item.setText(2, text)
            item.setToolTip(2, text)
            tuned.setdefault(cell.folder, [0, 0])
            tuned[cell.folder][0] += bool(sets)
            tuned[cell.folder][1] += 1
        for folder, _ in self.session.sessions:
            done, total = tuned.get(folder, (0, 0))
            text = f"{os.path.basename(folder)}  —  {done} of {total} saved"
            tip = folder
            skipped = self.session.skipped.get(folder)
            if skipped:
                text += f",  {len(skipped)} left out"
                tip += ("\n\nLeft out because another open folder has a cell with the "
                        "same name:\n" + ", ".join(skipped))
            self._folder_items[folder].setText(0, text)
            self._folder_items[folder].setToolTip(0, tip)
        n = len(self.session.cells)
        self.count.setText(
            f"{self.session.n_tuned()} of {n} cells have saved parameters" if n else "Cells")
        has_any = bool(self.session.parameters)
        self.save_btn.setEnabled(has_any)
        self.clear_btn.setEnabled(has_any)

    def _row_selected(self, item, _previous=None):
        """A cell row shows that cell; a folder row only arms 'Remove folder'."""
        self.remove_btn.setEnabled(bool(item and item.data(0, FOLDER_ROLE)))
        path = item.data(0, CELL_ROLE) if item else None
        if path:
            cell = next(c for c in self.session.cells if c.path == path)
            if cell != self.tune.current_cell():
                self.tune.set_cell(cell)

    def _first_untuned(self, start):
        """Index of the first cell without parameters at or after `start`, wrapping."""
        n = len(self.session.cells)
        for step in range(n):
            i = (start + step) % n
            if not self.session.parameters.get(self.session.cells[i].name):
                return i
        return None

    def _next_untuned(self):
        cell = self.tune.current_cell()
        current = self.session.cells.index(cell) if cell in self.session.cells else -1
        target = self._first_untuned(current + 1)
        if target is None:
            self._show_status("Every cell has saved parameters. "
                              "Go to the Detect & export tab when you are ready.")
        else:
            self.tree.setCurrentItem(self._cell_items[self.session.cells[target].path])

    # ----------------------------------------------------------- parameter files

    def _parameter_dir(self):
        return QSettings().value("parameter_dir", self.session.start_dir, str)

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
        names = {cell.name for cell in self.session.cells}
        here = sum(1 for cell in parameters if cell in names)
        self._show_status(f"Loaded parameters for {len(parameters)} cells "
                          f"({here} of them are in the open folders).")

    def _save_parameters(self):
        from datetime import date
        default = os.path.join(self._parameter_dir(),
                               f"{date.today():%Y%m%d}_optimized_parameters.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", default, "JSON files (*.json)")
        if path:
            self.save_parameters_to(path)

    def save_parameters_to(self, path):
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


def _plural(n, noun):
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


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
