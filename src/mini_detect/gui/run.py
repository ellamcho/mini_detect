"""The Detect & export tab: run detection on every cell with saved parameters,
then write the CSV files."""
from __future__ import annotations

import os
from datetime import date

import numpy as np
from PySide6.QtCore import QSettings, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import analysis
from .state import Job

COLUMNS = ["Cell", "Event type", "Parameters", "Sweeps", "Events", "Frequency (Hz)",
           "Amplitude (pA)", "Rs (MΩ)", "Rm (MΩ)", "Cm (pF)"]
EXPORT_LABELS = [
    ("summary", "Mean amplitude and frequency per cell"),
    ("passive", "Rs, Rm and Cm per cell"),
    ("ipi", "Inter-event intervals (every event)"),
    ("amplitudes", "Event amplitudes (every event)"),
]
CELL_COLUMN_NAMES = ("cell", "cell_id", "cell_name", "cellid")
GROUP_COLUMN_NAMES = ("condition", "group", "treatment")


class RunPanel(QWidget):
    status = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.results = None
        self._stale = False
        self._job = None
        self._metadata = None
        self._build()
        self._restore()
        session.parameters_changed.connect(self._parameters_changed)
        session.cells_changed.connect(self._folder_changed)
        self._refresh_buttons()

    # ------------------------------------------------------------------ layout

    def _build(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        layout.addLayout(top)
        left = QVBoxLayout()
        top.addLayout(left, 1)

        box = QGroupBox("Where to save")
        form = QFormLayout(box)
        self.output = QLineEdit()
        self.output.setPlaceholderText("Folder for the CSV files")
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose_output)
        row = QHBoxLayout()
        row.addWidget(self.output, 1)
        row.addWidget(choose)
        form.addRow("Output folder", row)
        self.tag = QLineEdit(date.today().strftime("%Y%m%d"))
        self.tag.setToolTip("Added to the name of every file that is written.")
        form.addRow("File name tag", self.tag)
        left.addWidget(box)

        box = QGroupBox("Conditions (optional)")
        form = QFormLayout(box)
        self.metadata_path = QLineEdit()
        self.metadata_path.setReadOnly(True)
        self.metadata_path.setPlaceholderText("CSV listing each cell folder and its condition")
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose_metadata)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._set_metadata(""))
        row = QHBoxLayout()
        row.addWidget(self.metadata_path, 1)
        row.addWidget(choose)
        row.addWidget(clear)
        form.addRow("Metadata file", row)
        self.cell_column = QComboBox()
        self.group_column = QComboBox()
        row = QHBoxLayout()
        row.addWidget(QLabel("Cells in"))
        row.addWidget(self.cell_column, 1)
        row.addWidget(QLabel("conditions in"))
        row.addWidget(self.group_column, 1)
        form.addRow("Columns", row)
        self.match = QLabel("Without a metadata file, all cells are exported as one group.")
        self.match.setWordWrap(True)
        form.addRow("", self.match)
        left.addWidget(box)

        box = QGroupBox("What to export")
        inner = QVBoxLayout(box)
        self.kinds = {}
        for key, label in EXPORT_LABELS:
            check = QCheckBox(label)
            check.setChecked(True)
            self.kinds[key] = check
            inner.addWidget(check)
        row = QHBoxLayout()
        self.rs_enabled = QCheckBox("Leave out cells with Rs above")
        self.rs_enabled.setChecked(True)
        self.rs_enabled.setToolTip(
            "Cells whose mean series resistance is above this value, or whose "
            "series resistance could not be measured, are left out of every file.")
        self.rs_value = QDoubleSpinBox()
        self.rs_value.setRange(1, 500)
        self.rs_value.setDecimals(0)
        self.rs_value.setValue(25)
        self.rs_value.setSuffix(" MΩ")
        self.rs_enabled.toggled.connect(self.rs_value.setEnabled)
        row.addWidget(self.rs_enabled)
        row.addWidget(self.rs_value)
        row.addStretch(1)
        inner.addLayout(row)
        inner.addStretch(1)
        top.addWidget(box)

        row = QHBoxLayout()
        self.run = QPushButton("Run detection")
        self.cancel = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.export = QPushButton("Export CSV files")
        self.plot = QPushButton("Plot results")
        self.plot.setToolTip(
            "Shows one dot per cell with a bar at the mean, for every measurement, "
            "by condition or by day. Uses the metadata file and the Rs limit chosen "
            "above, and can save the figures as PDF and PNG.")
        self.open_folder = QPushButton("Open output folder")
        row.addWidget(self.run)
        row.addWidget(self.cancel)
        row.addWidget(self.progress, 1)
        row.addWidget(self.export)
        row.addWidget(self.plot)
        row.addWidget(self.open_folder)
        layout.addLayout(row)

        self.note = QLabel("")
        layout.addWidget(self.note)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeaderItem(5).setToolTip("Events per second of recording.")
        self.table.horizontalHeaderItem(6).setToolTip(
            "Mean event amplitude: the average of each sweep's mean.")
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 3)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress and the names of written files appear here.")
        layout.addWidget(self.log, 1)

        self.run.clicked.connect(self._run)
        self.cancel.clicked.connect(self._cancel)
        self.export.clicked.connect(self._export)
        self.plot.clicked.connect(self._plot)
        self.open_folder.clicked.connect(self._open_output)
        self.output.textChanged.connect(self._refresh_buttons)
        self.cell_column.currentIndexChanged.connect(self._update_match)
        self.group_column.currentIndexChanged.connect(self._update_match)

    # ---------------------------------------------------------------- settings

    def _restore(self):
        settings = QSettings()
        self.output.setText(settings.value("output_folder", "", str))
        self.rs_enabled.setChecked(settings.value("rs_enabled", True, bool))
        self.rs_value.setValue(settings.value("rs_value", 25.0, float))
        self.rs_value.setEnabled(self.rs_enabled.isChecked())
        metadata = settings.value("metadata_path", "", str)
        if metadata and os.path.isfile(metadata):
            self._set_metadata(metadata)

    def remember(self):
        settings = QSettings()
        settings.setValue("output_folder", self.output.text().strip())
        settings.setValue("rs_enabled", self.rs_enabled.isChecked())
        settings.setValue("rs_value", self.rs_value.value())
        settings.setValue("metadata_path", self.metadata_path.text())

    # ---------------------------------------------------------------- metadata

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the output folder", self.output.text() or self.session.start_dir)
        if folder:
            self.output.setText(folder)

    def _choose_metadata(self):
        start = os.path.dirname(self.metadata_path.text()) or self.session.start_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the metadata file", start, "CSV files (*.csv);;All files (*)")
        if path:
            self._set_metadata(path)

    def _set_metadata(self, path):
        self._metadata = None
        self.cell_column.clear()
        self.group_column.clear()
        self.metadata_path.setText("")
        if path:
            try:
                table = analysis.read_metadata(path)
            except Exception as exc:
                QMessageBox.warning(self, "Metadata file", f"Could not read this file:\n{exc}")
                table = None
            if table is not None and len(table.columns) < 2:
                QMessageBox.warning(
                    self, "Metadata file",
                    "This file needs at least two columns: one with the cell folder "
                    "names and one with each cell's condition.")
                table = None
            if table is not None:
                columns = [str(c) for c in table.columns]
                table.columns = columns
                self._metadata = table
                self.metadata_path.setText(path)
                for combo in (self.cell_column, self.group_column):
                    combo.blockSignals(True)
                    combo.addItems(columns)
                    combo.blockSignals(False)
                self.cell_column.setCurrentIndex(_guess(columns, CELL_COLUMN_NAMES, 0))
                self.group_column.setCurrentIndex(_guess(columns, GROUP_COLUMN_NAMES, 1))
        self._update_match()

    def conditions(self):
        """{cell: condition} from the metadata file, or None when there is none."""
        if self._metadata is None:
            return None
        cell, group = self.cell_column.currentText(), self.group_column.currentText()
        if not cell or not group or cell == group:
            return None
        return analysis.conditions_from_metadata(self._metadata, cell, group)

    def _update_match(self):
        if self._metadata is None:
            self.match.setText("Without a metadata file, all cells are exported as one group.")
            return
        conditions = self.conditions()
        if conditions is None:
            self.match.setText("Pick two different columns.")
            return
        names = [cell.name for cell in self.session.cells]
        matched = sum(1 for name in names if name in conditions)
        groups = len(set(conditions.values()))
        self.match.setText(
            f"{matched} of {len(names)} cell folders are listed, in {groups} conditions. "
            "Cells that are not listed are exported as 'Unknown'.")

    # --------------------------------------------------------------- detection

    def _folder_changed(self):
        self.results = None
        self._stale = False
        self.table.setRowCount(0)
        self._update_match()
        self._refresh_buttons()

    def _parameters_changed(self):
        if self.results is not None:
            self._stale = True
        self._refresh_buttons()

    def _refresh_buttons(self):
        running = self._job is not None
        tuned = self.session.n_tuned()
        self.run.setEnabled(not running and tuned > 0)
        self.cancel.setVisible(running)
        self.progress.setVisible(running)
        ready = self.results is not None and not self._stale and not running
        self.export.setEnabled(ready and bool(self.output.text().strip()))
        self.plot.setEnabled(ready)
        self.open_folder.setEnabled(os.path.isdir(self.output.text().strip()))
        if running:
            return
        if not self.session.cells:
            self.note.setText("Add a session folder first.")
        elif tuned == 0:
            self.note.setText("No cell has saved parameters yet. Save some on the Tune tab.")
        elif self.results is None:
            self.note.setText(
                f"{tuned} of {len(self.session.cells)} cells have saved parameters "
                "and will be analysed. Cells without parameters are skipped.")
        elif self._stale:
            self.note.setText("Parameters changed since the last run. "
                              "Run detection again before exporting.")
        elif not self.output.text().strip():
            self.note.setText("Choose an output folder to export.")
        else:
            self.note.setText("Detection finished. Ready to export.")

    def _run(self):
        cells = list(self.session.cells)
        n_folders = len({cell.folder for cell in cells})
        # Snapshot, so edits made while the job runs cannot change it mid-way.
        parameters = {cell: [dict(p) for p in sets]
                      for cell, sets in self.session.parameters.items()}

        def work(job):
            return analysis.detect_cells(
                cells, parameters,
                progress=job.progress.emit, cancelled=job.cancelled, log=job.message.emit)

        self._job = Job(self, work)
        self._job.progress.connect(self._on_progress)
        self._job.message.connect(self.log.appendPlainText)
        self._job.done.connect(self._on_done)
        self._job.failed.connect(self._on_failed)
        self._stale = False
        self._run_parameters = parameters
        self.progress.setRange(0, 0)
        self.note.setText("Running detection…")
        self.log.appendPlainText(
            f"Running detection on {len(cells)} cells in {n_folders} session folder"
            + ("" if n_folders == 1 else "s"))
        self._refresh_buttons()
        self._job.start()

    def _cancel(self):
        if self._job is not None:
            self._job.cancel()
            self.note.setText("Stopping…")

    def _on_progress(self, done, total, cell):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"{cell}   %v of %m sweeps")

    def _on_done(self, _token, results):
        self._job = None
        if results is None:
            self.log.appendPlainText("Detection cancelled.")
        else:
            self.results = results
            self._fill_table()
            n_events = sum(r.n_events for r in results)
            n_cells = len({r.cell for r in results})
            self.log.appendPlainText(f"Done: {n_events} events in {n_cells} cells.")
            self.status.emit(f"Detection finished: {n_events} events in {n_cells} cells")
        self._refresh_buttons()

    def _on_failed(self, _token, text):
        self._job = None
        self.log.appendPlainText(text)
        QMessageBox.critical(self, "Detection failed", text.strip().splitlines()[-1])
        self._refresh_buttons()

    def _fill_table(self):
        self.table.setRowCount(len(self.results))
        for row, r in enumerate(self.results):
            p = r.params
            cells = [
                r.cell, r.mode,
                f"{p['window_size']} / {p['sample_step']} / {p['interval']}",
                str(len(r.sweeps)), str(r.n_events),
                _num(r.frequency_hz, 2), _num(r.mean_amplitude, 1),
                _num(r.mean_rs, 1), _num(r.mean_rm, 1), _num(r.mean_cm, 1),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column >= 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)

    # ------------------------------------------------------------------ export

    def _export(self):
        output = self.output.text().strip()
        tag = self.tag.text().strip() or date.today().strftime("%Y%m%d")
        kinds = [key for key, check in self.kinds.items() if check.isChecked()]
        threshold = self.rs_value.value() if self.rs_enabled.isChecked() else None
        self.log.appendPlainText(f"Exporting to {output}")
        try:
            written = analysis.export_results(
                self.results, output, tag, conditions=self.conditions(),
                rs_threshold=threshold, kinds=kinds, log=self.log.appendPlainText)
            for name, save in (
                (f"{tag}_optimized_parameters.json",
                 lambda p: analysis.save_parameters(p, self._run_parameters)),
                (f"{tag}_optimized_results.pkl",
                 lambda p: analysis.save_legacy_results(p, self.results)),
            ):
                save(os.path.join(output, name))
                written.append(os.path.join(output, name))
                self.log.appendPlainText(f"  wrote {name}")
        except Exception as exc:
            self.log.appendPlainText(f"Export failed: {exc}")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.remember()
        self.status.emit(f"Wrote {len(written)} files to {output}")
        self._refresh_buttons()

    def _plot(self):
        """Open the plot window for the current results."""
        threshold = self.rs_value.value() if self.rs_enabled.isChecked() else None
        kept = analysis.apply_rs_threshold(self.results, threshold, log=lambda _: None)
        if not kept:
            QMessageBox.information(
                self, "Plot results",
                "There is nothing to plot: every cell is above the Rs limit, or its "
                "Rs could not be measured. Untick the Rs limit to plot them anyway.")
            return
        left_out = len(self.results) - len(kept)
        note = ""
        if left_out:
            note = (f"{left_out} left out: Rs above {threshold:g} M\u03a9 "
                    "or not measurable.")
        from .plot_window import PlotWindow  # loads matplotlib only when it is needed
        self.plot_window = PlotWindow(
            kept, self.conditions(), self.tag.text().strip(), self.output.text().strip(),
            note, self)
        self.plot_window.show()

    def _open_output(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output.text().strip()))


def _guess(columns, candidates, default):
    for i, name in enumerate(columns):
        if name.strip().lower() in candidates:
            return i
    return min(default, len(columns) - 1)


def _num(value, decimals):
    return "—" if value is None or np.isnan(value) else f"{value:.{decimals}f}"
