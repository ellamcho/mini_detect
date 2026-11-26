"""Window that shows the summary plots and saves them."""
from __future__ import annotations

import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout,
)

from .. import plots


class PlotWindow(QDialog):
    def __init__(self, results, conditions, tag, output_dir, note="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plots")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.results = results
        self.conditions = conditions
        self.tag = tag
        self.output_dir = output_dir
        self.limits = plots.axis_limits(results)
        paired = [r for pairs in plots.find_pairs(results).values() for pair in pairs
                  for r in pair]
        self.pair_limits = plots.axis_limits(paired, headroom=0.25)
        self.note = note
        self.specs = []
        self.saved_to = ""

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Plot"))
        self.by = QComboBox()
        self.by.addItem("By condition  (all days together)", plots.BY_CONDITION)
        self.by.addItem("By day  (conditions compared within each day)", plots.BY_DAY)
        self.by.addItem("Before and after  (cells recorded twice, paired t test)",
                        plots.BY_PAIR)
        self.by.setToolTip(
            "A day is one session folder. By day gives one figure per session folder, "
            "comparing the conditions recorded in it.\n\n"
            "Before and after pairs a cell with a second recording of it: a cell whose "
            "name is another cell's name plus an underscore and a label, such as "
            "20261117-2 and 20261117-2_picro. Each treatment gets a figure.")
        row.addWidget(self.by)
        row.addSpacing(16)
        row.addWidget(QLabel("Figure"))
        self.which = QComboBox()
        self.which.setMinimumWidth(260)
        row.addWidget(self.which)
        row.addStretch(1)
        self.save_btn = QPushButton("Save all as PDF and PNG")
        self.save_btn.setToolTip(
            "Saves every figure in the list. The PDF keeps text editable for Illustrator.")
        row.addWidget(self.save_btn)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        row.addWidget(self.open_btn)
        layout.addLayout(row)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("QScrollArea { background: white; }")
        layout.addWidget(self.scroll, 1)

        self.by.currentIndexChanged.connect(self._rebuild)
        self.which.currentIndexChanged.connect(self._show)
        self.save_btn.clicked.connect(self._save)
        self.open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.saved_to)))
        self._rebuild()

        width, height = plots.figure_size(self.specs[0]) if self.specs else (9, 3)
        screen = self.screen().availableGeometry().width() if self.screen() else 1500
        self.resize(min(int(width * 100) + 60, screen - 80), int(height * 100) + 150)

    def _limits(self):
        return self.pair_limits if self.by.currentData() == plots.BY_PAIR else self.limits

    def _rebuild(self):
        self.specs = plots.build_figures(self.results, self.conditions, self.by.currentData())
        self.save_btn.setEnabled(bool(self.specs))
        if not self.specs:
            message = QLabel(
                "No cells were recorded twice.\n\nA pair is two cells where one is named "
                "after the other with an underscore and a label added,\nfor example "
                "20261117-2 and 20261117-2_picro.")
            message.setAlignment(Qt.AlignCenter)
            self.scroll.setWidget(message)
            self.info.setText(self.note)
        self.which.blockSignals(True)
        self.which.clear()
        self.which.addItems([spec.title for spec in self.specs])
        self.which.blockSignals(False)
        self._show(0)

    def _show(self, index):
        if not 0 <= index < len(self.specs):
            return
        spec = self.specs[index]
        if spec.pairs:
            tests = [f"{row['Measurement'].split(' (')[0]} {_p_text(row['p'])}"
                     for row in plots.paired_stats(spec) if row["N_Pairs"] >= 2]
            text = ("Each line joins one cell before and after. Paired t test:  "
                    + ("   \u00b7   ".join(tests) if tests else "needs at least two pairs."))
        else:
            text = "One dot per cell; the black bar is the mean."
        self.info.setText(f"{text}  {self.note}".strip())
        figure = plots.make_figure(spec, self._limits())
        canvas = FigureCanvasQTAgg(figure)
        width, height = figure.get_size_inches()
        canvas.setFixedSize(int(width * figure.dpi), int(height * figure.dpi))
        self.scroll.setWidget(canvas)  # the scroll area deletes the previous canvas

    def _save(self):
        folder = os.path.join(self.output_dir, "plots") if self.output_dir else ""
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Choose where to save the plots")
            if not folder:
                return
        try:
            written = plots.save_figures(self.specs, self._limits(), folder, self.tag)
            if self.by.currentData() == plots.BY_PAIR:
                stats_path = os.path.join(folder, f"{self.tag}_paired_t_tests.csv".strip("_"))
                plots.save_paired_stats(self.specs, stats_path)
                written.append(stats_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save plots", f"Could not save the plots:\n{exc}")
            return
        self.saved_to = folder
        self.open_btn.setEnabled(True)
        self.info.setText(f"Saved {len(written)} files to {folder}")


def _p_text(p) -> str:
    if p != p:  # not a number: the test could not be done
        return "p not available"
    return ("p < 0.0001" if p < 0.0001 else f"p = {p:.3g}") + f" ({plots.stars(p)})"

