"""The Tune tab: look at a few sweeps from one cell and adjust the detection
parameters until the marked events look right."""
from __future__ import annotations

import os
import random
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from .. import analysis
from .. import detection as det
from .state import Job

PRESETS = [(60, 15, 5), (100, 25, 15), (130, 35, 20), (200, 50, 50), (300, 50, 70)]
MODE_LABELS = [
    ("EPSC  (inward events)", "EPSC"),
    ("IPSC  (outward events)", "IPSC"),
    ("IPSC, high Cl⁻  (inward events)", "IPSC_high_Cl"),
]
PEAK_COLOR = "#e03131"
CACHE_SIZE = 40


class TunePanel(QWidget):
    next_cell_requested = Signal()
    status = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._cell = None           # (name, path, n sweeps)
        self._all_files = {}        # cell path -> every sweep file
        self._samples = {}          # cell path -> the sweeps currently shown
        self._cache = OrderedDict() # file path -> loaded Sweep
        self._shown = []            # Sweeps on screen
        self._plots = []            # (plot, scatter) per shown sweep
        self._token = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._update_detection)

        self._build()
        session.parameters_changed.connect(self._refresh_saved)
        self._set_enabled(False)

    # ------------------------------------------------------------------ layout

    def _build(self):
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        self.header = QLabel("Choose a data folder, then pick a cell.")
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left.addWidget(self.header)
        self.graphics = pg.GraphicsLayoutWidget()
        left.addWidget(self.graphics, 1)
        hint = QLabel("Scroll to zoom, drag to pan. Right-click a trace and choose "
                      "View All to see the whole sweep. The grey band is the "
                      "RC-check pulse, which is ignored.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        left.addWidget(hint)
        layout.addLayout(left, 1)

        side = QVBoxLayout()
        layout.addLayout(side)

        box = QGroupBox("Detection parameters")
        form = QFormLayout(box)
        self.mode = QComboBox()
        for label, value in MODE_LABELS:
            self.mode.addItem(label, value)
        form.addRow("Event type", self.mode)
        self.preset = QComboBox()
        self.preset.addItem("Custom", None)
        for p in PRESETS:
            self.preset.addItem(" / ".join(map(str, p)), p)
        self.preset.setToolTip("Commonly used combinations of window / step / peak search.")
        form.addRow("Preset", self.preset)
        self.window_size = self._spin(2, 5000, 60,
            "Length of the stretch after an event that must decay steadily back "
            "toward baseline. Longer windows are stricter and suit slower events.")
        form.addRow("Decay window", self.window_size)
        self.sample_step = self._spin(1, 1000, 15,
            "Within the decay window the trace is checked every this many samples. "
            "Smaller steps are stricter, larger steps tolerate more noise.")
        form.addRow("Sample step", self.sample_step)
        self.interval = self._spin(1, 1000, 5,
            "How far from the start of the decay window to look for the event's peak.")
        form.addRow("Peak search", self.interval)
        side.addWidget(box)

        box = QGroupBox("Sweeps shown")
        row = QHBoxLayout(box)
        self.n_sweeps = QSpinBox()
        self.n_sweeps.setRange(1, 6)
        self.n_sweeps.setValue(3)
        row.addWidget(self.n_sweeps)
        self.shuffle = QPushButton("Show different sweeps")
        self.shuffle.setToolTip("Pick another random set of sweeps from this cell.")
        row.addWidget(self.shuffle, 1)
        side.addWidget(box)

        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        side.addWidget(self.stats)

        self.save = QPushButton("Save for this cell")
        self.save_next = QPushButton("Save and go to next cell")
        self.save_next.setDefault(True)
        self.save_next.setToolTip("Save, then move to the next cell without parameters.")
        side.addWidget(self.save)
        side.addWidget(self.save_next)

        box = QGroupBox("Saved for this cell")
        inner = QVBoxLayout(box)
        self.saved = QListWidget()
        self.saved.setMaximumHeight(90)
        self.saved.setToolTip("One set per event type. Click a set to load it.")
        inner.addWidget(self.saved)
        self.remove = QPushButton("Remove selected")
        inner.addWidget(self.remove)
        side.addWidget(box)
        side.addStretch(1)

        self.mode.currentIndexChanged.connect(self._params_edited)
        for spin in (self.window_size, self.sample_step, self.interval):
            spin.valueChanged.connect(self._params_edited)
        self.preset.activated.connect(self._preset_chosen)
        self.n_sweeps.valueChanged.connect(self._n_sweeps_changed)
        self.shuffle.clicked.connect(self._shuffle)
        self.save.clicked.connect(self._save)
        self.save_next.clicked.connect(self._save_and_next)
        self.saved.itemClicked.connect(self._saved_clicked)
        self.remove.clicked.connect(self._remove_saved)

    @staticmethod
    def _spin(low, high, value, tip):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSuffix(" samples")
        spin.setToolTip(tip)
        return spin

    def _set_enabled(self, enabled):
        for w in (self.mode, self.preset, self.window_size, self.sample_step, self.interval,
                  self.n_sweeps, self.shuffle, self.save, self.save_next, self.saved,
                  self.remove):
            w.setEnabled(enabled)

    # ----------------------------------------------------------- current values

    def current_set(self) -> dict:
        return analysis.make_param_set(
            self.window_size.value(), self.sample_step.value(), self.interval.value(),
            self.mode.currentData(),
        )

    def _apply_set(self, p: dict):
        widgets = (self.mode, self.window_size, self.sample_step, self.interval)
        for w in widgets:
            w.blockSignals(True)
        self.mode.setCurrentIndex(max(0, self.mode.findData(det.normalize_mode(p["mode"]))))
        self.window_size.setValue(p["window_size"])
        self.sample_step.setValue(p["sample_step"])
        self.interval.setValue(p["interval"])
        for w in widgets:
            w.blockSignals(False)
        self._params_edited()

    def _params_edited(self):
        current = (self.window_size.value(), self.sample_step.value(), self.interval.value())
        self.preset.setCurrentIndex(PRESETS.index(current) + 1 if current in PRESETS else 0)
        self._timer.start()

    def _preset_chosen(self, index):
        preset = self.preset.itemData(index)
        if preset:
            self._apply_set({**self.current_set(), "window_size": preset[0],
                             "sample_step": preset[1], "interval": preset[2]})

    # ------------------------------------------------------------ choosing cell

    def set_cell(self, cell):
        self._cell = cell
        self._refresh_saved()
        if cell is None:
            self._token += 1
            self._set_enabled(False)
            self._clear_plots()
            self.header.setText("Choose a data folder, then pick a cell.")
            self.stats.setText("")
            return
        self._set_enabled(True)
        sets = self.session.parameters.get(cell[0])
        if sets:
            self._apply_set(sets[0])
        self._load()

    def _load(self):
        """Make sure the sample sweeps for the current cell are loaded, then draw."""
        name, path, _ = self._cell
        files = self._samples.get(path)
        if files is not None and all(f in self._cache for f in files):
            self._draw()
            return
        self._token += 1
        self.header.setText(f"<b>{name}</b> &nbsp; loading sweeps…")
        known = self._all_files.get(path)
        n = self.n_sweeps.value()
        cached = set(self._cache)

        def work(job):
            all_files = known if known is not None else det.list_sweeps(path)
            chosen = files if files is not None else _sample(all_files, n)
            loaded = {}
            for f in chosen:
                if f in cached:
                    continue
                try:
                    loaded[f] = det.load_sweep(f)
                except Exception as exc:
                    loaded[f] = exc
            return path, all_files, chosen, loaded

        job = Job(self, work, token=self._token)
        job.done.connect(self._loaded)
        job.failed.connect(self._load_failed)
        job.start()

    def _loaded(self, token, result):
        path, all_files, chosen, loaded = result
        self._all_files[path] = all_files
        self._samples[path] = chosen
        for f, sweep in loaded.items():
            self._cache[f] = sweep
            self._cache.move_to_end(f)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        if token == self._token and self._cell and self._cell[1] == path:
            self._draw()

    def _load_failed(self, token, text):
        if token == self._token:
            self.header.setText("Could not load the sweeps for this cell.")
            self.status.emit(text.strip().splitlines()[-1])

    def _n_sweeps_changed(self, n):
        if not self._cell:
            return
        path = self._cell[1]
        files = self._samples.get(path)
        all_files = self._all_files.get(path)
        if files is None or all_files is None:
            return
        if n < len(files):
            self._samples[path] = files[:n]
        else:
            others = [f for f in all_files if f not in files]
            self._samples[path] = files + _sample(others, n - len(files))
        self._load()

    def _shuffle(self):
        if not self._cell:
            return
        path = self._cell[1]
        all_files = self._all_files.get(path)
        if all_files is None:
            return
        current = self._samples.get(path) or []
        n = self.n_sweeps.value()
        others = [f for f in all_files if f not in current]
        # Prefer sweeps that are not on screen; fall back to any when too few remain.
        self._samples[path] = _sample(others if len(others) >= n else all_files, n)
        self._load()

    # ---------------------------------------------------------------- plotting

    def _clear_plots(self):
        self.graphics.clear()
        self._plots = []
        self._shown = []

    def _draw(self):
        name, path, n_total = self._cell
        self._clear_plots()
        files = self._samples.get(path, [])
        sweeps = [self._cache[f] for f in files if isinstance(self._cache.get(f), det.Sweep)]
        failed = [os.path.basename(f) for f in files
                  if not isinstance(self._cache.get(f), det.Sweep)]
        if failed:
            self.status.emit(f"Could not read: {', '.join(failed)}")
        self.header.setText(
            f"<b>{name}</b> &nbsp; showing {len(sweeps)} of {n_total} sweeps")
        self._shown = sweeps
        if not sweeps:
            self.stats.setText("")
            return

        dark = self.palette().window().color().lightness() < 128
        trace_pen = pg.mkPen("#74a9f7" if dark else "#1c5fb8", width=1)
        y_low, y_high = np.inf, -np.inf
        first = None
        for row, sweep in enumerate(sweeps):
            plot = self.graphics.addPlot(row=row, col=0)
            plot.showGrid(x=False, y=True, alpha=0.15)
            plot.setLabel("left", "pA")
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)
            time = np.arange(len(sweep.filtered)) / sweep.fs
            plot.plot(time, sweep.filtered, pen=trace_pen)

            lo, hi = sweep.control_window or det.fallback_pulse_window("early", sweep.fs)
            region = pg.LinearRegionItem(values=(lo / sweep.fs, hi / sweep.fs), movable=False,
                                         brush=(128, 128, 128, 70), pen=pg.mkPen(None))
            region.setZValue(-10)
            plot.addItem(region)

            scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None),
                                         brush=pg.mkBrush(PEAK_COLOR))
            scatter.setZValue(10)
            plot.addItem(scatter)

            # Scale to the trace outside the RC-check pulse so minis are visible.
            keep = np.ones(len(sweep.filtered), dtype=bool)
            keep[max(lo - int(0.002 * sweep.fs), 0):hi + int(0.03 * sweep.fs)] = False
            if keep.any():
                y_low = min(y_low, float(np.nanmin(sweep.filtered[keep])))
                y_high = max(y_high, float(np.nanmax(sweep.filtered[keep])))

            if first is None:
                first = plot
            else:
                plot.setXLink(first)
                plot.setYLink(first)
            if row == len(sweeps) - 1:
                plot.setLabel("bottom", "Time", units="s")
            self._plots.append((plot, scatter))

        if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
            pad = 0.05 * (y_high - y_low)
            first.setYRange(y_low - pad, y_high + pad, padding=0)
        first.setXRange(0, max(s.duration_s for s in sweeps), padding=0.01)
        self._update_detection()

    def _update_detection(self):
        if not self._shown:
            return
        p = self.current_set()
        amplitudes = []
        duration = 0.0
        for sweep, (plot, scatter) in zip(self._shown, self._plots):
            values, indices = det.detect_events(
                sweep, p["window_size"], p["sample_step"], p["interval"], p["mode"])
            scatter.setData(x=np.asarray(indices) / sweep.fs, y=np.asarray(values))
            rate = len(values) / sweep.duration_s if sweep.duration_s else 0.0
            plot.setTitle(f"{sweep.name} — {_count(len(values))} ({rate:.1f} Hz)",
                          size="10pt")
            amplitudes.extend(values)
            duration += sweep.duration_s
        if amplitudes:
            self.stats.setText(
                f"<b>{_count(len(amplitudes))}</b> in {len(self._shown)} sweeps<br>"
                f"{len(amplitudes) / duration:.2f} Hz · "
                f"mean {np.mean(amplitudes):.1f} pA")
        else:
            self.stats.setText("<b>No events</b> with these parameters.")

    # ------------------------------------------------------------- saved sets

    def _refresh_saved(self):
        self.saved.clear()
        if self._cell:
            for p in self.session.parameters.get(self._cell[0], []):
                self.saved.addItem(analysis.describe_param_set(p))

    def _saved_clicked(self, item):
        sets = self.session.parameters.get(self._cell[0], [])
        row = self.saved.row(item)
        if 0 <= row < len(sets):
            self._apply_set(sets[row])

    def _remove_saved(self):
        row = self.saved.currentRow()
        if self._cell and row >= 0:
            self.session.remove_set(self._cell[0], row)

    def _save(self):
        if not self._cell:
            return False
        p = self.current_set()
        self.session.save_set(self._cell[0], p)
        self.status.emit(f"Saved {analysis.describe_param_set(p)} for {self._cell[0]}")
        return True

    def _save_and_next(self):
        if self._save():
            self.next_cell_requested.emit()


def _count(n):
    return "1 event" if n == 1 else f"{n} events"


def _sample(files, n):
    files = list(files)
    if len(files) <= n:
        return files
    return sorted(random.sample(files, n), key=lambda f: det.natural_key(os.path.basename(f)))
