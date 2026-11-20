"""Shared state for the window: the open data folder and the per-cell parameters,
plus a small helper for running slow work off the UI thread."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback

from PySide6.QtCore import QObject, QStandardPaths, Signal

from .. import analysis


class Job(QObject):
    """Run `fn(job)` on a background thread and report back on the UI thread."""

    done = Signal(object, object)    # token, result
    failed = Signal(object, str)     # token, traceback text
    progress = Signal(int, int, str)
    message = Signal(str)

    def __init__(self, parent, fn, token=None):
        super().__init__(parent)
        self._fn = fn
        self.token = token
        self._cancel = threading.Event()
        self.done.connect(self._finished)
        self.failed.connect(self._finished)

    def _finished(self, _token=None, _result=None):
        self.deleteLater()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _run(self):
        try:
            result = self._fn(self)
        except Exception:
            self._emit(self.failed, self.token, traceback.format_exc())
        else:
            self._emit(self.done, self.token, result)

    @staticmethod
    def _emit(signal, *args):
        try:
            signal.emit(*args)
        except RuntimeError:  # the window was closed while the job was running
            pass


class Session(QObject):
    cells_changed = Signal()
    parameters_changed = Signal()

    def __init__(self):
        super().__init__()
        self.data_folder = ""
        self.cells = []        # (name, path, number of sweeps)
        self.parameters = {}   # {cell name: [param set, ...]}

    # -- data folder

    def open_folder(self, folder: str, cells: list) -> int:
        """Switch to a new data folder. Returns how many cells' parameters were
        restored from the last session on that folder."""
        self.data_folder = folder
        self.cells = cells
        self.parameters = self._read_autosave()
        self.cells_changed.emit()
        self.parameters_changed.emit()
        return len(self.parameters)

    def n_tuned(self) -> int:
        return sum(1 for name, _, _ in self.cells if self.parameters.get(name))

    # -- parameters

    def save_set(self, cell: str, param_set: dict) -> None:
        """Store a parameter set for a cell, replacing any set with the same event type."""
        sets = self.parameters.setdefault(cell, [])
        for i, existing in enumerate(sets):
            if existing["mode"] == param_set["mode"]:
                sets[i] = param_set
                break
        else:
            sets.append(param_set)
        self._changed()

    def remove_set(self, cell: str, index: int) -> None:
        sets = self.parameters.get(cell, [])
        if 0 <= index < len(sets):
            del sets[index]
            if not sets:
                del self.parameters[cell]
            self._changed()

    def replace_parameters(self, parameters: dict, merge: bool = False) -> None:
        self.parameters = {**self.parameters, **parameters} if merge else dict(parameters)
        self._changed()

    def _changed(self):
        self._write_autosave()
        self.parameters_changed.emit()

    # -- autosave, so tuning work survives closing the window

    def _autosave_path(self):
        if not self.data_folder:
            return None
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        key = hashlib.sha1(os.path.abspath(self.data_folder).encode()).hexdigest()[:16]
        return os.path.join(base, "sessions", f"{key}.json")

    def _read_autosave(self) -> dict:
        path = self._autosave_path()
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as f:
                return analysis.normalize_parameters(json.load(f))
        except (OSError, ValueError):
            return {}

    def _write_autosave(self):
        path = self._autosave_path()
        if not path:
            return
        try:
            analysis.save_parameters(path, self.parameters)
        except OSError:
            pass
