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
from .. import detection as det


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
        # Tidy up only now that the result has been handed over in full. Deleting
        # the job from a slot on `done` could run while this thread was still
        # delivering `done` to the other slots, and they then never got it.
        try:
            self.deleteLater()
        except RuntimeError:
            pass

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
        self.sessions = []     # [(session folder, [Cell, ...])] in the order added
        self.missing = []      # folders from last time that could not be opened
        self.cells = []        # every loaded Cell, across all session folders
        self.skipped = {}      # {session folder: [cell names left out as duplicates]}
        self.parameters = {}   # {cell name: [param set, ...]}

    # -- session folders

    @property
    def folders(self) -> list:
        return [folder for folder, _ in self.sessions]

    @property
    def start_dir(self) -> str:
        """Where file dialogs open: next to the first session folder."""
        return os.path.dirname(self.sessions[0][0]) if self.sessions else ""

    def restore(self) -> list:
        """Load the parameters autosaved last time. Returns the session folders
        that were open then, for the caller to scan and add again."""
        folders, self.parameters = self._read_autosave()
        self.parameters_changed.emit()
        return folders

    def add_sessions(self, found, missing=()) -> list:
        """Add scanned session folders, as [(folder, cells)]; a folder that is
        already open is refreshed. Returns the folders that are new."""
        known = dict(self.sessions)
        added = [folder for folder, _ in found if folder not in known]
        known.update(found)
        self.sessions = list(known.items())
        self.missing = [f for f in dict.fromkeys([*self.missing, *missing])
                        if f not in known]
        adopted = any([self._adopt_old_autosave(folder) for folder in added])
        self._folders_changed()
        if adopted:
            self.parameters_changed.emit()
        return added

    def remove_folder(self, folder: str) -> None:
        """Take a session folder off the list. Saved parameters are kept."""
        self.sessions = [(f, cells) for f, cells in self.sessions if f != folder]
        self.missing = [f for f in self.missing if f != folder]
        self._folders_changed()

    def _folders_changed(self):
        self.cells, self.skipped = det.merge_sessions(self.sessions)
        self._write_autosave()
        self.cells_changed.emit()

    def n_tuned(self) -> int:
        return sum(1 for cell in self.cells if self.parameters.get(cell.name))

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
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        return os.path.join(base, "workspace.json")

    def _read_autosave(self) -> tuple:
        """(session folders, parameters) as they were when the app last closed."""
        try:
            with open(self._autosave_path(), "r") as f:
                data = json.load(f)
            folders = [f for f in data.get("folders", []) if isinstance(f, str)]
            return folders, analysis.normalize_parameters(data.get("parameters"))
        except (OSError, ValueError, AttributeError):
            return [], {}

    def _write_autosave(self):
        data = {"folders": self.folders + self.missing, "parameters": self.parameters}
        try:
            path = self._autosave_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _adopt_old_autosave(self, folder: str) -> bool:
        """Version 0.1 autosaved parameters per data folder. Move any such file
        for `folder` into the current parameters, without overwriting anything."""
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        key = hashlib.sha1(os.path.abspath(folder).encode()).hexdigest()[:16]
        path = os.path.join(base, "sessions", f"{key}.json")
        try:
            with open(path, "r") as f:
                old = analysis.normalize_parameters(json.load(f))
            os.remove(path)
        except (OSError, ValueError):
            return False
        new = {cell: sets for cell, sets in old.items() if cell not in self.parameters}
        self.parameters.update(new)
        return bool(new)
