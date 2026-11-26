"""Batch detection across cells, per-cell summaries and CSV export.

File names and column layouts match what the original notebook wrote, so
downstream analysis keeps working. No GUI code here.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import detection as det

EXPORT_KINDS = ("summary", "passive", "ipi", "amplitudes")
MIN_INTERVAL_MS = 1.0


# ------------------------------------------------------------------ parameters

def make_param_set(window_size, sample_step, interval, mode, control_pulse_timing="early"):
    return {
        "window_size": int(window_size),
        "sample_step": int(sample_step),
        "interval": int(interval),
        "mode": det.normalize_mode(mode),
        "control_pulse_timing": control_pulse_timing or "early",
    }


def normalize_parameters(raw) -> dict:
    """Turn a loaded JSON mapping into {cell: [param_set, ...]}.

    Accepts the older one-dict-per-cell layout as well as lists of sets.
    """
    out = {}
    for cell, entry in (raw or {}).items():
        entries = entry if isinstance(entry, list) else [entry]
        sets = []
        for p in entries:
            if not isinstance(p, dict):
                continue
            try:
                sets.append(make_param_set(
                    p["window_size"], p["sample_step"], p["interval"],
                    p.get("mode", "EPSC"), p.get("control_pulse_timing", "early"),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        if sets:
            out[str(cell)] = sets
    return out


def load_parameters(path: str) -> dict:
    with open(path, "r") as f:
        return normalize_parameters(json.load(f))


def save_parameters(path: str, parameters: dict) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(parameters, f, indent=2)


def describe_param_set(p: dict) -> str:
    if "window_size" not in p:  # results loaded from a file that did not say
        return p["mode"]
    return f"{p['mode']} {p['window_size']}/{p['sample_step']}/{p['interval']}"


# --------------------------------------------------------------------- results

@dataclass
class SweepResult:
    path: str
    values: list = field(default_factory=list)   # event amplitudes (pA)
    indices: list = field(default_factory=list)  # event positions (samples)
    n_samples: int = 0
    fs: float = det.DEFAULT_FS
    error: str | None = None


@dataclass
class CellResult:
    """Detection output for one cell with one parameter set."""

    cell: str
    mode: str
    set_index: int  # 1-based position among this cell's parameter sets
    n_sets: int
    params: dict
    sweeps: list
    passive: list  # per sweep: (Rs, Rm, Cm, tau)
    session: str = ""  # name of the session folder (the recording day) the cell is in

    @property
    def n_events(self) -> int:
        return sum(len(s.values) for s in self.sweeps)

    @property
    def duration_s(self) -> float:
        return sum(s.n_samples / s.fs for s in self.sweeps)

    @property
    def frequency_hz(self) -> float:
        duration = self.duration_s
        return self.n_events / duration if duration > 0 else np.nan

    @property
    def mean_amplitude(self) -> float:
        """Mean of the per-sweep mean amplitudes (sweeps without events ignored)."""
        means = [np.mean(s.values) for s in self.sweeps if len(s.values)]
        return float(np.mean(means)) if means else np.nan

    @property
    def amplitudes(self) -> list:
        return [v for s in self.sweeps for v in s.values]

    def intervals_ms(self, min_interval_ms=MIN_INTERVAL_MS) -> list:
        """Inter-event intervals within each sweep, concatenated across sweeps."""
        out = []
        for s in self.sweeps:
            if len(s.indices) < 2:
                continue
            ms = np.diff(np.sort(s.indices)) * 1000.0 / s.fs
            if min_interval_ms is not None:
                ms = ms[ms >= min_interval_ms]
            out.extend(ms.tolist())
        return out

    def _passive_stats(self, column: int):
        valid = [p[column] for p in self.passive if not np.isnan(p[column])]
        if not valid:
            return np.nan, np.nan
        return float(np.mean(valid)), float(np.std(valid))

    @property
    def mean_rs(self) -> float:
        return self._passive_stats(0)[0]

    @property
    def mean_rm(self) -> float:
        return self._passive_stats(1)[0]

    @property
    def mean_cm(self) -> float:
        return self._passive_stats(2)[0]

    @property
    def label(self) -> str:
        return f"{self.cell} ({self.mode})" if self.n_sets > 1 else self.cell


def _try_load(path):
    try:
        return det.load_sweep(path), None
    except Exception as exc:
        return None, exc


def run_detection(folders, parameters, progress=None, cancelled=None, log=print,
                  name_with_folder=False):
    """Detect events in every cell that has saved parameters, in one session
    folder or a list of them. With `name_with_folder`, cells are named
    sessionfolder-cellfolder (as the app does by default) rather than by their
    own folder name. See detect_cells for the other arguments."""
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]
    cells, skipped = det.merge_sessions(
        [(str(folder), det.scan_cells(str(folder))) for folder in folders],
        name_with_folder)
    for folder, names in skipped.items():
        log(f"Left out of {os.path.basename(folder)} (same name as a cell in another "
            f"folder): {', '.join(names)}")
    return detect_cells(cells, parameters, progress, cancelled, log)


def detect_cells(cells, parameters, progress=None, cancelled=None, log=print):
    """Detect events in each of `cells` (detection.Cell) that has saved parameters.

    progress(done, total, cell_name) is called after each sweep; cancelled() is
    polled between sweeps. Returns a list of CellResult, or None if cancelled.
    """
    jobs = [
        (cell.name, det.list_sweeps(cell.path),
         os.path.basename(os.path.normpath(cell.folder)))
        for cell in cells
        if parameters.get(cell.name)
    ]
    total = sum(len(files) for _, files, _ in jobs)
    done = 0
    results = []
    # Reading over a network drive dominates the run time, so read a few sweeps at once.
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        for name, files, session in jobs:
            param_sets = parameters[name]
            per_set = [[] for _ in param_sets]
            passive = []
            for path, (sweep, error) in zip(files, pool.map(_try_load, files)):
                if cancelled is not None and cancelled():
                    return None
                if sweep is None:  # unreadable file: record it and carry on
                    log(f"  Could not read {name}/{os.path.basename(path)}: {error}")
                    for bucket in per_set:
                        bucket.append(SweepResult(path, error=str(error)))
                    passive.append(det._NAN4)
                else:
                    passive.append(det.compute_rs_rm_cm(sweep.raw, sweep.header, sweep.fs))
                    for bucket, p in zip(per_set, param_sets):
                        values, indices = det.detect_events(
                            sweep, p["window_size"], p["sample_step"], p["interval"],
                            p["mode"], p.get("control_pulse_timing"),
                        )
                        bucket.append(
                            SweepResult(path, values, indices, len(sweep.raw), sweep.fs))
                done += 1
                if progress is not None:
                    progress(done, total, name)
            for i, (p, bucket) in enumerate(zip(param_sets, per_set), start=1):
                result = CellResult(name, p["mode"], i, len(param_sets), p, bucket, passive,
                                    session)
                results.append(result)
                log(f"{result.label}: {result.n_events} events in {len(files)} sweeps "
                    f"({describe_param_set(p)})")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


def save_legacy_results(path: str, results) -> None:
    """Pickle results in the layout the notebook called `optimized_results`:
    a list of (mat_files, [(amplitudes, indices), ...]) per cell and parameter set."""
    legacy = [
        ([s.path for s in r.sweeps], [(s.values, s.indices) for s in r.sweeps])
        for r in results
    ]
    with open(path, "wb") as f:
        pickle.dump(legacy, f, protocol=pickle.HIGHEST_PROTOCOL)


# -- results saved before the app wrote its own results file, or by the notebook

class _PlainUnpickler(pickle.Unpickler):
    """Reads lists, numbers, text and NumPy arrays, and refuses anything else, so
    opening a .pkl file cannot run code that was hidden in it."""

    _ALLOWED = {
        "_reconstruct", "scalar", "ndarray", "dtype", "_frombuffer",
        "float64", "float32", "int64", "int32", "bool_",
    }

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"unexpected content ({module}.{name})")


def read_legacy_results(path: str) -> list:
    """The detections in a .pkl written by save_legacy_results or by the notebook:
    [(sweep files, [(amplitudes, positions) for each sweep])], one per cell and
    parameter set."""
    not_results = ValueError("this .pkl file does not hold mini detection results")
    try:
        with open(path, "rb") as f:
            raw = _PlainUnpickler(f).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError):
        raise not_results from None
    entries = []
    try:
        for files, sweeps in raw:
            files = [str(file) for file in files]
            sweeps = [([float(v) for v in values], [int(i) for i in indices])
                      for values, indices in sweeps]
            if len(files) != len(sweeps):
                raise not_results
            if files:
                entries.append((files, sweeps))
    except (TypeError, ValueError):
        raise not_results from None
    if not entries:
        raise not_results
    return entries


def _measure(path):
    """What a sweep's results need besides its events: length, sampling rate and
    (Rs, Rm, Cm, tau). Returns (those, None) or (None, the error)."""
    try:
        raw, header = det.read_mat(path)
        fs = det.sampling_rate(header)
        return (len(raw), fs, det.compute_rs_rm_cm(raw, header, fs)), None
    except Exception as exc:
        return None, exc


def load_legacy_results(path, parameters=None, progress=None, cancelled=None, log=print):
    """Rebuild full results from a .pkl that holds only the detected events.

    The events are taken from the file as they are; detection is not run again.
    Each recording is read once for its length, Rs, Rm and Cm, so the recordings
    must still be where they were. `parameters` ({cell: [param set]}) says which
    event type and settings each cell was detected with.

    progress(done, total, cell_name) is called after each recording; cancelled()
    is polled between them. Returns a list of CellResult, or None if cancelled.
    """
    entries = read_legacy_results(path)
    parameters = parameters or {}

    # One cell can have several entries, one per parameter set, in a row.
    cells = []  # [(cell folder, [entry, ...])]
    for entry in entries:
        folder = os.path.dirname(entry[0][0])
        if cells and cells[-1][0] == folder:
            cells[-1][1].append(entry)
        else:
            cells.append((folder, [entry]))

    def names(with_folder):
        return [det.cell_name(os.path.dirname(folder), folder, with_folder)
                for folder, _ in cells]

    # Name the cells the way the parameters do.
    with_folder = (sum(name in parameters for name in names(True))
                   >= sum(name in parameters for name in names(False)))

    files = list(dict.fromkeys(file for entry in entries for file in entry[0]))
    owner = {file: name for name, (_, group) in zip(names(with_folder), cells)
             for entry in group for file in entry[0]}
    measured = {}
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for done, (file, outcome) in enumerate(zip(files, pool.map(_measure, files)), 1):
            if cancelled is not None and cancelled():
                return None
            measured[file] = outcome
            if progress is not None:
                progress(done, len(files), owner[file])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results = []
    unknown = [name for name in names(with_folder) if not parameters.get(name)]
    if unknown:
        log("No saved parameters for these cells, so their event type was taken from "
            f"the direction of their events: {', '.join(unknown)}")
    for name, (folder, group) in zip(names(with_folder), cells):
        sets = parameters.get(name, [])
        if sets and len(sets) != len(group):
            log(f"{name}: the parameters list {len(sets)} event type(s) but the file "
                f"has {len(group)}; matched them in order")
        session = os.path.basename(os.path.dirname(folder))
        for i, (entry_files, entry_sweeps) in enumerate(group):
            if i < len(sets):
                params = sets[i]
            else:  # not in the parameters: tell the event type from the events
                amplitudes = [v for values, _ in entry_sweeps for v in values]
                inward = not amplitudes or np.median(amplitudes) < 0
                params = {"mode": "EPSC" if inward else "IPSC"}
            sweeps, passive = [], []
            for file, (values, indices) in zip(entry_files, entry_sweeps):
                found, error = measured[file]
                if found is not None:
                    n_samples, fs, rc = found
                    sweeps.append(SweepResult(file, values, indices, n_samples, fs))
                    passive.append(rc)
                elif values:
                    raise ValueError(
                        f"could not open the recording {file} ({error}). The recordings "
                        "are read once, for their length, Rs, Rm and Cm. If they are on "
                        "a drive that is not connected, connect it and try again")
                else:  # it had no events, so leaving it out changes nothing
                    sweeps.append(SweepResult(file, error=str(error)))
                    passive.append(det._NAN4)
            results.append(CellResult(name, params["mode"], i + 1, len(group), params,
                                      sweeps, passive, session))
    return results


RESULTS_FORMAT = "mini_detect results"


def save_results(path: str, results) -> None:
    """Save everything about a detection run to one JSON file, so it can be
    loaded again later without re-reading the recordings."""
    data = {
        "format": RESULTS_FORMAT,
        "version": 1,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "cells": [{
            "cell": r.cell,
            "mode": r.mode,
            "set_index": r.set_index,
            "n_sets": r.n_sets,
            "params": r.params,
            "session": r.session,
            "passive": [[float(v) for v in values] for values in r.passive],
            "sweeps": [{
                "path": s.path,
                "values": [float(v) for v in s.values],
                "indices": [int(i) for i in s.indices],
                "n_samples": int(s.n_samples),
                "fs": float(s.fs),
                "error": s.error,
            } for s in r.sweeps],
        } for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_results(path: str):
    """Read a file written by save_results. Returns (results, when it was saved)."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != RESULTS_FORMAT:
        raise ValueError("this is not a results file saved by Mini Detect")
    results = []
    for c in data["cells"]:
        passive = [tuple(values) for values in c["passive"]]
        sweeps = [SweepResult(s["path"], s["values"], s["indices"], s["n_samples"],
                              s["fs"], s.get("error")) for s in c["sweeps"]]
        results.append(CellResult(c["cell"], c["mode"], c["set_index"], c["n_sets"],
                                  c["params"], sweeps, passive, c.get("session", "")))
    return results, data.get("saved", "")


# ---------------------------------------------------------------------- export

def read_metadata(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, skipinitialspace=True)


def conditions_from_metadata(metadata: pd.DataFrame, cell_col: str, group_col: str) -> dict:
    table = metadata[[cell_col, group_col]].dropna()
    return {
        str(cell).strip(): str(group).strip()
        for cell, group in zip(table[cell_col], table[group_col])
    }


def _safe(name) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip() or "Unnamed"


def _fmt_list(values) -> str:
    return ", ".join("NaN" if np.isnan(v) else f"{v:.2f}" for v in values)


def _padded_table(columns: dict, index_name: str) -> pd.DataFrame:
    longest = max((len(v) for v in columns.values()), default=0)
    data = {k: list(v) + [np.nan] * (longest - len(v)) for k, v in columns.items()}
    df = pd.DataFrame(data)
    df.insert(0, index_name, np.arange(1, longest + 1))
    return df


def _unique(name: str, taken) -> str:
    candidate, n = name, 1
    while candidate in taken:
        n += 1
        candidate = f"{name}_{n}"
    return candidate


def apply_rs_threshold(results, rs_threshold, log=print) -> list:
    """Results without the cells whose mean series resistance is above
    `rs_threshold` or could not be measured. None keeps everything."""
    if rs_threshold is None:
        return list(results)
    kept = []
    for r in results:
        rs = r.mean_rs
        if np.isnan(rs):
            log(f"Excluded {r.label}: series resistance could not be measured")
        elif rs > rs_threshold:
            log(f"Excluded {r.label}: mean Rs {rs:.1f} MOhm > {rs_threshold:g} MOhm")
        else:
            kept.append(r)
    return kept


def passive_table(results, condition_of) -> pd.DataFrame:
    rows = []
    for r in results:
        rs, rs_sd = r._passive_stats(0)
        rm, rm_sd = r._passive_stats(1)
        cm, cm_sd = r._passive_stats(2)
        rows.append({
            "Cell_Name": r.cell,
            "Condition": condition_of(r.cell),
            "Mode": r.mode,
            "Param_Set_Index": r.set_index,
            "Control_Pulse_Timing": "from_header",
            "Mean_Rs_MOhm": rs, "Std_Rs_MOhm": rs_sd,
            "Mean_Rm_MOhm": rm, "Std_Rm_MOhm": rm_sd,
            "Mean_Cm_pF": cm, "Std_Cm_pF": cm_sd,
            "N_Files": len(r.passive),
            "Rs_Values": _fmt_list([p[0] for p in r.passive]),
            "Rm_Values": _fmt_list([p[1] for p in r.passive]),
            "Cm_Values": _fmt_list([p[2] for p in r.passive]),
            "Tau_Values_ms": _fmt_list([p[3] for p in r.passive]),
        })
    return pd.DataFrame(rows)


def export_results(results, output_dir, tag, conditions=None, rs_threshold=None,
                   kinds=EXPORT_KINDS, log=print) -> list:
    """Write the CSV files. Returns the list of paths written.

    conditions: {cell name: condition}, or None to export everything as one group.
    rs_threshold: if given, cells whose mean series resistance is above it (or
        could not be measured) are left out of every file.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []

    def condition_of(cell):
        if conditions is None:
            return "All"
        return conditions.get(cell, "Unknown")

    def write(df, filename):
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        written.append(path)
        log(f"  wrote {filename}")

    if conditions is not None:
        missing = sorted({r.cell for r in results if r.cell not in conditions},
                         key=det.natural_key)
        if missing:
            log(f"Not in the metadata file, exported as 'Unknown': {', '.join(missing)}")

    results = apply_rs_threshold(results, rs_threshold, log)

    grouped = {}
    for r in results:
        grouped.setdefault(condition_of(r.cell), []).append(r)

    if "passive" in kinds:
        table = passive_table(results, condition_of)
        write(table, f"Rs_Rm_Cm_all_cells_{tag}.csv")
        if conditions is not None and len(table):
            for condition, part in table.groupby("Condition", sort=False):
                write(part.drop(columns="Condition"), f"Rs_Rm_Cm_{_safe(condition)}_{tag}.csv")

    if "summary" in kinds:
        for data_type, getter in (("Amplitude", lambda r: r.mean_amplitude),
                                  ("Frequency", lambda r: r.frequency_hz)):
            column = f"{data_type}_Value"
            combined = []
            for condition, members in grouped.items():
                rows = [{
                    "Cell_Name": r.cell,
                    "Condition": condition,
                    "Mode": r.mode,
                    "Param_Set_Index": r.set_index,
                    column: getter(r),
                } for r in members]
                combined.extend(rows)
                write(pd.DataFrame(rows).drop(columns="Condition"),
                      f"{_safe(condition)}_{data_type}_{tag}.csv")
            write(pd.DataFrame(combined, columns=["Cell_Name", "Condition", "Mode",
                                                  "Param_Set_Index", column]),
                  f"{data_type}_combined_{tag}.csv")

    per_event = (
        ("ipi", "Interval_Number", "IPI_by_cell", lambda r: r.intervals_ms()),
        ("amplitudes", "Amplitude_Number", "ampls_by_cell", lambda r: r.amplitudes),
    )
    for kind, index_name, suffix, getter in per_event:
        if kind not in kinds:
            continue
        for condition, members in grouped.items():
            columns = {}
            for r in members:
                columns[_unique(r.label, columns)] = getter(r)
            write(_padded_table(columns, index_name), f"{_safe(condition)}_{suffix}_{tag}.csv")

    return written
