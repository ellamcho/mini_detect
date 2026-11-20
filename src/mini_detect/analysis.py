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


def run_detection(parent_folder, parameters, progress=None, cancelled=None, log=print):
    """Detect events in every cell folder that has saved parameters.

    progress(done, total, cell_name) is called after each sweep; cancelled() is
    polled between sweeps. Returns a list of CellResult, or None if cancelled.
    """
    jobs = [
        (name, det.list_sweeps(folder))
        for name, folder, _ in det.scan_cells(parent_folder)
        if parameters.get(name)
    ]
    total = sum(len(files) for _, files in jobs)
    done = 0
    results = []
    # Reading over a network drive dominates the run time, so read a few sweeps at once.
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        for name, files in jobs:
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
                result = CellResult(name, p["mode"], i, len(param_sets), p, bucket, passive)
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

    if rs_threshold is not None:
        kept = []
        for r in results:
            rs = r.mean_rs
            if np.isnan(rs):
                log(f"Excluded {r.label}: series resistance could not be measured")
            elif rs > rs_threshold:
                log(f"Excluded {r.label}: mean Rs {rs:.1f} MOhm > {rs_threshold:g} MOhm")
            else:
                kept.append(r)
        results = kept

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
