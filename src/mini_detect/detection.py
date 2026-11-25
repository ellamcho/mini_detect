"""Core mini (mEPSC / mIPSC) detection for ScanImage .mat sweeps.

This is the same algorithm as the original notebook (functions.py / param_tune.py):

1. band-pass filter the sweep (1-3000 Hz),
2. find non-overlapping windows in which the trace decays monotonically,
3. take the peak just after the start of each window,
4. keep peaks that clear the noise and amplitude thresholds and are not inside
   the RC-check (control) pulse.

Nothing in this module depends on the GUI.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.ndimage import median_filter

DEFAULT_FS = 10000.0
BANDPASS_HZ = (1.0, 3000.0)
BASELINE_WINDOW = 1000  # samples; median filter used for the noise estimate
NOISE_SD_FACTOR = 2.0
MIN_AMPLITUDE_PA = 8.0
MAX_IPSC_AMPLITUDE_PA = 60.0

# Used only when a file's header has no RC-check pulse description.
FALLBACK_PULSE_WINDOWS_MS = {
    "early": (200, 250),
    "early_WW_2": (100, 200),
    "early_WW": (200, 300),
    "late": (9800, 9900),
}

MODES = ("EPSC", "IPSC", "IPSC_high_Cl")
_MODE_LOOKUP = {m.upper(): m for m in MODES}

_RC_PATTERN = re.compile(r"state\.phys\.internal\.pulseString_RCCheck='([^']*)'")
_RATE_PATTERN = re.compile(r"state\.phys\.settings\.inputRate=([0-9.]+)")


def normalize_mode(mode) -> str:
    """Return the canonical spelling of a mode; unknown values become 'EPSC'."""
    return _MODE_LOOKUP.get(str(mode).upper(), "EPSC")


def is_inward(mode) -> bool:
    """True for modes whose events are negative-going (EPSC, high-Cl IPSC)."""
    return normalize_mode(mode) != "IPSC"


def natural_key(text: str):
    """Sort key so that AD0_2 comes before AD0_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def list_sweeps(cell_folder: str) -> list[str]:
    """All .mat files in a cell folder, in natural order."""
    names = [
        n for n in os.listdir(cell_folder)
        if n.lower().endswith(".mat") and not n.startswith(".")
    ]
    return [os.path.join(cell_folder, n) for n in sorted(names, key=natural_key)]


class Cell(NamedTuple):
    name: str      # the cell's folder name; also its key in parameter files
    path: str
    n_sweeps: int
    folder: str    # the session folder the cell is in


def _subfolders(folder: str) -> list[tuple]:
    with os.scandir(folder) as entries:
        found = [(e.name, e.path) for e in entries
                 if not e.name.startswith(".") and e.is_dir()]
    return sorted(found, key=lambda item: natural_key(item[0]))


def _count_sweeps(folder: str) -> int:
    try:
        with os.scandir(folder) as entries:
            return sum(1 for e in entries
                       if e.name.lower().endswith(".mat") and not e.name.startswith("."))
    except OSError:
        return 0


def scan_cells(session_folder: str) -> list[Cell]:
    """Every sub-folder of a session folder that holds .mat files."""
    folders = _subfolders(session_folder)
    # Listing hundreds of folders on a network drive is slow one at a time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(_count_sweeps, [path for _, path in folders]))
    return [Cell(name, path, n, session_folder)
            for (name, path), n in zip(folders, counts) if n]


def list_cells(session_folder: str) -> list[str]:
    """Sub-folders of `session_folder` that contain at least one .mat file."""
    return [cell.path for cell in scan_cells(session_folder)]


def find_sessions(folder: str) -> list[tuple]:
    """Session folders at or just below `folder`, as [(session_folder, cells)].

    If `folder` holds cell folders it is itself the session. Otherwise each of
    its sub-folders that holds cell folders is one, so a folder of sessions can
    be opened in one go.
    """
    folder = os.path.normpath(os.path.abspath(folder))
    cells = scan_cells(folder)
    if cells:
        return [(folder, cells)]
    found = []
    for _, path in _subfolders(folder):
        cells = scan_cells(path)
        if cells:
            found.append((path, cells))
    return found


def cell_name(session_folder: str, cell_folder: str, name_with_folder: bool) -> str:
    """The name a cell goes by: its own folder name, or, with `name_with_folder`,
    the session folder's name joined to it (folder 1 inside EC123 is EC123-1)."""
    own = os.path.basename(os.path.normpath(cell_folder))
    if not name_with_folder:
        return own
    return f"{os.path.basename(os.path.normpath(session_folder))}-{own}"


def merge_sessions(sessions, name_with_folder: bool = False) -> tuple:
    """Flatten [(session_folder, cells)] into one list of cells, named as
    cell_name() describes.

    Parameters, metadata and exports all identify a cell by its name, so a name
    can be used only once: the first session that has it keeps it.
    Returns (cells, skipped) where skipped is {session_folder: [names left out]}.
    """
    cells, seen, skipped = [], set(), {}
    for folder, session_cells in sessions:
        for cell in session_cells:
            name = cell_name(folder, cell.path, name_with_folder)
            if name in seen:
                skipped.setdefault(folder, []).append(name)
            else:
                seen.add(name)
                cells.append(cell._replace(name=name))
    return cells, skipped


# ---------------------------------------------------------------- file loading

def parse_pulse_params(pulse_string: str) -> dict:
    """Parse 'amplitude=5;duration=3000;...' into a dict."""
    params = {}
    for part in pulse_string.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            try:
                params[key] = float(value) if "." in value else int(value)
            except ValueError:
                params[key] = value
    return params


def rc_pulse_params(header: str) -> dict | None:
    match = _RC_PATTERN.search(header)
    return parse_pulse_params(match.group(1)) if match else None


def control_pulse_window(header: str, fs: float = DEFAULT_FS):
    """(start, end) sample indices of the RC-check pulse, or None if not in the header."""
    params = rc_pulse_params(header)
    if not params:
        return None
    delay_ms = params.get("delay")
    width_ms = params.get("pulseWidth")
    if not isinstance(delay_ms, (int, float)) or not isinstance(width_ms, (int, float)):
        return None
    return int(delay_ms * fs / 1000), int((delay_ms + width_ms) * fs / 1000)


def fallback_pulse_window(timing: str | None, fs: float = DEFAULT_FS):
    start_ms, end_ms = FALLBACK_PULSE_WINDOWS_MS.get(
        timing or "early", FALLBACK_PULSE_WINDOWS_MS["early"]
    )
    return int(start_ms * fs / 1000), int(end_ms * fs / 1000)


def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    high = min(highcut, 0.9 * nyquist)
    b, a = signal.butter(order, [lowcut / nyquist, high / nyquist], btype="band")
    return signal.filtfilt(b, a, data)


def read_mat(path: str):
    """Return (raw trace, header string) from a ScanImage wave .mat file."""
    mat = sio.loadmat(path)
    name = os.path.splitext(os.path.basename(path))[0]
    if name not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        if len(keys) != 1:
            raise ValueError(f"cannot find the recording inside {os.path.basename(path)}")
        name = keys[0]
    wave = mat[name]
    raw = np.asarray(wave["data"][0][0], dtype=float).ravel()
    try:
        header = str(wave["UserData"][0][0]["headerString"][0][0][0])
    except Exception:
        try:
            header = str(wave["UserData"][0][0][0][0][2])
        except Exception:
            header = ""
    return raw, header


@dataclass
class Sweep:
    """One loaded sweep plus everything about it that does not depend on the
    detection parameters, so parameters can be changed without re-reading it."""

    path: str
    raw: np.ndarray
    filtered: np.ndarray
    fs: float
    header: str
    control_window: tuple | None  # from the header; None if the header lacks it
    noise_median: float
    noise_std: float

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def duration_s(self) -> float:
        return len(self.raw) / self.fs


def load_sweep(path: str) -> Sweep:
    raw, header = read_mat(path)
    rate = _RATE_PATTERN.search(header)
    fs = float(rate.group(1)) if rate and float(rate.group(1)) > 0 else DEFAULT_FS
    filtered = bandpass_filter(raw, BANDPASS_HZ[0], BANDPASS_HZ[1], fs, order=2)
    noise = filtered - median_filter(filtered, size=BASELINE_WINDOW)
    return Sweep(
        path=path,
        raw=raw,
        filtered=filtered,
        fs=fs,
        header=header,
        control_window=control_pulse_window(header, fs),
        noise_median=float(np.median(noise)),
        noise_std=float(np.std(noise)),
    )


# ------------------------------------------------------------------- detection

def decay_windows(trace, window_size: int, sample_step: int, mode) -> list[int]:
    """Start indices of non-overlapping windows in which the trace, sampled every
    `sample_step` points, moves monotonically back toward baseline (falling for
    IPSCs, rising for EPSCs).

    Vectorised, but returns exactly what the notebook's sample-by-sample scan did:
    the scan advances one point at a time and jumps a full window after each hit.
    """
    window_size = int(window_size)
    sample_step = int(sample_step)
    n = len(trace)
    if window_size < 1 or sample_step < 1 or n < window_size:
        return []
    offsets = np.arange(0, window_size, sample_step)
    if len(offsets) < 2:
        return []
    n_starts = n - window_size + 1
    rising = is_inward(mode)
    ok = np.ones(n_starts, dtype=bool)
    for a, b in zip(offsets[:-1], offsets[1:]):
        step = trace[b:b + n_starts] - trace[a:a + n_starts]
        ok &= (step > 0) if rising else (step < 0)
    candidates = np.flatnonzero(ok)

    starts = []
    position = 0
    while True:
        j = int(np.searchsorted(candidates, position))
        if j >= len(candidates):
            break
        start = int(candidates[j])
        starts.append(start)
        position = start + window_size
    return starts


def peaks_near_decay(trace, starts, interval: int, mode) -> list[int]:
    """Index of the peak within `interval` samples after each decay-window start."""
    inward = is_inward(mode)
    n = len(trace)
    peaks = []
    for start in starts:
        segment = trace[start:min(start + int(interval), n)]
        offset = np.argmin(segment) if inward else np.argmax(segment)
        peaks.append(start + int(offset))
    return peaks


def filter_peaks(sweep: Sweep, peak_indices, mode, fallback_timing: str | None = "early"):
    """Keep peaks that clear the noise / amplitude thresholds and lie outside the
    control pulse. Returns (values, indices)."""
    window = sweep.control_window or fallback_pulse_window(fallback_timing, sweep.fs)
    lo, hi = window
    trace = sweep.filtered
    values, indices = [], []
    if is_inward(mode):
        threshold = sweep.noise_median - NOISE_SD_FACTOR * sweep.noise_std
        for idx in peak_indices:
            v = trace[idx]
            if v < threshold and v < -MIN_AMPLITUDE_PA and not (lo <= idx <= hi):
                values.append(float(v))
                indices.append(int(idx))
    else:
        threshold = sweep.noise_median + NOISE_SD_FACTOR * sweep.noise_std
        for idx in peak_indices:
            v = trace[idx]
            if (v > threshold and MIN_AMPLITUDE_PA < v < MAX_IPSC_AMPLITUDE_PA
                    and not (lo <= idx <= hi)):
                values.append(float(v))
                indices.append(int(idx))
    return values, indices


def detect_events(sweep: Sweep, window_size, sample_step, interval, mode,
                  fallback_timing: str | None = "early"):
    """Run the full detection on one loaded sweep. Returns (amplitudes, indices)."""
    mode = normalize_mode(mode)
    starts = decay_windows(sweep.filtered, window_size, sample_step, mode)
    peaks = peaks_near_decay(sweep.filtered, starts, interval, mode)
    return filter_peaks(sweep, peaks, mode, fallback_timing)


# ----------------------------------------------------------- passive properties

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_NAN4 = (np.nan, np.nan, np.nan, np.nan)


def compute_rs_rm_cm(raw, header: str, fs: float = DEFAULT_FS):
    """Series resistance, membrane resistance (MOhm), capacitance (pF) and tau (ms)
    from the RC-check pulse described in the file header. All NaN if unusable."""
    params = rc_pulse_params(header)
    if not params:
        return _NAN4
    delay_ms = params.get("delay")
    width_ms = params.get("pulseWidth")
    amplitude_mv = params.get("amplitude")
    if not all(isinstance(v, (int, float)) for v in (delay_ms, width_ms, amplitude_mv)):
        return _NAN4
    if np.isclose(amplitude_mv, 0):
        return _NAN4

    start = int(delay_ms * fs / 1000)
    end = int((delay_ms + width_ms) * fs / 1000)
    base_start = max(start - int(round(0.005 * fs)), 0)  # 5 ms pre-pulse baseline
    steady_start = start + min(int(round(0.015 * fs)), int(0.6 * (end - start)))
    if end > len(raw) or steady_start >= end:
        return _NAN4

    baseline_segment = raw[base_start:start]
    peak_segment = raw[base_start:steady_start]
    steady_segment = raw[steady_start:end]
    if not (len(baseline_segment) and len(peak_segment) and len(steady_segment)):
        return _NAN4

    base = float(np.mean(baseline_segment))
    peak = float(peak_segment[int(np.argmin(peak_segment))])
    steady = float(np.mean(steady_segment))
    d_peak = abs(base - peak)
    d_steady = abs(base - steady)
    if np.isclose(d_peak, 0) or np.isclose(d_steady, 0):
        return _NAN4

    rs = abs(amplitude_mv) / d_peak * 1000.0
    rm = abs(amplitude_mv) / d_steady * 1000.0 - rs

    dt_ms = 1000.0 / fs
    base_minus_steady = base - steady
    charge = float(_trapezoid(raw[start:steady_start] - base_minus_steady, dx=dt_ms))
    amplitude_pa = peak + base_minus_steady
    if np.isclose(amplitude_pa, 0) or np.isclose(rm, 0):
        return _NAN4
    tau_ms = charge / amplitude_pa
    cm = 1000.0 * tau_ms / rm
    return rs, rm, cm, tau_ms
