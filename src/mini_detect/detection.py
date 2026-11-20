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
from dataclasses import dataclass

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


def scan_cells(parent_folder: str) -> list[tuple]:
    """(name, path, number of sweeps) for every sub-folder holding .mat files."""
    cells = []
    for name in sorted(os.listdir(parent_folder), key=natural_key):
        path = os.path.join(parent_folder, name)
        if name.startswith(".") or not os.path.isdir(path):
            continue
        n_sweeps = len(list_sweeps(path))
        if n_sweeps:
            cells.append((name, path, n_sweeps))
    return cells


def list_cells(parent_folder: str) -> list[str]:
    """Sub-folders of `parent_folder` that contain at least one .mat file."""
    return [path for _, path, _ in scan_cells(parent_folder)]


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
