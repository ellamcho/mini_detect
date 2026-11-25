"""Synthetic ScanImage-style sweeps shared by the tests."""
import numpy as np
import pytest
import scipy.io as sio

FS = 10000
HEADER = (
    "state.phys.internal.pulseString_RCCheck='amplitude=-5;duration=1000;offset=0;"
    "numPulses=1;isi=0;pulseWidth=50;delay=200;ramp=0;patternRepeats=0;patternISI=0;'\r"
    "state.phys.settings.inputRate=10000\r"
)


def synthetic_trace(seed, n=60000, sign=-1.0):
    """Noise plus a handful of fast-rise / exponential-decay events."""
    rng = np.random.default_rng(seed)
    trace = rng.normal(0, 0.3, n)
    t = np.arange(400)
    for start in rng.choice(np.arange(4000, n - 1000), size=25, replace=False):
        amplitude = rng.uniform(12, 40)
        trace[start:start + 400] += sign * amplitude * np.exp(-t / 60.0)
    trace[2000:2500] -= 40.0  # RC-check pulse
    return trace


def write_sweep(path, trace):
    wave = {
        "data": trace.reshape(1, -1),
        "UserData": {"info": "", "name": path.stem, "headerString": np.array([HEADER])},
    }
    sio.savemat(str(path), {path.stem: wave})


@pytest.fixture
def data_folder(tmp_path):
    for cell, seeds in (("cellA", (1, 2)), ("cellB", (3,))):
        (tmp_path / cell).mkdir()
        for i, seed in enumerate(seeds):
            write_sweep(tmp_path / cell / f"AD0_{i + 1}.mat", synthetic_trace(seed))
    (tmp_path / "notes").mkdir()  # a folder without sweeps is ignored
    return tmp_path


@pytest.fixture
def session_folders(tmp_path):
    """Two session folders side by side. Both have a cell called cellC."""
    parent = tmp_path / "experiment"
    for session, cells in (("day1", {"cellC": 4}), ("day2", {"cellC": 5, "cellD": 6})):
        for cell, seed in cells.items():
            (parent / session / cell).mkdir(parents=True)
            write_sweep(parent / session / cell / "AD0_1.mat", synthetic_trace(seed))
    (parent / "figures").mkdir()  # not a session folder: ignored
    return parent
