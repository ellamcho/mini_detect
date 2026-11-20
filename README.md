# mini_detect

A desktop app for detecting miniature synaptic currents (mEPSCs / mIPSCs) in
ScanImage `.mat` sweeps. You tune the detection parameters for each recording by
eye, run detection on every recording, and export the results as CSV files.

It runs on macOS, Windows and Linux.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which
installs Python for you. Install uv once, then open a new terminal and run:

```
uv tool install git+https://github.com/ellamcho/mini_detect
```

If the computer does not have git, use this instead:

```
uv tool install https://github.com/ellamcho/mini_detect/archive/refs/heads/main.zip
```

Start the app from any terminal:

```
mini-detect
```

To update to the newest version later, run the install command again with
`--reinstall` added, for example
`uv tool install --reinstall git+https://github.com/ellamcho/mini_detect`.

<details>
<summary>Installing with pip or conda instead</summary>

In any Python 3.10+ environment:

```
pip install git+https://github.com/ellamcho/mini_detect
mini-detect
```

If `mini-detect` is not found, `python -m mini_detect` does the same thing.
</details>

## What you need

1. **A data folder with one sub-folder per recording.** Each sub-folder holds
   that recording's sweeps as `.mat` files. The sub-folder name is the cell name.

   ```
   my_experiment/
       EC101-1/   AD0_1.mat  AD0_2.mat  ...
       EC101-1b/  AD0_1.mat  AD0_2.mat  ...
   ```

2. **Optionally, a metadata CSV** that lists each cell's condition, so results
   are grouped by condition. Tuning never shows the condition, so detection
   stays blind. See [metadata_template.csv](metadata_template.csv):

   ```
   Cell,Condition
   EC101-1,Control
   EC101-1b,Drug
   ```

## How to use it

**1. Tune.** Choose the data folder, then click a cell. The app shows a few
random sweeps from that cell with the detected events marked in red. Change the
event type and the three parameters until the marks look right; the plot updates
as you type. Scroll to zoom and drag to pan.

| Parameter | What it does |
| --- | --- |
| Decay window | Length (in samples) of the stretch after an event that must decay steadily back toward baseline. Longer is stricter and suits slower events. |
| Sample step | The trace is checked every this many samples inside the window. Smaller is stricter; larger tolerates more noise. |
| Peak search | How far from the start of the decay window to look for the event's peak. |

Click **Save and go to next cell** to store the parameters and move on. A cell
can have one saved set per event type, for example one for EPSCs and one for
IPSCs. Cells without saved parameters are skipped later.

Your saved parameters are remembered if you close the app and reopen the same
data folder. Use **Save as…** under the cell list to keep them as a JSON file,
and **Load…** to bring a JSON file back in.

**2. Detect & export.** Choose an output folder, optionally a metadata file, and
click **Run detection**. When it finishes, the table shows each cell's event
count, frequency, mean amplitude, series resistance (Rs), membrane resistance
(Rm) and capacitance (Cm). Click **Export CSV files** to write:

| File | Contents |
| --- | --- |
| `<condition>_Amplitude_<tag>.csv`, `Amplitude_combined_<tag>.csv` | Mean event amplitude per cell |
| `<condition>_Frequency_<tag>.csv`, `Frequency_combined_<tag>.csv` | Event frequency (Hz) per cell |
| `Rs_Rm_Cm_all_cells_<tag>.csv`, `Rs_Rm_Cm_<condition>_<tag>.csv` | Rs, Rm and Cm per cell |
| `<condition>_IPI_by_cell_<tag>.csv` | Every inter-event interval (ms), one column per cell |
| `<condition>_ampls_by_cell_<tag>.csv` | Every event amplitude (pA), one column per cell |
| `<tag>_optimized_parameters.json` | The parameters used for each cell |
| `<tag>_optimized_results.pkl` | Raw detections, in the format the notebooks read |

With "Leave out cells with Rs above" ticked, cells whose mean series resistance
is above the limit, or could not be measured, are left out of every file.

## How detection works

Each sweep is band-pass filtered (1–3000 Hz). The app looks for stretches where
the trace decays steadily back toward baseline, takes the peak at the start of
each one, and keeps peaks that are at least 8 pA, larger than twice the noise
standard deviation, and outside the RC-check pulse. The RC-check pulse timing
and the sampling rate are read from each file's header.

## For developers

```
git clone https://github.com/ellamcho/mini_detect
cd mini_detect
pip install -e ".[test]"
pytest
```

The app lives in `src/mini_detect`: `detection.py` (file loading and the
detection algorithm), `analysis.py` (batch detection and CSV export) and `gui/`
(the windows). The notebooks and scripts at the top level of the repository are
the original analysis code and are not part of the installed app.
# mini_detect: Overview

This Python-based algorithm allows you to detect mini PSCs ("minis") with the help of graphical parameter fine-tuning in Jupyter notebook. This algorithm is particularly good at detection in recordings with heterogenuous taus and amplitudes, and has been tested with low and high Hz mini recordings. This code is optimized for data obtained using ScanImage software. 

**What you need:** 
1. Each recording consists of one folder with all of the sweeps (.mat files)
2. For blind detection, each recording's condition should be recorded on the metadata csv (see [template](https://github.com/ellamcho/mini_detect/blob/main/sample_metadata.csv))

**What you'll get:**
1. A JSON file of all the parameters you used for each cell for future reference
2. The avg mini amplitude per cell recording
3. The mini Hz per cell recording
4. csvs separated by condition and also combined with the information from 2 and 3

# How to Use
1. Make sure your data is in the correct format and fill out the [metadata.csv file](https://github.com/ellamcho/mini_detect/blob/main/sample_metadata.csv)
2. Download the Code folder to your local directory (make sure all files are in the same folder)
3. Open the main.ipynb file and follow the steps written
4. Analyze your data as needed (resutls compatible in Prism, more Python, R, etc.)
