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

1. **One or more session folders.** A session folder has one sub-folder per
   cell, and each of those holds that cell's sweeps as `.mat` files.

   ```
   my_experiment/
       EC123/
           1/    AD0_1.mat  AD0_2.mat  ...
           1b/   AD0_1.mat  AD0_2.mat  ...
       EC124/
           1/    AD0_1.mat  AD0_2.mat  ...
   ```

   You can open several session folders at once.

   **Cell names.** A cell is named after its session folder and its own folder,
   joined with a hyphen: the cells above are `EC123-1`, `EC123-1b` and `EC124-1`.
   That name is what appears in parameter files, the metadata file and the
   exported files. If your cell folders already carry the full name (a folder
   called `EC123-1`), untick **Name cells folder-cell** at the top of the window
   and each cell is named by its own folder alone. When you load a parameter
   file whose names only fit the other setting, the app switches the box for
   you and says so at the bottom of the window. Each name can be used only
   once, so with the box unticked, if two open folders have a cell folder with
   the same name, the folder added first keeps it and the app shows how many
   were left out.

2. **Optionally, a metadata CSV** that lists each cell's condition, so results
   are grouped by condition. Tuning never shows the condition, so detection
   stays blind. See [metadata_template.csv](metadata_template.csv):

   ```
   Cell,Condition
   EC123-1,Control
   EC123-1b,Drug
   ```

## How to use it

**1. Add your folders.** Click **Add folder…** and choose a session folder. To
open several at once, choose the folder that contains them (`my_experiment`
above), or drag the session folders onto the window. Click **Add folder…** again
to add folders that live somewhere else. Each session folder appears in the list
on the left with its cells below it. **Remove folder** takes the selected one
off the list without deleting anything.

**2. Tune.** Click a cell. The app shows a few
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

The open folders and your saved parameters are remembered when you close the app
and come back the next time you start it. Use **Save as…** under the cell list
to write the parameters of every cell, from all session folders, to one JSON
file, and **Load…** to bring a JSON file back in.

**3. Detect & export.** Detection runs on the cells of every open folder. Choose an output folder, optionally a metadata file, and
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
| `<tag>_detection_results.json` | Everything from the run, for **Load results…** |
| `<tag>_optimized_results.pkl` | Raw detections, in the format the notebooks read |

With "Leave out cells with Rs above" ticked, cells whose mean series resistance
is above the limit, or could not be measured, are left out of every file.

**Coming back to earlier results.** Click **Load results…** and choose the
`_detection_results.json` file from an earlier export (or the `.pkl` saved next
to it). The table fills in straight away, and you can plot or export again
without re-running detection or opening the data folders.

A `.pkl` from an older version of the app, or from the notebook, has no
`_detection_results.json` beside it and holds only the detected events. It loads
too: the events are kept as they are, and each recording is read once for its
length, Rs, Rm and Cm, so the recordings must be where they were (about a minute
for 1500 sweeps on a network drive). The app then saves the
`_detection_results.json` next to the `.pkl`, and from then on it loads straight
away. Each cell's event type and settings come from the
`_optimized_parameters.json` saved with the `.pkl`, or else from the parameters
saved in the app.

**4. Plot the results.** After detection (or after loading results), click **Plot results** to see every
measurement (frequency, amplitude, Rs, Rm and Cm) with one dot per cell and a
black bar at the mean, conditions side by side. Choose how to split the cells:

- **By condition** puts the cells from all days together, one figure per event
  type.
- **By day** makes one figure per session folder, comparing the conditions
  recorded that day. Every figure uses the same axis ranges so days can be
  compared.

- **Before and after** is for cells recorded twice. A cell whose name is another
  cell's name plus an underscore and a label is treated as the second recording
  of that cell, and the label as the treatment: `20261117-2` and
  `20261117-2_picro` are one cell before and after picro. Each treatment gets a
  figure with a line joining each cell's two recordings and the result of a
  two-sided paired t test above (`ns` for p ≥ 0.05, `*` < 0.05, `**` < 0.01,
  `***` < 0.001, `****` < 0.0001). The exact p values are shown above the
  figure, and saving also writes them to `<tag>_paired_t_tests.csv`.

The plots use the metadata file and the Rs limit chosen on the tab; without a
metadata file all cells form one group. A pair is only used when both of its
recordings pass the Rs limit. **Save all as PDF and PNG** writes every
figure in the list to a `plots` folder inside the output folder. Text in the PDF
stays editable in Illustrator.

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
