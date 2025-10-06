# mini_detect: Overview

This Python-based algorithm allows you to detect mini PSCs ("minis") with the help of graphical parameter fine-tuning in Jupyter notebook. This algorithm is particularly good at detection in recordings with heterogenuous taus and amplitudes, and has been tested with low and high Hz mini recordings. 

**What you need:** 
1. Each recording consists of one folder with all of the sweeps (.mat files)
2. For blind detection, each recording's condition should be recorded on the metadata csv (see [template](https://github.com/ellamcho/mini_detect/blob/main/sample_metadata.csv))

**What you'll get:**
1. A JSON file of all the parameters you used for each cell for future reference
2. The avg mini amplitude per cell recording
3. The mini Hz per cell recording
4. csvs separated by condition and also combined with the information from 2 and 3

Below are some examples of what you'll see when performing parameter tuning.
 
**Example: Poor Detection with "Reasonable" Parameters**
![Poor Detection with "Reasonable" Parameters](https://github.com/ellamcho/mini_detect/blob/main/Images/poor_detection.png)

**The same trace with refined parameters:**
![Better Detection with Refined Parameters](https://github.com/ellamcho/mini_detect/blob/main/Images/better_detection.png)

**Live summary table comparing parameters:**
![Summary Table for Ease of Comparison](https://github.com/ellamcho/mini_detect/blob/main/Images/summary_table.png)

# How to Use
1. Make sure your data is in the correct format and fill out the [metadata.csv file](https://github.com/ellamcho/mini_detect/blob/main/sample_metadata.csv)
2. Download the [Code folder](https://github.com/ellamcho/mini_detect/tree/main/Code) to your local directory (make sure all files are in the same folder)
3. Open the [main.ipynb file](https://github.com/ellamcho/mini_detect/blob/main/Code/main.ipynb) and follow the steps written
4. Analyze your data as needed (resutls compatible in Prism, more Python, R, etc.)
