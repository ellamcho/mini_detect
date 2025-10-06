### Custom basic functions for main.py ###

# Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
import os
import glob


# Function to extract acq data from .mat file 
def mat_to_arr(mat_file, name):
    mat = sio.loadmat(mat_file)
    arr = mat[name][0][0][0][0]
    return arr

# Bandpass filter (units in Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)
    return filtered

# Find monotonic decay windows
def decay_window(file, window_size=None, sample_step=None, mode='IPSC'):
    """
    If window_size or sample_step are None, they default by mode:
      IPSC -> (600, 100), EPSC -> (300, 50)
    If you pass values, they are respected regardless of mode.
    """
    if window_size is None or sample_step is None:
        if mode == 'EPSC':
            window_size = 150 if window_size is None else window_size
            sample_step = 25  if sample_step is None else sample_step
        else:  # IPSC (default)
            window_size = 600 if window_size is None else window_size
            sample_step = 100 if sample_step is None else sample_step

    monotonic_indices = []
    i = 0
    n = len(file)
    while i <= n - window_size:
        window = file[i:i+window_size]
        sampled_window = window[::sample_step]

        if len(sampled_window) > 1:
            if mode == 'IPSC' and np.all(np.diff(sampled_window) < 0):
                # Ensure no interior peak: max at start, min at end
                if np.argmax(sampled_window) == 0 and np.argmin(sampled_window) == len(sampled_window)-1:
                    monotonic_indices.append(i)
                    i += window_size
                    continue
            elif mode == 'EPSC' and np.all(np.diff(sampled_window) > 0):
                # Ensure no interior peak: min at start, max at end
                if np.argmin(sampled_window) == 0 and np.argmax(sampled_window) == len(sampled_window)-1:
                    monotonic_indices.append(i)
                    i += window_size
                    continue
        i += 1

    print(f"Found {len(monotonic_indices)} monotonic windows of size {window_size} "
          f"(non-overlapping, sampled every {sample_step} points, mode={mode})")
    return monotonic_indices

# Find peak near start of decay window
def find_peak_near_decay(signal, decay_starts, interval=None, mode='IPSC'):
    # Default intervals by mode
    if interval is None:
        if mode == 'EPSC':
            interval = 10
        else:  # IPSC or anything else
            interval = 50

    peak_indices = []
    peak_values = []

    for start_idx in decay_starts:
        end_idx = min(start_idx + interval, len(signal))
        segment = signal[start_idx:end_idx]

        if mode == 'IPSC':
            peak_in_segment = np.argmax(segment)
        elif mode == 'EPSC':
            peak_in_segment = np.argmin(segment)
        else:  # absolute peak
            peak_in_segment = np.argmax(np.abs(segment))

        peak_idx = start_idx + peak_in_segment
        peak_indices.append(peak_idx)
        peak_values.append(signal[peak_idx])

    print(f"Found {len(peak_indices)} peaks near decay starts "
          f"(interval={interval}, mode={mode})")

    return peak_indices, peak_values

    # Usage:
    # peaks, values = find_peak_near_decay(ad0_66_filt, ad0_66_mono, interval=50, mode='IPSC')

# Plotting function
def plotter(data_filt, specific_ind):
    if specific_ind != None:
        plt.figure(figsize=(24, 12))
        plt.plot(data_filt, linewidth=0.1)
        plt.scatter(specific_ind, data_filt[specific_ind], color='red', s=10, label=f'{specific_ind}')
        plt.tight_layout()
        plt.show()

    elif specific_ind == None:
        plt.figure(figsize=(24, 6))
        plt.plot(data_filt, linewidth=0.1)
        plt.tight_layout()
        plt.show()

# Median filter for baseline estimation
from scipy.ndimage import median_filter

def filter_peaks_by_noise(signal, peak_indices, baseline=None, mode='IPSC', fs=10000, exclude_control_pulse=True):
    if baseline is None:
        baseline = median_filter(signal, size=1000)
    noise = signal - baseline
    noise_median = np.median(noise)
    noise_std = np.std(noise)
    
    # Define control pulse exclusion window (200 to 250 ms at given sampling rate)
    if exclude_control_pulse:
        control_start = 2000  # 200 ms in sample points (2000 samples at 10kHz)
        control_end = 2500    # 250 ms in sample points (2500 samples at 10kHz)
        print(f"Excluding control pulse region: samples {control_start} to {control_end} ({control_start/fs*1000:.1f} to {control_end/fs*1000:.1f} ms)")
    
    if mode == 'IPSC':
        threshold = noise_median + 2 * noise_std
        if exclude_control_pulse:
            filtered_indices = [idx for idx in peak_indices if signal[idx] > threshold and signal[idx] < 300 
                              and not (control_start <= idx <= control_end)]
        else:
            filtered_indices = [idx for idx in peak_indices if signal[idx] > threshold and signal[idx] < 300]
    elif mode == 'EPSC':
        threshold = noise_median - 2 * noise_std
        if exclude_control_pulse:
            filtered_indices = [idx for idx in peak_indices if signal[idx] < threshold and signal[idx] > -300 
                              and not (control_start <= idx <= control_end)]
        else:
            filtered_indices = [idx for idx in peak_indices if signal[idx] < threshold and signal[idx] > -300]
    else:
        threshold = noise_median + 2 * noise_std
        if exclude_control_pulse:
            filtered_indices = [idx for idx in peak_indices if abs(signal[idx]) > abs(threshold) and signal[idx] < 300 
                              and not (control_start <= idx <= control_end)]
        else:
            filtered_indices = [idx for idx in peak_indices if abs(signal[idx]) > abs(threshold) and signal[idx] < 300]
    
    filtered_peaks = [signal[idx] for idx in filtered_indices]
    excluded_count = len(peak_indices) - len(filtered_indices)
    print(f"Kept {len(filtered_peaks)} peaks above threshold ({threshold:.2f}) for mode {mode}")
    if exclude_control_pulse and excluded_count > 0:
        print(f"Excluded {excluded_count} peaks in control pulse region")
    return filtered_peaks, filtered_indices

# mini detection for a single .mat file
def mini_detect_single(mat_file, mode): # Input is a single .mat file path
    file_name = os.path.basename(mat_file)
    name_without_ext = os.path.splitext(file_name)[0]
    mat_arr = mat_to_arr(mat_file, name_without_ext)
    filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
    mono_indices = decay_window(filtered_mat, 600, 100, mode=mode)
    mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, interval=50, mode=mode)
    filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], baseline=None, mode=mode, fs=10000)
    print(f"Detected {len(filtered_peaks[0])} events in {mat_file}")
    return filtered_peaks # output will be ([peak values], [peak indices])

# mini detection for all .mat files in a folder
def mini_detect(folder_path, mode):
    mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
    filtered_peaks_all = []
    for mat_file in mat_files:
        try:
            print(f"Processing {mat_file}")
            file_name = os.path.basename(mat_file)
            name_without_ext = os.path.splitext(file_name)[0]
            mat_arr = mat_to_arr(mat_file, name_without_ext)
            filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
            mono_indices = decay_window(filtered_mat, 150, 25, mode=mode)
            if not mono_indices:
                print(f"No monotonic windows found in {mat_file}, skipping.")
                continue
            mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, interval=10, mode=mode)
            if not mono_peaks[0]:
                print(f"No peaks found in {mat_file}, skipping.")
                continue
            filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], baseline=None, mode=mode, fs=10000)
            print(f"Detected {len(filtered_peaks[0])} events in {mat_file}")
            filtered_peaks_all.append(filtered_peaks)
        except Exception as e:
            print(f"Error processing {mat_file}: {e}")
            continue
    return mat_files, filtered_peaks_all

        # output:
        # filename[0] = 'file path'; filename [2] is the same for the next folder 
        # filename[1] = ([peak values], [peak indices]); filename[3] is the same for the next folder

# mini detection for all .mat files in all subfolders of a parent folder
def batch_mini_detect(parent_folder, mode):
    folder_paths = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    all_results = []
    for folder in folder_paths:
        print(f"Scanning folder: {folder}")
        all_results.append(mini_detect(folder, mode))
    return all_results

    #OUTPUT: 
    # no_TTX_EPSC[n] is cell #
    # no_TTX_EPSC[n][0] is .mat files for that cell
    # no_TTX_EPSC[n][1] is data for that cell (all .mat files)
    # no_TTX_EPSC[n][1][j] is data for a single .mat file 
    # no_TTX_EPSC[n][1][j][0] is peak amplitdues for a single .mat file , no_TTX_EPSC[n][1][j][1] is peak indices for a single .mat file

# Extract cell names from detected data
def get_cell_names(detected_data):
    cell_names = []
    for i in range(len(detected_data)):
        # Extract cell name from first file path
        file_path = detected_data[i][0][0]  # First file path for this cell
        # Split path and get the folder name (cell name)
        path_parts = file_path.split('/')
        cell_name = path_parts[-2]  # Second to last part is the cell folder name
        cell_names.append(cell_name)
    return cell_names

# get avg mini amp per cell
def avg_amp_per_cell(detected_data):
    cell_amp_means = []
    for i in range(len(detected_data)):  # go through each cell/folder
        mean_amp_per_mat = []
        for j in range(len(detected_data[i][1])):  # go through each .mat file
            peaks = detected_data[i][1][j][0]  # peak amplitudes for this .mat file
            if peaks is not None and len(peaks) > 0:
                mean_amp_per_mat.append(np.mean(peaks))
            else:
                mean_amp_per_mat.append(np.nan)  # or skip, but np.nan is safer for plotting
        # Only average non-nan values
        valid_means = [m for m in mean_amp_per_mat if not np.isnan(m)]
        if valid_means:
            cell_amp_means.append(np.mean(valid_means))
        else:
            cell_amp_means.append(np.nan)
    return cell_amp_means

# get hz per cell
def hz_per_cell(detected_data):
    cell_hz = []
    for i in range(len(detected_data)):
        fname = detected_data[i][0]    # filename is at the even index before this
        num_peaks = []
        for j in range(len(detected_data[i][1])):  # each .mat file in the cell
            num_peaks.append(len(detected_data[i][1][j][0]))
    
        total_peaks_per_cell = np.sum(num_peaks)

        if any(tag in fname for tag in ["EC35-3", "EC42-1", "EC46-5"]):
            scale = 3
        else:
            scale = 10

        cell_hz.append(total_peaks_per_cell / (scale * len(detected_data[i][1])))  # mean Hz for this cell
    return cell_hz

# Export data function
def export_data(condition_labels, big_data, data_type, date, output_dir, detected_data_list=None):
    # Export each condition as a separate CSV
    for i, (data, label) in enumerate(zip(big_data, condition_labels)):
        # Get actual cell names if detected_data_list is provided
        if detected_data_list is not None and i < len(detected_data_list):
            cell_names = get_cell_names(detected_data_list[i])
        else:
            # Fallback to generic cell IDs
            cell_names = [f"Cell_{j+1}" for j in range(len(data))]
        
        # Convert to DataFrame - use dynamic column name based on data_type
        column_name = f'{data_type}_Value'
        df = pd.DataFrame({
            'Cell_Name': cell_names,
            column_name: data
        })
        # Create filename with data_type included
        filename = f"{output_dir}/{label}_{data_type}_{date}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Exported {label}: {len(data)} values to {filename}")

    # Also create a combined CSV with all conditions
    combined_data = []
    for i, (data, label) in enumerate(zip(big_data, condition_labels)):
        # Get actual cell names if detected_data_list is provided
        if detected_data_list is not None and i < len(detected_data_list):
            cell_names = get_cell_names(detected_data_list[i])
        else:
            # Fallback to generic cell IDs
            cell_names = [f"Cell_{j+1}" for j in range(len(data))]
            
        for j, value in enumerate(data):
            combined_data.append({
                'Cell_Name': cell_names[j] if j < len(cell_names) else f"Cell_{j+1}",
                'Condition': label,
                f'{data_type}_Value': value
            })

    combined_df = pd.DataFrame(combined_data)
    combined_filename = f"{output_dir}{data_type}_combined_{date}.csv"
    combined_df.to_csv(combined_filename, index=False)
    print(f"Exported combined data: {len(combined_data)} rows to {combined_filename}")

    print("\nCSV export completed!")

def separate_data_by_groups(optimized_results, metadata, cell_id_col='Cell_ID', group_col='Group'):
    """
    Separate detection results into groups based on metadata
    
    Args:
        optimized_results: Results from tuner.run_full_detection()
        metadata: DataFrame with cell IDs and group assignments
        cell_id_col: Column name for cell IDs in metadata
        group_col: Column name for group assignments in metadata
    
    Returns:
        dict: Results separated by group
    """
    # Get cell names from optimized results
    cell_names = get_cell_names(optimized_results)
    
    # Create a mapping from metadata
    metadata_dict = dict(zip(metadata[cell_id_col], metadata[group_col]))
    
    # Initialize group containers
    grouped_results = {}
    ungrouped_cells = []
    
    print(f"Found {len(cell_names)} cells in detection results")
    print(f"Found {len(metadata)} entries in metadata")
    
    # Group the results
    for i, cell_name in enumerate(cell_names):
        if cell_name in metadata_dict:
            group = metadata_dict[cell_name]
            if group not in grouped_results:
                grouped_results[group] = []
            grouped_results[group].append(optimized_results[i])
            print(f"  {cell_name} -> {group}")
        else:
            ungrouped_cells.append(cell_name)
            print(f"  {cell_name} -> NOT FOUND in metadata")
    
    if ungrouped_cells:
        print(f"\nWarning: {len(ungrouped_cells)} cells not found in metadata:")
        for cell in ungrouped_cells:
            print(f"  - {cell}")
    
    print(f"\nGrouping summary:")
    for group, results in grouped_results.items():
        print(f"  {group}: {len(results)} cells")
    
    return grouped_results, ungrouped_cells