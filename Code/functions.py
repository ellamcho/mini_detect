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

def filter_peaks_by_noise(signal, peak_indices, baseline=None, mode='IPSC', fs=10000, exclude_control_pulse=True, control_pulse_timing='early'):
    if baseline is None:
        baseline = median_filter(signal, size=1000)
    noise = signal - baseline
    noise_median = np.median(noise)
    noise_std = np.std(noise)
    
    # Define control pulse exclusion window based on timing
    if exclude_control_pulse:
        if control_pulse_timing == 'early':
            control_start = 2000  # 200 ms in sample points (2000 samples at 10kHz)
            control_end = 2500    # 250 ms in sample points (2500 samples at 10kHz)
        
        elif control_pulse_timing == 'early_WW':
            control_start = 2000  # 9.8 s in sample points (98000 samples at 10kHz)
            control_end = 3000    # 9.9 s in sample points (99000 samples at 10kHz)
        
        elif control_pulse_timing == 'late':
            control_start = 98000  # 9.8 s in sample points (98000 samples at 10kHz)
            control_end = 99000    # 9.9 s in sample points (99000 samples at 10kHz)
        else:
            raise ValueError("control_pulse_timing must be 'early' or 'late'")
        
        print(f"Excluding {control_pulse_timing} control pulse region: samples {control_start} to {control_end} ({control_start/fs*1000:.1f} to {control_end/fs*1000:.1f} ms)")
    
    if mode == 'IPSC':
        threshold = noise_median + 2 * noise_std
        if exclude_control_pulse:
            filtered_indices = [idx for idx in peak_indices if signal[idx] > threshold and signal[idx] < 100 and signal[idx] > 8
                              and not (control_start <= idx <= control_end)]
        else:
            filtered_indices = [idx for idx in peak_indices if signal[idx] > threshold and signal[idx] < 100 and signal[idx] > 8]
    elif mode == 'EPSC':
        threshold = noise_median - 2 * noise_std
        if exclude_control_pulse:
            filtered_indices = [idx for idx in peak_indices if signal[idx] < threshold and signal[idx] > -100 and signal[idx] < -8
                              and not (control_start <= idx <= control_end)]
        else:
            filtered_indices = [idx for idx in peak_indices if signal[idx] < threshold and signal[idx] > -100 and signal[idx] < -8]
    else:
        threshold = noise_median + 2 * noise_std
        if exclude_control_pulse:
            filtered_indices = [idx for idx in peak_indices if abs(signal[idx]) > abs(threshold) and signal[idx] < 100 and signal[idx] > 8
                              and not (control_start <= idx <= control_end)]
        else:
            filtered_indices = [idx for idx in peak_indices if abs(signal[idx]) > abs(threshold) and signal[idx] < 100 and signal[idx] > 8]
    
    filtered_peaks = [signal[idx] for idx in filtered_indices]
    excluded_count = len(peak_indices) - len(filtered_indices)
    print(f"Kept {len(filtered_peaks)} peaks above threshold ({threshold:.2f}) for mode {mode}")
    if exclude_control_pulse and excluded_count > 0:
        print(f"Excluded {excluded_count} peaks in control pulse region")
    return filtered_peaks, filtered_indices

# mini detection for a single .mat file
def mini_detect_single(mat_file, mode, control_pulse_timing='early'): # Input is a single .mat file path
    file_name = os.path.basename(mat_file)
    name_without_ext = os.path.splitext(file_name)[0]
    mat_arr = mat_to_arr(mat_file, name_without_ext)
    filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
    mono_indices = decay_window(filtered_mat, 600, 100, mode=mode)
    mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, interval=50, mode=mode)
    filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], baseline=None, mode=mode, fs=10000, control_pulse_timing=control_pulse_timing)
    print(f"Detected {len(filtered_peaks[0])} events in {mat_file}")
    return filtered_peaks # output will be ([peak values], [peak indices])

# mini detection for all .mat files in a folder
def mini_detect(folder_path, mode, control_pulse_timing='early'):
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
            filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], baseline=None, mode=mode, fs=10000, control_pulse_timing=control_pulse_timing)
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
def batch_mini_detect(parent_folder, mode, control_pulse_timing='early'):
    folder_paths = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    all_results = []
    for folder in folder_paths:
        print(f"Scanning folder: {folder}")
        all_results.append(mini_detect(folder, mode, control_pulse_timing))
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

# Calculate inter-peak intervals
def calculate_inter_peak_intervals(peak_indices, fs=10000, min_interval_ms=None):
    """
    Calculate inter-peak intervals from detected peak indices.
    
    Args:
        peak_indices: List of peak indices (sample points)
        fs: Sampling frequency in Hz (default: 10000)
        min_interval_ms: Minimum interval in milliseconds to include (optional filtering)
    
    Returns:
        tuple: (intervals_ms, intervals_samples)
            - intervals_ms: Inter-peak intervals in milliseconds
            - intervals_samples: Inter-peak intervals in sample points
    """
    if len(peak_indices) < 2:
        print("Warning: Need at least 2 peaks to calculate intervals")
        return [], []
    
    # Sort peak indices to ensure chronological order
    sorted_peaks = np.sort(peak_indices)
    
    # Calculate intervals in sample points
    intervals_samples = np.diff(sorted_peaks)
    
    # Convert to milliseconds
    intervals_ms = intervals_samples * 1000.0 / fs
    
    # Apply minimum interval filter if specified
    if min_interval_ms is not None:
        valid_mask = intervals_ms >= min_interval_ms
        intervals_ms = intervals_ms[valid_mask]
        intervals_samples = intervals_samples[valid_mask]
        print(f"Filtered intervals: kept {len(intervals_ms)}/{len(np.diff(sorted_peaks))} intervals >= {min_interval_ms} ms")
    
    print(f"Calculated {len(intervals_ms)} inter-peak intervals")
    if len(intervals_ms) > 0:
        print(f"  Mean interval: {np.mean(intervals_ms):.2f} ms")
        print(f"  Median interval: {np.median(intervals_ms):.2f} ms")
        print(f"  Range: {np.min(intervals_ms):.2f} - {np.max(intervals_ms):.2f} ms")
    
    return intervals_ms.tolist(), intervals_samples.tolist()

# Calculate inter-peak intervals for a single detection result
def ipi_single_file(detected_peaks, fs=10000, min_interval_ms=None):
    """
    Calculate inter-peak intervals for a single file's detection results.
    
    Args:
        detected_peaks: Tuple of ([peak_values], [peak_indices]) from mini detection
        fs: Sampling frequency in Hz (default: 10000)
        min_interval_ms: Minimum interval in milliseconds to include
    
    Returns:
        tuple: (intervals_ms, intervals_samples)
    """
    peak_values, peak_indices = detected_peaks
    return calculate_inter_peak_intervals(peak_indices, fs, min_interval_ms)

# Calculate inter-peak intervals for all files in a cell
def ipi_per_cell(detected_data, fs=10000, min_interval_ms=None, concatenate=True):
    """
    Calculate inter-peak intervals for all detection results from a single cell.
    
    Args:
        detected_data: Cell detection data in format [mat_files, detection_results]
        fs: Sampling frequency in Hz (default: 10000)
        min_interval_ms: Minimum interval in milliseconds to include
        concatenate: If True, combine intervals from all files; if False, return per-file
    
    Returns:
        If concatenate=True: tuple of (all_intervals_ms, all_intervals_samples)
        If concatenate=False: list of tuples, one per file
    """
    mat_files, detection_results = detected_data
    
    if concatenate:
        all_intervals_ms = []
        all_intervals_samples = []
        
        for i, detected_peaks in enumerate(detection_results):
            intervals_ms, intervals_samples = ipi_single_file(detected_peaks, fs, min_interval_ms)
            all_intervals_ms.extend(intervals_ms)
            all_intervals_samples.extend(intervals_samples)
            print(f"  File {i+1}/{len(detection_results)}: {len(intervals_ms)} intervals")
        
        print(f"Total intervals for cell: {len(all_intervals_ms)}")
        return all_intervals_ms, all_intervals_samples
    
    else:
        file_intervals = []
        for i, detected_peaks in enumerate(detection_results):
            intervals_ms, intervals_samples = ipi_single_file(detected_peaks, fs, min_interval_ms)
            file_intervals.append((intervals_ms, intervals_samples))
            print(f"  File {i+1}/{len(detection_results)}: {len(intervals_ms)} intervals")
        
        return file_intervals

# Export raw IPI data for each cell as separate CSV files
def export_ipi_per_cell(detected_data_list, output_dir, date, metadata=None, cell_id_col='Cell_ID', group_col='Group', fs=10000, min_interval_ms=None, condition_label=""):
    """
    Export raw inter-peak intervals for each cell as separate CSV files.
    
    Args:
        detected_data_list: List of detection results from batch_mini_detect()
        output_dir: Directory to save CSV files
        date: Date string for filename
        metadata: DataFrame with cell IDs and group assignments (optional)
        cell_id_col: Column name for cell IDs in metadata (default: 'Cell_ID')
        group_col: Column name for group assignments in metadata (default: 'Group')
        fs: Sampling frequency in Hz (default: 10000)
        min_interval_ms: Minimum interval in milliseconds to include
        condition_label: Fallback label if no metadata provided
    
    Returns:
        dict: Summary of exported data per cell
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cell names
    cell_names = get_cell_names(detected_data_list)
    export_summary = {}
    
    # Create metadata mapping if provided
    metadata_dict = {}
    if metadata is not None:
        metadata_dict = dict(zip(metadata[cell_id_col], metadata[group_col]))
        print(f"Using metadata with {len(metadata_dict)} cell-condition mappings")
    
    for i, (cell_data, cell_name) in enumerate(zip(detected_data_list, cell_names)):
        print(f"Processing cell {i+1}/{len(detected_data_list)}: {cell_name}")
        
        # Get condition from metadata or use fallback
        if cell_name in metadata_dict:
            cell_condition = metadata_dict[cell_name]
            print(f"  Condition from metadata: {cell_condition}")
        else:
            cell_condition = condition_label if condition_label else 'Unknown'
            if metadata is not None:
                print(f"  Warning: {cell_name} not found in metadata, using: {cell_condition}")
        
        # Get all intervals for this cell
        intervals_ms, intervals_samples = ipi_per_cell(cell_data, fs, min_interval_ms, concatenate=True)
        
        if len(intervals_ms) > 0:
            # Create DataFrame with raw IPI data
            df = pd.DataFrame({
                'IPI_ms': intervals_ms,
                'IPI_samples': intervals_samples,
                'Cell_Name': cell_name,
                'Condition': cell_condition
            })
            
            # Create filename
            condition_suffix = f"_{cell_condition}" if cell_condition != 'Unknown' else ""
            filename = f"{output_dir}/{cell_name}_IPI{condition_suffix}_{date}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            # Store summary
            export_summary[cell_name] = {
                'filename': filename,
                'condition': cell_condition,
                'n_intervals': len(intervals_ms),
                'mean_ipi_ms': np.mean(intervals_ms),
                'median_ipi_ms': np.median(intervals_ms),
                'min_ipi_ms': np.min(intervals_ms),
                'max_ipi_ms': np.max(intervals_ms)
            }
            
            print(f"  Exported {len(intervals_ms)} intervals to {filename}")
            print(f"  Mean IPI: {np.mean(intervals_ms):.2f} ms")
        
        else:
            print(f"  No valid intervals found for {cell_name}")
            export_summary[cell_name] = {
                'filename': None,
                'condition': cell_condition,
                'n_intervals': 0,
                'mean_ipi_ms': np.nan,
                'median_ipi_ms': np.nan,
                'min_ipi_ms': np.nan,
                'max_ipi_ms': np.nan
            }
    
    print(f"\nIPI export completed! Files saved to {output_dir}")
    return export_summary

# Export raw IPI data with file-level detail
def export_ipi_detailed(detected_data_list, output_dir, date, metadata=None, cell_id_col='Cell_ID', group_col='Group', fs=10000, min_interval_ms=None, condition_label=""):
    """
    Export raw inter-peak intervals with file-level detail for each cell.
    
    Args:
        detected_data_list: List of detection results from batch_mini_detect()
        output_dir: Directory to save CSV files
        date: Date string for filename
        metadata: DataFrame with cell IDs and group assignments (optional)
        cell_id_col: Column name for cell IDs in metadata (default: 'Cell_ID')
        group_col: Column name for group assignments in metadata (default: 'Group')
        fs: Sampling frequency in Hz (default: 10000)
        min_interval_ms: Minimum interval in milliseconds to include
        condition_label: Fallback label if no metadata provided
    
    Returns:
        dict: Summary of exported data per cell
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cell names
    cell_names = get_cell_names(detected_data_list)
    export_summary = {}
    
    # Create metadata mapping if provided
    metadata_dict = {}
    if metadata is not None:
        metadata_dict = dict(zip(metadata[cell_id_col], metadata[group_col]))
        print(f"Using metadata with {len(metadata_dict)} cell-condition mappings")
    
    for i, (cell_data, cell_name) in enumerate(zip(detected_data_list, cell_names)):
        print(f"Processing cell {i+1}/{len(detected_data_list)}: {cell_name}")
        
        # Get condition from metadata or use fallback
        if cell_name in metadata_dict:
            cell_condition = metadata_dict[cell_name]
            print(f"  Condition from metadata: {cell_condition}")
        else:
            cell_condition = condition_label if condition_label else 'Unknown'
            if metadata is not None:
                print(f"  Warning: {cell_name} not found in metadata, using: {cell_condition}")
        
        mat_files, detection_results = cell_data
        all_intervals = []
        
        # Process each file separately to track source
        for j, detected_peaks in enumerate(detection_results):
            intervals_ms, intervals_samples = ipi_single_file(detected_peaks, fs, min_interval_ms)
            
            # Get filename for this recording
            file_path = mat_files[j]
            file_name = os.path.basename(file_path).replace('.mat', '')
            
            # Add intervals with file information
            for ipi_ms, ipi_samples in zip(intervals_ms, intervals_samples):
                all_intervals.append({
                    'IPI_ms': ipi_ms,
                    'IPI_samples': ipi_samples,
                    'Cell_Name': cell_name,
                    'File_Name': file_name,
                    'File_Index': j,
                    'Condition': cell_condition
                })
        
        if len(all_intervals) > 0:
            # Create DataFrame
            df = pd.DataFrame(all_intervals)
            
            # Create filename
            condition_suffix = f"_{cell_condition}" if cell_condition != 'Unknown' else ""
            filename = f"{output_dir}/{cell_name}_IPI_detailed{condition_suffix}_{date}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            # Store summary
            ipi_values = df['IPI_ms'].values
            export_summary[cell_name] = {
                'filename': filename,
                'condition': cell_condition,
                'n_intervals': len(ipi_values),
                'n_files': len(detection_results),
                'mean_ipi_ms': np.mean(ipi_values),
                'median_ipi_ms': np.median(ipi_values),
                'min_ipi_ms': np.min(ipi_values),
                'max_ipi_ms': np.max(ipi_values)
            }
            
            print(f"  Exported {len(all_intervals)} intervals from {len(detection_results)} files to {filename}")
            print(f"  Mean IPI: {np.mean(ipi_values):.2f} ms")
        
        else:
            print(f"  No valid intervals found for {cell_name}")
            export_summary[cell_name] = {
                'filename': None,
                'condition': cell_condition,
                'n_intervals': 0,
                'n_files': len(detection_results),
                'mean_ipi_ms': np.nan,
                'median_ipi_ms': np.nan,
                'min_ipi_ms': np.nan,
                'max_ipi_ms': np.nan
            }
    
    print(f"\nDetailed IPI export completed! Files saved to {output_dir}")
    return export_summary

# Export IPI data with one CSV per condition, cells as columns
def export_ipi_by_condition(detected_data_list, output_dir, date, metadata=None, cell_id_col='Cell_ID', group_col='Group', fs=10000, min_interval_ms=None, pad_method='nan'):
    """
    Export raw inter-peak intervals with one CSV per condition, where each column is a cell.
    
    Args:
        detected_data_list: List of detection results from batch_mini_detect()
        output_dir: Directory to save CSV files
        date: Date string for filename
        metadata: DataFrame with cell IDs and group assignments (optional)
        cell_id_col: Column name for cell IDs in metadata (default: 'Cell_ID')
        group_col: Column name for group assignments in metadata (default: 'Group')
        fs: Sampling frequency in Hz (default: 10000)
        min_interval_ms: Minimum interval in milliseconds to include
        pad_method: How to handle unequal lengths ('nan', 'drop', or 'repeat_last')
    
    Returns:
        dict: Summary of exported files by condition
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cell names
    cell_names = get_cell_names(detected_data_list)
    
    # Create metadata mapping if provided
    metadata_dict = {}
    if metadata is not None:
        metadata_dict = dict(zip(metadata[cell_id_col], metadata[group_col]))
        print(f"Using metadata with {len(metadata_dict)} cell-condition mappings")
    
    # Organize data by condition
    condition_data = {}
    
    for i, (cell_data, cell_name) in enumerate(zip(detected_data_list, cell_names)):
        print(f"Processing cell {i+1}/{len(detected_data_list)}: {cell_name}")
        
        # Get condition from metadata or use fallback
        if cell_name in metadata_dict:
            cell_condition = metadata_dict[cell_name]
        else:
            cell_condition = 'Unknown'
            if metadata is not None:
                print(f"  Warning: {cell_name} not found in metadata")
        
        # Get all intervals for this cell
        intervals_ms, intervals_samples = ipi_per_cell(cell_data, fs, min_interval_ms, concatenate=True)
        
        # Add to condition dictionary
        if cell_condition not in condition_data:
            condition_data[cell_condition] = {}
        
        condition_data[cell_condition][cell_name] = intervals_ms
        print(f"  {cell_name} -> {cell_condition}: {len(intervals_ms)} intervals")
    
    # Export one CSV per condition
    export_summary = {}
    
    for condition, cells_data in condition_data.items():
        print(f"\nExporting condition: {condition}")
        
        if not cells_data:
            print(f"  No data for condition {condition}, skipping")
            continue
        
        # Find the maximum number of intervals across all cells in this condition
        max_intervals = max(len(intervals) for intervals in cells_data.values())
        print(f"  Max intervals in condition: {max_intervals}")
        
        # Create DataFrame with cells as columns
        condition_df_data = {}
        
        for cell_name, intervals in cells_data.items():
            if len(intervals) == 0:
                # Handle cells with no intervals
                condition_df_data[cell_name] = [np.nan] * max_intervals
            elif len(intervals) < max_intervals:
                # Handle unequal lengths
                if pad_method == 'nan':
                    # Pad with NaN
                    padded_intervals = intervals + [np.nan] * (max_intervals - len(intervals))
                elif pad_method == 'drop':
                    # Use only the length of the shortest cell (handled later)
                    padded_intervals = intervals
                elif pad_method == 'repeat_last':
                    # Repeat the last value
                    if len(intervals) > 0:
                        padded_intervals = intervals + [intervals[-1]] * (max_intervals - len(intervals))
                    else:
                        padded_intervals = [np.nan] * max_intervals
                else:
                    # Default to NaN padding
                    padded_intervals = intervals + [np.nan] * (max_intervals - len(intervals))
                
                condition_df_data[cell_name] = padded_intervals
            else:
                condition_df_data[cell_name] = intervals
        
        # Handle 'drop' method by finding minimum length
        if pad_method == 'drop':
            min_intervals = min(len(intervals) for intervals in cells_data.values() if len(intervals) > 0)
            if min_intervals > 0:
                condition_df_data = {cell: intervals[:min_intervals] 
                                   for cell, intervals in condition_df_data.items()}
                print(f"  Truncated all cells to {min_intervals} intervals")
        
        # Create DataFrame
        df = pd.DataFrame(condition_df_data)
        
        # Add an index column for interval number
        df.index.name = 'Interval_Number'
        df.reset_index(inplace=True)
        df['Interval_Number'] = df['Interval_Number'] + 1  # Start from 1 instead of 0
        
        # Create filename
        filename = f"{output_dir}/{condition}_IPI_by_cell_{date}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Calculate summary statistics
        valid_data = df.drop('Interval_Number', axis=1).values.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]
        
        export_summary[condition] = {
            'filename': filename,
            'n_cells': len(cells_data),
            'n_total_intervals': len(valid_data),
            'mean_ipi_ms': np.mean(valid_data) if len(valid_data) > 0 else np.nan,
            'median_ipi_ms': np.median(valid_data) if len(valid_data) > 0 else np.nan,
            'cells': list(cells_data.keys())
        }
        
        print(f"  Exported {len(cells_data)} cells to {filename}")
        print(f"  Total intervals: {len(valid_data)}")
        if len(valid_data) > 0:
            print(f"  Mean IPI: {np.mean(valid_data):.2f} ms")
    
    print(f"\nCondition-based IPI export completed! Files saved to {output_dir}")
    return export_summary

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

##### Rs, Rm, and Cm calculation functions #####

def compute_Rs_Rm_Cm(data, control_pulse_timing, fs=10000):
    """
    Calculate series resistance (Rs), membrane resistance (Rm), and membrane capacitance (Cm)
    matching the compute_Rs_Rm_Cm_OG method from the notebook.

    Args:
        data: Raw electrophysiology data array
        control_pulse_timing: 'early', 'early_WW', or 'late'
        fs: Sampling frequency in Hz (default: 10000)

    Returns:
        tuple: (Rs, Rm, Cm, tau_ms) - Rs and Rm in MΩ, Cm in pF, tau in ms
    """
    if control_pulse_timing == 'early':
        control_start = 2000  # 200 ms at 10 kHz
        control_end = 2500    # 250 ms at 10 kHz
    elif control_pulse_timing == 'early_WW':
        control_start = 2000  # 200 ms
        control_end = 3000    # 300 ms
    elif control_pulse_timing == 'late':
        control_start = 98000  # 9.8 s
        control_end = 99000    # 9.9 s
    else:
        raise ValueError("control_pulse_timing must be 'early', 'early_WW', or 'late'")

    # Include ~5 ms before step to get baseline
    control_start = control_start - 50

    # Peak and baseline are taken within the step window (EPSC assumed: negative peak)
    peak_segment = data[control_start:control_end]
    if peak_segment.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    peak_idx = int(np.argmin(peak_segment))
    base_idx = int(np.argmax(peak_segment))
    peak_value = peak_segment[peak_idx]
    base_value = peak_segment[base_idx]

    print(f"Baseline current: {base_value:.2f} pA")
    print(f"Peak current: {peak_value:.2f} pA")

    # Rs (MΩ) using a fixed 5 mV step
    delta_I_inst = (base_value - peak_value)
    if np.isclose(delta_I_inst, 0):
        Rs = np.nan
    else:
        Rs = (5.0 / delta_I_inst) * 1000.0

    # Steady state over the last ~15 ms of the step
    steady_state_start = control_end - 150
    steady_segment = data[steady_state_start:control_end]
    steady_value = np.mean(steady_segment) if steady_segment.size > 0 else np.nan
    print(f"Steady state current: {steady_value:.2f} pA")

    if np.isnan(steady_value) or np.isclose((base_value - steady_value), 0):
        steady_R = np.nan
    else:
        steady_R = (5.0 / (base_value - steady_value)) * 1000.0
    Rm = steady_R - Rs if (not np.isnan(steady_R) and not np.isnan(Rs)) else np.nan

    # Tau by threshold crossing: 63% toward steady state from peak
    delta_I = steady_value - peak_value
    target_current = peak_value + 0.63 * delta_I

    search_start = control_start + peak_idx
    search_end = steady_state_start
    search_segment = data[search_start:search_end]

    tau_candidates = np.where(search_segment >= target_current)[0]
    if tau_candidates.size > 0:
        tau_samples = int(tau_candidates[0])
        tau_ms = tau_samples / float(fs) * 1000.0
        Cm = (tau_ms / Rs) * 1e3 if not np.isnan(Rs) and not np.isclose(Rs, 0) else np.nan
    else:
        tau_ms = np.nan
        Cm = np.nan

    return Rs, Rm, Cm, tau_ms

# Compute Rs, Rm, and Cm for a single .mat file
def compute_Rs_Rm_Cm_single_file(mat_file, control_pulse_timing='early', fs=10000):
    """
    Calculate series resistance, membrane resistance, and membrane capacitance from a single .mat file.
    
    Args:
        mat_file: Path to .mat file
        control_pulse_timing: 'early', 'early_WW', or 'late'
        fs: Sampling frequency in Hz (default: 10000)
    
    Returns:
        tuple: (Rs, Rm, Cm, tau_ms) - Rs and Rm in MΩ, Cm in pF, tau in ms
    """
    file_name = os.path.basename(mat_file)
    name_without_ext = os.path.splitext(file_name)[0]
    mat_arr = mat_to_arr(mat_file, name_without_ext)
    
    Rs, Rm, Cm, tau_ms = compute_Rs_Rm_Cm(mat_arr, control_pulse_timing, fs)
    print(f"  {file_name}: Rs = {Rs:.2f} MΩ, Rm = {Rm:.2f} MΩ, Cm = {Cm:.2f} pF, tau = {tau_ms:.2f} ms")
    
    return Rs, Rm, Cm, tau_ms

# Compute Rs, Rm, and Cm for all files in a cell
def compute_Rs_Rm_Cm_per_cell(detected_data, control_pulse_timing='early', fs=10000):
    """
    Calculate series resistance, membrane resistance, and membrane capacitance for all files from a single cell.
    
    Args:
        detected_data: Cell detection data in format [mat_files, detection_results]
        control_pulse_timing: 'early', 'early_WW', or 'late'
        fs: Sampling frequency in Hz (default: 10000)
    
    Returns:
        tuple: (Rs_values, Rm_values, Cm_values, tau_values, mean_Rs, std_Rs, mean_Rm, std_Rm, mean_Cm, std_Cm)
    """
    mat_files, detection_results = detected_data
    Rs_values = []
    Rm_values = []
    Cm_values = []
    tau_values = []
    
    for mat_file in mat_files:
        try:
            Rs, Rm, Cm, tau_ms = compute_Rs_Rm_Cm_single_file(mat_file, control_pulse_timing, fs)
            Rs_values.append(Rs)
            Rm_values.append(Rm)
            Cm_values.append(Cm)
            tau_values.append(tau_ms)
        except Exception as e:
            print(f"  Error calculating Rs/Rm/Cm for {mat_file}: {e}")
            Rs_values.append(np.nan)
            Rm_values.append(np.nan)
            Cm_values.append(np.nan)
            tau_values.append(np.nan)
    
    # Calculate statistics
    valid_Rs = [r for r in Rs_values if not np.isnan(r)]
    valid_Rm = [r for r in Rm_values if not np.isnan(r)]
    valid_Cm = [c for c in Cm_values if not np.isnan(c)]
    
    if valid_Rs:
        mean_Rs = np.mean(valid_Rs)
        std_Rs = np.std(valid_Rs)
        print(f"  Mean Rs: {mean_Rs:.2f} ± {std_Rs:.2f} MΩ ({len(valid_Rs)} files)")
    else:
        mean_Rs = np.nan
        std_Rs = np.nan
        print(f"  No valid Rs values calculated")
    
    if valid_Rm:
        mean_Rm = np.mean(valid_Rm)
        std_Rm = np.std(valid_Rm)
        print(f"  Mean Rm: {mean_Rm:.2f} ± {std_Rm:.2f} MΩ ({len(valid_Rm)} files)")
    else:
        mean_Rm = np.nan
        std_Rm = np.nan
        print(f"  No valid Rm values calculated")
    
    if valid_Cm:
        mean_Cm = np.mean(valid_Cm)
        std_Cm = np.std(valid_Cm)
        print(f"  Mean Cm: {mean_Cm:.2f} ± {std_Cm:.2f} pF ({len(valid_Cm)} files)")
    else:
        mean_Cm = np.nan
        std_Cm = np.nan
        print(f"  No valid Cm values calculated")
    
    return Rs_values, Rm_values, Cm_values, tau_values, mean_Rs, std_Rs, mean_Rm, std_Rm, mean_Cm, std_Cm

# Compute Rs, Rm, and Cm for all cells in detected data
def compute_Rs_Rm_Cm_all_cells(detected_data_list, control_pulse_timing='early', parameters_dict=None, fs=10000):
    """
    Calculate series resistance, membrane resistance, and membrane capacitance for all cells in detected data.
    
    Args:
        detected_data_list: List of detection results from batch_mini_detect()
        control_pulse_timing: Default timing if not found in parameters_dict ('early', 'early_WW', or 'late')
        parameters_dict: Optional dict with cell-specific parameters (from JSON file)
        fs: Sampling frequency in Hz (default: 10000)
    
    Returns:
        dict: Dictionary with cell names as keys and Rs/Rm/Cm data as values
    """
    cell_names = get_cell_names(detected_data_list)
    Rs_Rm_Cm_summary = {}
    
    for i, (cell_data, cell_name) in enumerate(zip(detected_data_list, cell_names)):
        print(f"\nProcessing cell {i+1}/{len(detected_data_list)}: {cell_name}")
        
        # Get cell-specific control_pulse_timing from parameters if available
        if parameters_dict and cell_name in parameters_dict:
            cell_pulse_timing = parameters_dict[cell_name].get('control_pulse_timing', control_pulse_timing)
            print(f"  Using control_pulse_timing from parameters: {cell_pulse_timing}")
        else:
            cell_pulse_timing = control_pulse_timing
            print(f"  Using default control_pulse_timing: {cell_pulse_timing}")
        
        Rs_values, Rm_values, Cm_values, tau_values, mean_Rs, std_Rs, mean_Rm, std_Rm, mean_Cm, std_Cm = compute_Rs_Rm_Cm_per_cell(cell_data, cell_pulse_timing, fs)
        
        Rs_Rm_Cm_summary[cell_name] = {
            'Rs_per_file': Rs_values,
            'Rm_per_file': Rm_values,
            'Cm_per_file': Cm_values,
            'tau_per_file': tau_values,
            'mean_Rs': mean_Rs,
            'std_Rs': std_Rs,
            'mean_Rm': mean_Rm,
            'std_Rm': std_Rm,
            'mean_Cm': mean_Cm,
            'std_Cm': std_Cm,
            'n_files': len(Rs_values),
            'control_pulse_timing': cell_pulse_timing
        }
    
    print(f"\nRs/Rm/Cm calculation completed for {len(Rs_Rm_Cm_summary)} cells")
    return Rs_Rm_Cm_summary

# Export Rs, Rm, and Cm data to CSV
def export_Rs_Rm_Cm_data(detected_data_list, output_dir, date, metadata=None, cell_id_col='Cell_ID', group_col='Group', control_pulse_timing='early', parameters_dict=None, fs=10000):
    """
    Calculate and export series resistance, membrane resistance, and membrane capacitance data for all cells.
    
    Args:
        detected_data_list: List of detection results from batch_mini_detect()
        output_dir: Directory to save CSV files
        date: Date string for filename
        metadata: DataFrame with cell IDs and group assignments (optional)
        cell_id_col: Column name for cell IDs in metadata
        group_col: Column name for group assignments in metadata
        control_pulse_timing: Default timing if not found in parameters_dict
        parameters_dict: Optional dict with cell-specific parameters (from JSON file)
        fs: Sampling frequency in Hz (default: 10000)
    
    Returns:
        DataFrame: Summary of Rs, Rm, and Cm data for all cells
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Rs, Rm, and Cm for all cells
    Rs_Rm_Cm_summary = compute_Rs_Rm_Cm_all_cells(detected_data_list, control_pulse_timing, parameters_dict, fs)
    
    # Create metadata mapping if provided
    metadata_dict = {}
    if metadata is not None:
        metadata_dict = dict(zip(metadata[cell_id_col], metadata[group_col]))
    
    # Prepare data for export
    export_data = []
    for cell_name, data in Rs_Rm_Cm_summary.items():
        # Get condition from metadata or use 'Unknown'
        condition = metadata_dict.get(cell_name, 'Unknown')
        
        export_data.append({
            'Cell_Name': cell_name,
            'Condition': condition,
            'Control_Pulse_Timing': data['control_pulse_timing'],
            'Mean_Rs_MOhm': data['mean_Rs'],
            'Std_Rs_MOhm': data['std_Rs'],
            'Mean_Rm_MOhm': data['mean_Rm'],
            'Std_Rm_MOhm': data['std_Rm'],
            'Mean_Cm_pF': data['mean_Cm'],
            'Std_Cm_pF': data['std_Cm'],
            'N_Files': data['n_files'],
            'Rs_Values': ', '.join([f"{r:.2f}" if not np.isnan(r) else 'NaN' for r in data['Rs_per_file']]),
            'Rm_Values': ', '.join([f"{r:.2f}" if not np.isnan(r) else 'NaN' for r in data['Rm_per_file']]),
            'Cm_Values': ', '.join([f"{c:.2f}" if not np.isnan(c) else 'NaN' for c in data['Cm_per_file']]),
            'Tau_Values_ms': ', '.join([f"{t:.2f}" if not np.isnan(t) else 'NaN' for t in data['tau_per_file']])
        })
    
    # Create DataFrame
    df = pd.DataFrame(export_data)
    
    # Save combined file
    combined_filename = f"{output_dir}/Rs_Rm_Cm_all_cells_{date}.csv"
    df.to_csv(combined_filename, index=False)
    print(f"\nExported combined Rs/Rm/Cm data to {combined_filename}")
    
    # Also export by condition if metadata provided
    if metadata is not None:
        for condition in df['Condition'].unique():
            condition_df = df[df['Condition'] == condition][['Cell_Name', 'Control_Pulse_Timing', 'Mean_Rs_MOhm', 'Std_Rs_MOhm', 
                                                               'Mean_Rm_MOhm', 'Std_Rm_MOhm', 'Mean_Cm_pF', 'Std_Cm_pF', 
                                                               'N_Files', 'Rs_Values', 'Rm_Values', 'Cm_Values', 'Tau_Values_ms']]
            condition_filename = f"{output_dir}/Rs_Rm_Cm_{condition}_{date}.csv"
            condition_df.to_csv(condition_filename, index=False)
            print(f"Exported {condition}: {len(condition_df)} cells to {condition_filename}")
    
    return df