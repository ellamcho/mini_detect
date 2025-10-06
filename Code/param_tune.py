## for Parameter Tuning ##

import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
import glob
from functions import (mat_to_arr, bandpass_filter, decay_window, 
                       find_peak_near_decay, filter_peaks_by_noise)

class ParameterTuner:
    def __init__(self, parent_folder, mode='IPSC'):
        """
        Initialize parameter tuner for mini detection optimization
        
        Args:
            parent_folder: Path to folder containing cell subfolders
            mode: Detection mode ('IPSC' or 'EPSC')
        """
        self.parent_folder = parent_folder
        self.mode = mode
        self.cell_folders = [os.path.join(parent_folder, d) 
                           for d in os.listdir(parent_folder) 
                           if os.path.isdir(os.path.join(parent_folder, d))]
        self.parameters = {}  # Store optimized parameters per cell
        
        print(f"Found {len(self.cell_folders)} cell folders:")
        for folder in self.cell_folders:
            cell_name = os.path.basename(folder)
            mat_files = glob.glob(os.path.join(folder, "*.mat"))
            print(f"  {cell_name}: {len(mat_files)} files")
    
    def get_random_files(self, cell_folder, n_files=3):
        """Get random sample of .mat files from a cell folder"""
        mat_files = glob.glob(os.path.join(cell_folder, "*.mat"))
        if len(mat_files) <= n_files:
            return mat_files
        return random.sample(mat_files, n_files)
    
    def test_parameters(self, cell_folder, window_size, sample_step, interval, n_files=3):
        """
        Test detection parameters on random files from a cell
        
        Returns:
            dict: Results including detection counts, peak amplitudes, and file info
        """
        files = self.get_random_files(cell_folder, n_files)
        results = {
            'cell_name': os.path.basename(cell_folder),
            'parameters': {
                'window_size': window_size,
                'sample_step': sample_step, 
                'interval': interval
            },
            'files_tested': [os.path.basename(f) for f in files],
            'detections': [],
            'peak_amplitudes': [],
            'peak_indices': []
        }
        
        total_peaks = 0
        all_amplitudes = []
        
        for mat_file in files:
            try:
                # Process single file with given parameters
                file_name = os.path.basename(mat_file)
                name_without_ext = os.path.splitext(file_name)[0]
                mat_arr = mat_to_arr(mat_file, name_without_ext)
                filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
                
                # Use custom parameters
                mono_indices = decay_window(filtered_mat, window_size, sample_step, mode=self.mode)
                if not mono_indices:
                    results['detections'].append(0)
                    results['peak_amplitudes'].append([])
                    results['peak_indices'].append([])
                    continue
                    
                mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, interval=interval, mode=self.mode)
                if not mono_peaks[0]:
                    results['detections'].append(0)
                    results['peak_amplitudes'].append([])
                    results['peak_indices'].append([])
                    continue
                    
                filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], 
                                                     baseline=None, mode=self.mode, fs=10000)
                
                n_peaks = len(filtered_peaks[0])
                results['detections'].append(n_peaks)
                results['peak_amplitudes'].append(filtered_peaks[0])
                results['peak_indices'].append(filtered_peaks[1])
                
                total_peaks += n_peaks
                all_amplitudes.extend(filtered_peaks[0])
                
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                results['detections'].append(0)
                results['peak_amplitudes'].append([])
                results['peak_indices'].append([])
        
        results['total_peaks'] = total_peaks
        results['mean_peaks_per_file'] = total_peaks / len(files) if files else 0
        results['mean_amplitude'] = np.mean(all_amplitudes) if all_amplitudes else 0
        results['amplitude_std'] = np.std(all_amplitudes) if all_amplitudes else 0
        
        return results
    
    def test_parameters_on_files(self, files, window_size, sample_step, interval):
        """
        Test detection parameters on a specific list of files
        
        Returns:
            dict: Results including detection counts, peak amplitudes, and file info
        """
        results = {
            'cell_name': os.path.basename(os.path.dirname(files[0])) if files else 'unknown',
            'parameters': {
                'window_size': window_size,
                'sample_step': sample_step, 
                'interval': interval
            },
            'files_tested': [os.path.basename(f) for f in files],
            'detections': [],
            'peak_amplitudes': [],
            'peak_indices': []
        }
        
        total_peaks = 0
        all_amplitudes = []
        
        for mat_file in files:
            try:
                # Process single file with given parameters
                file_name = os.path.basename(mat_file)
                name_without_ext = os.path.splitext(file_name)[0]
                mat_arr = mat_to_arr(mat_file, name_without_ext)
                filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
                
                # Use custom parameters
                mono_indices = decay_window(filtered_mat, window_size, sample_step, mode=self.mode)
                if not mono_indices:
                    results['detections'].append(0)
                    results['peak_amplitudes'].append([])
                    results['peak_indices'].append([])
                    continue
                    
                mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, interval=interval, mode=self.mode)
                if not mono_peaks[0]:
                    results['detections'].append(0)
                    results['peak_amplitudes'].append([])
                    results['peak_indices'].append([])
                    continue
                    
                filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], 
                                                     baseline=None, mode=self.mode, fs=10000)
                
                n_peaks = len(filtered_peaks[0])
                results['detections'].append(n_peaks)
                results['peak_amplitudes'].append(filtered_peaks[0])
                results['peak_indices'].append(filtered_peaks[1])
                
                total_peaks += n_peaks
                all_amplitudes.extend(filtered_peaks[0])
                
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                results['detections'].append(0)
                results['peak_amplitudes'].append([])
                results['peak_indices'].append([])
        
        results['total_peaks'] = total_peaks
        results['mean_peaks_per_file'] = total_peaks / len(files) if files else 0
        results['mean_amplitude'] = np.mean(all_amplitudes) if all_amplitudes else 0
        results['amplitude_std'] = np.std(all_amplitudes) if all_amplitudes else 0
        
        return results
    
    def test_parameters_on_files(self, files, window_size, sample_step, interval):
        """
        Test detection parameters on a specific list of files
        
        Returns:
            dict: Results including detection counts, peak amplitudes, and file info
        """
        results = {
            'cell_name': os.path.basename(os.path.dirname(files[0])) if files else 'unknown',
            'parameters': {
                'window_size': window_size,
                'sample_step': sample_step, 
                'interval': interval
            },
            'files_tested': [os.path.basename(f) for f in files],
            'detections': [],
            'peak_amplitudes': [],
            'peak_indices': []
        }
        
        total_peaks = 0
        all_amplitudes = []
        
        for mat_file in files:
            try:
                # Process single file with given parameters
                file_name = os.path.basename(mat_file)
                name_without_ext = os.path.splitext(file_name)[0]
                mat_arr = mat_to_arr(mat_file, name_without_ext)
                filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
                
                # Use custom parameters
                mono_indices = decay_window(filtered_mat, window_size, sample_step, mode=self.mode)
                if not mono_indices:
                    results['detections'].append(0)
                    results['peak_amplitudes'].append([])
                    results['peak_indices'].append([])
                    continue
                    
                mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, interval=interval, mode=self.mode)
                if not mono_peaks[0]:
                    results['detections'].append(0)
                    results['peak_amplitudes'].append([])
                    results['peak_indices'].append([])
                    continue
                    
                filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], 
                                                     baseline=None, mode=self.mode, fs=10000)
                
                n_peaks = len(filtered_peaks[0])
                results['detections'].append(n_peaks)
                results['peak_amplitudes'].append(filtered_peaks[0])
                results['peak_indices'].append(filtered_peaks[1])
                
                total_peaks += n_peaks
                all_amplitudes.extend(filtered_peaks[0])
                
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                results['detections'].append(0)
                results['peak_amplitudes'].append([])
                results['peak_indices'].append([])
        
        results['total_peaks'] = total_peaks
        results['mean_peaks_per_file'] = total_peaks / len(files) if files else 0
        results['mean_amplitude'] = np.mean(all_amplitudes) if all_amplitudes else 0
        results['amplitude_std'] = np.std(all_amplitudes) if all_amplitudes else 0
        
        return results
    
    def visualize_results(self, results, cell_folder=None):
        """
        Visualize detection results with sample traces
        """
        cell_name = results['cell_name']
        params = results['parameters']
        
        print(f"\n=== Results for {cell_name} ===")
        print(f"Parameters: window_size={params['window_size']}, sample_step={params['sample_step']}, interval={params['interval']}")
        print(f"Files tested: {results['files_tested']}")
        print(f"Detections per file: {results['detections']}")
        print(f"Total peaks: {results['total_peaks']}")
        print(f"Mean peaks per file: {results['mean_peaks_per_file']:.1f}")
        print(f"Mean amplitude: {results['mean_amplitude']:.1f} ± {results['amplitude_std']:.1f}")
        
        # Plot sample trace with detections
        if cell_folder and results['files_tested']:
            try:
                # Load first test file for visualization
                first_file = os.path.join(cell_folder, results['files_tested'][0])
                file_name = os.path.basename(first_file)
                name_without_ext = os.path.splitext(file_name)[0]
                mat_arr = mat_to_arr(first_file, name_without_ext)
                filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
                
                plt.figure(figsize=(20, 8))
                plt.plot(filtered_mat, 'b-', linewidth=0.5, alpha=0.7, label='Filtered signal')
                
                # Plot detected peaks
                if results['peak_indices'][0]:  # First file's peaks
                    peak_indices = results['peak_indices'][0]
                    peak_values = [filtered_mat[i] for i in peak_indices]
                    plt.scatter(peak_indices, peak_values, color='red', s=30, 
                              label=f'Detected peaks (n={len(peak_indices)})', zorder=5)
                
                # Highlight control pulse region
                plt.axvspan(2000, 2500, alpha=0.3, color='gray', label='Control pulse (excluded)')
                
                plt.title(f'{cell_name} - {file_name}\nParameters: ws={params["window_size"]}, ss={params["sample_step"]}, int={params["interval"]}')
                plt.xlabel('Sample index')
                plt.ylabel('Amplitude (pA)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Could not plot sample trace: {e}")
    
    def sample_and_test(self, cell_folder, parameter_sets, n_files=3, visualize=True):
        """
        Test multiple parameter combinations on a cell and compare results
        
        Args:
            cell_folder: Path to cell folder or cell name
            parameter_sets: List of (window_size, sample_step, interval) tuples
            n_files: Number of random files to test per parameter set
            visualize: Whether to show plots
        
        Returns:
            list: Results for each parameter set
        """
        # Handle both full path and cell name
        if not os.path.exists(cell_folder):
            # Assume it's a cell name, find the full path
            matching_folders = [f for f in self.cell_folders if os.path.basename(f) == cell_folder]
            if not matching_folders:
                print(f"Cell folder '{cell_folder}' not found")
                return []
            cell_folder = matching_folders[0]
        
        # Get the SAME random files for all parameter tests
        test_files = self.get_random_files(cell_folder, n_files)
        print(f"\nTesting {len(parameter_sets)} parameter sets on {os.path.basename(cell_folder)}")
        print(f"Using the SAME {len(test_files)} files for all parameter tests:")
        for f in test_files:
            print(f"  - {os.path.basename(f)}")
        
        all_results = []
        for i, (window_size, sample_step, interval) in enumerate(parameter_sets):
            print(f"\n--- Test {i+1}/{len(parameter_sets)} ---")
            results = self.test_parameters_on_files(test_files, window_size, sample_step, interval)
            all_results.append(results)
            
            if visualize:
                self.visualize_results(results, cell_folder)
        
        # Summary comparison
        print(f"\n=== SUMMARY COMPARISON for {os.path.basename(cell_folder)} ===")
        print("Set | Window | Step | Interval | Total Peaks | Mean/File | Mean Amp")
        print("----|--------|------|----------|-------------|-----------|----------")
        for i, result in enumerate(all_results):
            p = result['parameters']
            print(f"{i+1:2d}  | {p['window_size']:6d} | {p['sample_step']:4d} | {p['interval']:8d} | "
                  f"{result['total_peaks']:11d} | {result['mean_peaks_per_file']:9.1f} | {result['mean_amplitude']:8.1f}")
        
        return all_results
    
    def save_parameters(self, cell_name, window_size, sample_step, interval):
        """Save optimal parameters for a cell"""
        self.parameters[cell_name] = {
            'window_size': window_size,
            'sample_step': sample_step,
            'interval': interval,
            'mode': self.mode
        }
        print(f"Saved parameters for {cell_name}: ws={window_size}, ss={sample_step}, int={interval}")
    
    def export_parameters(self, filename, output_dir=None):
        """
        Export all saved parameters to JSON file
        
        Args:
            filename: Name of the JSON file to create
            output_dir: Optional directory to save the file to. If None, saves to current directory.
        """
        if output_dir is not None:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            # Construct full path
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        with open(full_path, 'w') as f:
            json.dump(self.parameters, f, indent=2)
        print(f"Exported parameters for {len(self.parameters)} cells to {full_path}")
    
    def load_parameters(self, filename):
        """Load parameters from JSON file"""
        with open(filename, 'r') as f:
            self.parameters = json.load(f)
        print(f"Loaded parameters for {len(self.parameters)} cells from {filename}")
    
    def run_full_detection(self, output_dir="./results/"):
        """
        Run full detection on all cells using their optimized parameters
        """
        if not self.parameters:
            print("No parameters saved! Use save_parameters() first.")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        
        for cell_folder in self.cell_folders:
            cell_name = os.path.basename(cell_folder)
            
            if cell_name not in self.parameters:
                print(f"No parameters saved for {cell_name}, skipping")
                continue
            
            params = self.parameters[cell_name]
            print(f"\nProcessing {cell_name} with optimized parameters...")
            
            # Run detection with saved parameters
            mat_files = glob.glob(os.path.join(cell_folder, "*.mat"))
            cell_results = []
            
            for mat_file in mat_files:
                try:
                    file_name = os.path.basename(mat_file)
                    name_without_ext = os.path.splitext(file_name)[0]
                    mat_arr = mat_to_arr(mat_file, name_without_ext)
                    filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
                    
                    mono_indices = decay_window(filtered_mat, 
                                              params['window_size'], 
                                              params['sample_step'], 
                                              mode=self.mode)
                    if mono_indices:
                        mono_peaks = find_peak_near_decay(filtered_mat, mono_indices, 
                                                        interval=params['interval'], 
                                                        mode=self.mode)
                        if mono_peaks[0]:
                            filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], 
                                                                 baseline=None, mode=self.mode, fs=10000)
                            cell_results.append(filtered_peaks)
                        else:
                            cell_results.append(([], []))
                    else:
                        cell_results.append(([], []))
                        
                except Exception as e:
                    print(f"Error processing {mat_file}: {e}")
                    cell_results.append(([], []))
            
            all_results.append((mat_files, cell_results))
            
            # Calculate summary stats
            total_peaks = sum(len(result[0]) for result in cell_results)
            print(f"Detected {total_peaks} total peaks in {len(mat_files)} files")
        
        return all_results