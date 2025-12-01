## for Parameter Tuning ##

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import random
import json
import os
import glob
from functions import (mat_to_arr, bandpass_filter, decay_window, 
                       find_peak_near_decay, filter_peaks_by_noise)

# Set plotly to display in notebook
pio.renderers.default = 'notebook'

class ParameterTuner:
    def __init__(self, parent_folder, mode='IPSC', control_pulse_timing='early'):
        """
        Initialize parameter tuner for mini detection optimization
        
        Args:
            parent_folder: Path to folder containing cell subfolders
            mode: Detection mode ('IPSC' or 'EPSC')
            control_pulse_timing: Control pulse timing ('early' for 200-250ms or 'late' for 9.8-9.9s)
        """
        self.parent_folder = parent_folder
        self.mode = mode
        self.control_pulse_timing = control_pulse_timing
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
                                                     baseline=None, mode=self.mode, fs=10000,
                                                     control_pulse_timing=self.control_pulse_timing)
                
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
                                                     baseline=None, mode=self.mode, fs=10000,
                                                     control_pulse_timing=self.control_pulse_timing)
                
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
        Visualize detection results with interactive Plotly sample traces
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
                
                # Create time axis (assuming 10kHz sampling)
                time_axis = np.arange(len(filtered_mat)) / 10000 * 1000  # Convert to ms
                
                # Create interactive Plotly figure
                fig = go.Figure()
                
                # Add filtered signal trace
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=filtered_mat,
                    mode='lines',
                    name='Filtered signal',
                    line=dict(color='blue', width=1),
                    hovertemplate='Time: %{x:.1f}ms<br>Amplitude: %{y:.1f}pA<extra></extra>'
                ))
                
                # Plot detected peaks
                if results['peak_indices'][0]:  # First file's peaks
                    peak_indices = results['peak_indices'][0]
                    peak_times = [time_axis[i] for i in peak_indices]
                    peak_values = [filtered_mat[i] for i in peak_indices]
                    
                    fig.add_trace(go.Scatter(
                        x=peak_times,
                        y=peak_values,
                        mode='markers',
                        name=f'Detected peaks (n={len(peak_indices)})',
                        marker=dict(color='red', size=8, symbol='circle'),
                        hovertemplate='Peak Time: %{x:.1f}ms<br>Amplitude: %{y:.1f}pA<extra></extra>'
                    ))
                
                # Add control pulse region based on timing
                if self.control_pulse_timing == 'early':
                    fig.add_vrect(
                        x0=200, x1=250,  # Convert sample indices to ms
                        fillcolor="gray", opacity=0.3,
                        annotation_text="Control pulse (excluded)",
                        annotation_position="top left"
                    )
                elif self.control_pulse_timing == 'early_WW':
                    fig.add_vrect(
                        x0=200, x1=300,  # Convert sample indices to ms
                        fillcolor="gray", opacity=0.3,
                        annotation_text="Control pulse (excluded)",
                        annotation_position="top left"
                    )
                elif self.control_pulse_timing == 'late':
                    fig.add_vrect(
                        x0=9800, x1=9900,  # Convert sample indices to ms
                        fillcolor="gray", opacity=0.3,
                        annotation_text="Control pulse (excluded)",
                        annotation_position="top left"
                    )
                
                # Update layout for better interactivity
                fig.update_layout(
                    title=f'{cell_name} - {file_name}<br>Parameters: ws={params["window_size"]}, ss={params["sample_step"]}, int={params["interval"]}',
                    xaxis_title='Time (ms)',
                    yaxis_title='Amplitude (pA)',
                    width=1200,
                    height=500,
                    hovermode='x unified',
                    showlegend=True
                )
                
                # Enable zoom and pan
                fig.update_xaxes(showspikes=True, spikecolor="green", spikethickness=2)
                fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
                
                # Show the interactive plot
                try:
                    fig.show()
                except Exception as e:
                    print(f"Could not display interactive plot: {e}")
                    print("Consider running: pip install nbformat ipywidgets")
                    # Fallback to basic text output
                    print(f"Plot would show {len(filtered_mat)} data points with {len(peak_indices) if results['peak_indices'][0] else 0} detected peaks")
                
            except Exception as e:
                print(f"Could not plot sample trace: {e}")
    
    def create_interactive_comparison(self, cell_folder, test_files, parameter_sets):
        """
        Create interactive comparison plot showing all parameter sets side by side
        """
        n_params = len(parameter_sets)
        n_files = len(test_files)
        
        # Create subplots - one row per parameter set, one column per file
        subplot_titles = []
        for i, (ws, ss, iv) in enumerate(parameter_sets):
            for j, file in enumerate(test_files):
                file_name = os.path.basename(file)
                subtitle = f"Param {i+1}: {file_name}" if n_files > 1 else f"WS:{ws}, SS:{ss}, IV:{iv}"
                subplot_titles.append(subtitle)
        
        fig = make_subplots(
            rows=n_params,
            cols=n_files,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.03
        )
        
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink', 'gray']
        
        # Process each parameter set
        for param_idx, (window_size, sample_step, interval) in enumerate(parameter_sets):
            # Test this parameter set on all files
            results = self.test_parameters_on_files(test_files, window_size, sample_step, interval)
            
            # Plot each file for this parameter set
            for file_idx, file_path in enumerate(test_files):
                try:
                    # Load and process the file
                    file_name = os.path.basename(file_path)
                    name_without_ext = os.path.splitext(file_name)[0]
                    mat_arr = mat_to_arr(file_path, name_without_ext)
                    filtered_mat = bandpass_filter(mat_arr, 1, 3000, fs=10000, order=2)
                    
                    # Create time axis
                    time_axis = np.arange(len(filtered_mat)) / 10000 * 1000  # Convert to ms
                    
                    row = param_idx + 1
                    col = file_idx + 1
                    color = colors[param_idx % len(colors)]
                    
                    # Add filtered signal
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=filtered_mat,
                            mode='lines',
                            name=f'P{param_idx+1}F{file_idx+1}',
                            line=dict(color=color, width=0.8),
                            showlegend=False,
                            hovertemplate='Time: %{x:.1f}ms<br>Amplitude: %{y:.1f}pA<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    
                    # Add detected peaks for this file
                    if results['peak_indices'][file_idx]:  # Check if this file has peaks
                        peak_indices = results['peak_indices'][file_idx]
                        peak_times = [time_axis[i] for i in peak_indices]
                        peak_values = [filtered_mat[i] for i in peak_indices]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode='markers',
                                name=f'Peaks P{param_idx+1}F{file_idx+1}',
                                marker=dict(color='red', size=6),
                                showlegend=False,
                                hovertemplate='Peak: %{x:.1f}ms, %{y:.1f}pA<extra></extra>'
                            ),
                            row=row, col=col
                        )
                    
                    # Add control pulse region
                    if self.control_pulse_timing == 'early':
                        fig.add_vrect(
                            x0=200, x1=250,
                            fillcolor="gray", opacity=0.2,
                            row=row, col=col
                        )

                    elif self.control_pulse_timing == 'early_WW':
                        fig.add_vrect(
                            x0=200, x1=300,
                            fillcolor="gray", opacity=0.2,
                            row=row, col=col
                        )
                    elif self.control_pulse_timing == 'late':
                        fig.add_vrect(
                            x0=9800, x1=9900,
                            fillcolor="gray", opacity=0.2,
                            row=row, col=col
                        )
                        
                except Exception as e:
                    print(f"Error plotting {file_path}: {e}")
        
        # Add parameter annotations on the left
        for i, (ws, ss, iv) in enumerate(parameter_sets):
            fig.add_annotation(
                text=f"WS:{ws}<br>SS:{ss}<br>IV:{iv}",
                xref="paper", yref="paper",
                x=-0.1, y=1 - (i + 0.5) / n_params,
                showarrow=False,
                font=dict(size=10),
                xanchor="right"
            )
        
        # Update layout
        cell_name = os.path.basename(cell_folder)
        fig.update_layout(
            title=f"Interactive Parameter Comparison - {cell_name}",
            height=300 * n_params,
            width=400 * n_files if n_files > 1 else 1000,
            showlegend=False
        )
        
        # Enable interactive features
        fig.update_xaxes(showspikes=True, spikecolor="green", spikethickness=1)
        fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=1)
        
        # Set axis labels only for bottom and left subplots
        fig.update_xaxes(title_text="Time (ms)", row=n_params, col=1)
        fig.update_yaxes(title_text="Amplitude (pA)", row=1, col=1)
        
        return fig

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
        
        # Create interactive comparison plot
        if visualize and len(parameter_sets) > 1:
            print("\nCreating interactive comparison plot...")
            try:
                comparison_fig = self.create_interactive_comparison(cell_folder, test_files, parameter_sets)
                # Try different rendering methods
                try:
                    comparison_fig.show()
                except Exception as show_error:
                    print(f"Note: Could not display interactive plot ({show_error})")
                    print("Individual parameter plots were shown above.")
            except Exception as plot_error:
                print(f"Could not create comparison plot: {plot_error}")
                print("Individual parameter plots were shown above.")
        
        # Summary comparison
        print(f"\n=== SUMMARY COMPARISON for {os.path.basename(cell_folder)} ===")
        print("Set | Window | Step | Interval | Total Peaks | Mean/File | Mean Amp")
        print("----|--------|------|----------|-------------|-----------|----------")
        for i, result in enumerate(all_results):
            p = result['parameters']
            print(f"{i+1:2d}  | {p['window_size']:6d} | {p['sample_step']:4d} | {p['interval']:8d} | "
                  f"{result['total_peaks']:11d} | {result['mean_peaks_per_file']:9.1f} | {result['mean_amplitude']:8.1f}")
        
        return all_results
    
    def save_parameters(self, cell_name, window_size, sample_step, interval, control_pulse_timing=None):
        """Save optimal parameters for a cell"""
        # Use provided control_pulse_timing or fall back to instance default
        pulse_timing = control_pulse_timing if control_pulse_timing is not None else self.control_pulse_timing
        
        self.parameters[cell_name] = {
            'window_size': window_size,
            'sample_step': sample_step,
            'interval': interval,
            'mode': self.mode,
            'control_pulse_timing': pulse_timing
        }
        print(f"Saved parameters for {cell_name}: ws={window_size}, ss={sample_step}, int={interval}, pulse_timing={pulse_timing}")
    
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
                            # Use the stored control_pulse_timing for this specific cell, fall back to tuner default
                            cell_pulse_timing = params.get('control_pulse_timing', self.control_pulse_timing)
                            filtered_peaks = filter_peaks_by_noise(filtered_mat, mono_peaks[0], 
                                                                 baseline=None, mode=self.mode, fs=10000,
                                                                 control_pulse_timing=cell_pulse_timing)
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