DRUG_COLORS = {
    "TTX": (209, 169, 153),
    "BIMU": (38, 77, 117),
    "SKF": (235, 172, 104),
    "Iso": (161, 185, 214),
}

def plot_trace_animation(
    data,
    fs,
    win_size,
    drug=None,
    color_rgb=None,
    skip_samples=0,
    save_path=None,
    fps=None,
    dpi=150,
    figsize=None,
    width_px=None,
    height_px=None,
    minimal_axes=False,
    x_scale_ms=100,
    y_scale_pA=50,
    show_scale_labels=False,
    scale_bar_y_offset=0.07,
    peak_indices=None,
    peak_color='red',
    peak_size=6,
    peak_offset_pA=0,
):
    """
    Create animated trace with optional detected peak overlay.
    
    Parameters
    ----------
    data : array_like
        1D array of current trace data (pA)
    fs : int
        Sampling frequency in Hz
    win_size : int
        Width of visible window in milliseconds
    drug : str, optional
        Drug name for color mapping ('TTX', 'BIMU', 'SKF', 'Iso')
    color_rgb : tuple, optional
        RGB color tuple (0-255) for trace, overrides drug color
    skip_samples : int, optional
        Number of initial samples to skip
    save_path : str, optional
        Path to save MP4 file. If None, displays interactively
    fps : int, optional
        Frames per second for video export
    dpi : int, optional
        Resolution for video export (default: 150)
    figsize : tuple, optional
        Figure size in inches (width, height)
    width_px : int, optional
        Figure width in pixels (overrides figsize)
    height_px : int, optional
        Figure height in pixels (overrides figsize)
    minimal_axes : bool, optional
        Use minimal axis styling with scale bars (default: False)
    x_scale_ms : int, optional
        Scale bar width in milliseconds (for minimal_axes mode)
    y_scale_pA : int, optional
        Scale bar height in pA (for minimal_axes mode)
    show_scale_labels : bool, optional
        Show labels on scale bars (for minimal_axes mode)
    scale_bar_y_offset : float, optional
        Vertical position of scale bar as fraction from bottom (0.0=bottom, 1.0=top).
        Default is 0.07 (7% from bottom).
    peak_indices : array_like or list of array_like, optional
        Single array of peak indices OR list of arrays for multiple peak sets.
        Red dots will appear at these locations.
    peak_color : str or list of str, optional
        Color(s) for peak markers. Single color or list matching peak_indices.
        (default: 'red', or ['red', 'blue', 'green', ...] for multiple sets)
    peak_size : int or list of int, optional
        Size(s) of peak markers. Single size or list matching peak_indices.
    peak_offset_pA : float, optional
        Vertical offset in pA to apply to peak markers. Positive moves markers up,
        negative moves them down. Default is 0 (markers at actual peak values).
        (default: 8)
    
    Examples
    --------
    # Basic usage with detected peaks
    >>> from param_tune import ParameterTuner
    >>> tuner = ParameterTuner(parent_folder, mode='IPSC')
    >>> results = tuner.test_parameters(cell_folder, 10, 2, 30, n_files=1)
    >>> peak_indices = results['peak_indices'][0]
    >>> plot_trace_animation(data, fs=10000, win_size=500, peak_indices=peak_indices)
    
    # Save video with peaks
    >>> plot_trace_animation(
    ...     data, fs=10000, win_size=500,
    ...     peak_indices=peak_indices,
    ...     save_path='trace_with_peaks.mp4',
    ...     fps=60, dpi=300
    ... )
    
    # Compare two parameter sets on same trace
    >>> results1 = tuner.test_parameters_on_files([file], 10, 2, 30)
    >>> results2 = tuner.test_parameters_on_files([file], 15, 3, 40)
    >>> plot_trace_animation(
    ...     data, fs=10000, win_size=500,
    ...     peak_indices=[results1['peak_indices'][0], results2['peak_indices'][0]],
    ...     peak_color=['red', 'blue'],
    ...     peak_size=[10, 8],
    ...     save_path='comparison.mp4'
    ... )
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib as mpl
    mpl.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'

    # Optionally skip initial samples
    if skip_samples and skip_samples > 0:
        data = data[skip_samples:]
    t = np.arange(len(data)) / fs
    
    # Normalize peak_indices, peak_color, peak_size to lists
    if peak_indices is None:
        peak_sets = []
    elif isinstance(peak_indices, list) and len(peak_indices) > 0 and isinstance(peak_indices[0], (list, np.ndarray)):
        # Multiple peak sets
        peak_sets = peak_indices
    else:
        # Single peak set
        peak_sets = [peak_indices]
    
    # Normalize colors and sizes
    if isinstance(peak_color, list):
        colors = peak_color
    else:
        colors = [peak_color] * len(peak_sets)
    
    if isinstance(peak_size, list):
        sizes = peak_size
    else:
        sizes = [peak_size] * len(peak_sets)
    
    # Default colors if not enough provided
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    while len(colors) < len(peak_sets):
        colors.append(default_colors[len(colors) % len(default_colors)])
    while len(sizes) < len(peak_sets):
        sizes.append(8)
    
    # Prepare peak data for each set
    peak_data = []
    for i, indices in enumerate(peak_sets):
        if indices is not None and len(indices) > 0:
            indices = np.array(indices)
            # Adjust for skip_samples
            if skip_samples and skip_samples > 0:
                indices = indices - skip_samples
            # Filter to valid indices
            indices = indices[(indices >= 0) & (indices < len(data))]
            if len(indices) > 0:
                peak_data.append({
                    'indices': indices,
                    'times': t[indices],
                    'values': data[indices] + peak_offset_pA,
                    'color': colors[i],
                    'size': sizes[i]
                })
    
    if len(peak_data) == 0:
        peak_data = [{'indices': np.array([]), 'times': np.array([]), 'values': np.array([]), 'color': 'red', 'size': 8}]

    window_ms = win_size          # size of visible window in ms
    window_samples = int(window_ms * fs / 1000)
    step = int(5 * fs / 1000)  # advance 5 ms per frame

    # Figure sizing: priority width/height in pixels > figsize inches > default
    figsize_inches = (6, 4)
    if width_px is not None and height_px is not None:
        try:
            w_px = int(width_px)
            h_px = int(height_px)
            # Ensure even pixel dims for common FFmpeg codecs
            if w_px % 2 != 0:
                w_px += 1
            if h_px % 2 != 0:
                h_px += 1
            figsize_inches = (w_px / dpi, h_px / dpi)
        except Exception:
            pass
    elif figsize is not None:
        figsize_inches = figsize

    fig, ax = plt.subplots(figsize=figsize_inches)
    print(f"Figure size (inches): {figsize_inches}, dpi={dpi}")
    # Determine line color: manual RGB override > drug mapping > default
    if color_rgb is not None:
        r, g, b = color_rgb
        plot_color = (r/255, g/255, b/255)
    elif drug is not None and drug in DRUG_COLORS:
        r, g, b = DRUG_COLORS[drug]
        plot_color = (r/255, g/255, b/255)
    else:
        plot_color = None  # Matplotlib default

    line, = ax.plot([], [], lw=0.8, color=plot_color)
    
    # Add scatter plots for detected peaks (one per peak set)
    peak_scatters = []
    for pdata in peak_data:
        scatter = ax.scatter([], [], c=pdata['color'], s=pdata['size'], zorder=5, marker='o')
        peak_scatters.append(scatter)

    # y-limits fixed to full-trace min/max
    # ymin, ymax = -200, 10
    # pad = 0.05 * (ymax - ymin if ymax > ymin else 1)
    ax.set_ylim(-350, 350)
    if not minimal_axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (pA)")
    else:
        # Minimal axis styling: no spines, ticks, or labels
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Prepare scale bars if minimal axes is requested
    xbar = ybar = None
    xbar_text = ybar_text = None
    if minimal_axes:
        bar_color = 'k'
        xbar, = ax.plot([], [], color=bar_color, lw=1.0)
        ybar, = ax.plot([], [], color=bar_color, lw=1.0)
        if show_scale_labels:
            xbar_text = ax.text(0, 0, "", fontsize=6, va='top', ha='center')
            ybar_text = ax.text(0, 0, "", fontsize=6, va='center', ha='right')

    def init():
        line.set_data([], [])
        for scatter in peak_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        artists = [line] + peak_scatters
        if minimal_axes:
            if xbar is not None:
                xbar.set_data([], [])
                artists.append(xbar)
            if ybar is not None:
                ybar.set_data([], [])
                artists.append(ybar)
            if show_scale_labels and xbar_text is not None and ybar_text is not None:
                xbar_text.set_text("")
                ybar_text.set_text("")
                artists.extend([xbar_text, ybar_text])
        return tuple(artists)

    def update(frame):
        start = frame * step
        end = start + window_samples
        if end > len(data):
            end = len(data)
        if start >= end:
            return tuple([line] + peak_scatters)

        ax.set_xlim(t[start], t[end-1])
        line.set_data(t[start:end], data[start:end])
        
        # Update peaks in current window for each peak set
        for scatter, pdata in zip(peak_scatters, peak_data):
            if len(pdata['indices']) > 0:
                # Find peaks within current window
                in_window = (pdata['indices'] >= start) & (pdata['indices'] < end)
                window_peak_times = pdata['times'][in_window]
                window_peak_values = pdata['values'][in_window]
                if len(window_peak_times) > 0:
                    scatter.set_offsets(np.c_[window_peak_times, window_peak_values])
                else:
                    scatter.set_offsets(np.empty((0, 2)))
            else:
                scatter.set_offsets(np.empty((0, 2)))
        
        if minimal_axes:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            # Place bars near bottom-left with configurable padding
            pad_frac = 0.07  # horizontal padding
            x0 = xmin + pad_frac * (xmax - xmin)
            y0 = ymin + scale_bar_y_offset * (ymax - ymin)
            # Convert desired scales to data units
            x_len = max(0, (x_scale_ms or 0) / 1000.0)
            y_len = max(0, (y_scale_pA or 0))
            if xbar is not None:
                xbar.set_data([x0, x0 + x_len], [y0, y0])
            if ybar is not None:
                ybar.set_data([x0, x0], [y0, y0 + y_len])
            if show_scale_labels and xbar_text is not None and ybar_text is not None:
                # Label strings
                xlbl = f"{int(x_scale_ms)} ms" if x_scale_ms else ""
                ylbl = f"{int(y_scale_pA)} pA" if y_scale_pA else ""
                # Position labels slightly offset from bars
                xbar_text.set_position((x0 + x_len/2.0, y0 - 0.02 * (ymax - ymin)))
                xbar_text.set_text(xlbl)
                # Place y label slightly to the left of the vertical bar
                ybar_text.set_position((x0 - 0.005 * (xmax - xmin), y0 + y_len/2.0))
                ybar_text.set_text(ylbl)
            artists = [line] + peak_scatters
            if xbar is not None:
                artists.append(xbar)
            if ybar is not None:
                artists.append(ybar)
            if show_scale_labels and xbar_text is not None and ybar_text is not None:
                artists.extend([xbar_text, ybar_text])
            return tuple(artists)
        return tuple([line] + peak_scatters)

    n_frames = int(np.ceil((len(data) - window_samples) / step))
    if n_frames < 1:
        n_frames = 1
    print(f"Frames: {n_frames}, window_samples={window_samples}, step={step}")

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=True,
        interval=1000 * step / fs  # ms per frame
    )

    plt.tight_layout()

    # Save to MP4 if requested
    if save_path:
        # Frames per second defaults to sampling step rate
        effective_fps = fps if fps is not None else max(1, int(fs / step))
        try:
            import os
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            # Use non-interactive backend for file export
            import matplotlib as mpl
            try:
                mpl.use('Agg')
            except Exception:
                pass
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=effective_fps, metadata={"artist": "mini_detect"})
            def _progress(i, n):
                if i % max(1, n//10) == 0:
                    print(f"Export progress: {i}/{n} frames")
            ani.save(save_path, writer=writer, dpi=dpi, progress_callback=_progress)
            print(f"Saved MP4 to: {save_path} at {effective_fps} fps, dpi={dpi}")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except FileNotFoundError as e:
            print(f"Failed to save MP4 (path not found): {e}\nCheck write permissions and path: {save_path}")
        except RuntimeError as e:
            print("RuntimeError while saving MP4. This often means FFmpeg is missing or not found.")
            print(f"Details: {e}\nInstall FFmpeg via Homebrew and try again:")
            print("  brew install ffmpeg")
            print("If FFmpeg is installed, set its path in Matplotlib, e.g.:")
            print("  import matplotlib as mpl; mpl.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'")
            try:
                from matplotlib.animation import PillowWriter
                pillow_out = save_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=effective_fps)
                ani.save(pillow_out, writer=writer, dpi=dpi)
                print(f"FFmpeg fallback: saved GIF to {pillow_out}")
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e2:
                print(f"GIF fallback failed: {e2}")
        except KeyboardInterrupt:
            print("Export interrupted by user.")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"Failed to save MP4: {e}")

    # Show interactively if not saving only
    if not save_path:
        plt.show()

from functions import mat_to_arr 
# EC99-4
EC99_4_AD0_125 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4/AD0_125.mat', 'AD0_125')
EC99_4_AD0_126 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4/AD0_126.mat', 'AD0_126')
EC99_4_AD0_127 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4/AD0_127.mat', 'AD0_127')
# EC99-4b
EC99_4b_AD0_195 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4b/AD0_195.mat', 'AD0_195')
EC99_4b_AD0_196 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4b/AD0_196.mat', 'AD0_196')
EC99_4b_AD0_197 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4b/AD0_197.mat', 'AD0_197')
EC99_4b_AD0_206 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4b/AD0_206.mat', 'AD0_206')
EC99_4b_AD0_212 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-4b/AD0_212.mat', 'AD0_212')
# EC97-6
EC97_6_AD0_256 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC97-6/AD0_256.mat', 'AD0_256')
EC97_6_AD0_257 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC97-6/AD0_257.mat', 'AD0_257')
EC97_6_AD0_258 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC97-6/AD0_258.mat', 'AD0_258')
# EC97-6b
EC97_6b_AD0_227 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC97-6b/AD0_227.mat', 'AD0_227')
EC97_6b_AD0_228 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC97-6b/AD0_228.mat', 'AD0_228')
EC97_6b_AD0_229 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC97-6b/AD0_229.mat', 'AD0_229')
# EC99-1b
EC99_1b_AD0_1 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1b/AD0_1.mat', 'AD0_1')
EC99_1b_AD0_2 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1b/AD0_2.mat', 'AD0_2')
EC99_1b_AD0_3 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1b/AD0_3.mat', 'AD0_3')
# EC99-1
EC99_1_AD0_54 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1/AD0_54.mat', 'AD0_54')
EC99_1_AD0_55 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1/AD0_55.mat', 'AD0_55')
EC99_1_AD0_56 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1/AD0_56.mat', 'AD0_56')
EC99_1_AD0_40 = mat_to_arr('/Volumes/Neurobio/MICROSCOPE/Ella/2025/Ephys/Experiments/LHb_minis_Gs_pharmacology/data/EC99-1/AD0_40.mat', 'AD0_40')

from functions import bandpass_filter
# # Apply bandpass filter to all loaded traces (10 kHz, 50-5000 Hz)
# EC99_4_AD0_125_filt = bandpass_filter(EC99_4_AD0_125, 1, 3000, fs=10000, order=2)
# EC99_4_AD0_126_filt = bandpass_filter(EC99_4_AD0_126, fs=10000, lowcut=50, highcut=5000)
# EC99_4_AD0_127_filt = bandpass_filter(EC99_4_AD0_127, fs=10000, lowcut=50, highcut=5000)

# EC99_4b_AD0_195_filt = bandpass_filter(EC99_4b_AD0_195, fs=10000, lowcut=50, highcut=5000)
EC99_4b_AD0_206_filt = bandpass_filter(EC99_4b_AD0_206, 1, 3000, fs=10000, order=2)
# EC99_4b_AD0_212_filt = bandpass_filter(EC99_4b_AD0_212, 1, 3000, fs=10000, order=2)
# EC99_4b_AD0_195_filt = bandpass_filter(EC99_4b_AD0_195, 1, 3000, fs=10000, order=2)
# EC99_4b_AD0_196_filt = bandpass_filter(EC99_4b_AD0_196, 1, 3000, fs=10000, order=2)
# EC99_4b_AD0_197_filt = bandpass_filter(EC99_4b_AD0_197, fs=10000, lowcut=50, highcut=5000)

# EC97_6_AD0_256_filt = bandpass_filter(EC97_6_AD0_256, fs=10000, lowcut=50, highcut=5000)
# EC97_6_AD0_257_filt = bandpass_filter(EC97_6_AD0_257, fs=10000, lowcut=50, highcut=5000)
# EC97_6_AD0_258_filt = bandpass_filter(EC97_6_AD0_258, fs=10000, lowcut=50, highcut=5000)

# EC97_6b_AD0_227_filt = bandpass_filter(EC97_6b_AD0_227, fs=10000, lowcut=50, highcut=5000)
# EC97_6b_AD0_228_filt = bandpass_filter(EC97_6b_AD0_228, fs=10000, lowcut=50, highcut=5000)
# EC97_6b_AD0_229_filt = bandpass_filter(EC97_6b_AD0_229, fs=10000, lowcut=50, highcut=5000)

# EC99_1b_AD0_1_filt = bandpass_filter(EC99_1b_AD0_1, fs=10000, lowcut=50, highcut=5000)
# EC99_1b_AD0_2_filt = bandpass_filter(EC99_1b_AD0_2, fs=10000, lowcut=50, highcut=5000)
# EC99_1b_AD0_3_filt = bandpass_filter(EC99_1b_AD0_3, fs=10000, lowcut=50, highcut=5000)

# EC99_1_AD0_54_filt = bandpass_filter(EC99_1_AD0_54, fs=10000, lowcut=50, highcut=5000)
# EC99_1_AD0_55_filt = bandpass_filter(EC99_1_AD0_55, fs=10000, lowcut=50, highcut=5000)
# EC99_1_AD0_56_filt = bandpass_filter(EC99_1_AD0_56, fs=10000, lowcut=50, highcut=5000)



# TTX: 244 210 196
# BIMU: 38, 77, 117
# SKF 235 172 104
# Iso: 161 185 214 

# plot_trace_animation(
#     EC99_4_AD0_125,
#     fs=10000,
#     win_size=500,
#     drug="TTX",
#     skip_samples=2500,
#     save_path="/Users/ellacho/Downloads/EC99_4_AD0_125_TTX.mp4",
#     fps=60,
#     dpi=300
#     # Example: 1280x720 export in pixels
#     # width_px=1280,
#     # height_px=720,
#     # Example: custom inches (overrides default 6x3 in)
#     # figsize=(6, 6),
# )

plot_trace_animation(
    EC99_1_AD0_55,
    fs=10000,
    win_size=750,
    drug="BIMU",
    skip_samples=2550,
    save_path="/Users/ellacho/Downloads/EC99_1_AD0_55_BIMU_nofilt.mp4",
    fps=60,
    dpi=300,
    minimal_axes=True,
    x_scale_ms=50,
    y_scale_pA=50,
    show_scale_labels=True
    # Example: 1280x720 export in pixels
    # width_px=1280,
    # height_px=720,
    # Example: custom inches (overrides default 6x3 in)
    # figsize=(6, 6),
)


# plot_trace_animation(EC97_6b_AD0_228, fs=10000, win_size=500, drug="SKF", skip_samples=2600)
# plot_trace_animation(EC97_6_AD0_258, fs=10000, win_size=500, drug="Iso", skip_samples=2600)

# # plot_trace_animation(EC99_1b_AD0_1, fs=10000, win_size=500, color_rgb=(200, 50, 50), skip_samples=2600)
# plot_trace_animation(EC99_1_AD0_56, fs=10000, win_size=500, drug="TTX", skip_samples=2600)