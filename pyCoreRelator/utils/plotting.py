"""
Core plotting functions for DTW correlation visualization.

## Included Functions

- **plot_segment_pair_correlation**: Main function for plotting correlation between log segments
- **plot_multilog_segment_pair_correlation**: Specialized function for multilogs with images  
- **visualize_combined_segments**: Combine and visualize multiple segment pairs
- **plot_correlation_distribution**: Plot quality metric distributions from CSV results

This module provides functions for visualizing dynamic time warping (DTW) correlations
between core log data, including support for multi-segment correlations, RGB/CT images,
age constraints, and quality indicators. Works seamlessly with both original and 
ML-processed core data.
"""

import os
import math
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed
from IPython.display import display, Image as IPImage
from PIL import Image
import imageio
from ..analysis.path_combining import combine_segment_dtw_results
from .matrix_plots import plot_dtw_matrix_with_paths
from .data_loader import load_and_prepare_quality_data, reconstruct_raw_data_from_histogram
from ..analysis.quality import cohens_d

def plot_segment_pair_correlation(log_a, log_b, md_a, md_b, 
                                  segment_pairs=None, dtw_results=None, segments_a=None, segments_b=None,
                                  depth_boundaries_a=None, depth_boundaries_b=None,
                                  wp=None, a_start=None, a_end=None, b_start=None, b_end=None,
                                  step=5, picked_datum_a=None, picked_datum_b=None, 
                                  quality_indicators=None, combined_quality=None,
                                  age_consideration=False, datum_ages_a=None, datum_ages_b=None,
                                  all_constraint_depths_a=None, all_constraint_depths_b=None,
                                  all_constraint_ages_a=None, all_constraint_ages_b=None,
                                  all_constraint_pos_errors_a=None, all_constraint_pos_errors_b=None,
                                  all_constraint_neg_errors_a=None, all_constraint_neg_errors_b=None,
                                  color_function=None, save_path=None,
                                  visualize_pairs=True, visualize_segment_labels=False,
                                  mark_depths=True, mark_ages=False,
                                  single_segment_mode=None,
                                  available_columns_a=None, available_columns_b=None,
                                  rgb_img_a=None, ct_img_a=None, rgb_img_b=None, ct_img_b=None,
                                  color_style_map=None, dpi=None,
                                  fig_format=None, correlation_figsize=None,
                                  show_quality_indicators=False,
                                  invert_colormap=None):
    """
    Enhanced unified function to plot correlation between log segments for both single and multiple segment pairs.
    Supports both single logs and multilogs with RGB and CT images.
    
    Parameters
    ----------
    log_a, log_b : array-like
        The full log data arrays (can be single log or multilogs)
    md_a, md_b : array-like
        Measured depth arrays for the full logs
    segment_pairs : list of tuples, optional
        List of tuples (a_idx, b_idx) for segment pairs to visualize (multi-segment mode)
    dtw_results : dict, optional
        Dictionary containing DTW results for each segment pair (multi-segment mode)
    segments_a, segments_b : list, optional
        Lists of segments in log_a and log_b (multi-segment mode)
    depth_boundaries_a, depth_boundaries_b : list, optional
        Depth boundaries for log_a and log_b (multi-segment mode)
    wp : array-like, optional
        Warping path array with coordinates in global index space (single-segment mode)
    a_start, a_end : int, optional
        Start and end indices of segment A in the full log (single-segment mode)
    b_start, b_end : int, optional
        Start and end indices of segment B in the full log (single-segment mode)
    step : int, default=5
        Sampling interval for visualization
    picked_datum_a, picked_datum_b : array-like, optional
        Arrays of picked depth indices to display as markers
    quality_indicators : dict, optional
        Quality indicators for single pair
    combined_quality : dict, optional
        Combined quality indicators for multiple pairs
    age_consideration : bool, default=False
        Whether age data should be displayed
    datum_ages_a, datum_ages_b : dict, optional
        Dictionaries containing age data for picked depths
    all_constraint_depths_a, all_constraint_depths_b : array-like, optional
        Depths of age constraints
    all_constraint_ages_a, all_constraint_ages_b : array-like, optional
        Ages of constraints
    all_constraint_pos_errors_a, all_constraint_pos_errors_b : array-like, optional
        Positive errors for constraints
    all_constraint_neg_errors_a, all_constraint_neg_errors_b : array-like, optional
        Negative errors for constraints
    color_function : callable, optional
        Function to map log values to colors (consistent across all segments)
    save_path : str, optional
        Path to save the figure
    visualize_pairs : bool, default=True
        Whether to show segment pairs with colors (True) or use log value coloring (False)
    visualize_segment_labels : bool, default=False
        Whether to show segment labels
    mark_depths : bool, default=True
        Whether to mark picked depth boundaries on the logs
    mark_ages : bool, default=False
        Whether to show age information
    single_segment_mode : bool, optional
        Explicitly set mode to single segment (True) or multi-segment (False)
    available_columns_a, available_columns_b : list, optional
        Lists of column names for multilogs
    rgb_img_a, ct_img_a, rgb_img_b, ct_img_b : array-like, optional
        RGB and CT images for cores
    color_style_map : dict, optional
        Dictionary mapping log column names to colors and styles
    dpi : int, optional
        Resolution for saved figures in dots per inch. If None, uses default (150)
    fig_format : list of str, optional
        Output figure formats (e.g. ['png'], ['png', 'svg']). Accepts 'png', 'jpeg',
        'jpg', 'svg', 'pdf'. Defaults to ['png']. Invalid entries fall back to ['png'].
    correlation_figsize : tuple, optional
        Figure size (width, height) for the correlation plot. Default is (6, 20).
    show_quality_indicators : bool, default=False
        If True, display the DTW Quality Indicators text box on the figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot
        
    Examples
    --------
    Single segment mode:
    >>> fig = plot_segment_pair_correlation(
    ...     log_a, log_b, md_a, md_b,
    ...     wp=warping_path, a_start=0, a_end=100, b_start=50, b_end=150,
    ...     single_segment_mode=True
    ... )
    
    Multi-segment mode:
    >>> fig = plot_segment_pair_correlation(
    ...     log_a, log_b, md_a, md_b,
    ...     segment_pairs=[(0, 1), (1, 2)], dtw_results=results_dict,
    ...     segments_a=segs_a, segments_b=segs_b,
    ...     depth_boundaries_a=bounds_a, depth_boundaries_b=bounds_b,
    ...     single_segment_mode=False
    ... )
    """
    
    # Default color style map for multilogs
    if color_style_map is None:
        color_style_map = {
            'R': {'color': 'red', 'linestyle': '--'},
            'G': {'color': 'green', 'linestyle': '--'},
            'B': {'color': 'blue', 'linestyle': '--'},
            'Lumin': {'color': 'darkgray', 'linestyle': '--'},
            'hiresMS': {'color': 'black', 'linestyle': '-'},
            'MS': {'color': 'gray', 'linestyle': '-'},
            'Den_gm/cc': {'color': 'orange', 'linestyle': '-'},
            'CT': {'color': 'purple', 'linestyle': '-'}
        }
    
    def get_yl_br_color(log_value):
        """Generate yellow-brown color spectrum based on log value."""
        color = np.array([1-0.4*log_value, 1-0.7*log_value, 0.6-0.6*log_value])
        color[color > 1] = 1
        color[color < 0] = 0
        return color
    
    # Determine if we have multilogs
    is_multilog_a = log_a.ndim > 1 and log_a.shape[1] > 1
    is_multilog_b = log_b.ndim > 1 and log_b.shape[1] > 1
    
    # Determine operating mode
    if single_segment_mode is not None:
        multi_segment_mode = not single_segment_mode
    else:
        multi_segment_mode = segment_pairs is not None and dtw_results is not None
        single_segment_mode = wp is not None and a_start is not None and a_end is not None and b_start is not None and b_end is not None
    
    # Validate required parameters for each mode
    if single_segment_mode:
        if wp is None or a_start is None or a_end is None or b_start is None or b_end is None:
            print("Error: In single segment mode, the following parameters are required:")
            print("- wp: Warping path array")
            print("- a_start, a_end: Start and end indices of segment A")
            print("- b_start, b_end: Start and end indices of segment B")
            return None
    
    elif multi_segment_mode:
        if segment_pairs is None or dtw_results is None or segments_a is None or segments_b is None or depth_boundaries_a is None or depth_boundaries_b is None:
            print("Error: In multi-segment mode, the following parameters are required:")
            print("- segment_pairs: List of segment pair indices")
            print("- dtw_results: Dictionary of DTW results")
            print("- segments_a, segments_b: Lists of segments")
            print("- depth_boundaries_a, depth_boundaries_b: Depth boundaries")
            return None
    
    else:
        print("Error: Unable to determine operating mode. Please provide either:")
        print("- For single segment mode: wp, a_start, a_end, b_start, b_end")
        print("- For multi-segment mode: segment_pairs, dtw_results, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b")
        print("- Or explicitly set single_segment_mode parameter")
        return None
    
    # Use default color function if not provided
    if color_function is None:
        color_function = get_yl_br_color
    
    # Setup figure layout based on presence of images
    has_rgb_a = rgb_img_a is not None
    has_ct_a = ct_img_a is not None
    has_rgb_b = rgb_img_b is not None
    has_ct_b = ct_img_b is not None
    
    img_rows_a = sum([has_rgb_a, has_ct_a])
    img_rows_b = sum([has_rgb_b, has_ct_b])
    max_img_rows = max(img_rows_a, img_rows_b)
    
    # Resolve correlation figure size
    _corr_w = correlation_figsize[0] if correlation_figsize else 6
    _corr_h = correlation_figsize[1] if correlation_figsize else 20

    # Create figure with appropriate size
    if max_img_rows > 0:
        fig_height = _corr_h + (max_img_rows * 3)
        fig = plt.figure(figsize=(_corr_w, fig_height))
        gs = GridSpec(max_img_rows + 1, 2, height_ratios=[1]*max_img_rows + [3])
        
        # Create image axes
        current_row = 0
        
        # Core A images
        if has_rgb_a:
            ax_rgb_a = plt.subplot(gs[current_row, 0])
            ax_rgb_a.imshow(rgb_img_a.transpose(1, 0, 2), aspect='auto', 
                           extent=[0, 1, np.max(md_a), np.min(md_a)])
            ax_rgb_a.set_ylabel('RGB A')
            ax_rgb_a.set_xticks([])
            current_row += 1
        
        if has_ct_a and current_row < max_img_rows:
            ax_ct_a = plt.subplot(gs[current_row, 0])
            ct_display = ct_img_a.transpose(1, 0, 2) if len(ct_img_a.shape) == 3 else ct_img_a.transpose()
            ax_ct_a.imshow(ct_display, aspect='auto', extent=[0, 1, np.max(md_a), np.min(md_a)], cmap='gray')
            ax_ct_a.set_ylabel('CT A')
            ax_ct_a.set_xticks([])
            current_row += 1
        
        # Core B images
        current_row = 0
        if has_rgb_b:
            ax_rgb_b = plt.subplot(gs[current_row, 1])
            ax_rgb_b.imshow(rgb_img_b.transpose(1, 0, 2), aspect='auto', 
                           extent=[2, 3, np.max(md_b), np.min(md_b)])
            ax_rgb_b.set_ylabel('RGB B')
            ax_rgb_b.set_xticks([])
            current_row += 1
        
        if has_ct_b and current_row < max_img_rows:
            ax_ct_b = plt.subplot(gs[current_row, 1])
            ct_display = ct_img_b.transpose(1, 0, 2) if len(ct_img_b.shape) == 3 else ct_img_b.transpose()
            ax_ct_b.imshow(ct_display, aspect='auto', extent=[2, 3, np.max(md_b), np.min(md_b)], cmap='gray')
            ax_ct_b.set_ylabel('CT B')
            ax_ct_b.set_xticks([])
            current_row += 1
        
        ax = plt.subplot(gs[-1, :])
    else:
        fig = plt.figure(figsize=(_corr_w, _corr_h))
        ax = fig.add_subplot(111)
    
    # Plot log data
    if is_multilog_a:
        column_names_a = available_columns_a if available_columns_a else [f"Log A{i+1}" for i in range(log_a.shape[1])]
        for i, col_name in enumerate(column_names_a):
            style = color_style_map.get(col_name, {'color': f'C{i}', 'linestyle': '-'})
            ax.plot(log_a[:, i], md_a, color=style['color'], linestyle=style['linestyle'], 
                   linewidth=1, label=f'A: {col_name}')
    else:
        ax.plot(log_a, md_a, 'b', linewidth=1)
    
    if is_multilog_b:
        column_names_b = available_columns_b if available_columns_b else [f"Log B{i+1}" for i in range(log_b.shape[1])]
        for i, col_name in enumerate(column_names_b):
            style = color_style_map.get(col_name, {'color': f'C{i}', 'linestyle': '-'})
            ax.plot(log_b[:, i] + 2, md_b, color=style['color'], linestyle=style['linestyle'], 
                   linewidth=1, label=f'B: {col_name}')
    else:
        ax.plot(log_b + 2, md_b, 'b', linewidth=1)
    
    # Add dividing lines and set limits
    ax.plot([1, 1], [0, np.max(md_a)], 'k', linewidth=0.5)
    ax.plot([2, 2], [0, np.max(md_b)], 'k', linewidth=0.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, max(np.max(md_a), np.max(md_b)))
    
    # Prepare logs for coloring (invert specified dimensions before averaging)
    if invert_colormap is not None and is_multilog_a:
        log_a_color = log_a.copy()
        for dim_i, inv in enumerate(invert_colormap):
            if (inv == 'True' or inv is True) and dim_i < log_a_color.shape[1]:
                log_a_color[:, dim_i] = 1 - log_a_color[:, dim_i]
        log_a_inv = np.mean(log_a_color, axis=1)
    elif is_multilog_a:
        log_a_inv = np.mean(log_a, axis=1)
    elif invert_colormap is not None and len(invert_colormap) > 0 and (invert_colormap[0] == 'True' or invert_colormap[0] is True):
        log_a_inv = 1 - log_a
    else:
        log_a_inv = log_a

    if invert_colormap is not None and is_multilog_b:
        log_b_color = log_b.copy()
        for dim_i, inv in enumerate(invert_colormap):
            if (inv == 'True' or inv is True) and dim_i < log_b_color.shape[1]:
                log_b_color[:, dim_i] = 1 - log_b_color[:, dim_i]
        log_b_inv = np.mean(log_b_color, axis=1)
    elif is_multilog_b:
        log_b_inv = np.mean(log_b, axis=1)
    elif invert_colormap is not None and len(invert_colormap) > 0 and (invert_colormap[0] == 'True' or invert_colormap[0] is True):
        log_b_inv = 1 - log_b
    else:
        log_b_inv = log_b

    # Rescale to [0, 1] for multi-log: averaging compressed the range
    if is_multilog_a or is_multilog_b:
        combined_min = min(np.nanmin(log_a_inv), np.nanmin(log_b_inv))
        combined_max = max(np.nanmax(log_a_inv), np.nanmax(log_b_inv))
        if combined_max > combined_min:
            log_a_inv = (log_a_inv - combined_min) / (combined_max - combined_min)
            log_b_inv = (log_b_inv - combined_min) / (combined_max - combined_min)
    
    # Add legend for multilogs
    if is_multilog_a or is_multilog_b:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    def adapt_step_size(step, wp):
        """Adapt step size based on warping path length."""
        if wp is None:
            return step
            
        local_step = step
        while len(wp) <= local_step and local_step > 1:
            local_step = local_step // 2
        return local_step
    
    def visualize_single_point_segment(wp, a_start, b_start, log_a_inv, log_b_inv, md_a, md_b, color_function):
        """Handle visualization of single point segments."""
        if wp is None:
            return
            
        if a_end - a_start == 0:  # log_a has single point
            single_idx = a_start
            single_depth = md_a[single_idx]
            single_value = log_a_inv[single_idx]
            
            wp_filtered = wp[wp[:, 0] == a_start] if len(wp) > 0 else wp
            for i in range(len(wp_filtered)):
                if i+1 >= len(wp_filtered):
                    continue
                    
                b_idx = min(max(0, int(wp_filtered[i, 1])), len(md_b)-1)
                b_depth = md_b[b_idx]
                b_value = log_b_inv[b_idx]
                
                next_b_idx = min(max(0, int(wp_filtered[i+1, 1])), len(md_b)-1)
                next_b_depth = md_b[next_b_idx]
                
                # Draw filled polygons for correlation
                x = [2, 3, 3, 2]
                y = [b_depth, b_depth, next_b_depth, next_b_depth]
                b_fill_value = np.mean(log_b_inv[min(b_idx,next_b_idx):max(b_idx,next_b_idx)+1])
                fill_color = color_function(b_fill_value)
                ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
                
                mean_value = (single_value + b_value) * 0.5
                x = [1, 2, 2, 1]
                y = [single_depth, b_depth, next_b_depth, single_depth]
                fill_color = color_function(mean_value)
                ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
        
        else:  # log_b has single point
            single_idx = b_start
            single_depth = md_b[single_idx]
            single_value = log_b_inv[single_idx]
            
            wp_filtered = wp[wp[:, 1] == b_start] if len(wp) > 0 else wp
            for i in range(len(wp_filtered)):
                if i+1 >= len(wp_filtered):
                    continue
                    
                a_idx = min(max(0, int(wp_filtered[i, 0])), len(md_a)-1)
                a_depth = md_a[a_idx]
                a_value = log_a_inv[a_idx]
                
                next_a_idx = min(max(0, int(wp_filtered[i+1, 0])), len(md_a)-1)
                next_a_depth = md_a[next_a_idx]
                
                # Draw filled polygons for correlation
                x = [0, 1, 1, 0]
                y = [a_depth, a_depth, next_a_depth, next_a_depth]
                a_fill_value = np.mean(log_a_inv[min(a_idx,next_a_idx):max(a_idx,next_a_idx)+1])
                fill_color = color_function(a_fill_value)
                ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
                
                mean_value = (single_value + a_value) * 0.5
                x = [1, 2, 2, 1]
                y = [a_depth, single_depth, single_depth, next_a_depth]
                fill_color = color_function(mean_value)
                ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
    
    def visualize_normal_segments(wp, a_start, a_end, b_start, b_end, log_a_inv, log_b_inv, md_a, md_b, step_size, color_function):
        """Visualize normal (non-single-point) segments with correlation coloring."""
        if wp is None:
            return
            
        # Filter wp to segment if in multi-segment mode
        if multi_segment_mode:
            mask = ((wp[:, 0] >= a_start) & (wp[:, 0] <= a_end) & 
                    (wp[:, 1] >= b_start) & (wp[:, 1] <= b_end))
            wp_segment = wp[mask]
        else:
            wp_segment = wp
            
        if len(wp_segment) < 2:
            return
        
        # Draw correlation intervals with proper coloring
        i_max = -1
        for i in range(0, len(wp_segment)-step_size, step_size):
            try:
                i_max = i
                
                # Ensure indices are within valid ranges
                p_i = min(max(0, int(wp_segment[i, 0])), len(md_a)-1)
                p_i_step = min(max(0, int(wp_segment[i+step_size, 0])), len(md_a)-1)
                q_i = min(max(0, int(wp_segment[i, 1])), len(md_b)-1)
                q_i_step = min(max(0, int(wp_segment[i+step_size, 1])), len(md_b)-1)
                
                # Draw correlation intervals for both logs
                depth1_base = md_a[p_i]
                depth1_top = md_a[p_i_step]
                if p_i_step < p_i:
                    mean_log1 = np.mean(log_a_inv[p_i_step:p_i+1])
                    x = [0, 1, 1, 0]
                    y = [depth1_base, depth1_base, depth1_top, depth1_top]
                    fill_color = color_function(mean_log1)
                    ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
                else:
                    mean_log1 = log_a_inv[p_i]
                    
                depth2_base = md_b[q_i]
                depth2_top = md_b[q_i_step]
                if q_i_step < q_i:  
                    mean_log2 = np.mean(log_b_inv[q_i_step:q_i+1])
                    x = [2, 3, 3, 2]
                    y = [depth2_base, depth2_base, depth2_top, depth2_top]
                    fill_color = color_function(mean_log2)
                    ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
                else:
                    mean_log2 = log_b_inv[q_i]
                    
                # Draw correlation between logs
                if (p_i_step < p_i) or (q_i_step < q_i):
                    mean_logs = (mean_log1 + mean_log2)*0.5
                    x = [1, 2, 2, 1]
                    y = [depth1_base, depth2_base, depth2_top, depth1_top]
                    fill_color = color_function(mean_logs)
                    ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
            except Exception as e:
                print(f"Error processing segment at i={i}: {e}")
        
        # Draw the last segment if needed
        if len(wp_segment) > step_size and i_max >= 0:
            try:
                i = i_max
                
                p_i_step = min(max(0, int(wp_segment[i+step_size, 0])), len(md_a)-1)
                p_last = min(max(0, int(wp_segment[-1, 0])), len(md_a)-1)
                q_i_step = min(max(0, int(wp_segment[i+step_size, 1])), len(md_b)-1)
                q_last = min(max(0, int(wp_segment[-1, 1])), len(md_b)-1)
                
                depth1_base = md_a[p_i_step]
                depth1_top = md_a[p_last]
                if p_last < p_i_step:
                    mean_log1 = np.mean(log_a_inv[p_last:p_i_step+1])
                    x = [0, 1, 1, 0]
                    y = [depth1_base, depth1_base, depth1_top, depth1_top]
                    fill_color = color_function(mean_log1)
                    ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
                
                depth2_base = md_b[q_i_step]
                depth2_top = md_b[q_last]
                if q_last < q_i_step:  
                    mean_log2 = np.mean(log_b_inv[q_last:q_i_step+1])
                    x = [2, 3, 3, 2]
                    y = [depth2_base, depth2_base, depth2_top, depth2_top]
                    fill_color = color_function(mean_log2)
                    ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
                
                if (p_last < p_i_step) or (q_last < q_i_step):
                    mean_logs = (mean_log1 + mean_log2)*0.5
                    x = [1, 2, 2, 1]
                    y = [depth1_base, depth2_base, depth2_top, depth1_top]
                    fill_color = color_function(mean_logs)
                    ax.fill(x, y, facecolor=fill_color, edgecolor=fill_color, linewidth=0)
            except Exception as e:
                print(f"Error processing last segment: {e}")
    
    def add_age_constraints(constraint_depths, constraint_ages, constraint_pos_errors, constraint_neg_errors, 
                           md_array, is_core_a=True):
        """Add age constraint markers and annotations to the plot."""
        if constraint_depths is None or constraint_ages is None:
            return
            
        constraint_depths = np.array(constraint_depths) if not isinstance(constraint_depths, np.ndarray) else constraint_depths
        constraint_ages = np.array(constraint_ages) if not isinstance(constraint_ages, np.ndarray) else constraint_ages
        constraint_pos_errors = np.array(constraint_pos_errors) if not isinstance(constraint_pos_errors, np.ndarray) else constraint_pos_errors
        constraint_neg_errors = np.array(constraint_neg_errors) if not isinstance(constraint_neg_errors, np.ndarray) else constraint_neg_errors
        
        # Set position based on core A or B
        xmin = 0.0 if is_core_a else 0.67
        xmax = 0.33 if is_core_a else 1.0
        text_pos = 0.5 if is_core_a else 2.5
        
        for i, depth_cm in enumerate(constraint_depths):
            nearest_idx = np.argmin(np.abs(md_array - depth_cm))
            adj_depth = md_array[nearest_idx]
            
            # Draw red horizontal line at constraint depth
            ax.axhline(y=adj_depth, xmin=xmin, xmax=xmax, color='r', linestyle='--', linewidth=1)
            
            # Add age annotation
            age_text = f"{constraint_ages[i]:.0f} ({constraint_pos_errors[i]:.0f}/{constraint_neg_errors[i]:.0f})"
            ax.text(text_pos, adj_depth+5, age_text, fontsize=8, color='r', ha='center', va='top',
                   bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    def add_picked_depths(picked_datum, md_array, ages=None, is_core_a=True):
        """Add picked depth markers and age annotations to the plot."""
        if picked_datum is None or len(picked_datum) == 0:
            return
        
        xmin = 0.0 if is_core_a else 0.67
        xmax = 0.33 if is_core_a else 1.0
        text_pos = 0.5 if is_core_a else 2.5
        
        for depth in picked_datum:
            # Handle tuple case (depth, category)
            if isinstance(depth, tuple) and len(depth) >= 1:
                depth_value = depth[0]
            else:
                depth_value = depth
            
            # Convert to actual depth value
            if isinstance(depth_value, (int, np.integer)) and depth_value < len(md_array):
                adj_depth = md_array[depth_value]
            else:
                try:
                    adj_depth = float(depth_value)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {depth_value} to a valid depth. Skipping.")
                    continue
            
            # Draw horizontal line at picked depth
            ax.axhline(y=adj_depth, xmin=xmin, xmax=xmax, color='black', linestyle=':', linewidth=1)
            
            # Add age annotation if enabled
            if mark_ages and ages and 'depths' in ages and 'ages' in ages:
                age_depths = np.array(ages['depths'])
                closest_idx = np.argmin(np.abs(age_depths - adj_depth))
                age = ages['ages'][closest_idx]
                pos_err = ages['pos_uncertainties'][closest_idx]
                neg_err = ages['neg_uncertainties'][closest_idx]
                
                age_text = f"{age:.0f} (+{pos_err:.0f}/-{neg_err:.0f})"
                ax.text(text_pos, adj_depth-2, age_text, fontsize=7, color='black', ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    def visualize_segment_pair(wp, a_start, a_end, b_start, b_end, color=None, segment_label=None):
        """Process visualization for a single segment pair."""
        # Highlight the segments being correlated
        segment_a_depths = md_a[a_start:a_end+1]
        segment_b_depths = md_b[b_start:b_end+1]
        
        # Highlight segments
        if len(segment_a_depths) > 0:
            ax.axhspan(min(segment_a_depths), max(segment_a_depths), 
                      xmin=0, xmax=0.33, alpha=0.2, 
                      color='green' if color is None else color)
        
        if len(segment_b_depths) > 0:
            ax.axhspan(min(segment_b_depths), max(segment_b_depths), 
                      xmin=0.67, xmax=1.0, alpha=0.2, 
                      color='green' if color is None else color)
        
        # Add segment labels if provided
        if segment_label is not None:
            center_a = (min(segment_a_depths) + max(segment_a_depths)) / 2 if len(segment_a_depths) > 0 else a_start
            center_b = (min(segment_b_depths) + max(segment_b_depths)) / 2 if len(segment_b_depths) > 0 else b_start
            
            ax.text(0.5, center_a, segment_label, color=color, fontweight='bold', ha='center')
            ax.text(2.5, center_b, segment_label, color=color, fontweight='bold', ha='center')

        if wp is None:
            return

        # Handle different segment types
        if a_end - a_start == 0 or b_end - b_start == 0:
            try:
                visualize_single_point_segment(wp, a_start, b_start, log_a_inv, log_b_inv, md_a, md_b, color_function)
            except Exception as e:
                print(f"Error processing single-point correlation: {e}")
        else:
            local_step = adapt_step_size(step, wp)
            visualize_normal_segments(wp, a_start, a_end, b_start, b_end, log_a_inv, log_b_inv, md_a, md_b, local_step, color_function)

    # Main visualization logic
    if single_segment_mode:
        visualize_segment_pair(wp, a_start, a_end, b_start, b_end)
    else:
        if visualize_pairs:
            # Highlight each segment pair with unique color
            for idx, (a_idx, b_idx) in enumerate(segment_pairs):
                color = plt.cm.Set1(idx % 9)
                
                a_start = depth_boundaries_a[segments_a[a_idx][0]]
                a_end = depth_boundaries_a[segments_a[a_idx][1]]
                b_start = depth_boundaries_b[segments_b[b_idx][0]]
                b_end = depth_boundaries_b[segments_b[b_idx][1]]
                
                segment_label = f"({a_idx+1}, {b_idx+1})" if visualize_segment_labels else None
                
                visualize_segment_pair(None, a_start, a_end, b_start, b_end, color=color, segment_label=segment_label)
                
                # Draw warping path for visualization
                paths, _, _ = dtw_results.get((a_idx, b_idx), ([], [], []))
                if paths and len(paths) > 0:
                    wp_segment = paths[0]
                    if len(wp_segment) > 0:
                        mask = ((wp_segment[:, 0] >= a_start) & (wp_segment[:, 0] <= a_end) & 
                                (wp_segment[:, 1] >= b_start) & (wp_segment[:, 1] <= b_end))
                        wp_segment = wp_segment[mask]
                        
                        if len(wp_segment) > 0:
                            step_size = max(1, len(wp_segment) // 15)
                            for i in range(0, len(wp_segment), step_size):
                                p_idx = int(wp_segment[i, 0])
                                q_idx = int(wp_segment[i, 1])
                                p_depth = md_a[p_idx]
                                q_depth = md_b[q_idx]
                                plt.plot([1, 2], [p_depth, q_depth], color=color, linestyle=':', linewidth=0.7)
        else:
            # Process each segment pair with coloring
            for a_idx, b_idx in segment_pairs:
                a_start = depth_boundaries_a[segments_a[a_idx][0]]
                a_end = depth_boundaries_a[segments_a[a_idx][1]]
                b_start = depth_boundaries_b[segments_b[b_idx][0]]
                b_end = depth_boundaries_b[segments_b[b_idx][1]]
                
                paths, _, _ = dtw_results.get((a_idx, b_idx), ([], [], []))
                if not paths or len(paths) == 0:
                    continue
                    
                wp = paths[0]
                
                if a_end - a_start == 0 or b_end - b_start == 0:
                    visualize_single_point_segment(wp, a_start, b_start, log_a_inv, log_b_inv, md_a, md_b, color_function)
                else:
                    local_step = adapt_step_size(step, wp)
                    visualize_normal_segments(wp, a_start, a_end, b_start, b_end, log_a_inv, log_b_inv, md_a, md_b, local_step, color_function)
    
    # Add age constraint markers
    if mark_ages and age_consideration and all_constraint_depths_a is not None and all_constraint_ages_a is not None:
        add_age_constraints(
            all_constraint_depths_a, all_constraint_ages_a, 
            all_constraint_pos_errors_a, all_constraint_neg_errors_a, 
            md_a, is_core_a=True
        )

    if mark_ages and age_consideration and all_constraint_depths_b is not None and all_constraint_ages_b is not None:
        add_age_constraints(
            all_constraint_depths_b, all_constraint_ages_b, 
            all_constraint_pos_errors_b, all_constraint_neg_errors_b, 
            md_b, is_core_a=False
        )
    
    # Add picked depth markers
    if mark_depths:
        if multi_segment_mode:
            if picked_datum_a is not None:
                add_picked_depths(picked_datum_a, md_a, datum_ages_a if mark_ages else None, is_core_a=True)
            elif depth_boundaries_a is not None:
                converted_depths_a = [md_a[idx] for idx in depth_boundaries_a]
                add_picked_depths(converted_depths_a, md_a, datum_ages_a if mark_ages else None, is_core_a=True)
            
            if picked_datum_b is not None:
                add_picked_depths(picked_datum_b, md_b, datum_ages_b if mark_ages else None, is_core_a=False)
            elif depth_boundaries_b is not None:
                converted_depths_b = [md_b[idx] for idx in depth_boundaries_b]
                add_picked_depths(converted_depths_b, md_b, datum_ages_b if mark_ages else None, is_core_a=False)
        else:
            add_picked_depths(picked_datum_a, md_a, datum_ages_a if mark_ages else None, is_core_a=True)
            add_picked_depths(picked_datum_b, md_b, datum_ages_b if mark_ages else None, is_core_a=False)
    
    # Setup axes and labels
    ax.set_xticks([])
    ax.invert_yaxis()
    
    # Create depth labels
    labels = []
    start_depth = 0
    end_depth = np.floor(np.max(md_a)/100) * 100
    for label in np.arange(start_depth, end_depth+1, 100):
        labels.append(str(int(label)))
    ax.set_yticks(np.arange(start_depth, end_depth+1, 100))
    ax.set_yticklabels(labels)
    ax.set_ylabel('depth (cm)', fontsize=12)
    
    # Create depth labels for core B on right side
    ax2 = ax.twinx()
    labels = []
    start_depth = 0
    end_depth = np.floor(np.max(md_b)/100) * 100
    for label in np.arange(start_depth, end_depth+1, 100):
        labels.append(str(int(label)))
    ax2.set_yticks(np.arange(start_depth, end_depth+1, 100))
    ax2.set_yticklabels(labels)
    ax2.set_ylim(0, max(np.max(md_a), np.max(md_b)))
    ax2.invert_yaxis()

    # Add legend for age markers
    if mark_ages:
        legend_elements = [
            Line2D([0], [0], color='black', linestyle=':', label='Selected Depths'),
            Line2D([0], [0], color='red', linestyle='--', label='Age Constraints'),
        ]
        ax.legend(handles=legend_elements, loc='lower center', fontsize=8, title="Ages (Year BP)")
    
    # Add quality indicators (only when show_quality_indicators is True)
    if show_quality_indicators and (quality_indicators is not None or combined_quality is not None):
        qi = combined_quality if multi_segment_mode else quality_indicators
        if qi:
            quality_text = (
                "DTW Quality Indicators: \n"
                f"Normalized DTW Cost: {qi.get('norm_dtw', 0):.3f} (lower is better)\n "
                f"DTW Warping Ratio: {qi.get('dtw_ratio', 0):.3f} (lower is better)\n"
                f"Diagonality: {qi.get('perc_diag', 0):.1f}% (higher is better)\n"
                f"DTW Warping Efficiency: {qi.get('dtw_warp_eff', 0):.1f}% (higher is better)\n"
                f"Pearson's r: {qi.get('corr_coef', 0):.3f} (higher is better)\n"
                f"Age Overlap: {qi.get('perc_age_overlap', 0):.1f}% (higher is better)"
            )
            plt.figtext(0.97, 0.97, quality_text, 
                       fontsize=12, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))

    # Save figure if path provided
    if save_path:
        from .helpers import save_figure_formats
        save_dpi = dpi if dpi is not None else 150
        save_figure_formats(fig, save_path, fig_format=fig_format, dpi=save_dpi)
    
    return fig


def plot_multilog_segment_pair_correlation(log_a, log_b, md_a, md_b, 
                                  wp, a_start, a_end, b_start, b_end,
                                  step=5, quality_indicators=None, 
                                  available_columns=None,
                                  rgb_img_a=None, ct_img_a=None,
                                  rgb_img_b=None, ct_img_b=None,
                                  picked_datum_a=None, picked_datum_b=None,
                                  picked_categories_a=None, picked_categories_b=None,
                                  category_colors=None,
                                  title=None,
                                  show_quality_indicators=False,
                                  invert_colormap=None):
    """
    Plot correlation between two multilogs (multiple log curves) with RGB and CT images.
    
    Parameters
    ----------
    log_a, log_b : array-like
        Multidimensional log data arrays with shape (n_samples, n_logs)
    md_a, md_b : array-like
        Measured depth arrays
    wp : array-like
        Warping path as sequence of index pairs
    a_start, a_end : int
        Start and end indices for segment in log_a
    b_start, b_end : int
        Start and end indices for segment in log_b
    step : int, default=5
        Sampling interval for visualization
    quality_indicators : dict, optional
        Dictionary containing quality indicators
    available_columns : list of str, optional
        Names of the logs being displayed
    rgb_img_a, rgb_img_b : array-like, optional
        RGB images for cores A and B
    ct_img_a, ct_img_b : array-like, optional
        CT images for cores A and B
    picked_datum_a, picked_datum_b : list, optional
        Lists of picked depths to mark on the plots
    picked_categories_a, picked_categories_b : list, optional
        Categories for picked depths (for coloring)
    category_colors : dict, optional
        Mapping of category codes to colors
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
        
    Examples
    --------
    >>> fig = plot_multilog_segment_pair_correlation(
    ...     multilog_a, multilog_b, depths_a, depths_b,
    ...     warping_path, 0, 100, 50, 150,
    ...     available_columns=['R', 'G', 'B', 'MS'],
    ...     rgb_img_a=rgb_image_a, ct_img_a=ct_image_a
    ... )
    """
    
    # Check for images
    has_rgb_a = rgb_img_a is not None
    has_ct_a = ct_img_a is not None
    has_rgb_b = rgb_img_b is not None
    has_ct_b = ct_img_b is not None
    
    num_img_rows_a = (1 if has_rgb_a else 0) + (1 if has_ct_a else 0)
    num_img_rows_b = (1 if has_rgb_b else 0) + (1 if has_ct_b else 0)
    num_img_rows = max(num_img_rows_a, num_img_rows_b)
    
    # Define color and line style mapping for each column
    color_style_map = {
        'R': {'color': 'red', 'linestyle': '--'},
        'G': {'color': 'green', 'linestyle': '--'},
        'B': {'color': 'blue', 'linestyle': '--'},
        'Lumin': {'color': 'darkgray', 'linestyle': '--'},
        'hiresMS': {'color': 'black', 'linestyle': '-'},
        'MS': {'color': 'gray', 'linestyle': '-'},
        'Den_gm/cc': {'color': 'orange', 'linestyle': '-'},
        'CT': {'color': 'purple', 'linestyle': '-'}
    }
    
    # Default category colors if not provided
    if category_colors is None:
        category_colors = {
            1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 
            5: 'orange', 6: 'cyan', 7: 'magenta', 8: 'yellow', 9: 'black'
        }
    
    # Function for yellow-brown color mapping for correlation intervals
    def get_yl_br_color(log_value):
        color = np.array([1-0.4*log_value, 1-0.7*log_value, 0.6-0.6*log_value])
        color[color > 1] = 1
        color[color < 0] = 0
        return color
    
    # Create figure with appropriate height
    fig_height = 20 + 4 * num_img_rows  # Base height plus space for images
    fig = plt.figure(figsize=(12, fig_height))
    
    # Create GridSpec for layout
    if num_img_rows > 0:
        gs = gridspec.GridSpec(num_img_rows + 1, 2, height_ratios=[1] * num_img_rows + [5])
        
        # Setup image subplots for core A
        img_row = 0
        if has_rgb_a:
            ax_rgb_a = fig.add_subplot(gs[img_row, 0])
            ax_rgb_a.imshow(rgb_img_a, aspect='auto', extent=[0, 1, np.max(md_a), np.min(md_a)])
            ax_rgb_a.set_title(f"RGB Image - Core A")
            ax_rgb_a.set_xticks([])
            img_row += 1
        
        if has_ct_a:
            ax_ct_a = fig.add_subplot(gs[img_row if img_row < num_img_rows else 0, 0])
            ax_ct_a.imshow(ct_img_a, aspect='auto', extent=[0, 1, np.max(md_a), np.min(md_a)], cmap='gray')
            ax_ct_a.set_title(f"CT Image - Core A")
            ax_ct_a.set_xticks([])
            
        # Setup image subplots for core B
        img_row = 0
        if has_rgb_b:
            ax_rgb_b = fig.add_subplot(gs[img_row, 1])
            ax_rgb_b.imshow(rgb_img_b, aspect='auto', extent=[0, 1, np.max(md_b), np.min(md_b)])
            ax_rgb_b.set_title(f"RGB Image - Core B")
            ax_rgb_b.set_xticks([])
            img_row += 1
        
        if has_ct_b:
            ax_ct_b = fig.add_subplot(gs[img_row if img_row < num_img_rows else 0, 1])
            ax_ct_b.imshow(ct_img_b, aspect='auto', extent=[0, 1, np.max(md_b), np.min(md_b)], cmap='gray')
            ax_ct_b.set_title(f"CT Image - Core B")
            ax_ct_b.set_xticks([])
        
        # Main correlation plot
        ax = fig.add_subplot(gs[-1, :])
    else:
        # Just create a single plot if no images
        ax = fig.add_subplot(111)
    
    # Plot each log type with appropriate color/style
    if log_a.ndim > 1 and log_a.shape[1] > 1:
        # Plot multidimensional logs
        column_names = available_columns if available_columns else [f"Log {i+1}" for i in range(log_a.shape[1])]
        
        for i, col_name in enumerate(column_names):
            if col_name in color_style_map:
                color = color_style_map[col_name]['color']
                linestyle = color_style_map[col_name]['linestyle']
            else:
                color = f'C{i}'
                linestyle = '-'
                
            # Plot log for core A
            ax.plot(log_a[:, i], md_a, color=color, linestyle=linestyle, 
                   linewidth=1, label=f'Core A {col_name}')
            
            # Plot log for core B (shifted to the right)
            ax.plot(log_b[:, i] + 2, md_b, color=color, linestyle=linestyle, 
                   linewidth=1, label=f'Core B {col_name}')
    else:
        # Plot single-dimensional logs
        ax.plot(log_a, md_a, 'b', linewidth=1)
        ax.plot(log_b + 2, md_b, 'b', linewidth=1)
    
    # Add vertical lines to separate logs
    ax.plot([1, 1], [np.min(md_a), np.max(md_a)], 'k', linewidth=0.5)
    ax.plot([2, 2], [np.min(md_b), np.max(md_b)], 'k', linewidth=0.5)
    
    # Set plot limits
    ax.set_xlim(0, 3)
    ax.set_ylim(min(np.min(md_a), np.min(md_b)), max(np.max(md_a), np.max(md_b)))
    
    # Draw correlation intervals using avg across all dimensions for coloring
    # Invert specified dimensions before averaging
    if invert_colormap is not None and log_a.ndim > 1:
        log_a_color = log_a.copy()
        for dim_i, inv in enumerate(invert_colormap):
            if (inv == 'True' or inv is True) and dim_i < log_a_color.shape[1]:
                log_a_color[:, dim_i] = 1 - log_a_color[:, dim_i]
        log_a_inv = np.mean(log_a_color, axis=1)
    elif log_a.ndim > 1:
        log_a_inv = np.mean(log_a, axis=1)
    elif invert_colormap is not None and len(invert_colormap) > 0 and (invert_colormap[0] == 'True' or invert_colormap[0] is True):
        log_a_inv = 1 - log_a
    else:
        log_a_inv = 1 - log_a

    if invert_colormap is not None and log_b.ndim > 1:
        log_b_color = log_b.copy()
        for dim_i, inv in enumerate(invert_colormap):
            if (inv == 'True' or inv is True) and dim_i < log_b_color.shape[1]:
                log_b_color[:, dim_i] = 1 - log_b_color[:, dim_i]
        log_b_inv = np.mean(log_b_color, axis=1)
    elif log_b.ndim > 1:
        log_b_inv = np.mean(log_b, axis=1)
    elif invert_colormap is not None and len(invert_colormap) > 0 and (invert_colormap[0] == 'True' or invert_colormap[0] is True):
        log_b_inv = 1 - log_b
    else:
        log_b_inv = 1 - log_b

    # Rescale to [0, 1] for multi-log: averaging compressed the range
    if log_a.ndim > 1 or log_b.ndim > 1:
        combined_min = min(np.nanmin(log_a_inv), np.nanmin(log_b_inv))
        combined_max = max(np.nanmax(log_a_inv), np.nanmax(log_b_inv))
        if combined_max > combined_min:
            log_a_inv = (log_a_inv - combined_min) / (combined_max - combined_min)
            log_b_inv = (log_b_inv - combined_min) / (combined_max - combined_min)
    
    # If we have a valid warping path, process it
    if wp is not None and len(wp) > 0:
        # Adjust the step size if necessary
        effective_step = min(step, len(wp) // 10) if len(wp) > 0 else step
        effective_step = max(1, effective_step)
        
        # Draw correlation intervals (similar to Testing9 but with avg across dimensions)
        for i in range(0, len(wp)-effective_step, effective_step):
            # Get indices from warping path
            p_i = min(max(0, int(wp[i, 0])), len(md_a)-1)
            p_i_step = min(max(0, int(wp[i+effective_step, 0])), len(md_a)-1)
            q_i = min(max(0, int(wp[i, 1])), len(md_b)-1)
            q_i_step = min(max(0, int(wp[i+effective_step, 1])), len(md_b)-1)
            
            # Get corresponding depths
            depth1_base = md_a[p_i]
            depth1_top = md_a[p_i_step]
            depth2_base = md_b[q_i]
            depth2_top = md_b[q_i_step]
            
            # Fill intervals for log A
            if p_i_step < p_i:
                mean_log1 = np.mean(log_a_inv[p_i_step:p_i+1])
                x = [0, 1, 1, 0]
                y = [depth1_base, depth1_base, depth1_top, depth1_top]
                ax.fill(x, y, facecolor=get_yl_br_color(mean_log1), edgecolor=None)
            else:
                mean_log1 = log_a_inv[p_i]
                
            # Fill intervals for log B
            if q_i_step < q_i:
                mean_log2 = np.mean(log_b_inv[q_i_step:q_i+1])
                x = [2, 3, 3, 2]
                y = [depth2_base, depth2_base, depth2_top, depth2_top]
                ax.fill(x, y, facecolor=get_yl_br_color(mean_log2), edgecolor=None)
            else:
                mean_log2 = log_b_inv[q_i]
                
            # Fill intervals between the logs
            if (p_i_step < p_i) or (q_i_step < q_i):
                mean_logs = (mean_log1 + mean_log2) * 0.5
                x = [1, 2, 2, 1]
                y = [depth1_base, depth2_base, depth2_top, depth1_top]
                ax.fill(x, y, facecolor=get_yl_br_color(mean_logs), edgecolor=None)
    
    # Add picked depths if provided
    if picked_datum_a is not None:
        for i, depth in enumerate(picked_datum_a):
            # Get the category if available
            category = 1  # Default category
            if picked_categories_a and i < len(picked_categories_a):
                category = picked_categories_a[i]
            
            # Get color for this category
            color = category_colors.get(category, 'red')
            
            # Add a line at this depth
            ax.axhline(y=depth, xmin=0, xmax=0.33, color=color, linestyle='--', linewidth=1.0)
            
            # Add category label
            ax.text(0.1, depth, f"#{category}", fontsize=8, color=color, 
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    if picked_datum_b is not None:
        for i, depth in enumerate(picked_datum_b):
            # Get the category if available
            category = 1  # Default category
            if picked_categories_b and i < len(picked_categories_b):
                category = picked_categories_b[i]
            
            # Get color for this category
            color = category_colors.get(category, 'red')
            
            # Add a line at this depth
            ax.axhline(y=depth, xmin=0.67, xmax=1.0, color=color, linestyle='--', linewidth=1.0)
            
            # Add category label
            ax.text(2.9, depth, f"#{category}", fontsize=8, color=color, 
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Add quality indicators if provided (only when show_quality_indicators is True)
    if show_quality_indicators and quality_indicators is not None:
        quality_text = (
            "DTW Quality Indicators: \n"
            f"Normalized DTW Cost: {quality_indicators.get('norm_dtw', 0):.3f} (lower is better)\n "
            f"DTW Warping Ratio: {quality_indicators.get('dtw_ratio', 0):.3f} (lower is better)\n"
            f"Warping Deviation: variance {quality_indicators.get('variance_deviation', 0):.2f} (lower is better)\n"
            f"Diagonality: {quality_indicators.get('perc_diag', 0):.1f}% (higher is better)\n"
            f"Pearson's r: {quality_indicators.get('corr_coef', 0):.3f} (higher is better)\n"
            f"Matching Function: {quality_indicators.get('match_min', 0):.3f}; mean {quality_indicators.get('match_mean', 0):.3f} (lower is better)\n"
            f"Age Overlap: {quality_indicators.get('perc_age_overlap', 0):.1f}% (higher is better)"
        )
        plt.figtext(0.97, 0.97, quality_text, 
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend for log curves
    if log_a.ndim > 1 and log_a.shape[1] > 1:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',
                 ncol=len(available_columns) if available_columns else 1)
    
    # Add title
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    else:
        plt.suptitle(f"Multilog Correlation with {len(available_columns) if available_columns else 1} Log Types", 
                     fontsize=16, y=0.98)
    
    # Invert y-axis for proper depth display
    ax.invert_yaxis()
    
    # Add depth labels
    ax.set_ylabel('Depth (cm)', fontsize=12)
    ax.set_xticks([])  # Hide x-ticks
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
    
    return fig


def visualize_combined_segments(dtw_result, log_a, log_b, md_a, md_b, segment_pairs_to_combine, 
                              correlation_save_path='CombinedSegmentPairs_DTW_correlation.png',
                              matrix_save_path='CombinedSegmentPairs_DTW_matrix.png',
                              color_interval_size=None,
                              visualize_pairs=True,
                              visualize_segment_labels=True,
                              mark_depths=True, mark_ages=False,
                              datum_ages_a=None, datum_ages_b=None,
                              core_a_age_data=None, core_b_age_data=None,
                              core_a_name=None,
                              core_b_name=None,
                              # Bed correlation parameters
                              core_a_interpreted_beds=None,
                              core_b_interpreted_beds=None,
                              dpi=None,
                              fig_format=None,
                              correlation_figsize=None,
                              matrix_figsize=None,
                              show_quality_indicators=False,
                              invert_colormap=None):
    """
    Combine selected segment pairs and visualize the results.
    
    Parameters:
    -----------
    dtw_result : dict
        Dictionary containing DTW analysis results from run_comprehensive_dtw_analysis().
        Expected keys: 'dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b',
        'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full'
    log_a, log_b : array-like
        Log data arrays
    md_a, md_b : array-like
        Measured depth arrays
    segment_pairs_to_combine : list
        List of tuples (a_idx, b_idx) for segment pairs to combine
    core_a_interpreted_beds, core_b_interpreted_beds : array-like, optional
        Arrays of interpreted bed names corresponding to depth boundaries (excluding first and last).
        When both provided with matching bed names, correlation lines will be drawn between cores.
    [... other parameters remain the same ...]
        
    Returns:
    --------
    tuple
        (combined_wp, combined_quality)
        - combined_wp : numpy.ndarray
            Combined warping path spanning all selected segments
        - combined_quality : dict
            Aggregated quality metrics for the combined correlation
    """
    
    # Extract variables from unified dictionary
    dtw_results = dtw_result['dtw_correlation']
    valid_dtw_pairs = dtw_result['valid_dtw_pairs']
    segments_a = dtw_result['segments_a']
    segments_b = dtw_result['segments_b']
    depth_boundaries_a = dtw_result['depth_boundaries_a']
    depth_boundaries_b = dtw_result['depth_boundaries_b']
    dtw_distance_matrix_full = dtw_result['dtw_distance_matrix_full']
    
    # Extract age constraint data from core_a_age_data and core_b_age_data if provided
    if core_a_age_data is not None:
        all_constraint_depths_a = core_a_age_data.get('in_sequence_depths')
        all_constraint_ages_a = core_a_age_data.get('in_sequence_ages')
        all_constraint_pos_errors_a = core_a_age_data.get('in_sequence_pos_errors')
        all_constraint_neg_errors_a = core_a_age_data.get('in_sequence_neg_errors')
        age_constraint_a_source_cores = core_a_age_data.get('core')
    else:
        all_constraint_depths_a = None
        all_constraint_ages_a = None
        all_constraint_pos_errors_a = None
        all_constraint_neg_errors_a = None
        age_constraint_a_source_cores = None
    
    if core_b_age_data is not None:
        all_constraint_depths_b = core_b_age_data.get('in_sequence_depths')
        all_constraint_ages_b = core_b_age_data.get('in_sequence_ages')
        all_constraint_pos_errors_b = core_b_age_data.get('in_sequence_pos_errors')
        all_constraint_neg_errors_b = core_b_age_data.get('in_sequence_neg_errors')
        age_constraint_b_source_cores = core_b_age_data.get('core')
    else:
        all_constraint_depths_b = None
        all_constraint_ages_b = None
        all_constraint_pos_errors_b = None
        all_constraint_neg_errors_b = None
        age_constraint_b_source_cores = None
    
    # Helper function: Create a global colormap function that uses the full log data
    def create_global_colormap(log_a, log_b):
        """
        Create a global colormap function based on the full normalized log data.
        
        Parameters:
            log_a (array): First normalized log data (0-1 range)
            log_b (array): Second normalized log data (0-1 range)
            
        Returns:
            function: A function that maps log values to consistent colors
        """
        # Since logs are already normalized to 0-1 range, we don't need to recalculate
        # the range but can directly create a function that maps to the yellow-brown spectrum
        # with consistent coloring across all segments
        
        def global_color_function(log_value):
            """
            Generate a color in the yellow-brown spectrum based on log value using global mapping.
            
            Parameters:
                log_value (float): Value between 0-1 to determine color
                
            Returns:
                array: RGB color values in range 0-1
            """

            log_value = 1 - log_value

            color = np.array([1-0.4*log_value, 1-0.7*log_value, 0.6-0.6*log_value])
            color[color > 1] = 1
            color[color < 0] = 0
            return color
        
        return global_color_function

    # Helper function: Validate and process bed correlation arrays
    def validate_bed_arrays(interpreted_bed_a, interpreted_bed_b, depth_boundaries_a, depth_boundaries_b):
        """
        Validate bed correlation arrays and find matching bed names.
        
        Returns:
            list: List of tuples (bed_name, depth_a, depth_b) for matching beds
        """
        # Check if both arrays are provided
        if interpreted_bed_a is None or interpreted_bed_b is None:
            return []
        
        # Convert to numpy arrays for easier processing
        bed_a = np.array(interpreted_bed_a)
        bed_b = np.array(interpreted_bed_b)
        
        # Check if arrays are empty or contain only empty strings
        if len(bed_a) == 0 or len(bed_b) == 0:
            return []
        
        if all(bed == '' or bed is None for bed in bed_a) or all(bed == '' or bed is None for bed in bed_b):
            return []
        
        # Expected length is number of depth boundaries minus first and last
        expected_len_a = len(depth_boundaries_a) - 2
        expected_len_b = len(depth_boundaries_b) - 2
        
        # Check array lengths
        if len(bed_a) != expected_len_a or len(bed_b) != expected_len_b:
            return []
        
        # Find matching bed names (ignoring empty/None names)
        matches = []
        for i, bed_name_a in enumerate(bed_a):
            if bed_name_a and bed_name_a != '':
                for j, bed_name_b in enumerate(bed_b):
                    if bed_name_b and bed_name_b != '' and bed_name_a == bed_name_b:
                        # Convert bed index to depth boundary index (adding 1 to skip first boundary)
                        depth_idx_a = i + 1
                        depth_idx_b = j + 1
                        
                        # Get actual depth values
                        depth_a = md_a[depth_boundaries_a[depth_idx_a]]
                        depth_b = md_b[depth_boundaries_b[depth_idx_b]]
                        
                        matches.append((bed_name_a, depth_a, depth_b))
                        break  # Only match each bed once
        
        return matches

    # Helper function: Draw bed correlation lines
    def draw_bed_correlations(matches, ax):
        """
        Draw dashed correlation lines between matching beds.
        
        Parameters:
            matches: List of tuples (bed_name, depth_a, depth_b)
            ax: matplotlib axis object
        """
        for bed_name, depth_a, depth_b in matches:
            # Draw dashed black line between the two cores
            ax.plot([1, 2], [depth_a, depth_b], 
                   color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Add bed name label at the middle of the line
            mid_depth = (depth_a + depth_b) / 2
            ax.text(1.5, mid_depth, bed_name, 
                   fontweight='bold', color='black', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='none'))

    # Combine segment pairs
    combined_wp, combined_quality = combine_segment_dtw_results(
        dtw_results, segment_pairs_to_combine, segments_a, segments_b,
        depth_boundaries_a, depth_boundaries_b, log_a, log_b, dtw_distance_matrix_full
    )
    
    if combined_wp is None:
        print("Failed to combine segment pairs. No visualization created.")
        return None, None
    
    # Use color_interval_size if provided, otherwise use default calculation
    if color_interval_size is not None:
        step_size = int(color_interval_size)
    else:
        # Use smaller step size for efficiency
        step_size = max(1, min(5, len(combined_wp) // 5))
    
    # Create the global colormap
    global_color_func = create_global_colormap(log_a, log_b)

    # Create directory structure if needed for save paths
    if correlation_save_path:
        save_dir = os.path.dirname(correlation_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    if matrix_save_path:
        save_dir = os.path.dirname(matrix_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # Create correlation plot in multi-segment mode
    correlation_fig = plot_segment_pair_correlation(
        log_a, log_b, md_a, md_b,
        single_segment_mode=False,  # Explicitly set to multi-segment mode
        # Multi-segment mode parameters
        segment_pairs=segment_pairs_to_combine, 
        dtw_results=dtw_results, 
        segments_a=segments_a, 
        segments_b=segments_b,
        depth_boundaries_a=depth_boundaries_a, 
        depth_boundaries_b=depth_boundaries_b,
        combined_quality=combined_quality,
        # Common parameters
        step=step_size,
        save_path=correlation_save_path,
        visualize_pairs=visualize_pairs,
        visualize_segment_labels=visualize_segment_labels,
        color_function=global_color_func,
        mark_depths=mark_depths,
        mark_ages=mark_ages,
        # Age-related parameters
        age_consideration=mark_ages,  # Set age_consideration to match mark_ages
        datum_ages_a=datum_ages_a,
        datum_ages_b=datum_ages_b,
        all_constraint_depths_a=all_constraint_depths_a,
        all_constraint_depths_b=all_constraint_depths_b,
        all_constraint_ages_a=all_constraint_ages_a,
        all_constraint_ages_b=all_constraint_ages_b,
        all_constraint_pos_errors_a=all_constraint_pos_errors_a,
        all_constraint_pos_errors_b=all_constraint_pos_errors_b,
        all_constraint_neg_errors_a=all_constraint_neg_errors_a,
        all_constraint_neg_errors_b=all_constraint_neg_errors_b,
        dpi=dpi,
        fig_format=fig_format,
        correlation_figsize=correlation_figsize,
        show_quality_indicators=show_quality_indicators,
        invert_colormap=invert_colormap
    )

    # NEW: Add bed correlation lines if conditions are met
    if correlation_fig is not None:
        bed_matches = validate_bed_arrays(core_a_interpreted_beds, core_b_interpreted_beds, 
                                        depth_boundaries_a, depth_boundaries_b)
        
        if bed_matches:
            # Get the main axis from the correlation figure
            ax = correlation_fig.get_axes()[0]  # Assuming main plot is first axis
            draw_bed_correlations(bed_matches, ax)
            
            # Re-save the correlation figure with bed correlation lines
            from .helpers import save_figure_formats
            save_dpi = dpi if dpi is not None else 150
            save_figure_formats(correlation_fig, correlation_save_path, fig_format=fig_format, dpi=save_dpi)

    # Create DTW matrix plot
    plot_dtw_matrix_with_paths(
        dtw_distance_matrix_full, 
        mode='combined_path',
        segment_pairs=segment_pairs_to_combine, 
        dtw_results=dtw_results,
        combined_wp=combined_wp, 
        segments_a=segments_a, 
        segments_b=segments_b,
        depth_boundaries_a=depth_boundaries_a, 
        depth_boundaries_b=depth_boundaries_b,
        output_filename=matrix_save_path,
        visualize_pairs=visualize_pairs,
        visualize_segment_labels=visualize_segment_labels,
        core_a_age_data=core_a_age_data if mark_ages else None,
        core_b_age_data=core_b_age_data if mark_ages else None,
        md_a=md_a,
        md_b=md_b,
        core_a_name=core_a_name,
        core_b_name=core_b_name,
        dpi=dpi,
        fig_format=fig_format,
        matrix_figsize=matrix_figsize
    )

    return combined_wp, combined_quality


def plot_correlation_distribution(mapping_csv, target_mapping_id=None, quality_index=None, save_png=True, png_filename=None, core_a_name=None, core_b_name=None, bin_width=None, pdf_method='normal', kde_bandwidth=0.05, mute_mode=False, targeted_binsize=None, dpi=None, invert_norm_dtw=True, fig_format=None):
    """
    UPDATED: Handle new CSV format with different column structure.
    Plot distribution of a specified quality index.
    
    Parameters:
    - mapping_csv: path to the CSV/Parquet file containing mapping results
    - quality_index: required parameter specifying which quality index to plot
    - target_mapping_id: optional mapping ID to highlight in the plot
    - save_png: whether to save the plot as PNG (default: True)
    - png_filename: optional custom filename for saving PNG
    - bin_width: width of histogram bins (default: None for automatic estimation based on quality_index)
    - pdf_method: str, method for probability density function overlay: 'KDE', 'skew-normal', or 'normal' (default)
    - kde_bandwidth: float, bandwidth for KDE when pdf_method='KDE' (default: 0.05)
    - mute_mode: bool, if True, suppress all print statements (default: False)
    - targeted_binsize: tuple or None, (synthetic_bins, bin_width) for consistent bin sizing with synthetic data (default: None)
    - dpi: int, optional resolution for saved figures in dots per inch. If None, uses default (150)
    - invert_norm_dtw: bool, if True and quality_index is 'norm_dtw', plot 1 - norm_dtw as nDTWc (default: True)
    """
    
    # Define quality index display names and descriptions
    quality_index_mapping = {
        'norm_dtw': 'Normalized DTW Cost',
        'norm_dtw_inverted': r'nDTW$_c$',
        'norm_dtw_sect': 'Normalized DTW Cost (Correlated Section)',
        'dtw_ratio': 'DTW Warping Ratio', 
        'variance_deviation': 'Warping Deviation variance',
        'perc_diag': 'Diagonality %',
        'corr_coef': 'Pearson\'s r',
        'corr_coef_sect': 'Pearson\'s r (Correlated Section)',
        'match_min': 'Matching Function min',
        'match_mean': 'Matching Function mean',
        'perc_age_overlap': 'Age Overlap %'
    }
    
    # Check if quality_index is provided
    if quality_index is None:
        if not mute_mode:
            print("Error: quality_index parameter is required")
            print("Available quality indices: perc_diag, norm_dtw, dtw_ratio, corr_coef, wrapping_deviation, mean_matching_function, perc_age_overlap")
        return None, None, {}
    
    # Load the file - auto-detect format
    try:
        if mapping_csv.lower().endswith('.pkl'):
            df = pd.read_pickle(mapping_csv)
        else:
            df = pd.read_csv(mapping_csv)
    except FileNotFoundError:
        if not mute_mode:
            print(f"File not found: {mapping_csv}")
        return None, None, {}
    except Exception as e:
        if not mute_mode:
            print(f"Error reading file {mapping_csv}: {e}")
        return None, None, {}
    
    # Check if quality_index column exists
    if quality_index not in df.columns:
        if not mute_mode:
            print(f"Error: '{quality_index}' column not found in the file")
            print(f"Available columns: {list(df.columns)}")
        return None, None, {}
    
    # Convert to numpy array and remove NaN values
    quality_values = np.array(df[quality_index])
    quality_values = quality_values[~np.isnan(quality_values)]
    
    # Track if we're using inverted norm_dtw for display purposes
    using_inverted_norm_dtw = False
    if invert_norm_dtw and quality_index == 'norm_dtw':
        quality_values = 1 - quality_values
        using_inverted_norm_dtw = True
    # Check if there are enough unique values for proper distribution
    if len(quality_values) == 0:
        if not mute_mode:
            print("Error: No valid quality values found (all NaN)")
        return None, None, {}
    
    # Determine bin_width and calculate number of bins
    if bin_width is None:
        # Set default bin widths based on quality index
        # Sectional metrics use the same bin widths as their non-sectional counterparts
        if quality_index in ['corr_coef', 'corr_coef_sect']:
            bin_width = 0.025
        elif quality_index in ['norm_dtw', 'norm_dtw_sect']:
            bin_width = 0.0025
        else:
            # Automatically determine bin width for other quality indices
            data_range = quality_values.max() - quality_values.min()
            if data_range > 0:
                # Use Freedman-Diaconis rule to estimate optimal bin width
                if len(quality_values) > 1:
                    q75, q25 = np.percentile(quality_values, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        bin_width = 2 * iqr / (len(quality_values) ** (1/3))
                    else:
                        # Fallback to simple rule if IQR is 0
                        bin_width = data_range / max(10, min(int(np.sqrt(len(quality_values))), 100))
                else:
                    bin_width = 0.1  # Default for single value
            else:
                bin_width = 0.1  # Default for zero range
    
    # Calculate number of bins based on bin_width
    data_range = quality_values.max() - quality_values.min()
    if data_range > 0 and bin_width > 0:
        no_bins = max(1, int(np.ceil(data_range / bin_width)))
        # Constrain to reasonable range
        no_bins = max(10, min(no_bins, 200))
    else:
        no_bins = 10  # Default fallback
    
    # Calculate total count for percentage
    total_count = len(quality_values)
    
    # Calculate histogram data as counts, then convert to percentages
    hist, bins = np.histogram(quality_values, bins=no_bins, density=False)
    hist = hist * 100 / total_count  # Convert to percentage
    
    # Initialize fit_params dictionary with common statistics
    fit_params = {
        'data_min': quality_values.min(),
        'data_max': quality_values.max(),
        'median': np.median(quality_values),
        'std': quality_values.std(),
        'n_points': total_count,
        'hist_area': np.sum(hist),  # Should sum to 100% for percentage
        'bins': bins,
        'hist': hist,
        'bin_width': bin_width
    }
    
    # Add probability density function curve based on method
    if len(quality_values) > 1:  # Only plot PDF if we have multiple values
        x = np.linspace(quality_values.min(), quality_values.max(), 1000)
        
        # Calculate actual bin width from the histogram bins for PDF scaling
        actual_bin_width = bins[1] - bins[0]
        # PDF curves should be scaled to match percentage histogram
        # Scale PDF by bin_width * 100 to convert from density to percentage per bin
        pdf_scale_factor = actual_bin_width * 100
        
        if pdf_method.upper() == 'KDE':
            # Use Kernel Density Estimation
            kde = stats.gaussian_kde(quality_values, bw_method=kde_bandwidth)
            y = kde(x) * pdf_scale_factor
            fit_params['method'] = 'KDE'
            fit_params['bandwidth'] = kde_bandwidth
            fit_params['kde_object'] = kde
            fit_params['x_range'] = x
            fit_params['y_values'] = y
        
        elif pdf_method.upper() == 'SKEW-NORMAL':
            # Fit skew-normal distribution
            try:
                # Fit skew-normal distribution using maximum likelihood estimation
                shape, location, scale = stats.skewnorm.fit(quality_values)
                
                # Generate PDF
                y = stats.skewnorm.pdf(x, shape, location, scale) * pdf_scale_factor
                
                # Calculate skewness
                skewness = shape / np.sqrt(1 + shape**2) * np.sqrt(2/np.pi)
                
                fit_params['method'] = 'skew-normal'
                fit_params['shape'] = shape
                fit_params['location'] = location
                fit_params['scale'] = scale
                fit_params['skewness'] = skewness
                fit_params['x_range'] = x
                fit_params['y_values'] = y
                
            except (RuntimeError, ValueError, TypeError) as e:
                if not mute_mode:
                    print(f"Warning: Skew-normal fitting failed: {e}")
                    print("Falling back to normal distribution fit")
                pdf_method = 'normal'
        
        if pdf_method.upper() == 'NORMAL':
            # Fit normal distribution
            mean_val, std_val = stats.norm.fit(quality_values)
            
            # Use targeted bin sizing if provided, otherwise use default
            if targeted_binsize is not None:
                # targeted_binsize now contains (synthetic_bins, bin_width)
                synthetic_bins, bin_width_targeted = targeted_binsize
                n_bins_targeted = len(synthetic_bins) - 1
                
                # Use the exact same bin edges as synthetic data
                bins_targeted = synthetic_bins
                
                # Compute histogram using targeted bins (as percentages)
                hist_values, _ = np.histogram(quality_values, bins=bins_targeted, density=False)
                hist_percentages = (hist_values / len(quality_values)) * 100
                
                # Generate PDF curve from tail to tail (6 sigma range) with percentage scaling
                x_min = mean_val - 6 * std_val
                x_max = mean_val + 6 * std_val
                x = np.linspace(x_min, x_max, 1000)
                y = stats.norm.pdf(x, mean_val, std_val) * bin_width_targeted * 100
                
                # Store targeted bin information
                fit_params['bins'] = bins_targeted
                fit_params['hist'] = hist_percentages
                fit_params['n_bins'] = n_bins_targeted
                fit_params['bin_width'] = bin_width_targeted
                fit_params['n_points'] = len(quality_values)
            else:
                # Fallback to default approach
                # Generate PDF
                y = stats.norm.pdf(x, mean_val, std_val) * pdf_scale_factor
                
                # Use histogram computation as percentages
                hist_values, bins_values = np.histogram(quality_values, bins=no_bins, density=False)
                hist_percentages = (hist_values / len(quality_values)) * 100
                
                # Store default bin information
                fit_params['bins'] = bins_values
                fit_params['hist'] = hist_percentages
                fit_params['n_bins'] = no_bins
                fit_params['bin_width'] = bin_width
                fit_params['n_points'] = len(quality_values)
            
            fit_params['method'] = 'normal'
            fit_params['mean'] = mean_val
            fit_params['std'] = std_val
            fit_params['x_range'] = x
            fit_params['y_values'] = y
    
    # Add median value
    fit_params['median'] = np.median(quality_values)
    
    # If target_mapping_id is provided, calculate its statistics
    if target_mapping_id is not None:
        target_row = df[df['mapping_id'] == target_mapping_id]
        if not target_row.empty:
            target_value = target_row[quality_index].values[0]
            # Apply inversion if using inverted norm_dtw
            if using_inverted_norm_dtw:
                target_value = 1 - target_value
            percentile = (quality_values < target_value).mean() * 100
            fit_params['target_value'] = target_value
            fit_params['target_percentile'] = percentile
    
    # Get the display name for the quality index
    if using_inverted_norm_dtw:
        quality_display_name = quality_index_mapping.get('norm_dtw_inverted', r'nDTW$_c$')
    else:
        quality_display_name = quality_index_mapping.get(quality_index, quality_index)
    
    # If not in mute mode, create and show the plot
    if not mute_mode:
        # Create the figure
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot histogram of quality index as percentages
        # Use weights to convert counts to percentages (similar to Cell 10)
        ax.hist(quality_values, bins=no_bins, alpha=0.7, color='skyblue', 
                density=False,
                weights=np.ones(len(quality_values)) * 100 / len(quality_values))
        
        # Add probability density function curve if available
        if 'x_range' in fit_params and 'y_values' in fit_params:
            x = fit_params['x_range']
            y = fit_params['y_values']
            
            if fit_params['method'] == 'KDE':
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8, 
                        label=f'KDE\n(bandwidth = {fit_params["bandwidth"]})\n(n = {total_count:,})')
            elif fit_params['method'] == 'skew-normal':
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8, 
                        label=f'Skew-Normal Fit\n(α = {fit_params["shape"]:.3f})\n(μ = {fit_params["location"]:.3f})\n(σ = {fit_params["scale"]:.3f})\nn = {total_count:,}')
            elif fit_params['method'] == 'normal':
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8, 
                        label=f'Normal Fit\n(mean = {fit_params["mean"]:.3f})\n(σ = {fit_params["std"]:.3f})\nn = {total_count:,}')
        
        # Add vertical line for median
        median_value = fit_params['median']
        ax.axvline(median_value, color='green', linestyle='dashed', linewidth=2, 
                   label=f'Median: {median_value:.3f}')
        
        # If target_mapping_id is provided, highlight it
        if target_mapping_id is not None and 'target_value' in fit_params:
            target_value = fit_params['target_value']
            percentile = fit_params['target_percentile']
            ax.axvline(target_value, color='purple', linestyle='solid', linewidth=2,
                      label=f'Mapping {target_mapping_id}: {target_value:.3f}\n({percentile:.3f}th percentile)')
        
        # Set x-axis based on quality index
        # Sectional metrics use the same x-axis range as their non-sectional counterparts
        if quality_index in ['corr_coef', 'corr_coef_sect']:
            ax.set_xlim(0, 1.0)
        elif using_inverted_norm_dtw:
            ax.set_xlim(0.6, 1.0)
        
        # Add labels and title
        ax.set_xlabel(quality_display_name)
        ax.set_ylabel('Percentage (%)')
        title_index = quality_display_name if using_inverted_norm_dtw else quality_index
        title = f'Distribution of {title_index}'
        if core_a_name and core_b_name:
            title += f'\n{core_a_name} vs {core_b_name}'
        plt.title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the figure if requested
        if save_png:
            if png_filename is None:
                png_filename = f'{quality_index}_distribution_{pdf_method.lower()}.png'

            from .helpers import save_figure_formats
            save_dpi = dpi if dpi is not None else 150
            save_figure_formats(plt.gcf(), png_filename, fig_format=fig_format, dpi=save_dpi)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"Summary Statistics for {quality_display_name}:")
        print(f"Median: {fit_params['median']:.3f}")
        print(f"Min: {quality_values.min():.3f}")
        print(f"Max: {quality_values.max():.3f}")
        print(f"Standard Deviation: {quality_values.std():.3f}")
        print(f"Number of data points: {total_count}")
        
        # Print distribution fitting results
        if 'method' in fit_params:
            print(f"\nDistribution Fitting Results ({fit_params['method']}):")
            print(f"{'='*60}")
            if fit_params['method'] == 'KDE':
                print(f"Bandwidth: {fit_params['bandwidth']} (Controls smoothness of the density curve)")
            elif fit_params['method'] == 'skew-normal':
                print(f"Shape parameter (α): {fit_params['shape']:.5f} (Asymmetry; α=0 gives normal distribution; positive α = right skew, negative α = left skew; magnitude indicates strength of skew)")
                print(f"Location parameter (μ): {fit_params['location']:.5f} (Center/peak position of the distribution; larger magnitude shifts distribution)")
                print(f"Scale parameter (σ): {fit_params['scale']:.5f} (Width/spread of the distribution; magnitude indicates the degree of spread)")
                print(f"Skewness: {fit_params['skewness']:.5f} (Measure of asymmetry; positive = longer right tail; magnitude indicates severity of skew)")
            elif fit_params['method'] == 'normal':
                print(f"Mean (μ): {fit_params['mean']:.5f} (Center of the distribution; magnitude indicates distance from zero)")
                print(f"Standard deviation (σ): {fit_params['std']:.5f} (Width/spread of the distribution; larger magnitude increases variability)")
        
        # If target_mapping_id is provided, show its percentile
        if target_mapping_id is not None and 'target_value' in fit_params:
            target_value = fit_params['target_value']
            percentile = fit_params['target_percentile']
            print(f"\nMapping ID {target_mapping_id} {quality_display_name}: {target_value:.3f}")
            print(f"Percentile: {percentile:.3f}%")
        
        return fit_params
    else:
        # In mute mode, just return fit_params
        return fit_params
    
def process_single_row_parallel(row_data, combined_data, debug=False, use_inverted=False):
    """
    Helper function to process a single row using pre-calculated t-statistic and Cohen's d from CSV.
    
    Parameters:
    -----------
    row_data : tuple
        (idx, row, age_consideration, core_b_constraints_count)
    combined_data : array-like
        Reference data (not used anymore, kept for compatibility)
    debug : bool
        Whether to print debug information
    use_inverted : bool
        If True, use t_statistic_inv and cohens_d_inv columns (for inverted norm_dtw)
    
    Returns:
    --------
    tuple : (x_value, t_stat, cohens_d, effect_size_category, n_points, success)
    """
    idx, row, age_consideration, core_b_constraints_count = row_data
    
    # Determine constraint mapping for x-axis
    if age_consideration:
        x_value = core_b_constraints_count
    else:
        x_value = 0
    
    # Use pre-calculated t-statistic and Cohen's d from CSV columns
    try:
        # Get pre-calculated values from CSV columns (use inverted columns if requested)
        if use_inverted:
            t_stat = row.get('t_inv', row.get('t', 0.0))
            cohens_d_value = row.get('d_inv', row.get('d', 0.0))
        else:
            t_stat = row.get('t', 0.0)
            cohens_d_value = row.get('d', 0.0)
        effect_size_category = row.get('effect_size', 'negligible')
        n_points = row.get('n_points', 0)
        
        # Handle NaN values
        if pd.isna(t_stat):
            t_stat = 0.0
        if pd.isna(cohens_d_value):
            cohens_d_value = 0.0
        if pd.isna(effect_size_category) or effect_size_category == '':
            effect_size_category = 'negligible'
        if pd.isna(n_points):
            n_points = 0
            
        return x_value, t_stat, cohens_d_value, effect_size_category, n_points, True
        
    except Exception as e:
        if debug:
            print(f"Error processing row {idx}: {e}")
        return x_value, 0.0, 0.0, "negligible", 0, False


def calculate_improvement_scores_parallel(constraint_data, quality_index):
    """
    Helper function to calculate improvement scores in parallel.
    
    Parameters:
    -----------
    constraint_data : tuple
        (current_constraint, next_constraint, current_t_stats, next_t_stats)
    quality_index : str
        Quality index name for determining improvement direction
    
    Returns:
    --------
    list : List of improvement scores
    """
    current_constraint, next_constraint, current_t_stats, next_t_stats = constraint_data
    improvement_scores = []
    
    for curr_t in current_t_stats:
        for next_t in next_t_stats:
            t_change = next_t - curr_t
            
            # Determine improvement/deterioration based on quality index
            # Sectional metrics use the same improvement logic as their non-sectional counterparts
            if quality_index in ['norm_dtw', 'norm_dtw_sect']:
                improvement_score = -t_change  # Negative change is improvement
            else:
                improvement_score = t_change   # Positive change is improvement
            
            improvement_scores.append(improvement_score)
    
    return improvement_scores


def get_effect_size_color(effect_size_category):
    """
    Get color for effect size category using light gray to green scheme.
    
    Parameters:
    -----------
    effect_size_category : str
        Effect size category ('negligible', 'small', 'medium', 'large')
    
    Returns:
    --------
    str : Color code for the category
    """
    color_map = {
        'negligible': '#D3D3D3',  # Light gray
        'small': '#FFFFFF',       # White
        'medium': '#cee072',      # Light yellow-green  
        'large': '#3fd445'        # Green
    }
    return color_map.get(effect_size_category, '#D3D3D3')

def calculate_dynamic_sizes(total_points, total_connections):
    """
    Calculate dynamic sizes for plot elements based on data volume.
    
    Parameters:
    -----------
    total_points : int
        Total number of data points
    total_connections : int
        Total number of connections/lines
    
    Returns:
    --------
    dict : Dictionary with sizing parameters
    """
    # Base sizes
    base_dot_size = 60
    base_line_width = 1.0
    base_alpha = 0.8
    
    # Dynamic scaling based on data volume
    # Reduce sizes as data volume increases
    if total_points <= 50:
        dot_size = base_dot_size
        line_width = base_line_width
        alpha = base_alpha
    elif total_points <= 200:
        dot_size = max(40, base_dot_size * 0.8)
        line_width = max(0.7, base_line_width * 0.8)
        alpha = max(0.6, base_alpha * 0.8)
    elif total_points <= 500:
        dot_size = max(30, base_dot_size * 0.6)
        line_width = max(0.5, base_line_width * 0.6)
        alpha = max(0.4, base_alpha * 0.6)
    else:
        dot_size = max(20, base_dot_size * 0.4)
        line_width = max(0.3, base_line_width * 0.4)
        alpha = max(0.3, base_alpha * 0.4)
    
    # Further adjust based on connection density
    if total_connections > 1000:
        alpha *= 0.7
        line_width *= 0.8
    
    return {
        'dot_size': dot_size,
        'line_width': line_width,
        'alpha': alpha
    }


def _create_histogram_and_pdf_like_plot_correlation_distribution(quality_values, quality_index, 
                                                                targeted_binsize=None):
    """
    Create histogram and PDF using the exact same approach as plot_correlation_distribution.
    
    Returns a dictionary with histogram data and PDF parameters exactly like plot_correlation_distribution.
    """
    
    # Convert to numpy array and remove NaN values (same as plot_correlation_distribution)
    quality_values = np.array(quality_values)
    quality_values = quality_values[~np.isnan(quality_values)]
    
    if len(quality_values) == 0:
        return None
    
    # Determine bin_width (EXACT same logic as plot_correlation_distribution)
    bin_width = None
    if bin_width is None:
        # Set default bin widths based on quality index
        # Sectional metrics use the same bin width as their non-sectional counterparts
        if quality_index in ['corr_coef', 'corr_coef_sect']:
            bin_width = 0.025
        elif quality_index in ['norm_dtw', 'norm_dtw_sect']:
            bin_width = 0.0025
        else:
            # Automatically determine bin width for other quality indices
            data_range = quality_values.max() - quality_values.min()
            if data_range > 0:
                # Use Freedman-Diaconis rule to estimate optimal bin width
                if len(quality_values) > 1:
                    q75, q25 = np.percentile(quality_values, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        bin_width = 2 * iqr / (len(quality_values) ** (1/3))
                    else:
                        # Fallback to simple rule if IQR is 0
                        bin_width = data_range / max(10, min(int(np.sqrt(len(quality_values))), 100))
                else:
                    bin_width = 0.1  # Default for single value
            else:
                bin_width = 0.1  # Default for zero range
    
    # Calculate number of bins based on bin_width (EXACT same logic as plot_correlation_distribution)
    data_range = quality_values.max() - quality_values.min()
    if data_range > 0 and bin_width > 0:
        no_bins = max(1, int(np.ceil(data_range / bin_width)))
        # Constrain to reasonable range
        no_bins = max(10, min(no_bins, 200))
    else:
        no_bins = 10  # Default fallback
    
    # Calculate total count for percentage
    total_count = len(quality_values)
    
    # Calculate histogram data as counts, then convert to percentages (EXACT same as plot_correlation_distribution)
    hist, bins = np.histogram(quality_values, bins=no_bins, density=False)
    hist = hist * 100 / total_count  # Convert to percentage
    
    # Initialize fit_params dictionary with common statistics (EXACT same as plot_correlation_distribution)
    fit_params = {
        'data_min': quality_values.min(),
        'data_max': quality_values.max(),
        'median': np.median(quality_values),
        'std': quality_values.std(),
        'n_points': total_count,
        'hist_area': np.sum(hist),  # Should sum to 100% for percentage
        'bins': bins,
        'hist': hist,
        'bin_width': bin_width
    }
    
    # Add probability density function curve based on method (EXACT same as plot_correlation_distribution)
    if len(quality_values) > 1:  # Only plot PDF if we have multiple values
        x = np.linspace(quality_values.min(), quality_values.max(), 1000)
        
        # Calculate actual bin width from the histogram bins for PDF scaling
        actual_bin_width = bins[1] - bins[0]
        # PDF curves should be scaled to match percentage histogram
        # Scale PDF by bin_width * 100 to convert from density to percentage per bin
        pdf_scale_factor = actual_bin_width * 100
        
        # Use normal distribution fitting (same as plot_correlation_distribution default)
        # Fit normal distribution
        mean_val, std_val = stats.norm.fit(quality_values)
        
        # Use targeted bin sizing if provided, otherwise use default (EXACT same as plot_correlation_distribution)
        if targeted_binsize is not None:
            # targeted_binsize now contains (synthetic_bins, bin_width)
            synthetic_bins, bin_width_targeted = targeted_binsize
            n_bins_targeted = len(synthetic_bins) - 1
            
            # Use the exact same bin edges as synthetic data
            bins_targeted = synthetic_bins
            
            # Compute histogram using targeted bins (as percentages)
            hist_values, _ = np.histogram(quality_values, bins=bins_targeted, density=False)
            hist_percentages = (hist_values / len(quality_values)) * 100
            
            # Generate PDF curve from tail to tail (6 sigma range) with percentage scaling
            x_min = mean_val - 6 * std_val
            x_max = mean_val + 6 * std_val
            x = np.linspace(x_min, x_max, 1000)
            y = stats.norm.pdf(x, mean_val, std_val) * bin_width_targeted * 100
            
            # Store targeted bin information
            fit_params['bins'] = bins_targeted
            fit_params['hist'] = hist_percentages
            fit_params['n_bins'] = n_bins_targeted
            fit_params['bin_width'] = bin_width_targeted
            fit_params['n_points'] = len(quality_values)
        else:
            # Fallback to default approach
            # Generate PDF
            y = stats.norm.pdf(x, mean_val, std_val) * pdf_scale_factor
            
            # Use histogram computation as percentages
            hist_values, bins_values = np.histogram(quality_values, bins=no_bins, density=False)
            hist_percentages = (hist_values / len(quality_values)) * 100
            
            # Store default bin information
            fit_params['bins'] = bins_values
            fit_params['hist'] = hist_percentages
            fit_params['n_bins'] = no_bins
            fit_params['bin_width'] = bin_width
            fit_params['n_points'] = len(quality_values)
        
        fit_params['method'] = 'normal'
        fit_params['mean'] = mean_val
        fit_params['std'] = std_val
        fit_params['x_range'] = x
        fit_params['y_values'] = y
    
    # Add median value
    fit_params['median'] = np.median(quality_values)
    
    return fit_params


def plot_quality_distributions(quality_data, target_quality_indices, output_figure_filenames, 
                               CORE_A, CORE_B, debug=True, return_plot_info=False,
                               plot_real_data_histogram=True, plot_age_removal_step_pdf=True,
                               synthetic_csv_filenames=None, best_datum_values=None, 
                               fig_format=['png'], dpi=None, invert_norm_dtw=True):
    """
    Plot quality index distributions comparing real data vs synthetic null hypothesis.
    
    Brief summary: This function creates distribution plots showing PDF curves for different
    parameter combinations overlaid with synthetic null hypothesis data and mean value indicators.
    
    Parameters:
    -----------
    quality_data : dict
        Preprocessed quality data from load_and_prepare_quality_data function
    target_quality_indices : list
        List of quality indices to process (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    output_figure_filenames : dict
        Dictionary mapping quality_index to output figure filename paths
    CORE_A : str
        Name of core A
    CORE_B : str
        Name of core B
    debug : bool, default True
        If True, only print essential messages. If False, print all detailed messages.
    return_plot_info : bool, default False
        If True, return plotting information for gif creation
    plot_real_data_histogram : bool, default True
        If True, plot histograms for real data (no age constraints and all age constraints cases)
    plot_age_removal_step_pdf : bool, default True
        If True, plot all PDF curves including dashed lines for partially removed age constraints
    synthetic_csv_filenames : dict, optional
        Dictionary mapping quality_index to synthetic CSV filename paths for loading histogram data
    best_datum_values : dict, optional
        Dictionary mapping quality_index to best datum match values to plot as vertical lines
    dpi : int, optional
        Resolution for saved figures in dots per inch. If None, uses default (150)
    
    Returns:
    --------
    dict or None
        If return_plot_info=True, returns plotting information for gif creation
    """
    
    # Initialize plot info dictionary for gif creation (ONLY NEW ADDITION)
    plot_info_dict = {}
    
    # Loop through all quality indices
    for quality_index in target_quality_indices:
        if quality_index not in quality_data:
            print(f"Skipping {quality_index} - no data available")
            continue
            
        data = quality_data[quality_index]
        df_all_params = data['df_all_params']
        combined_data = data['combined_data']
        fitted_mean = data['fitted_mean']
        fitted_std = data['fitted_std']
        max_core_a_constraints = data['max_core_a_constraints']
        max_core_b_constraints = data['max_core_b_constraints']
        unique_combinations = data['unique_combinations']

        # Reconstruct synthetic data from binned CSV data EXACTLY like Cell 10
        synthetic_quality_values = None
        if synthetic_csv_filenames and quality_index in synthetic_csv_filenames:
            try:
                # Load fit params from synthetic CSV (same as Cell 10)
                df_fit_params = pd.read_csv(synthetic_csv_filenames[quality_index])
                
                # Initialize list to collect all raw synthetic data points (same as Cell 10)
                all_raw_data = []
                
                # Process each iteration to reconstruct raw data from binned data (same as Cell 10)
                for _, row in df_fit_params.iterrows():
                    # Extract binned data (same as Cell 10)
                    bins = np.fromstring(row['bins'].strip('[]'), sep=' ') if 'bins' in row and pd.notna(row['bins']) else None
                    hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ') if 'hist' in row and pd.notna(row['hist']) else None
                    n_points = row['n_points'] if 'n_points' in row and pd.notna(row['n_points']) else None
                    
                    if bins is not None and hist_percentages is not None and n_points is not None:
                        # IMPORTANT FIX: The hist values in CSV are not actually percentages but 
                        # raw histogram counts from the old buggy normalization code.
                        # We need to properly normalize them to reconstruct the correct data. (same as Cell 10)
                        
                        # First, normalize the histogram values so they represent proper proportions
                        hist_sum = np.sum(hist_percentages)
                        if hist_sum > 0:
                            # Normalize to get proper proportions, then scale by n_points to get counts
                            raw_counts = (hist_percentages / hist_sum) * n_points
                        else:
                            raw_counts = np.zeros_like(hist_percentages)
                        
                        # Reconstruct data points by sampling from each bin (same as Cell 10)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        bin_width = bins[1] - bins[0]
                        
                        for i, count in enumerate(raw_counts):
                            if count > 0:
                                # Generate random points within each bin
                                n_samples = int(round(count))
                                if n_samples > 0:
                                    # Sample uniformly within the bin
                                    bin_samples = np.random.uniform(
                                        bins[i], bins[i+1], n_samples
                                    )
                                    all_raw_data.extend(bin_samples)
                
                # Convert to numpy array (same as Cell 10)
                synthetic_quality_values = np.array(all_raw_data)
                
                # Transform norm_dtw values if invert_norm_dtw is True
                if invert_norm_dtw and quality_index in ['norm_dtw', 'norm_dtw_sect']:
                    synthetic_quality_values = 1 - synthetic_quality_values
                
            except Exception as e:
                print(f"Error reconstructing synthetic data: {e}")
                synthetic_quality_values = None
        
        # Create synthetic histogram and PDF using the EXACT same approach as Cell 10 and plot_correlation_distribution
        if synthetic_quality_values is not None and len(synthetic_quality_values) > 0:
            synthetic_fit_params = _create_histogram_and_pdf_like_plot_correlation_distribution(
                synthetic_quality_values, quality_index
            )
            
            no_bins = synthetic_fit_params['n_bins']
            x_fitted = synthetic_fit_params['x_range']
            y_fitted = synthetic_fit_params['y_values']
            fitted_mean = synthetic_fit_params['mean']
            fitted_std = synthetic_fit_params['std']
            synthetic_bins = synthetic_fit_params['bins']
            actual_bin_width = synthetic_fit_params['bin_width']
            
        else:
            # Fallback if synthetic data not available
            synthetic_bins = np.linspace(fitted_mean - 3*fitted_std, fitted_mean + 3*fitted_std, 31)
            synthetic_hist = np.ones(30) * (100 / 30)
            x_fitted = np.linspace(fitted_mean - 6 * fitted_std, fitted_mean + 6 * fitted_std, 1000)
            actual_bin_width = synthetic_bins[1] - synthetic_bins[0]
            y_fitted = stats.norm.pdf(x_fitted, fitted_mean, fitted_std) * actual_bin_width * 100
            no_bins = 30

        # Extract plot info for gif creation (ONLY NEW ADDITION)
        if return_plot_info:
            constraint_col = 'core_b_constraints_count'
            unique_constraints = sorted(df_all_params[constraint_col].unique())
            curves_by_constraint = {}
            
            for n_constraints in unique_constraints:
                constraint_data = df_all_params[df_all_params[constraint_col] == n_constraints]
                curves_by_constraint[n_constraints] = []
                
                for idx, row in constraint_data.iterrows():
                    if 'x_range' in row and 'y_values' in row and pd.notna(row['x_range']) and pd.notna(row['y_values']):
                        try:
                            x_range = np.fromstring(row['x_range'].strip('[]'), sep=' ')
                            y_values = np.fromstring(row['y_values'].strip('[]'), sep=' ')
                            
                            if len(x_range) > 0 and len(y_values) > 0:
                                curves_by_constraint[n_constraints].append({
                                    'x_range': x_range,
                                    'y_values': y_values,
                                    'constraint_count': row[constraint_col],
                                    'is_max_constraints': (row['core_a_constraints_count'] == max_core_a_constraints and 
                                                          row['core_b_constraints_count'] == max_core_b_constraints),
                                    'age_consideration': row['age_consideration'],
                                    'restricted_age_correlation': row['restricted_age_correlation'],
                                    'shortest_path_search': row['shortest_path_search']
                                })
                        except:
                            continue
            
            # Create combo_stats_by_constraint structure for GIF creation
            combo_stats_by_constraint = {}
            for n_constraints in unique_constraints:
                combo_stats_by_constraint[n_constraints] = {}
                constraint_curves = curves_by_constraint[n_constraints]
                
                # Group curves by combination type
                for curve in constraint_curves:
                    combo_key = f"age_{curve['age_consideration']}_restricted_{curve['restricted_age_correlation']}"
                    if combo_key not in combo_stats_by_constraint[n_constraints]:
                        combo_stats_by_constraint[n_constraints][combo_key] = {'curves': []}
                    combo_stats_by_constraint[n_constraints][combo_key]['curves'].append(curve)
            
            # Store plot info
            plot_info_dict[quality_index] = {
                'quality_index': quality_index,
                'CORE_A': CORE_A,
                'CORE_B': CORE_B,
                'unique_constraints': unique_constraints,
                'curves_by_constraint': curves_by_constraint,
                'combo_stats_by_constraint': combo_stats_by_constraint,
                'x_fitted': x_fitted,
                'y_fitted': y_fitted,
                'x_synth': x_fitted,  # Add missing key - same as x_fitted
                'y_synth': y_fitted,  # Add missing key - same as y_fitted  
                'fitted_mean': fitted_mean,
                'fitted_std': fitted_std,
                'combined_data': synthetic_quality_values if synthetic_quality_values is not None else np.linspace(fitted_mean - 3*fitted_std, fitted_mean + 3*fitted_std, 100),
                'n_bins': no_bins if synthetic_quality_values is not None else 30,
                'synthetic_bins': synthetic_bins,  # Actual bin edges for consistent histogram plotting
                'max_core_a_constraints': max_core_a_constraints,
                'max_core_b_constraints': max_core_b_constraints,
                'unique_combinations': unique_combinations,
                'min_core_b': df_all_params['core_b_constraints_count'].min(),
                'max_core_b': df_all_params['core_b_constraints_count'].max(),
                'invert_norm_dtw': invert_norm_dtw,
                'plot_limits': {  # Initial plot_limits - will be updated after plotting real data
                    # Sectional metrics use the same x-axis range as their non-sectional counterparts
                    # corr_coef and corr_coef_sect range from 0 to 1, norm_dtw and norm_dtw_sect start from 0
                    'x_min': 0 if quality_index in ['norm_dtw', 'norm_dtw_sect', 'corr_coef', 'corr_coef_sect'] else x_fitted.min(),
                    'x_max': 1.0 if quality_index in ['corr_coef', 'corr_coef_sect'] else x_fitted.max(),
                    'y_min': 0,
                    'y_max': y_fitted.max() * 1.1  # Add some headroom
                },
                'legend_elements': None,  # Will be populated after legend creation
                'legend_labels': None,
                'individual_curves': []  # Will store all individual curves for row-by-row animation
            }

        # Create combined plot
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # Plot synthetic histogram and PDF EXACTLY like plot_correlation_distribution
        if synthetic_quality_values is not None and len(synthetic_quality_values) > 0:
            # Use ax.hist with the actual bin edges to ensure consistent binning
            ax.hist(synthetic_quality_values, bins=synthetic_bins, alpha=0.3, color='gray', 
                    density=False,
                    weights=np.ones(len(synthetic_quality_values)) * 100 / len(synthetic_quality_values),
                    label='Synthetic data histogram')
        else:
            # Fallback: use bar chart for synthetic histogram
            bin_centers = (synthetic_bins[:-1] + synthetic_bins[1:]) / 2
            ax.bar(bin_centers, synthetic_hist, width=actual_bin_width*0.8, alpha=0.3, color='gray', 
                   label='Synthetic data histogram')

        # Plot fitted normal curve as gray dotted line (EXACT same as plot_correlation_distribution)
        ax.plot(x_fitted, y_fitted, color='gray', linestyle=':', linewidth=2, alpha=0.8,
                label='Synthetic data PDF (Null Hypothesis)')

        # Set up colormap based on core_b_constraints_count
        min_core_b = df_all_params['core_b_constraints_count'].min()
        max_core_b = df_all_params['core_b_constraints_count'].max()
        
        # Create colormap
        cmap = cm.get_cmap('Spectral_r')
        norm = colors.Normalize(vmin=min_core_b, vmax=max_core_b)

        # PDF mean values plotting removed per user request

        # Collect real data for statistical tests - separate by individual distributions
        solid_real_data_by_combo = {}
        dash_real_data_by_combo = {}

        # Track which constraint levels have been plotted for legend
        plotted_constraint_levels = set()

        # Sort dataframe by core_b_constraints_count to plot in ascending order
        df_all_params_sorted = df_all_params.sort_values('core_b_constraints_count', ascending=True)

        for idx, row in df_all_params_sorted.iterrows():
            # Extract parameter values directly from CSV columns
            age_consideration = row['age_consideration']
            restricted_age_correlation = row['restricted_age_correlation']
            shortest_path_search = row['shortest_path_search']
            core_a_constraints = row['core_a_constraints_count']
            core_b_constraints = row['core_b_constraints_count']
            
            # Extract the raw quality values for this row EXACTLY like plot_correlation_distribution does
            # We need to reconstruct the original quality values to plot histogram exactly like plot_correlation_distribution
            row_quality_values = None
            if 'bins' in row and 'hist' in row and 'n_points' in row and pd.notna(row['bins']) and pd.notna(row['hist']) and pd.notna(row['n_points']):
                try:
                    bins_data = np.fromstring(row['bins'].strip('[]'), sep=' ')
                    hist_data = np.fromstring(row['hist'].strip('[]'), sep=' ')
                    n_points = int(row['n_points'])
                    
                    # Reconstruct quality values from histogram (for exact plotting)
                    row_quality_values = reconstruct_raw_data_from_histogram(bins_data, hist_data, n_points)
                    
                    # Transform norm_dtw values if invert_norm_dtw is True
                    if invert_norm_dtw and quality_index in ['norm_dtw', 'norm_dtw_sect']:
                        row_quality_values = 1 - np.array(row_quality_values)
                except:
                    pass
            
            # Create histogram and PDF using EXACT same approach as plot_correlation_distribution
            row_fit_params = None
            if row_quality_values is not None and len(row_quality_values) > 0:
                # Use the synthetic data binning for consistency (targeted bin sizing like plot_correlation_distribution)
                targeted_binsize = (synthetic_bins, actual_bin_width) if synthetic_quality_values is not None else None
                row_fit_params = _create_histogram_and_pdf_like_plot_correlation_distribution(
                    row_quality_values, quality_index, targeted_binsize
                )
                if row_fit_params is not None and 'x_range' in row_fit_params and 'y_values' in row_fit_params:
                    x_range = row_fit_params['x_range']
                    y_values = row_fit_params['y_values']
                else:
                    x_range = y_values = None
            else:
                x_range = y_values = None
            
            # Create unique key for this parameter combination
            combo_key = f"age_{age_consideration}_restricted_{restricted_age_correlation}_shortest_{shortest_path_search}"
            
            # Store reconstructed quality values for statistical tests
            if row_quality_values is not None:
                # Group by search method based on shortest_path_search column
                if shortest_path_search:
                    if combo_key not in solid_real_data_by_combo:
                        solid_real_data_by_combo[combo_key] = []
                    solid_real_data_by_combo[combo_key].extend(row_quality_values)
                else:
                    if combo_key not in dash_real_data_by_combo:
                        dash_real_data_by_combo[combo_key] = []
                    dash_real_data_by_combo[combo_key].extend(row_quality_values)
            
            if x_range is not None and y_values is not None and len(x_range) > 0 and len(y_values) > 0:
                # Determine if this is a max constraint case
                is_max_constraints = (core_a_constraints == max_core_a_constraints and 
                                    core_b_constraints == max_core_b_constraints)
                
                # Get color from colormap based on core_b_constraints_count
                base_color = cmap(norm(core_b_constraints))
                
                # Darken the color by reducing brightness
                # Convert to HSV, reduce the V (value/brightness) component, then back to RGB
                base_color_rgb = base_color[:3]  # Remove alpha if present
                base_color_hsv = colors.rgb_to_hsv(base_color_rgb)
                darkened_hsv = (base_color_hsv[0], base_color_hsv[1], base_color_hsv[2] * 0.9)  # Reduce brightness by 10%
                color = colors.hsv_to_rgb(darkened_hsv)
                
                # Determine line style and transparency based on constraint levels
                if is_max_constraints or core_b_constraints == 0:
                    line_style = '-'
                    linewidth = 1.5
                    alpha = 0.9
                    zorder = 10  # Higher zorder to bring solid lines to front
                    constraint_label = f'{CORE_B} age constraints: {core_b_constraints}'
                    if core_b_constraints not in plotted_constraint_levels:
                        label = constraint_label
                        plotted_constraint_levels.add(core_b_constraints)
                    else:
                        label = None
                    should_plot = True  # Always plot solid lines
                else:
                    line_style = '--'
                    linewidth = 1
                    alpha = 0.9
                    zorder = 1  # Lower zorder for dashed lines
                    constraint_label = f'{CORE_B} age constraints: {core_b_constraints}'
                    if core_b_constraints not in plotted_constraint_levels:
                        label = constraint_label
                        plotted_constraint_levels.add(core_b_constraints)
                    else:
                        label = None
                    should_plot = plot_age_removal_step_pdf  # Only plot dashed lines if enabled
                
                # Plot PDF curve using stored data
                if should_plot:
                    ax.plot(x_range, y_values, 
                           color=color, 
                           linestyle=line_style,
                           linewidth=linewidth, alpha=alpha,
                           zorder=zorder,
                           label=label)
                    
                    # Store individual curve data for GIF creation if requested (for ALL plotted curves)
                    if return_plot_info:
                        plot_info_dict[quality_index]['individual_curves'].append({
                            'x_range': x_range,
                            'y_values': y_values,
                            'color': color,
                            'linestyle': line_style,
                            'linewidth': linewidth,
                            'alpha': alpha,
                            'zorder': zorder,
                            'label': label,
                            'constraint_count': core_b_constraints,
                            'is_max_constraints': is_max_constraints
                        })
                    
                    # Plot histogram for special cases: no age constraints (0) or all age constraints (max)
                    if plot_real_data_histogram and (core_b_constraints == 0 or (core_a_constraints == max_core_a_constraints and core_b_constraints == max_core_b_constraints)):
                        # Use the same approach as plot_correlation_distribution: reconstruct raw data from CSV and use ax.hist()
                        if row_quality_values is not None and len(row_quality_values) > 0:
                            # Use the SAME bin edges as synthetic data for consistent comparison
                            # This ensures histogram is comparable to the PDF curve (same bin_width scaling)
                            ax.hist(row_quality_values, bins=synthetic_bins, alpha=0.3, color=color, 
                                   edgecolor='none', density=False, zorder=zorder-1,
                                   weights=np.ones(len(row_quality_values)) * 100 / len(row_quality_values))

        # Update plot limits to include all plotted data (both synthetic and real data)
        # Get current axis limits after all plotting is done
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Update plot_info_dict with the actual plot limits that include all data
        if return_plot_info:
            plot_info_dict[quality_index]['plot_limits'].update({
                'x_min': current_xlim[0],
                'x_max': current_xlim[1],
                'y_min': current_ylim[0], 
                'y_max': current_ylim[1]
            })

        # Perform statistical tests for max constraint cases only
        solid_stats_by_combo = {}
        dash_stats_by_combo = {}

        # Filter data to only include max constraint cases for statistical analysis
        max_constraint_data = df_all_params[
            (df_all_params['core_a_constraints_count'] == max_core_a_constraints) &
            (df_all_params['core_b_constraints_count'] == max_core_b_constraints) 
        ]

        # Recalculate real data collections for max constraints only
        solid_real_data_by_combo_max = {}
        dash_real_data_by_combo_max = {}

        for idx, row in max_constraint_data.iterrows():
            age_consideration = row['age_consideration']
            restricted_age_correlation = row['restricted_age_correlation']
            shortest_path_search = row['shortest_path_search']
            
            combo_key = f"age_{age_consideration}_restricted_{restricted_age_correlation}_shortest_{shortest_path_search}"
            
            if 'bins' in row and 'hist' in row and 'n_points' in row and \
               pd.notna(row['bins']) and pd.notna(row['hist']) and pd.notna(row['n_points']):
                
                bins = np.fromstring(row['bins'].strip('[]'), sep=' ')
                hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ')
                n_points = row['n_points']
                
                raw_data_points = reconstruct_raw_data_from_histogram(bins, hist_percentages, n_points)
                
                # Transform norm_dtw values if invert_norm_dtw is True
                if invert_norm_dtw and quality_index in ['norm_dtw', 'norm_dtw_sect']:
                    raw_data_points = 1 - np.array(raw_data_points)
                
                if shortest_path_search:
                    if combo_key not in solid_real_data_by_combo_max:
                        solid_real_data_by_combo_max[combo_key] = []
                    solid_real_data_by_combo_max[combo_key].extend(raw_data_points)
                else:
                    if combo_key not in dash_real_data_by_combo_max:
                        dash_real_data_by_combo_max[combo_key] = []
                    dash_real_data_by_combo_max[combo_key].extend(raw_data_points)

        # Calculate statistics for optimal search combinations - max constraints only
        for combo_key, data in solid_real_data_by_combo_max.items():
            if len(data) > 1:
                data_array = np.array(data)
                cohens_d_val = cohens_d(data_array, combined_data)
                
                solid_stats_by_combo[combo_key] = {
                    'd': cohens_d_val,
                    'n_samples': len(data_array),
                    'mean': np.mean(data_array),
                    'std': np.std(data_array)
                }

        # Calculate statistics for random search combinations - max constraints only
        for combo_key, data in dash_real_data_by_combo_max.items():
            if len(data) > 1:
                data_array = np.array(data)
                cohens_d_val = cohens_d(data_array, combined_data)
                
                dash_stats_by_combo[combo_key] = {
                    'd': cohens_d_val,
                    'n_samples': len(data_array),
                    'mean': np.mean(data_array),
                    'std': np.std(data_array)
                }

        # Print detailed statistical results BEFORE creating the plot
        if not debug:  # Only show detailed analysis when debug=False (mute_mode=False)
            print(f"\n=== DETAILED STATISTICAL ANALYSIS FOR {quality_index} ===")
            print(f"Synth Data Distribution:")
            print(f"  Mean: {fitted_mean:.1f}, SD: {fitted_std:.1f}")
            print(f"  Sample size: {len(combined_data)}")
            print(f"  Interpretation: Baseline distribution from synthetic data representing no true correlation")
            print()

        if not debug:  # Only show optimal search results when debug=False (mute_mode=False)
            print(f"--- Optimal Search Results (Max Constraints Only) ---")
            
            for combo_key, stats_dict in solid_stats_by_combo.items():
                # Parse combo_key to get descriptive name
                if "age_True_restricted_True" in combo_key:
                    desc = "Consider age strictly"
                elif "age_True_restricted_False" in combo_key:
                    desc = "Consider age loosely"
                elif "age_False" in combo_key:
                    desc = "Neglect age"
                else:
                    desc = combo_key
                print(f"{desc}:")
                print(f"  Mean: {stats_dict['mean']:.1f}, SD: {stats_dict['std']:.1f}")
                print(f"  Cohen's d: {stats_dict['d']:.1f}")
                print(f"  Sample size: {stats_dict['n_samples']}")
                
                # Interpret effect size
                if abs(stats_dict['d']) < 0.2:
                    effect_size = "negligible"
                elif abs(stats_dict['d']) < 0.5:
                    effect_size = "small"
                elif abs(stats_dict['d']) < 0.8:
                    effect_size = "medium"
                else:
                    effect_size = "large"
                
                # Statistical interpretation
                if stats_dict['d'] > 0:
                    direction = "higher than"
                else:
                    direction = "lower than"
                
                print(f"  Effect size: {effect_size} difference between distributions")
                print(f"  Interpretation: {direction} null hypothesis with {effect_size} effect size")
                print()

        if not debug and dash_stats_by_combo:
            print(f"--- Random Search Results (Max Constraints Only) ---")
            for combo_key, stats_dict in dash_stats_by_combo.items():
                # Parse combo_key to get descriptive name
                if "age_True_restricted_True" in combo_key:
                    desc = "Consider age strictly"
                elif "age_True_restricted_False" in combo_key:
                    desc = "Consider age loosely"
                elif "age_False" in combo_key:
                    desc = "Neglect age"
                else:
                    desc = combo_key
                
                print(f"{desc}:")
                print(f"  Mean: {stats_dict['mean']:.1f}, SD: {stats_dict['std']:.1f}")
                print(f"  Cohen's d: {stats_dict['d']:.1f}")
                print(f"  Sample size: {stats_dict['n_samples']}")
                
                # Interpret effect size
                if abs(stats_dict['d']) < 0.2:
                    effect_size = "negligible"
                elif abs(stats_dict['d']) < 0.5:
                    effect_size = "small"
                elif abs(stats_dict['d']) < 0.8:
                    effect_size = "medium"
                else:
                    effect_size = "large"
                
                # Statistical interpretation
                if stats_dict['d'] > 0:
                    direction = "higher than"
                else:
                    direction = "lower than"
                
                print(f"  Effect size: {effect_size} difference between distributions")
                print(f"  Interpretation: {direction} null hypothesis with {effect_size} effect size")
                print()

        # Get display name for quality index
        def get_quality_display_name(quality_index):
            if quality_index == 'corr_coef':
                return "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                return "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                return "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                return "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                return quality_index
        
        display_name = get_quality_display_name(quality_index)
        
        # Set x-axis range for norm_dtw to 0.7 to 1.0
        # Sectional metrics use the same x-axis range as their non-sectional counterparts
        if quality_index in ['norm_dtw', 'norm_dtw_sect']:
            ax.set_xlim(0.7, 1.0)
        
        # Formatting
        ax.set_xlabel(f"{display_name}")
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{display_name}\n{CORE_A} vs {CORE_B}')

        # Create grouped legend based on plot parameters
        handles, labels = ax.get_legend_handles_labels()

        # Separate handles and labels by groups
        synthetic_handles = []
        synthetic_labels = []
        constraint_handles = []
        constraint_labels = []
        histogram_handles = []
        histogram_labels = []

        for handle, label in zip(handles, labels):
            if 'Null' in label or 'Synthetic' in label:
                synthetic_handles.append(handle)
                synthetic_labels.append(label)
            elif 'age constraints' in label:
                constraint_handles.append(handle)
                constraint_labels.append(label)

        # Create grouped legend with titles
        legend_elements = []
        legend_labels = []

        # Add synthetic group
        if synthetic_handles:
            legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
            legend_labels.append('Synthetic Data Correlations')
            legend_elements.extend(synthetic_handles)
            legend_labels.extend(synthetic_labels)
            legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
            legend_labels.append('')

        # Add constraint level group
        if constraint_handles:
            legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
            legend_labels.append('Real Data Correlations')
            
            # Filter to only show constraint 0 and max constraint
            filtered_constraint_pairs = []
            constraint_pairs = list(zip(constraint_handles, constraint_labels))
            
            for handle, label in constraint_pairs:
                # Handle both integer and float string formats (e.g., '0' or '0.0')
                constraint_count = int(float(label.split(': ')[-1]))
                if constraint_count == 0:
                    # Rename to use the core B name
                    new_label = f'PDF w/o {CORE_B}\'s age constraints'
                    filtered_constraint_pairs.append((handle, new_label, constraint_count))
                elif constraint_count == max_core_b_constraints:
                    # Rename to use the core B name
                    new_label = f'PDF w/ all {CORE_B}\'s age constraints'
                    filtered_constraint_pairs.append((handle, new_label, constraint_count))
            
            # Sort by constraint count (0 first, then max)
            filtered_constraint_pairs.sort(key=lambda x: x[2])  # Sort by constraint count
            
            # Add legend entries in the correct order: histogram first, then PDF for each constraint level
            if plot_real_data_histogram:
                # Get colors from colormap for histogram legend
                cmap = cm.get_cmap('Spectral_r')
                norm = colors.Normalize(vmin=min_core_b, vmax=max_core_b)
                
                for handle, pdf_label, constraint_count in filtered_constraint_pairs:
                    # First add histogram legend entry
                    constraint_color = cmap(norm(constraint_count))
                    constraint_color_rgb = constraint_color[:3]
                    constraint_color_hsv = colors.rgb_to_hsv(constraint_color_rgb)
                    constraint_darkened_hsv = (constraint_color_hsv[0], constraint_color_hsv[1], constraint_color_hsv[2] * 0.9)
                    constraint_darkened = colors.hsv_to_rgb(constraint_darkened_hsv)
                    
                    # Get sample count for this constraint level
                    # Match the actual histogram plotting logic from line 2361
                    if constraint_count == 0:
                        # For no age constraints case, find any row with core_b_constraints_count == 0
                        constraint_data = df_all_params[df_all_params['core_b_constraints_count'] == 0]
                    else:
                        # For all age constraints case, find max constraints for both cores
                        constraint_data = df_all_params[
                            (df_all_params['core_a_constraints_count'] == max_core_a_constraints) &
                            (df_all_params['core_b_constraints_count'] == max_core_b_constraints)
                        ]
                    n_points = constraint_data['n_points'].iloc[0] if not constraint_data.empty else 0
                    
                    if constraint_count == 0:
                        histogram_label = f'Histogram (n = {n_points}): No age constraints'
                        pdf_label = f'PDF (n = {n_points}): No age constraints'
                    else:
                        histogram_label = f'Histogram (n = {n_points}): With age constraints'
                        pdf_label = f'PDF (n = {n_points}): With age constraints'
                    
                    legend_elements.append(patches.Rectangle((0, 0), 1, 1, facecolor=constraint_darkened, alpha=0.3))
                    legend_labels.append(histogram_label)
                    
                    # Then add PDF legend entry
                    legend_elements.append(handle)
                    legend_labels.append(pdf_label)
            else:
                # Just add PDF entries when no histograms are plotted
                for handle, old_pdf_label, constraint_count in filtered_constraint_pairs:
                    # Get sample count for this constraint level
                    # Match the actual histogram plotting logic from line 2361
                    if constraint_count == 0:
                        # For no age constraints case, find any row with core_b_constraints_count == 0
                        constraint_data = df_all_params[df_all_params['core_b_constraints_count'] == 0]
                    else:
                        # For all age constraints case, find max constraints for both cores
                        constraint_data = df_all_params[
                            (df_all_params['core_a_constraints_count'] == max_core_a_constraints) &
                            (df_all_params['core_b_constraints_count'] == max_core_b_constraints)
                        ]
                    n_points = constraint_data['n_points'].iloc[0] if not constraint_data.empty else 0
                    
                    if constraint_count == 0:
                        pdf_label = f'PDF (n = {n_points}): No age constraints'
                    else:
                        pdf_label = f'PDF (n = {n_points}): With age constraints'
                        
                    legend_elements.append(handle)
                    legend_labels.append(pdf_label)
            
            # Add empty lines depending on plot parameters
            if plot_real_data_histogram and plot_age_removal_step_pdf:
                # Add 3 lines of spacing when both are enabled
                legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
                legend_labels.append('')
                legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
                legend_labels.append('')
                legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
                legend_labels.append('')
            elif not plot_real_data_histogram and plot_age_removal_step_pdf:
                # Add 3 lines of spacing when only PDF curves are shown
                legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
                legend_labels.append('')
                legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
                legend_labels.append('')
                legend_elements.append(Line2D([0], [0], color='white', linewidth=0, alpha=0))
                legend_labels.append('')
            # For plot_real_data_histogram=True and plot_age_removal_step_pdf=False, no empty lines

        # Best datum match is shown as text annotation, not in legend

        # Apply legend with grouping
        legend = ax.legend(legend_elements, legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left')

        # Style the group title labels
        for i, label in enumerate(legend_labels):
            if (label.startswith('Synthetic Data Correlations') or 
                label.startswith('Real Data Correlations') or 
                label.startswith('# of')):
                legend.get_texts()[i].set_weight('bold')
                legend.get_texts()[i].set_fontsize(10)
                legend.get_texts()[i].set_ha('left')
            else:
                # Make all other legend text smaller too
                legend.get_texts()[i].set_fontsize(9)

        # Add colorbar for age constraint levels (consistent with curve colors)
        # Create a colorbar that matches the distribution curves' color scheme
        # Apply same darkening as used for the curves
        cmap_darkened = cm.get_cmap('Spectral_r')
        
        # Create a custom colorbar with darkened colors to match the curves
        
        # Position colorbar directly under the legend box
        # First render the plot to get accurate legend positioning
        if plot_age_removal_step_pdf:
            plt.draw()
            
            # Get legend position in figure coordinates
            legend_bbox = legend.get_window_extent()
            legend_bbox_axes = legend_bbox.transformed(fig.transFigure.inverted())
            
            # Position colorbar directly below legend
            colorbar_height = 0.025
            colorbar_width = legend_bbox_axes.width * 0.9  # Slightly smaller than legend
            colorbar_x = legend_bbox_axes.x0 - legend_bbox_axes.width *.77  # Center it
            
            # Adjust colorbar position based on plot parameters
            if plot_real_data_histogram and plot_age_removal_step_pdf:
                # Shift down further when both histograms and PDF curves are shown
                colorbar_y = legend_bbox_axes.y0 - colorbar_height + 0.125  # Further down to account for histogram legend entries
            else:
                # Standard position for PDF-only case
                colorbar_y = legend_bbox_axes.y0 - colorbar_height + 0.127  # Below legend with gap
            
            cax = fig.add_axes([colorbar_x, colorbar_y, colorbar_width, colorbar_height])
            
            # Create a colorbar with the same normalization as the curves
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.set_label(f'Number of {CORE_B} Age Constraints', fontsize=8, fontweight='bold')
        
            # Set ticks to show actual constraint levels
            unique_constraints_sorted = sorted(df_all_params['core_b_constraints_count'].unique())
            cbar.set_ticks([min_core_b + (max_core_b - min_core_b) * i / (len(unique_constraints_sorted) - 1) 
                        for i in range(len(unique_constraints_sorted))])
            cbar.set_ticklabels([str(level) for level in unique_constraints_sorted])
            cbar.ax.tick_params(labelsize=8)

        # Store legend information for GIF creation if requested
        if return_plot_info:
            plot_info_dict[quality_index]['legend_elements'] = legend_elements
            plot_info_dict[quality_index]['legend_labels'] = legend_labels
            # Store actual plot limits after all adjustments
            plot_info_dict[quality_index]['actual_plot_limits'] = {
                'x_min': ax.get_xlim()[0],
                'x_max': ax.get_xlim()[1], 
                'y_min': ax.get_ylim()[0],
                'y_max': ax.get_ylim()[1]
            }
            # Store display name for GIF titles
            plot_info_dict[quality_index]['display_name'] = display_name

        ax.grid(True, alpha=0.3)
        # Sectional metrics use the same x-axis range as their non-sectional counterparts
        if quality_index in ['corr_coef', 'corr_coef_sect']:
            ax.set_xlim(0, 1.0)
            
        # Store actual axis limits for GIF creation if requested
        if return_plot_info:
            plot_info_dict[quality_index]['actual_plot_limits'] = {
                'x_min': ax.get_xlim()[0],
                'x_max': ax.get_xlim()[1],
                'y_min': ax.get_ylim()[0],
                'y_max': ax.get_ylim()[1]
            }

        # Add best datum match vertical line if available
        if best_datum_values and quality_index in best_datum_values:
            best_value = best_datum_values[quality_index]
            if pd.notna(best_value):  # Check if value is not NaN
                # Invert norm_dtw value to match the x-axis (1 - norm_dtw) when invert_norm_dtw is True
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                plot_value = (1 - best_value) if (invert_norm_dtw and is_norm_dtw_metric) else best_value
                
                # Plot line in dark green with long dash and highest zorder
                ax.axvline(plot_value, color='darkgreen', linestyle='--', linewidth=2, zorder=100)
                
                # Add text annotation next to the line
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                text_y = ax_ylim[0] + 0.90 * (ax_ylim[1] - ax_ylim[0])  # 90% up from bottom (higher position)
                
                # Position text based on arrow direction
                # When invert_norm_dtw=True, norm_dtw metrics use right-pointing arrows (like corr_coef)
                use_left_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_left_arrow:
                    # For left-pointing arrows, put text on left side of line
                    text_x = plot_value - 0.01 * (ax_xlim[1] - ax_xlim[0])
                    ha = 'right'
                else:
                    # For right-pointing arrows, put text on right side of line
                    text_x = plot_value + 0.01 * (ax_xlim[1] - ax_xlim[0])
                    ha = 'left'
                
                ax.text(text_x, text_y, 
                       f'Best\nDatum\nMatch\n({plot_value:.3f})', 
                       color='darkgreen', fontweight='bold', fontsize='x-small',
                       ha=ha, va='center', zorder=101)
        else:
            print("⚠️  WARNING: No valid correlatable datums to be marked.")

        # Add horizontal black arrow with 'Better Correlation Quality' text
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Position arrow in upper area of the plot
        arrow_y = ax_ylim[0] + 0.92 * (ax_ylim[1] - ax_ylim[0])  # 92% up from bottom
        
        # Determine arrow direction and position based on quality index
        # Sectional metrics use the same arrow direction as their non-sectional counterparts
        # When invert_norm_dtw=True, norm_dtw metrics use same direction as corr_coef (higher = better)
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_left_arrow = is_norm_dtw_metric and not invert_norm_dtw  # Only use left arrow if NOT inverted
        
        if use_left_arrow:
            # For norm_dtw (non-inverted), lower values are better (arrow points left) - position in upper right, moved slightly left
            arrow_start_x = ax_xlim[0] + 0.94 * (ax_xlim[1] - ax_xlim[0])  # Start 94% from left (moved left)
            arrow_end_x = ax_xlim[0] + 0.77 * (ax_xlim[1] - ax_xlim[0])    # End 77% from left  
            text_x = ax_xlim[0] + 0.855 * (ax_xlim[1] - ax_xlim[0])        # Text centered between 77% and 94%
        else:
            # For corr_coef and other indices (or inverted norm_dtw), higher values are better (arrow points right) - position in upper left
            arrow_start_x = ax_xlim[0] + 0.07 * (ax_xlim[1] - ax_xlim[0])  # Start 7% from left
            arrow_end_x = ax_xlim[0] + 0.24 * (ax_xlim[1] - ax_xlim[0])    # End 24% from left
            text_x = ax_xlim[0] + 0.155 * (ax_xlim[1] - ax_xlim[0])        # Text centered under arrow
            
        # Add horizontal arrow
        ax.annotate('', 
                   xy=(arrow_end_x, arrow_y), 
                   xytext=(arrow_start_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2),
                   zorder=5)
        
        # Add text under the arrow
        text_y = arrow_y - 0.03 * (ax_ylim[1] - ax_ylim[0])  # 3% below arrow
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='center', va='top',
               color='black', zorder=4)

        plt.tight_layout()
        
        # Save figure if output filename is provided
        if output_figure_filenames and quality_index in output_figure_filenames:
            output_filename_base = output_figure_filenames[quality_index]

            from .helpers import save_figure_formats
            save_dpi = dpi if dpi is not None else 150
            save_figure_formats(plt.gcf(), output_filename_base, fig_format=fig_format, dpi=save_dpi)

            if not debug:
                print(f"✓ Analysis complete for {quality_index}!")
        else:
            if debug:  # debug=True means mute_mode=True, show essential info only
                print(f"✓ Distribution plot completed for {quality_index}")
            else:  # debug=False means mute_mode=False, show detailed info
                print(f"✓ Analysis complete for {quality_index}!")
        
        # Show plot AFTER all printed texts - suppress when debug=True (i.e., mute_mode=True)
        if not debug:  # debug=True means mute_mode=True, so suppress display
            plt.show()

    if not debug:
        print(f"\n{'='*80}")
        print(f"ALL QUALITY INDICES PROCESSING COMPLETED")
        print(f"{'='*80}")
    
    # Return plot info if requested (ONLY NEW ADDITION)
    if return_plot_info:
        return plot_info_dict
    else:
        return None


def plot_t_statistics_vs_constraints(quality_data, target_quality_indices, output_figure_filenames,
                                     CORE_A, CORE_B, debug=True, n_jobs=-1, batch_size=None, return_plot_info=False, 
                                     fig_format=['png'], dpi=None, invert_norm_dtw=True):
    """
    Plot t-statistics vs number of age constraints for each quality index.
    OPTIMIZED with parallel processing and dynamic sizing.
    
    Parameters:
    -----------
    quality_data : dict
        Preprocessed quality data from load_and_prepare_quality_data function
    target_quality_indices : list
        List of quality indices to process (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    output_figure_filenames : dict
        Dictionary mapping quality_index to output figure filename paths
    CORE_A : str
        Name of core A
    CORE_B : str
        Name of core B
    debug : bool, default True
        If True, only print essential messages. If False, print all detailed messages.
    n_jobs : int, default -1
        Number of parallel jobs to run. -1 means using all available cores.
    batch_size : int, optional
        Batch size for parallel processing. If None, automatically determined.
    return_plot_info : bool, default False
        If True, return plotting information for gif creation
    
    Returns:
    --------
    dict or None
        If return_plot_info=True, returns plotting information for gif creation
    """
    
    plot_info_dict = {}
    
    for quality_index in target_quality_indices:
        if quality_index not in quality_data:
            print(f"Skipping {quality_index} - no data available")
            continue
            
        data = quality_data[quality_index]
        df_all_params = data['df_all_params']
        combined_data = data['combined_data']
        
        if not debug:  # Only show processing message when debug=False (mute_mode=False)
            print(f"Processing {quality_index} with {len(df_all_params)} scenarios...")
        
        # Initialize plot_info_dict entry if needed for GIF creation
        if return_plot_info:
            plot_info_dict[quality_index] = {
                'individual_points': [],  # Will store all individual points for row-by-row animation
                'individual_segments': []  # Will store all individual segments for row-by-row animation
            }
        
        # Create plot
        fig, ax = plt.subplots(figsize=(9.5, 5))
        
        # Cache repeated calculations
        combined_mean = np.mean(combined_data)
        combined_std = np.std(combined_data)
        
        # Prepare data for parallel processing
        if batch_size is None:
            batch_size = max(50, len(df_all_params) // (n_jobs if n_jobs > 0 else 4))
        
        # Prepare row data for parallel processing
        row_data_list = []
        for idx, row in df_all_params.iterrows():
            row_data_list.append((idx, row, row['age_consideration'], row['core_b_constraints_count']))
        
        # Process t-statistics in parallel
        if not debug:  # Only show processing message when debug=False (mute_mode=False)
            if n_jobs == -1:
                print(f"  Calculating t-statistics and Cohen's d using all available cores...")
            else:
                print(f"  Calculating t-statistics and Cohen's d using {n_jobs} cores...")
        
        # Determine if we should use inverted columns for norm_dtw metrics
        use_inverted = invert_norm_dtw and quality_index in ['norm_dtw', 'norm_dtw_sect']
        
        with tqdm(total=len(row_data_list), desc="  Processing t-statistics", disable=debug) as pbar:
            # Process in batches to manage memory
            all_results = []
            for i in range(0, len(row_data_list), batch_size):
                batch = row_data_list[i:i+batch_size]
                
                batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(process_single_row_parallel)(row_data, combined_data, debug, use_inverted)
                    for row_data in batch
                )
                all_results.extend(batch_results)
                pbar.update(len(batch))
        
        # Organize results by constraint count
        constraint_points = {}
        constraint_effect_sizes = {}
        constraint_cohens_d = {}
        constraint_n_points = {}
        constraint_t_ci_lower = {}
        constraint_t_ci_upper = {}
        
        for x_value, t_stat, cohens_d_val, effect_size_cat, n_points, success in all_results:
            if x_value not in constraint_points:
                constraint_points[x_value] = []
                constraint_effect_sizes[x_value] = []
                constraint_cohens_d[x_value] = []
                constraint_n_points[x_value] = []
                constraint_t_ci_lower[x_value] = []
                constraint_t_ci_upper[x_value] = []
            constraint_points[x_value].append(t_stat)
            constraint_effect_sizes[x_value].append(effect_size_cat)
            constraint_cohens_d[x_value].append(cohens_d_val)
            constraint_n_points[x_value].append(n_points)
        
        # Read CI values directly from dataframe
        for idx, row in df_all_params.iterrows():
            age_consideration = row['age_consideration']
            core_b_constraints_count = row['core_b_constraints_count']
            
            if age_consideration:
                x_value = core_b_constraints_count
            else:
                x_value = 0
            
            # Get t CI values (use inverted if needed for norm_dtw)
            if use_inverted:
                t_ci_lower = row.get('t_ci_lower_inv', row.get('t_ci_lower', np.nan))
                t_ci_upper = row.get('t_ci_upper_inv', row.get('t_ci_upper', np.nan))
            else:
                t_ci_lower = row.get('t_ci_lower', np.nan)
                t_ci_upper = row.get('t_ci_upper', np.nan)
            
            constraint_t_ci_lower[x_value].append(t_ci_lower)
            constraint_t_ci_upper[x_value].append(t_ci_upper)
        
        # Calculate improvement scores in parallel
        if not debug:  # Only show message when debug=False (mute_mode=False)
            print(f"  Calculating improvement scores...")
        sorted_constraints = sorted(constraint_points.keys())
        
        # Prepare data for parallel improvement score calculation
        constraint_pairs = []
        for i in range(len(sorted_constraints) - 1):
            current_constraint = sorted_constraints[i]
            next_constraint = sorted_constraints[i + 1]
            current_t_stats = constraint_points[current_constraint]
            next_t_stats = constraint_points[next_constraint]
            constraint_pairs.append((current_constraint, next_constraint, current_t_stats, next_t_stats))
        
        # Calculate improvement scores in parallel
        if constraint_pairs:
            with tqdm(total=len(constraint_pairs), desc="  Processing improvements", disable=debug) as pbar:
                improvement_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(calculate_improvement_scores_parallel)(pair_data, quality_index)
                    for pair_data in constraint_pairs
                )
                pbar.update(len(constraint_pairs))
            
            # Flatten results
            all_improvement_scores = []
            for scores in improvement_results:
                all_improvement_scores.extend(scores)
        else:
            all_improvement_scores = []
        
        # Normalize improvement scores
        if all_improvement_scores:
            max_abs_score = max(abs(score) for score in all_improvement_scores)
        else:
            max_abs_score = 1.0
        
        # Calculate total data points and connections for dynamic sizing
        total_points = sum(len(t_stats) for t_stats in constraint_points.values())
        total_connections = sum(len(current_t_stats) * len(next_t_stats) 
                              for current_t_stats, next_t_stats in 
                              [(constraint_points[sorted_constraints[i]], 
                                constraint_points[sorted_constraints[i+1]]) 
                               for i in range(len(sorted_constraints)-1)])
        
        # Get dynamic sizing parameters
        sizing = calculate_dynamic_sizes(total_points, total_connections)
        
        if not debug:  # Only show message when debug=False (mute_mode=False)
            print(f"  Drawing {total_points} points and {total_connections} connections...")
        
        # Move LinearSegmentedColormap import to top of file since we need it
        colors_list = ['#0066CC', '#e3e3e3', '#CC0000']  # Blue -> gray -> Red
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('improvement', colors_list, N=n_bins)
        
        # Prepare all line segments and colors for vectorized drawing
        if not debug:  # Only show message when debug=False (mute_mode=False)
            print(f"  Preparing line segments for vectorized drawing...")
        line_segments = []
        line_colors = []
        
        for i in range(len(sorted_constraints) - 1):
            current_constraint = sorted_constraints[i]
            next_constraint = sorted_constraints[i + 1]
            
            current_t_stats = constraint_points[current_constraint]
            next_t_stats = constraint_points[next_constraint]
            
            # Connect every dot in current constraint to every dot in next constraint
            for curr_t in current_t_stats:
                for next_t in next_t_stats:
                    # Calculate change in t-statistic
                    t_change = next_t - curr_t
                    
                    # Determine improvement/deterioration based on quality index
                    # When invert_norm_dtw=True, norm_dtw metrics follow corr_coef convention
                    is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                    use_negative_improvement = is_norm_dtw_metric and not invert_norm_dtw
                    
                    if use_negative_improvement:
                        improvement_score = -t_change  # Negative change is improvement (non-inverted norm_dtw)
                    else:
                        improvement_score = t_change   # Positive change is improvement
                    
                    # Normalize to [-1, 1] range for colormap
                    if max_abs_score > 0:
                        normalized_score = np.clip(improvement_score / max_abs_score, -1, 1)
                    else:
                        normalized_score = 0
                    
                    # Map to colormap (0 = blue/deterioration, 1 = red/improvement)
                    color_value = (normalized_score + 1) / 2  # Convert [-1,1] to [0,1]
                    color = cmap(color_value)
                    
                    # Store line segment and color
                    line_segments.append([[current_constraint, curr_t], [next_constraint, next_t]])
                    line_colors.append(color)
                    
                    # Store individual segment data for GIF creation if requested
                    if return_plot_info:
                        plot_info_dict[quality_index]['individual_segments'].append({
                            'segment': [[current_constraint, curr_t], [next_constraint, next_t]],
                            'color': color,
                            'alpha': sizing['alpha'],
                            'linewidth': sizing['line_width'],
                            'zorder': 1
                        })
        
        # Draw all connections at once using LineCollection (much faster than individual plot calls)
        if not debug:  # Only show message when debug=False (mute_mode=False)
            print(f"  Drawing {len(line_segments)} line segments using vectorized approach...")
        if line_segments:  # Only draw if we have segments
            lc = LineCollection(line_segments, colors=line_colors, alpha=sizing['alpha'], 
                              linewidths=sizing['line_width'], zorder=1)
            ax.add_collection(lc)
        
        # Draw light gray shaded area showing overall CI range across all constraint counts (zorder=0, lowest layer)
        # At each x position, find the absolute min and max across ALL CI values (both lower and upper bounds)
        ci_envelope_x = []
        ci_envelope_upper = []
        ci_envelope_lower = []
        
        for x_constraint in sorted(constraint_points.keys()):
            # Collect ALL CI values (both lower and upper bounds) and find the overall min/max
            all_ci_values = []
            for ci_low, ci_up in zip(constraint_t_ci_lower[x_constraint], constraint_t_ci_upper[x_constraint]):
                if not pd.isna(ci_low):
                    all_ci_values.append(ci_low)
                if not pd.isna(ci_up):
                    all_ci_values.append(ci_up)
            
            if all_ci_values:
                ci_envelope_x.append(x_constraint)
                ci_envelope_lower.append(min(all_ci_values))  # Absolute minimum of all CI values
                ci_envelope_upper.append(max(all_ci_values))  # Absolute maximum of all CI values
        
        # Draw continuous shaded polygon covering the full CI range from x=0 to x=max
        if len(ci_envelope_x) > 0:
            ax.fill_between(ci_envelope_x, ci_envelope_lower, ci_envelope_upper, 
                           color='lightgray', alpha=0.4, zorder=0, label='95% CI Range')
        
        # Draw dark gray error bars at each data point (zorder=2, under points but above shaded area)
        for x_constraint in constraint_points.keys():
            t_stats = constraint_points[x_constraint]
            ci_lowers = constraint_t_ci_lower[x_constraint]
            ci_uppers = constraint_t_ci_upper[x_constraint]
            
            for i, t_stat in enumerate(t_stats):
                ci_low = ci_lowers[i] if i < len(ci_lowers) else np.nan
                ci_up = ci_uppers[i] if i < len(ci_uppers) else np.nan
                
                if not pd.isna(ci_low) and not pd.isna(ci_up):
                    ax.plot([x_constraint, x_constraint], [ci_low, ci_up], 
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
                    # Add small horizontal caps at the ends
                    cap_width = 0.08
                    ax.plot([x_constraint - cap_width, x_constraint + cap_width], [ci_low, ci_low],
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
                    ax.plot([x_constraint - cap_width, x_constraint + cap_width], [ci_up, ci_up],
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Plot all individual points with white fill and black outline
        for x_constraint in constraint_points.keys():
            t_stats = constraint_points[x_constraint]
            
            for t_stat in t_stats:
                # Plot individual point with white fill and black outline
                ax.scatter(x_constraint, t_stat, color='white', edgecolor='black', 
                          linewidth=max(0.5, sizing['line_width']), s=sizing['dot_size'], zorder=3)
                
                # Store individual point data for GIF creation if requested
                if return_plot_info:
                    plot_info_dict[quality_index]['individual_points'].append({
                        'x': x_constraint,
                        'y': t_stat,
                        'color': 'white',
                        'edgecolor': 'black',
                        'linewidth': max(0.5, sizing['line_width']),
                        'size': sizing['dot_size'],
                        'zorder': 3
                    })
        
        # Add null hypothesis line
        ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        
        # Set x-axis
        all_constraints = df_all_params['core_b_constraints_count'].unique()
        x_min, x_max = 0, int(max(all_constraints))
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_xticks(range(0, x_max + 1))
        
        # Format plot
        ax.set_xlabel(f'Number of {CORE_B} Age Constraints')
        ax.set_ylabel('t-statistic')
        # Get display name for quality index
        def get_quality_display_name(quality_index):
            if quality_index == 'corr_coef':
                return "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                return "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                return "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                return "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                return quality_index
        
        display_name = get_quality_display_name(quality_index)
        ax.set_title(f'{display_name}\n{CORE_A} vs {CORE_B}')
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add inline text label for synthetic data line with white background box
        # Position text inside the graph at y=0 (on the dashed line), left-aligned with x=0
        ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
               color='dimgray', style='italic', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Store actual axis limits for GIF creation if requested  
        if return_plot_info:
            plot_info_dict[quality_index]['actual_plot_limits'] = {
                'x_min': ax.get_xlim()[0],
                'x_max': ax.get_xlim()[1], 
                'y_min': ax.get_ylim()[0],
                'y_max': ax.get_ylim()[1]
            }
            # Store display name for GIF titles
            plot_info_dict[quality_index]['display_name'] = display_name
            # Store CI data for GIF creation
            plot_info_dict[quality_index]['ci_envelope'] = {
                'x': ci_envelope_x,
                'lower': ci_envelope_lower,
                'upper': ci_envelope_upper
            }
            plot_info_dict[quality_index]['constraint_ci_data'] = {
                'ci_lower': dict(constraint_t_ci_lower),
                'ci_upper': dict(constraint_t_ci_upper)
            }
        
        # Add vertical colorbar for improvement/deterioration on right side
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-max_abs_score, vmax=max_abs_score))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.05)
        # Set bold label on the left side of the colorbar
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel('Change in t-statistic', fontweight='bold', labelpad=5, rotation=90)
        
        # Set colorbar ticks and labels
        cbar.set_ticks([-max_abs_score, 0, max_abs_score])
        cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
        
        # Add "Better Correlation" arrow pointing to the improvement direction
        # Position arrow vertically (parallel to y-axis) to the left of x=0
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Position arrow to the left of x=0
        arrow_x = -0.35  # Left of x=0
        arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2  # Center vertically
        
        # Determine arrow direction based on quality index
        # When invert_norm_dtw=True, norm_dtw metrics use upward arrow (like corr_coef)
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_downward_arrow:
            # For norm_dtw (non-inverted), lower values are better (downward arrow)
            arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
        else:
            # For other quality indices (or inverted norm_dtw), higher values are better (upward arrow)
            arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
            
        # Create gradient arrow using LineCollection
        n_segments = 100
        y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
        x_vals = np.full_like(y_vals, arrow_x)
        
        # Create line segments for gradient effect - exclude the last segment to leave space for arrowhead
        points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create colors for each segment - full spectrum of colormap
        colors_gradient = [cmap(i / (n_segments - 1)) for i in range(n_segments - 1)]
        
        # Create the gradient line
        lc = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
        ax.add_collection(lc)
        
        # Add arrowhead at the end
        arrow_color = cmap(1.0)
        arrowhead_start_y = y_vals[-2]
        arrowhead_end_y = arrow_y_end
        
        # Add just the arrowhead
        ax.annotate('', 
                   xy=(arrow_x, arrowhead_end_y), 
                   xytext=(arrow_x, arrowhead_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                   zorder=5)
        
        # Add text next to the arrow
        text_x = arrow_x + 0.1  # Slightly to the right of arrow
        text_y = arrow_y_center
        
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='left', va='center',
               rotation=90, color='black',
               zorder=4)
        
        plt.tight_layout()
        
        # Store plot info for gif creation if requested
        if return_plot_info:
            # Calculate axis limits for gif consistency
            if constraint_points:
                all_t_values = []
                for t_stats_list in constraint_points.values():
                    all_t_values.extend(t_stats_list)
                t_range = max(abs(min(all_t_values)), abs(max(all_t_values)))
                y_plot_min, y_plot_max = -t_range * 1.2, t_range * 1.2
            else:
                y_plot_min, y_plot_max = -5, 5
            
            # Get unique constraints
            unique_constraints = sorted(df_all_params['core_b_constraints_count'].unique())
            
            # Create constraint_t_stats mapping (use mean t-statistic for each constraint level)
            constraint_t_stats = {}
            for n_constraints in unique_constraints:
                if n_constraints in constraint_points:
                    # Use mean t-statistic for this constraint level
                    constraint_t_stats[n_constraints] = np.mean(constraint_points[n_constraints])
                else:
                    constraint_t_stats[n_constraints] = 0.0
            
            # Update the existing plot_info_dict entry (preserve individual_points and individual_segments)
            plot_info_dict[quality_index].update({
                'quality_index': quality_index,
                'CORE_A': CORE_A,
                'CORE_B': CORE_B,
                'unique_constraints': unique_constraints,
                'constraint_points': constraint_points,
                'constraint_effect_sizes': constraint_effect_sizes,
                'constraint_cohens_d': constraint_cohens_d,
                'constraint_n_points': constraint_n_points,
                'constraint_t_stats': constraint_t_stats,
                'line_segments': line_segments,
                'line_colors': line_colors,
                'sizing': sizing,
                'max_abs_score': max_abs_score,
                'cmap': cmap,
                'plot_limits': {
                    'x_min': -0.5,
                    'x_max': x_max + 0.5,
                    'y_min': y_plot_min,
                    'y_max': y_plot_max
                }
            })
        
        # Save figure
        if output_figure_filenames and quality_index in output_figure_filenames:
            t_stat_filename_base = output_figure_filenames[quality_index] + '_tstat'

            from .helpers import save_figure_formats
            save_dpi = dpi if dpi is not None else 150
            save_figure_formats(plt.gcf(), t_stat_filename_base, fig_format=fig_format, dpi=save_dpi)
        else:
            if debug:
                print(f"✓ t-statistics plot completed for {quality_index}")
            else:
                print(f"✓ t-statistics plot completed for {quality_index}")
        
        # Display the plot - suppress when debug=True (i.e., mute_mode=True)
        if not debug:  # debug=True means mute_mode=True, so suppress display
            plt.show()
    
    # Return plot info if requested
    if return_plot_info:
        return plot_info_dict
    else:
        return None


def plot_cohens_d_vs_constraints(quality_data, target_quality_indices, output_figure_filenames,
                                 CORE_A, CORE_B, debug=True, n_jobs=-1, batch_size=None, return_plot_info=False, 
                                 fig_format=['png'], dpi=None, invert_norm_dtw=True):
    """
    Plot Cohen's d vs number of age constraints for each quality index.
    Data points are colored by sample size (n_points) using a color spectrum.
    Connection lines are colored based on change in Cohen's d value.
    
    Parameters:
    -----------
    quality_data : dict
        Preprocessed quality data from load_and_prepare_quality_data function
    target_quality_indices : list
        List of quality indices to process (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    output_figure_filenames : dict
        Dictionary mapping quality_index to output figure filename paths
    CORE_A : str
        Name of core A
    CORE_B : str
        Name of core B
    debug : bool, default True
        If True, only print essential messages. If False, print all detailed messages.
    n_jobs : int, default -1
        Number of parallel jobs to run. -1 means using all available cores.
    batch_size : int, optional
        Batch size for parallel processing. If None, automatically determined.
    return_plot_info : bool, default False
        If True, return plotting information for gif creation
    fig_format : list, default ['png']
        List of file formats for saved figures
    dpi : int, optional
        Resolution for saved figures
    
    Returns:
    --------
    dict or None
        If return_plot_info=True, returns plotting information for gif creation
    """
    
    plot_info_dict = {}
    
    for quality_index in target_quality_indices:
        if quality_index not in quality_data:
            print(f"Skipping {quality_index} - no data available")
            continue
            
        data = quality_data[quality_index]
        df_all_params = data['df_all_params']
        combined_data = data['combined_data']
        
        if not debug:
            print(f"Processing Cohen's d for {quality_index} with {len(df_all_params)} scenarios...")
        
        # Initialize plot_info_dict entry if needed
        if return_plot_info:
            plot_info_dict[quality_index] = {
                'individual_points': [],
                'individual_segments': []
            }
        
        # Create plot
        fig, ax = plt.subplots(figsize=(9.5, 5))
        
        # Prepare data for parallel processing
        if batch_size is None:
            batch_size = max(50, len(df_all_params) // (n_jobs if n_jobs > 0 else 4))
        
        # Prepare row data for parallel processing
        row_data_list = []
        for idx, row in df_all_params.iterrows():
            row_data_list.append((idx, row, row['age_consideration'], row['core_b_constraints_count']))
        
        # Process data in parallel
        if not debug:
            if n_jobs == -1:
                print(f"  Calculating Cohen's d using all available cores...")
            else:
                print(f"  Calculating Cohen's d using {n_jobs} cores...")
        
        # Determine if we should use inverted columns for norm_dtw metrics
        use_inverted = invert_norm_dtw and quality_index in ['norm_dtw', 'norm_dtw_sect']
        
        with tqdm(total=len(row_data_list), desc="  Processing Cohen's d", disable=debug) as pbar:
            all_results = []
            for i in range(0, len(row_data_list), batch_size):
                batch = row_data_list[i:i+batch_size]
                
                batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(process_single_row_parallel)(row_data, combined_data, debug, use_inverted)
                    for row_data in batch
                )
                all_results.extend(batch_results)
                pbar.update(len(batch))
        
        # Organize results by constraint count
        constraint_cohens_d = {}
        constraint_n_points = {}
        constraint_d_ci_lower = {}
        constraint_d_ci_upper = {}
        
        for x_value, t_stat, cohens_d_val, effect_size_cat, n_points, success in all_results:
            if x_value not in constraint_cohens_d:
                constraint_cohens_d[x_value] = []
                constraint_n_points[x_value] = []
                constraint_d_ci_lower[x_value] = []
                constraint_d_ci_upper[x_value] = []
            constraint_cohens_d[x_value].append(cohens_d_val)
            constraint_n_points[x_value].append(n_points)
        
        # Read CI values directly from dataframe
        for idx, row in df_all_params.iterrows():
            age_consideration = row['age_consideration']
            core_b_constraints_count = row['core_b_constraints_count']
            
            if age_consideration:
                x_value = core_b_constraints_count
            else:
                x_value = 0
            
            # Get Cohen's d CI values (use inverted if needed for norm_dtw)
            if use_inverted:
                d_ci_lower = row.get('d_ci_lower_inv', row.get('d_ci_lower', np.nan))
                d_ci_upper = row.get('d_ci_upper_inv', row.get('d_ci_upper', np.nan))
            else:
                d_ci_lower = row.get('d_ci_lower', np.nan)
                d_ci_upper = row.get('d_ci_upper', np.nan)
            
            constraint_d_ci_lower[x_value].append(d_ci_lower)
            constraint_d_ci_upper[x_value].append(d_ci_upper)
        
        # Calculate improvement scores for Cohen's d
        sorted_constraints = sorted(constraint_cohens_d.keys())
        
        # Calculate improvement scores based on Cohen's d change
        all_improvement_scores = []
        for i in range(len(sorted_constraints) - 1):
            current_constraint = sorted_constraints[i]
            next_constraint = sorted_constraints[i + 1]
            current_d_values = constraint_cohens_d[current_constraint]
            next_d_values = constraint_cohens_d[next_constraint]
            
            for curr_d in current_d_values:
                for next_d in next_d_values:
                    d_change = next_d - curr_d
                    # When invert_norm_dtw=True, norm_dtw metrics follow corr_coef convention
                    is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                    use_negative_improvement = is_norm_dtw_metric and not invert_norm_dtw
                    
                    if use_negative_improvement:
                        improvement_score = -d_change  # Non-inverted norm_dtw: negative change is improvement
                    else:
                        improvement_score = d_change   # Positive change is improvement
                    all_improvement_scores.append(improvement_score)
        
        # Normalize improvement scores
        if all_improvement_scores:
            max_abs_score = max(abs(score) for score in all_improvement_scores)
        else:
            max_abs_score = 1.0
        
        # Calculate total data points and connections for dynamic sizing
        total_points = sum(len(d_vals) for d_vals in constraint_cohens_d.values())
        total_connections = sum(len(constraint_cohens_d[sorted_constraints[i]]) * 
                              len(constraint_cohens_d[sorted_constraints[i+1]]) 
                              for i in range(len(sorted_constraints)-1))
        
        # Get dynamic sizing parameters
        sizing = calculate_dynamic_sizes(total_points, total_connections)
        
        if not debug:
            print(f"  Drawing {total_points} points and {total_connections} connections...")
        
        # Create colormap for improvement/deterioration (connection lines)
        colors_list = ['#0066CC', '#e3e3e3', '#CC0000']  # Blue -> gray -> Red
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('improvement', colors_list, N=n_bins)
        
        # Prepare all line segments and colors for vectorized drawing
        if not debug:
            print(f"  Preparing line segments for vectorized drawing...")
        line_segments = []
        line_colors = []
        
        # Determine if we should use negative improvement logic for norm_dtw
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_negative_improvement = is_norm_dtw_metric and not invert_norm_dtw
        
        for i in range(len(sorted_constraints) - 1):
            current_constraint = sorted_constraints[i]
            next_constraint = sorted_constraints[i + 1]
            
            current_d_values = constraint_cohens_d[current_constraint]
            next_d_values = constraint_cohens_d[next_constraint]
            
            # Connect every dot in current constraint to every dot in next constraint
            for curr_d in current_d_values:
                for next_d in next_d_values:
                    # Calculate change in Cohen's d
                    d_change = next_d - curr_d
                    
                    # Determine improvement/deterioration based on quality index
                    if use_negative_improvement:
                        improvement_score = -d_change  # Non-inverted norm_dtw: negative change is improvement
                    else:
                        improvement_score = d_change   # Positive change is improvement
                    
                    # Normalize to [-1, 1] range for colormap
                    if max_abs_score > 0:
                        normalized_score = np.clip(improvement_score / max_abs_score, -1, 1)
                    else:
                        normalized_score = 0
                    
                    # Map to colormap (0 = blue/deterioration, 1 = red/improvement)
                    color_value = (normalized_score + 1) / 2  # Convert [-1,1] to [0,1]
                    color = cmap(color_value)
                    
                    # Store line segment and color
                    line_segments.append([[current_constraint, curr_d], [next_constraint, next_d]])
                    line_colors.append(color)
                    
                    # Store individual segment data for GIF creation if requested
                    if return_plot_info:
                        plot_info_dict[quality_index]['individual_segments'].append({
                            'segment': [[current_constraint, curr_d], [next_constraint, next_d]],
                            'color': color,
                            'alpha': sizing['alpha'],
                            'linewidth': sizing['line_width'],
                            'zorder': 1
                        })
        
        # Draw all connections at once using LineCollection
        if not debug:
            print(f"  Drawing {len(line_segments)} line segments using vectorized approach...")
        if line_segments:
            lc = LineCollection(line_segments, colors=line_colors, alpha=sizing['alpha'], 
                              linewidths=sizing['line_width'], zorder=1)
            ax.add_collection(lc)
        
        # Draw light gray shaded area showing overall CI range across all constraint counts (zorder=0, lowest layer)
        # At each x position, find the absolute min and max across ALL CI values (both lower and upper bounds)
        ci_envelope_x = []
        ci_envelope_upper = []
        ci_envelope_lower = []
        
        for x_constraint in sorted(constraint_cohens_d.keys()):
            # Collect ALL CI values (both lower and upper bounds) and find the overall min/max
            all_ci_values = []
            for ci_low, ci_up in zip(constraint_d_ci_lower[x_constraint], constraint_d_ci_upper[x_constraint]):
                if not pd.isna(ci_low):
                    all_ci_values.append(ci_low)
                if not pd.isna(ci_up):
                    all_ci_values.append(ci_up)
            
            if all_ci_values:
                ci_envelope_x.append(x_constraint)
                ci_envelope_lower.append(min(all_ci_values))  # Absolute minimum of all CI values
                ci_envelope_upper.append(max(all_ci_values))  # Absolute maximum of all CI values
        
        # Draw continuous shaded polygon covering the full CI range from x=0 to x=max
        if len(ci_envelope_x) > 0:
            ax.fill_between(ci_envelope_x, ci_envelope_lower, ci_envelope_upper, 
                           color='lightgray', alpha=0.4, zorder=0, label='95% CI Range')
        
        # Draw dark gray error bars at each data point (zorder=2, under points but above shaded area)
        for x_constraint in constraint_cohens_d.keys():
            d_values = constraint_cohens_d[x_constraint]
            ci_lowers = constraint_d_ci_lower[x_constraint]
            ci_uppers = constraint_d_ci_upper[x_constraint]
            
            for i, cohens_d_val in enumerate(d_values):
                ci_low = ci_lowers[i] if i < len(ci_lowers) else np.nan
                ci_up = ci_uppers[i] if i < len(ci_uppers) else np.nan
                
                if not pd.isna(ci_low) and not pd.isna(ci_up):
                    ax.plot([x_constraint, x_constraint], [ci_low, ci_up], 
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
                    # Add small horizontal caps at the ends
                    cap_width = 0.08
                    ax.plot([x_constraint - cap_width, x_constraint + cap_width], [ci_low, ci_low],
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
                    ax.plot([x_constraint - cap_width, x_constraint + cap_width], [ci_up, ci_up],
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Plot all individual points with white fill and black outline
        for x_constraint in constraint_cohens_d.keys():
            d_values = constraint_cohens_d[x_constraint]
            
            for cohens_d_val in d_values:
                # Plot individual point with white fill and black outline
                ax.scatter(x_constraint, cohens_d_val, color='white', edgecolor='black', 
                          linewidth=max(0.5, sizing['line_width']), s=sizing['dot_size'], zorder=3)
                
                # Store individual point data for GIF creation if requested
                if return_plot_info:
                    plot_info_dict[quality_index]['individual_points'].append({
                        'x': x_constraint,
                        'y': cohens_d_val,
                        'color': 'white',
                        'edgecolor': 'black',
                        'linewidth': max(0.5, sizing['line_width']),
                        'size': sizing['dot_size'],
                        'zorder': 3
                    })
        
        # Add null hypothesis line (d=0)
        ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        
        # Set x-axis
        all_constraints = df_all_params['core_b_constraints_count'].unique()
        x_min, x_max = 0, int(max(all_constraints))
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_xticks(range(0, x_max + 1))
        
        # Format plot
        ax.set_xlabel(f'Number of {CORE_B} Age Constraints')
        ax.set_ylabel("Cohen's d")
        
        # Get display name for quality index
        def get_quality_display_name(quality_index):
            if quality_index == 'corr_coef':
                return "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                return "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                return "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                return "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                return quality_index
        
        display_name = get_quality_display_name(quality_index)
        ax.set_title(f"{display_name}\n{CORE_A} vs {CORE_B}")
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add inline text label for synthetic data line with white background box
        # Position text inside the graph at y=0 (on the dashed line), left-aligned with x=0
        ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
               color='dimgray', style='italic', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Store actual axis limits for GIF creation if requested  
        if return_plot_info:
            plot_info_dict[quality_index]['actual_plot_limits'] = {
                'x_min': ax.get_xlim()[0],
                'x_max': ax.get_xlim()[1], 
                'y_min': ax.get_ylim()[0],
                'y_max': ax.get_ylim()[1]
            }
            plot_info_dict[quality_index]['display_name'] = display_name
            # Store CI data for GIF creation
            plot_info_dict[quality_index]['ci_envelope'] = {
                'x': ci_envelope_x,
                'lower': ci_envelope_lower,
                'upper': ci_envelope_upper
            }
            plot_info_dict[quality_index]['constraint_ci_data'] = {
                'ci_lower': dict(constraint_d_ci_lower),
                'ci_upper': dict(constraint_d_ci_upper)
            }
        
        # Add vertical colorbar for improvement/deterioration (connection lines) on right side
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-max_abs_score, vmax=max_abs_score))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.05)
        # Set bold label on the left side of the colorbar
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel("Change in Cohen's d", fontweight='bold', labelpad=5, rotation=90)
        
        # Set colorbar ticks and labels
        cbar.set_ticks([-max_abs_score, 0, max_abs_score])
        cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
        
        # Add "Better Correlation" arrow pointing to the improvement direction
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Position arrow to the left of x=0
        arrow_x = -0.35
        arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2
        
        # Determine arrow direction based on quality index
        # When invert_norm_dtw=True, norm_dtw metrics use upward arrow (like corr_coef)
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_downward_arrow:
            # For norm_dtw (non-inverted), lower values are better (downward arrow)
            arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
        else:
            # For other quality indices (or inverted norm_dtw), higher values are better (upward arrow)
            arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
            
        # Create gradient arrow using LineCollection
        n_segments = 100
        y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
        x_vals = np.full_like(y_vals, arrow_x)
        
        # Create line segments for gradient effect
        points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create colors for each segment
        colors_gradient = [cmap(i / (n_segments - 1)) for i in range(n_segments - 1)]
        
        # Create the gradient line
        lc_arrow = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
        ax.add_collection(lc_arrow)
        
        # Add arrowhead at the end
        arrow_color = cmap(1.0)
        arrowhead_start_y = y_vals[-2]
        arrowhead_end_y = arrow_y_end
        
        ax.annotate('', 
                   xy=(arrow_x, arrowhead_end_y), 
                   xytext=(arrow_x, arrowhead_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                   zorder=5)
        
        # Add text next to the arrow
        text_x = arrow_x + 0.1
        text_y = arrow_y_center
        
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='left', va='center',
               rotation=90, color='black',
               zorder=4)
        
        plt.tight_layout()
        
        # Store plot info for gif creation if requested
        if return_plot_info:
            # Calculate axis limits for gif consistency
            if constraint_cohens_d:
                all_d_values = []
                for d_list in constraint_cohens_d.values():
                    all_d_values.extend(d_list)
                d_range = max(abs(min(all_d_values)), abs(max(all_d_values)))
                y_plot_min, y_plot_max = -d_range * 1.2, d_range * 1.2
            else:
                y_plot_min, y_plot_max = -2, 2
            
            unique_constraints = sorted(df_all_params['core_b_constraints_count'].unique())
            
            plot_info_dict[quality_index].update({
                'quality_index': quality_index,
                'CORE_A': CORE_A,
                'CORE_B': CORE_B,
                'unique_constraints': unique_constraints,
                'constraint_cohens_d': constraint_cohens_d,
                'constraint_n_points': constraint_n_points,
                'line_segments': line_segments,
                'line_colors': line_colors,
                'sizing': sizing,
                'max_abs_score': max_abs_score,
                'cmap': cmap,
                'plot_limits': {
                    'x_min': -0.5,
                    'x_max': x_max + 0.5,
                    'y_min': y_plot_min,
                    'y_max': y_plot_max
                }
            })
        
        # Save figure
        if output_figure_filenames and quality_index in output_figure_filenames:
            base_filename = output_figure_filenames[quality_index]
            cohens_d_filename_base = base_filename + '_cohens_d'
            
            # Create directory if needed
            from .helpers import save_figure_formats
            save_dpi = dpi if dpi is not None else 150
            save_figure_formats(plt.gcf(), cohens_d_filename_base, fig_format=fig_format, dpi=save_dpi)
        else:
            if debug:
                print(f"✓ Cohen's d plot completed for {quality_index}")
            else:
                print(f"✓ Cohen's d plot completed for {quality_index}")
        
        # Display the plot - suppress when debug=True
        if not debug:
            plt.show()
    
    # Return plot info if requested
    if return_plot_info:
        return plot_info_dict
    else:
        return None


def plot_hedges_g_vs_constraints(quality_data, target_quality_indices, output_figure_filenames,
                                 CORE_A, CORE_B, debug=True, n_jobs=-1, batch_size=None, return_plot_info=False, 
                                 fig_format=['png'], dpi=None, invert_norm_dtw=True, linear_g_scale=False):
    """
    Plot Hedges' g vs number of age constraints for each quality index.
    Hedges' g is a bias-corrected version of Cohen's d, more appropriate for smaller samples.
    
    Parameters:
    -----------
    quality_data : dict
        Preprocessed quality data from load_and_prepare_quality_data function
    target_quality_indices : list
        List of quality indices to process (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    output_figure_filenames : dict
        Dictionary mapping quality_index to output figure filename paths
    CORE_A : str
        Name of core A
    CORE_B : str
        Name of core B
    debug : bool, default True
        If True, only print essential messages. If False, print all detailed messages.
    n_jobs : int, default -1
        Number of parallel jobs to run. -1 means using all available cores.
    batch_size : int, optional
        Batch size for parallel processing. If None, automatically determined.
    return_plot_info : bool, default False
        If True, return plotting information for gif creation
    fig_format : list, default ['png']
        List of file formats for saved figures
    dpi : int, optional
        Resolution for saved figures
    invert_norm_dtw : bool, default True
        If True, use inverted hedges_g_inv for norm_dtw metrics
    linear_g_scale : bool, default False
        If False, plot with linear Hedges' g scale on left y-axis.
        If True, rescale y-axis so Probability of Superiority (PS) is linear (0-100%),
        with right y-axis showing PS at 20% intervals, and left y-axis showing
        corresponding g values at each PS tick.
    
    Returns:
    --------
    dict or None
        If return_plot_info=True, returns plotting information for gif creation
    """
    
    plot_info_dict = {}
    
    for quality_index in target_quality_indices:
        if quality_index not in quality_data:
            print(f"Skipping {quality_index} - no data available")
            continue
            
        data = quality_data[quality_index]
        df_all_params = data['df_all_params']
        combined_data = data['combined_data']
        
        if not debug:
            print(f"Processing Hedges' g for {quality_index} with {len(df_all_params)} scenarios...")
        
        # Initialize plot_info_dict entry if needed
        if return_plot_info:
            plot_info_dict[quality_index] = {
                'individual_points': [],
                'individual_segments': []
            }
        
        # Create plot
        fig, ax = plt.subplots(figsize=(9.5, 5))
        
        # Prepare data for parallel processing
        if batch_size is None:
            batch_size = max(50, len(df_all_params) // (n_jobs if n_jobs > 0 else 4))
        
        # Determine if we should use inverted columns for norm_dtw metrics
        use_inverted = invert_norm_dtw and quality_index in ['norm_dtw', 'norm_dtw_sect']
        
        # Organize results by constraint count - read hedges_g from dataframe
        constraint_hedges_g = {}
        constraint_n_points = {}
        constraint_g_ci_lower = {}
        constraint_g_ci_upper = {}
        # Also read prob_superiority and PS CI bounds directly from CSV
        constraint_prob_superiority = {}
        constraint_ps_ci_lower = {}
        constraint_ps_ci_upper = {}
        
        for idx, row in df_all_params.iterrows():
            age_consideration = row['age_consideration']
            core_b_constraints_count = row['core_b_constraints_count']
            
            if age_consideration:
                x_value = core_b_constraints_count
            else:
                x_value = 0
            
            # Get hedges_g value from CSV (use inverted if needed)
            if use_inverted:
                hedges_g_val = row.get('g_inv', row.get('g', 0.0))
                g_ci_lower = row.get('g_ci_lower_inv', row.get('g_ci_lower', np.nan))
                g_ci_upper = row.get('g_ci_upper_inv', row.get('g_ci_upper', np.nan))
                # Get pre-calculated PS values from CSV (inverted)
                ps_val = row.get('ps_inv', row.get('ps', 0.5))
                ps_ci_lower = row.get('ps_ci_lower_inv', row.get('ps_ci_lower', np.nan))
                ps_ci_upper = row.get('ps_ci_upper_inv', row.get('ps_ci_upper', np.nan))
            else:
                hedges_g_val = row.get('g', 0.0)
                g_ci_lower = row.get('g_ci_lower', np.nan)
                g_ci_upper = row.get('g_ci_upper', np.nan)
                # Get pre-calculated PS values from CSV
                ps_val = row.get('ps', 0.5)
                ps_ci_lower = row.get('ps_ci_lower', np.nan)
                ps_ci_upper = row.get('ps_ci_upper', np.nan)
            
            if pd.isna(hedges_g_val):
                hedges_g_val = 0.0
            if pd.isna(ps_val):
                ps_val = 0.5
            
            n_points = row.get('n_points', 0)
            if pd.isna(n_points):
                n_points = 0
            
            if x_value not in constraint_hedges_g:
                constraint_hedges_g[x_value] = []
                constraint_n_points[x_value] = []
                constraint_g_ci_lower[x_value] = []
                constraint_g_ci_upper[x_value] = []
                constraint_prob_superiority[x_value] = []
                constraint_ps_ci_lower[x_value] = []
                constraint_ps_ci_upper[x_value] = []
            constraint_hedges_g[x_value].append(hedges_g_val)
            constraint_n_points[x_value].append(n_points)
            constraint_g_ci_lower[x_value].append(g_ci_lower)
            constraint_g_ci_upper[x_value].append(g_ci_upper)
            constraint_prob_superiority[x_value].append(ps_val)
            constraint_ps_ci_lower[x_value].append(ps_ci_lower)
            constraint_ps_ci_upper[x_value].append(ps_ci_upper)
        
        # Define conversion functions for linear PS scale (used for axis labels)
        def g_to_ps(g):
            """Convert Hedges' g to Probability of Superiority (0-100%)"""
            return stats.norm.cdf(g / np.sqrt(2)) * 100
        
        def ps_to_g(ps):
            """Convert Probability of Superiority (0-100%) to Hedges' g"""
            return stats.norm.ppf(ps / 100) * np.sqrt(2)
        
        # If linear_g_scale=False, use pre-calculated PS values from CSV; if True, use normal g scale
        if not linear_g_scale:
            # Use pre-calculated PS values from CSV (convert from 0-1 to 0-100% scale)
            # If PS CI values are not available in CSV, compute from g CI values as fallback
            constraint_plot_y = {}
            constraint_plot_ci_lower = {}
            constraint_plot_ci_upper = {}
            for x_val in constraint_hedges_g.keys():
                # PS values from CSV are in 0-1 scale, convert to 0-100% for plotting
                ps_y_values = []
                for i, ps in enumerate(constraint_prob_superiority[x_val]):
                    if not pd.isna(ps):
                        ps_y_values.append(ps * 100)
                    else:
                        # Fallback: compute from g value
                        g_val = constraint_hedges_g[x_val][i] if i < len(constraint_hedges_g[x_val]) else 0.0
                        ps_y_values.append(g_to_ps(g_val))
                constraint_plot_y[x_val] = ps_y_values
                
                # For CI bounds, use pre-calculated PS CI if available, otherwise compute from g CI
                plot_ci_lowers = []
                plot_ci_uppers = []
                for i in range(len(constraint_hedges_g[x_val])):
                    ps_ci_low = constraint_ps_ci_lower[x_val][i] if i < len(constraint_ps_ci_lower[x_val]) else np.nan
                    ps_ci_up = constraint_ps_ci_upper[x_val][i] if i < len(constraint_ps_ci_upper[x_val]) else np.nan
                    g_ci_low = constraint_g_ci_lower[x_val][i] if i < len(constraint_g_ci_lower[x_val]) else np.nan
                    g_ci_up = constraint_g_ci_upper[x_val][i] if i < len(constraint_g_ci_upper[x_val]) else np.nan
                    
                    # Use PS CI from CSV if available, otherwise compute from g CI
                    if not pd.isna(ps_ci_low):
                        plot_ci_lowers.append(ps_ci_low * 100)
                    elif not pd.isna(g_ci_low):
                        plot_ci_lowers.append(g_to_ps(g_ci_low))
                    else:
                        # Fallback to data point value if no CI available
                        plot_ci_lowers.append(constraint_plot_y[x_val][i] if i < len(constraint_plot_y[x_val]) else 50.0)
                    
                    if not pd.isna(ps_ci_up):
                        plot_ci_uppers.append(ps_ci_up * 100)
                    elif not pd.isna(g_ci_up):
                        plot_ci_uppers.append(g_to_ps(g_ci_up))
                    else:
                        # Fallback to data point value if no CI available
                        plot_ci_uppers.append(constraint_plot_y[x_val][i] if i < len(constraint_plot_y[x_val]) else 50.0)
                
                constraint_plot_ci_lower[x_val] = plot_ci_lowers
                constraint_plot_ci_upper[x_val] = plot_ci_uppers
        else:
            # Use original g values
            constraint_plot_y = constraint_hedges_g
            constraint_plot_ci_lower = constraint_g_ci_lower
            constraint_plot_ci_upper = constraint_g_ci_upper
        
        # Calculate improvement scores for Hedges' g
        sorted_constraints = sorted(constraint_hedges_g.keys())
        
        # Calculate improvement scores based on Hedges' g change
        all_improvement_scores = []
        for i in range(len(sorted_constraints) - 1):
            current_constraint = sorted_constraints[i]
            next_constraint = sorted_constraints[i + 1]
            current_g_values = constraint_hedges_g[current_constraint]
            next_g_values = constraint_hedges_g[next_constraint]
            
            for curr_g in current_g_values:
                for next_g in next_g_values:
                    g_change = next_g - curr_g
                    # When invert_norm_dtw=True, norm_dtw metrics follow corr_coef convention
                    is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                    use_negative_improvement = is_norm_dtw_metric and not invert_norm_dtw
                    
                    if use_negative_improvement:
                        improvement_score = -g_change
                    else:
                        improvement_score = g_change
                    all_improvement_scores.append(improvement_score)
        
        # Normalize improvement scores
        if all_improvement_scores:
            max_abs_score = max(abs(score) for score in all_improvement_scores)
        else:
            max_abs_score = 1.0
        
        # Calculate total data points and connections for dynamic sizing
        total_points = sum(len(g_vals) for g_vals in constraint_hedges_g.values())
        total_connections = sum(len(constraint_hedges_g[sorted_constraints[i]]) * 
                              len(constraint_hedges_g[sorted_constraints[i+1]]) 
                              for i in range(len(sorted_constraints)-1))
        
        # Get dynamic sizing parameters
        sizing = calculate_dynamic_sizes(total_points, total_connections)
        
        if not debug:
            print(f"  Drawing {total_points} points and {total_connections} connections...")
        
        # Create colormap for improvement/deterioration (connection lines)
        colors_list = ['#0066CC', '#e3e3e3', '#CC0000']  # Blue -> gray -> Red
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('improvement', colors_list, N=n_bins)
        
        # Prepare all line segments and colors for vectorized drawing
        if not debug:
            print(f"  Preparing line segments for vectorized drawing...")
        line_segments = []
        line_colors = []
        
        # Determine if we should use negative improvement logic for norm_dtw
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_negative_improvement = is_norm_dtw_metric and not invert_norm_dtw
        
        for i in range(len(sorted_constraints) - 1):
            current_constraint = sorted_constraints[i]
            next_constraint = sorted_constraints[i + 1]
            
            current_g_values = constraint_hedges_g[current_constraint]
            next_g_values = constraint_hedges_g[next_constraint]
            current_plot_values = constraint_plot_y[current_constraint]
            next_plot_values = constraint_plot_y[next_constraint]
            
            for idx_curr, curr_g in enumerate(current_g_values):
                curr_plot_y = current_plot_values[idx_curr]
                for idx_next, next_g in enumerate(next_g_values):
                    next_plot_y = next_plot_values[idx_next]
                    g_change = next_g - curr_g
                    
                    if use_negative_improvement:
                        improvement_score = -g_change
                    else:
                        improvement_score = g_change
                    
                    if max_abs_score > 0:
                        normalized_score = np.clip(improvement_score / max_abs_score, -1, 1)
                    else:
                        normalized_score = 0
                    
                    color_value = (normalized_score + 1) / 2
                    color = cmap(color_value)
                    
                    # Use plot y-values (transformed if linear_g_scale=True)
                    line_segments.append([[current_constraint, curr_plot_y], [next_constraint, next_plot_y]])
                    line_colors.append(color)
                    
                    if return_plot_info:
                        plot_info_dict[quality_index]['individual_segments'].append({
                            'segment': [[current_constraint, curr_plot_y], [next_constraint, next_plot_y]],
                            'color': color,
                            'alpha': sizing['alpha'],
                            'linewidth': sizing['line_width'],
                            'zorder': 1
                        })
        
        # Draw all connections at once using LineCollection
        if not debug:
            print(f"  Drawing {len(line_segments)} line segments using vectorized approach...")
        if line_segments:
            lc = LineCollection(line_segments, colors=line_colors, alpha=sizing['alpha'], 
                              linewidths=sizing['line_width'], zorder=1)
            ax.add_collection(lc)
        
        # Draw light gray shaded area showing overall CI range across all constraint counts (zorder=0, lowest layer)
        # At each x position, find the absolute min and max across ALL CI values (both lower and upper bounds)
        ci_envelope_x = []
        ci_envelope_upper = []
        ci_envelope_lower = []
        
        for x_constraint in sorted(constraint_plot_y.keys()):
            # Collect ALL CI values (both lower and upper bounds) and find the overall min/max
            # Use transformed CI values for plotting
            all_ci_values = []
            for ci_low, ci_up in zip(constraint_plot_ci_lower[x_constraint], constraint_plot_ci_upper[x_constraint]):
                if not pd.isna(ci_low):
                    all_ci_values.append(ci_low)
                if not pd.isna(ci_up):
                    all_ci_values.append(ci_up)
            
            if all_ci_values:
                ci_envelope_x.append(x_constraint)
                ci_envelope_lower.append(min(all_ci_values))  # Absolute minimum of all CI values
                ci_envelope_upper.append(max(all_ci_values))  # Absolute maximum of all CI values
        
        # Draw continuous shaded polygon covering the full CI range from x=0 to x=max
        if len(ci_envelope_x) > 0:
            ax.fill_between(ci_envelope_x, ci_envelope_lower, ci_envelope_upper, 
                           color='lightgray', alpha=0.4, zorder=0, label='95% CI Range')
        
        # Draw dark gray error bars at each data point (zorder=2, under points but above shaded area)
        for x_constraint in constraint_plot_y.keys():
            plot_y_values = constraint_plot_y[x_constraint]
            ci_lowers = constraint_plot_ci_lower[x_constraint]
            ci_uppers = constraint_plot_ci_upper[x_constraint]
            
            for i, plot_y_val in enumerate(plot_y_values):
                ci_low = ci_lowers[i] if i < len(ci_lowers) else np.nan
                ci_up = ci_uppers[i] if i < len(ci_uppers) else np.nan
                
                if not pd.isna(ci_low) and not pd.isna(ci_up):
                    ax.plot([x_constraint, x_constraint], [ci_low, ci_up], 
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
                    # Add small horizontal caps at the ends
                    cap_width = 0.08
                    ax.plot([x_constraint - cap_width, x_constraint + cap_width], [ci_low, ci_low],
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
                    ax.plot([x_constraint - cap_width, x_constraint + cap_width], [ci_up, ci_up],
                           color='dimgray', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Plot all individual points with white fill and black outline
        for x_constraint in constraint_plot_y.keys():
            plot_y_values = constraint_plot_y[x_constraint]
            
            for plot_y_val in plot_y_values:
                ax.scatter(x_constraint, plot_y_val, color='white', edgecolor='black', 
                          linewidth=max(0.5, sizing['line_width']), s=sizing['dot_size'], zorder=3)
                
                if return_plot_info:
                    plot_info_dict[quality_index]['individual_points'].append({
                        'x': x_constraint,
                        'y': plot_y_val,
                        'color': 'white',
                        'edgecolor': 'black',
                        'linewidth': max(0.5, sizing['line_width']),
                        'size': sizing['dot_size'],
                        'zorder': 3
                    })
        
        # Add null hypothesis line (g=0 corresponds to PS=50%)
        null_line_y = g_to_ps(0) if not linear_g_scale else 0
        ax.axhline(y=null_line_y, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        
        # Set x-axis
        all_constraints = df_all_params['core_b_constraints_count'].unique()
        x_min, x_max = 0, int(max(all_constraints))
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_xticks(range(0, x_max + 1))
        
        # Format plot based on linear_g_scale
        ax.set_xlabel(f'Number of {CORE_B} Age Constraints')
        
        if not linear_g_scale:
            # Linear PS scale: y-axis is 0.25% to 99.75% (PS) to avoid -inf/inf at edges
            ax.set_ylim(0.25, 99.75)
            # PS ticks at 10% intervals plus 0.25% and 99.75% at the ends
            ps_ticks = [0.25] + list(range(10, 100, 10)) + [99.75]  # [0.25, 10, 20, ..., 90, 99.75]
            ax.set_yticks(ps_ticks)
            # Left y-axis shows corresponding g values
            g_tick_labels = [f"{ps_to_g(ps):.2f}" for ps in ps_ticks]
            ax.set_yticklabels(g_tick_labels)
            ax.set_ylabel("Hedges' g")
        else:
            # Original g scale
            ax.set_ylabel("Hedges' g")
            ax.set_ylim(-4, 4)  # Set y-axis range for Hedges' g
            ax.set_yticks(np.arange(-4, 4.5, 1.0))  # Set y-axis ticks at every 1.0 interval
        
        # Get display name for quality index
        def get_quality_display_name(quality_index):
            if quality_index == 'corr_coef':
                return "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                return "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                return "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                return "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                return quality_index
        
        display_name = get_quality_display_name(quality_index)
        ax.set_title(f"{display_name}\n{CORE_A} vs {CORE_B}")
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add inline text label for synthetic data line
        synthetic_text_y = g_to_ps(0) if not linear_g_scale else 0
        ax.text(0, synthetic_text_y, 'Synthetic Data', fontsize=9, ha='left', va='center',
               color='dimgray', style='italic', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Store actual axis limits for GIF creation if requested  
        if return_plot_info:
            plot_info_dict[quality_index]['actual_plot_limits'] = {
                'x_min': ax.get_xlim()[0],
                'x_max': ax.get_xlim()[1], 
                'y_min': ax.get_ylim()[0],
                'y_max': ax.get_ylim()[1]
            }
            plot_info_dict[quality_index]['display_name'] = display_name
            # Store CI data for GIF creation
            plot_info_dict[quality_index]['ci_envelope'] = {
                'x': ci_envelope_x,
                'lower': ci_envelope_lower,
                'upper': ci_envelope_upper
            }
            plot_info_dict[quality_index]['constraint_ci_data'] = {
                'ci_lower': dict(constraint_g_ci_lower),
                'ci_upper': dict(constraint_g_ci_upper)
            }
        
        # Add vertical colorbar for improvement/deterioration on right side
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-max_abs_score, vmax=max_abs_score))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.14)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel("Change in Hedges' g", fontweight='bold', labelpad=5, rotation=90)
        
        cbar.set_ticks([-max_abs_score, 0, max_abs_score])
        cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
        
        # Add secondary y-axis for Probability of Superiority (as %)
        # PS = Φ(g / √2) where Φ is the standard normal CDF
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())  # Match the primary y-axis limits
        
        if not linear_g_scale:
            # For linear PS scale: right y-axis shows PS% directly at 10% intervals plus ends
            ps_ticks = [0.25] + list(range(10, 100, 10)) + [99.75]  # [0.25, 10, 20, ..., 90, 99.75]
            ax2.set_yticks(ps_ticks)
            # Format tick labels: show decimal for end ticks, whole number for middle ticks
            ps_labels = []
            for ps in ps_ticks:
                if ps == 0.25 or ps == 99.75:
                    ps_labels.append(f'{ps:.2f}%')
                else:
                    ps_labels.append(f'{ps:.0f}%')
            ax2.set_yticklabels(ps_labels)
            ax2.set_ylabel('Probability of Superiority', fontsize=10)
        else:
            # Original: Get the primary y-axis ticks and convert to PS values
            primary_ticks = ax.get_yticks()
            # Filter ticks to those within the axis limits
            y_min, y_max = ax.get_ylim()
            valid_ticks = [t for t in primary_ticks if y_min <= t <= y_max]
            
            # Calculate corresponding PS values for each g tick (as percentage)
            ps_values = [stats.norm.cdf(g / np.sqrt(2)) * 100 for g in valid_ticks]
            
            # Set secondary axis ticks at the same positions as primary
            ax2.set_yticks(valid_ticks)
            ax2.set_yticklabels([f'{ps:.0f}%' for ps in ps_values])
            ax2.set_ylabel('Probability of Superiority', fontsize=10)
        
        # Move the secondary y-axis spine inward so it doesn't overlap with colorbar
        ax2.spines['right'].set_position(('outward', 0))
        
        # Add "Better Correlation" arrow
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        arrow_x = -0.35
        arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2
        
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_downward_arrow:
            arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
        else:
            arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
            
        n_segments = 100
        y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
        x_vals = np.full_like(y_vals, arrow_x)
        
        points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        colors_gradient = [cmap(i / (n_segments - 1)) for i in range(n_segments - 1)]
        
        lc_arrow = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
        ax.add_collection(lc_arrow)
        
        arrow_color = cmap(1.0)
        arrowhead_start_y = y_vals[-2]
        arrowhead_end_y = arrow_y_end
        
        ax.annotate('', 
                   xy=(arrow_x, arrowhead_end_y), 
                   xytext=(arrow_x, arrowhead_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                   zorder=5)
        
        text_x = arrow_x + 0.1
        text_y = arrow_y_center
        
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='left', va='center',
               rotation=90, color='black',
               zorder=4)
        
        plt.tight_layout()
        
        # Store plot info for gif creation if requested
        if return_plot_info:
            if constraint_hedges_g:
                all_g_values = []
                for g_list in constraint_hedges_g.values():
                    all_g_values.extend(g_list)
                g_range = max(abs(min(all_g_values)), abs(max(all_g_values)))
                y_plot_min, y_plot_max = -g_range * 1.2, g_range * 1.2
            else:
                y_plot_min, y_plot_max = -2, 2
            
            unique_constraints = sorted(df_all_params['core_b_constraints_count'].unique())
            
            plot_info_dict[quality_index].update({
                'quality_index': quality_index,
                'CORE_A': CORE_A,
                'CORE_B': CORE_B,
                'unique_constraints': unique_constraints,
                'constraint_hedges_g': constraint_hedges_g,
                'constraint_n_points': constraint_n_points,
                'line_segments': line_segments,
                'line_colors': line_colors,
                'sizing': sizing,
                'max_abs_score': max_abs_score,
                'cmap': cmap,
                'plot_limits': {
                    'x_min': -0.5,
                    'x_max': x_max + 0.5,
                    'y_min': y_plot_min,
                    'y_max': y_plot_max
                }
            })
        
        # Save figure
        if output_figure_filenames and quality_index in output_figure_filenames:
            hedges_g_filename_base = output_figure_filenames[quality_index] + '_hedges_g'

            from .helpers import save_figure_formats
            save_dpi = dpi if dpi is not None else 150
            save_figure_formats(plt.gcf(), hedges_g_filename_base, fig_format=fig_format, dpi=save_dpi)
        else:
            if debug:
                print(f"✓ Hedges' g plot completed for {quality_index}")
            else:
                print(f"✓ Hedges' g plot completed for {quality_index}")
        
        if not debug:
            plt.show()
    
    if return_plot_info:
        return plot_info_dict
    else:
        return None


def _compute_effect_size_ci(t_obs, n1, n2, alpha=0.05):
    """
    Compute 95% confidence intervals for Cohen's d, Hedges' g, and t-statistic
    using SE-based normal approximation.
    
    Parameters:
    -----------
    t_obs : float
        Observed t-statistic from two-sample t-test
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
    alpha : float
        Significance level (default 0.05 for 95% CI)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - d_ci_lower, d_ci_upper: 95% CI for Cohen's d
        - g_ci_lower, g_ci_upper: 95% CI for Hedges' g
        - t_ci_lower, t_ci_upper: 95% CI for t-statistic
    """
    df = n1 + n2 - 2
    
    if df <= 0 or n1 <= 0 or n2 <= 0:
        return {
            'd_ci_lower': np.nan, 'd_ci_upper': np.nan,
            'g_ci_lower': np.nan, 'g_ci_upper': np.nan,
            't_ci_lower': np.nan, 't_ci_upper': np.nan
        }
    
    ncp_to_d_factor = np.sqrt(n1 * n2 / (n1 + n2))
    
    if n1 + n2 > 2:
        g_correction = 1 - 3 / (4 * (n1 + n2) - 9)
    else:
        g_correction = 1.0
    
    # Convert t to d
    d_obs = t_obs / ncp_to_d_factor
    g_obs = d_obs * g_correction
    
    z = stats.norm.ppf(1 - alpha/2)
    
    # SE for Cohen's d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d_obs**2 / (2 * df))
    se_g = se_d * g_correction
    se_t = np.sqrt(1 + t_obs**2 / (2 * df))
    
    d_lower = d_obs - z * se_d
    d_upper = d_obs + z * se_d
    
    g_lower = g_obs - z * se_g
    g_upper = g_obs + z * se_g
    
    t_lower = t_obs - z * se_t
    t_upper = t_obs + z * se_t
    
    return {
        'd_ci_lower': d_lower,
        'd_ci_upper': d_upper,
        'g_ci_lower': g_lower,
        'g_ci_upper': g_upper,
        't_ci_lower': t_lower,
        't_ci_upper': t_upper
    }


def calculate_quality_comparison_t_statistics(target_quality_indices, 
                                               output_csv_directory,
                                               input_syntheticPDF_directory,
                                               core_a_name, 
                                               core_b_name,
                                               log_columns=None,
                                               mute_mode=False):
    """
    Calculate t-statistics, Cohen's d, and effect size categories by comparing real core 
    correlation data against synthetic null hypothesis data. Appends statistical columns 
    to master CSV files and saves enhanced versions to the same directory.
    
    Parameters:
    -----------
    target_quality_indices : list
        List of quality indices to process (e.g., ['norm_dtw', 'corr_coef'])
    output_csv_directory : str
        Directory path where output CSV files are located and will be updated
    input_syntheticPDF_directory : str
        Directory path where synthetic/null hypothesis CSV files are located
    core_a_name : str
        Name identifier for the first core
    core_b_name : str
        Name identifier for the second core
    log_columns : list, optional
        List of log column names (e.g., ['hiresMS', 'CT', 'Lumin']). If provided, 
        creates subdirectory structure for file organization
    mute_mode : bool, optional
        If True, suppresses print statements (default: False)
    
    Returns:
    --------
    None
        Function saves enhanced CSV files with statistical columns to output_csv_directory
        
    Notes:
    ------
    The function calculates:
    - t_statistic: Statistical test comparing real vs synthetic data
    - cohens_d: Effect size measure
    - effect_size_category: Categorical interpretation of Cohen's d values
    
    CSV filenames are automatically generated based on:
    - Output: {output_csv_directory}/{log_columns_str}/{quality_index}_fit_params.csv
    - Input: {input_syntheticPDF_directory}/synthetic_PDFs_{log_columns_str}_{quality_index}.csv
    
    Enhanced CSV files are saved using the automatically generated filenames.
    """
    
    # Construct filenames from directories and parameters
    master_csv_filenames = {}
    synthetic_csv_filenames = {}
    log_cols_str = "_".join(log_columns) if log_columns else ""
    
    for quality_index in target_quality_indices:
        # Create master CSV path
        if log_cols_str:
            master_csv_filenames[quality_index] = os.path.join(
                output_csv_directory, log_cols_str, f"{quality_index}_fit_params.csv"
            )
        else:
            master_csv_filenames[quality_index] = os.path.join(
                output_csv_directory, f"{quality_index}_fit_params.csv"
            )
        
        # Create synthetic CSV path
        if log_cols_str:
            synthetic_csv_filenames[quality_index] = os.path.join(
                input_syntheticPDF_directory, f"synthetic_PDFs_{log_cols_str}_{quality_index}.csv"
            )
        else:
            synthetic_csv_filenames[quality_index] = os.path.join(
                input_syntheticPDF_directory, f"synthetic_PDFs_{quality_index}.csv"
            )
    
    # Keep original variable names for backward compatibility with rest of function
    CORE_A = core_a_name
    CORE_B = core_b_name
    
    # Define which categories to load - same filters as load_and_prepare_quality_data
    load_filters = {
        'age_consideration': [True, False],
        'restricted_age_correlation': [True, False],
        'shortest_path_search': [True]
    }
    
    for quality_index in target_quality_indices:
        # Suppress verbose processing message when mute_mode=True
        
        # Load master CSV
        master_csv_filename = master_csv_filenames[quality_index]
        if not os.path.exists(master_csv_filename):
            if not mute_mode:
                print(f"✗ Error: Master CSV file not found: {master_csv_filename}")
            continue
            
        try:
            df_master = pd.read_csv(master_csv_filename)
        except Exception as e:
            if not mute_mode:
                print(f"✗ Error loading master CSV {master_csv_filename}: {str(e)}")
            continue
        
        # Remove existing stat columns if present (will be recomputed)
        # Includes both new short names and legacy long names for backward compatibility
        stat_cols_to_remove = [
            't', 't_ci_lower', 't_ci_upper', 't_se',
            'd', 'd_ci_lower', 'd_ci_upper', 'd_se',
            'g', 'g_ci_lower', 'g_ci_upper', 'g_se',
            'ps', 'ps_ci_lower', 'ps_ci_upper', 'ps_se',
            'effect_size',
            't_inv', 't_ci_lower_inv', 't_ci_upper_inv', 't_se_inv',
            'd_inv', 'd_ci_lower_inv', 'd_ci_upper_inv', 'd_se_inv',
            'g_inv', 'g_ci_lower_inv', 'g_ci_upper_inv', 'g_se_inv',
            'ps_inv', 'ps_ci_lower_inv', 'ps_ci_upper_inv', 'ps_se_inv',
            # Legacy column names
            't_statistic', 't_statistic_inv',
            'cohens_d', 'cohens_d_inv', 'cohens_d_ci_lower', 'cohens_d_ci_upper',
            'cohens_d_ci_lower_inv', 'cohens_d_ci_upper_inv',
            'hedges_g', 'hedges_g_inv', 'hedges_g_ci_lower', 'hedges_g_ci_upper',
            'hedges_g_ci_lower_inv', 'hedges_g_ci_upper_inv',
            'prob_superiority', 'prob_superiority_inv',
            'prob_superiority_ci_lower', 'prob_superiority_ci_upper',
            'prob_superiority_ci_lower_inv', 'prob_superiority_ci_upper_inv',
            'effect_size_category',
        ]
        existing_stat_cols = [c for c in stat_cols_to_remove if c in df_master.columns]
        if existing_stat_cols:
            df_master = df_master.drop(columns=existing_stat_cols)
        
        # Apply same filters as load_and_prepare_quality_data
        mask = pd.Series([True] * len(df_master))
        for column, values in load_filters.items():
            if values is not None:
                if None in values:
                    mask &= (df_master[column].isin([v for v in values if v is not None]) | 
                            df_master[column].isna())
                else:
                    mask &= df_master[column].isin(values)
        
        df_filtered = df_master[mask].copy()
        
        # Load synthetic CSV for null hypothesis data
        synthetic_csv_filename = synthetic_csv_filenames[quality_index]
        if not os.path.exists(synthetic_csv_filename):
            if not mute_mode:
                print(f"✗ Error: Synthetic CSV file not found: {synthetic_csv_filename}")
            continue
            
        try:
            df_synthetic = pd.read_csv(synthetic_csv_filename)
        except Exception as e:
            if not mute_mode:
                print(f"✗ Error loading synthetic CSV {synthetic_csv_filename}: {str(e)}")
            continue
        
        # Reconstruct synthetic (null hypothesis) data
        combined_data = []
        for _, row in df_synthetic.iterrows():
            try:
                bins = np.fromstring(row['bins'].strip('[]'), sep=' ')
                hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ')
                n_points = row['n_points']
                raw_data_points = reconstruct_raw_data_from_histogram(bins, hist_percentages, n_points)
                combined_data.extend(raw_data_points)
            except:
                continue
        
        combined_data = np.array(combined_data)
        if len(combined_data) == 0:
            if not mute_mode:
                print(f"✗ Error: No valid synthetic data found for {quality_index}")
            continue
        
        # Initialize new columns
        t_statistics = []
        cohens_d_values = []
        effect_size_categories = []
        hedges_g_values = []
        prob_superiority_values = []
        
        # Initialize 95% CI and SE columns
        cohens_d_ci_lower_values = []
        cohens_d_ci_upper_values = []
        cohens_d_se_values = []
        hedges_g_ci_lower_values = []
        hedges_g_ci_upper_values = []
        hedges_g_se_values = []
        t_ci_lower_values = []
        t_ci_upper_values = []
        t_se_values = []
        prob_superiority_ci_lower_values = []
        prob_superiority_ci_upper_values = []
        prob_superiority_se_values = []
        
        # For norm_dtw and norm_dtw_sect, also compute inverted metrics (1 - value)
        # This makes higher values = better, consistent with corr_coef interpretation
        is_norm_dtw = quality_index in ['norm_dtw', 'norm_dtw_sect']
        if is_norm_dtw:
            t_statistics_inv = []
            cohens_d_values_inv = []
            hedges_g_values_inv = []
            prob_superiority_values_inv = []
            cohens_d_ci_lower_values_inv = []
            cohens_d_ci_upper_values_inv = []
            cohens_d_se_values_inv = []
            hedges_g_ci_lower_values_inv = []
            hedges_g_ci_upper_values_inv = []
            hedges_g_se_values_inv = []
            t_ci_lower_values_inv = []
            t_ci_upper_values_inv = []
            t_se_values_inv = []
            prob_superiority_ci_lower_values_inv = []
            prob_superiority_ci_upper_values_inv = []
            prob_superiority_se_values_inv = []
            # Transform synthetic data for inverted calculation
            combined_data_inv = 1 - combined_data
        
        # Calculate statistics for each row
        iterator = tqdm(df_filtered.iterrows(), total=len(df_filtered), 
                       desc=f"  Calculating statistics for {quality_index}", 
                       disable=mute_mode)
        
        for idx, row in iterator:
            # Check if row has histogram data
            has_histogram = ('bins' in row and 'hist' in row and 'n_points' in row and 
                           pd.notna(row['bins']) and pd.notna(row['hist']) and pd.notna(row['n_points']))
            
            if has_histogram:
                try:
                    # Use histogram data (now computed with consistent bin sizing if available)
                    bins = np.fromstring(row['bins'].strip('[]'), sep=' ')
                    hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ')
                    n_points = row['n_points']
                    
                    raw_data_points = reconstruct_raw_data_from_histogram(bins, hist_percentages, n_points)
                except Exception as e:
                    raw_data_points = []
            else:
                raw_data_points = []
            
            # Calculate t-statistic and Cohen's d using raw_data_points
            try:
                n1 = len(raw_data_points)
                n2 = len(combined_data)
                
                if n1 > 1:
                    t_stat, _ = stats.ttest_ind(raw_data_points, combined_data)
                    cohens_d_value = cohens_d(raw_data_points, combined_data)
                elif n1 == 1:
                    # Single point: use z-score as approximation for t-stat
                    combined_mean = np.mean(combined_data)
                    combined_std = np.std(combined_data)
                    t_stat = (raw_data_points[0] - combined_mean) / combined_std if combined_std > 0 else 0.0
                    cohens_d_value = cohens_d(raw_data_points, combined_data)
                else:
                    t_stat = 0.0
                    cohens_d_value = 0.0
                
                # Calculate Hedges' g (bias-corrected Cohen's d)
                # g = d * (1 - 3/(4*(n1+n2) - 9))
                if n1 + n2 > 2:
                    correction_factor = 1 - 3 / (4 * (n1 + n2) - 9)
                    hedges_g_value = cohens_d_value * correction_factor
                else:
                    hedges_g_value = cohens_d_value
                
                # Calculate Probability of Superiority based on Hedges' g
                # PS = Φ(g / √2) where Φ is the standard normal CDF
                prob_superiority = stats.norm.cdf(hedges_g_value / np.sqrt(2))
                
                # Categorize effect size based on Cohen's d
                abs_cohens_d = abs(cohens_d_value)
                if abs_cohens_d < 0.2:
                    effect_size_category = "negligible"
                elif abs_cohens_d < 0.5:
                    effect_size_category = "small"
                elif abs_cohens_d < 0.8:
                    effect_size_category = "medium"
                else:
                    effect_size_category = "large"
                
                # Compute 95% CI for d and g using non-central t-distribution
                if n1 >= 1 and n2 >= 1:
                    ci_results = _compute_effect_size_ci(t_stat, n1, n2, alpha=0.05)
                    d_ci_lower = ci_results['d_ci_lower']
                    d_ci_upper = ci_results['d_ci_upper']
                    g_ci_lower = ci_results['g_ci_lower']
                    g_ci_upper = ci_results['g_ci_upper']
                    t_ci_lower = ci_results['t_ci_lower']
                    t_ci_upper = ci_results['t_ci_upper']
                    ps_ci_lower = stats.norm.cdf(g_ci_lower / np.sqrt(2)) if not np.isnan(g_ci_lower) else np.nan
                    ps_ci_upper = stats.norm.cdf(g_ci_upper / np.sqrt(2)) if not np.isnan(g_ci_upper) else np.nan
                else:
                    d_ci_lower, d_ci_upper = np.nan, np.nan
                    g_ci_lower, g_ci_upper = np.nan, np.nan
                    t_ci_lower, t_ci_upper = np.nan, np.nan
                    ps_ci_lower, ps_ci_upper = np.nan, np.nan
                
                # Compute SE: SE(d) = sqrt((n1+n2)/(n1*n2) + d^2/(2*(n1+n2-2)))
                if n1 >= 1 and n2 >= 2 and (n1 + n2) > 2:
                    d_se = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d_value**2 / (2 * (n1 + n2 - 2)))
                    g_se = d_se * correction_factor if (n1 + n2 > 2) else d_se
                    t_se = d_se * np.sqrt(n1 * n2 / (n1 + n2))
                    ps_se = stats.norm.pdf(hedges_g_value / np.sqrt(2)) / np.sqrt(2) * g_se
                else:
                    d_se, g_se, t_se, ps_se = np.nan, np.nan, np.nan, np.nan
                
                # For norm_dtw/norm_dtw_sect, also compute inverted metrics (1 - value)
                if is_norm_dtw:
                    raw_data_points_inv = 1 - np.array(raw_data_points) if n1 > 0 else np.array([])
                    n1_inv = len(raw_data_points_inv)
                    
                    if n1_inv > 1:
                        t_stat_inv, _ = stats.ttest_ind(raw_data_points_inv, combined_data_inv)
                        cohens_d_value_inv = cohens_d(raw_data_points_inv, combined_data_inv)
                    elif n1_inv == 1:
                        combined_mean_inv = np.mean(combined_data_inv)
                        combined_std_inv = np.std(combined_data_inv)
                        t_stat_inv = (raw_data_points_inv[0] - combined_mean_inv) / combined_std_inv if combined_std_inv > 0 else 0.0
                        cohens_d_value_inv = cohens_d(raw_data_points_inv, combined_data_inv)
                    else:
                        t_stat_inv = 0.0
                        cohens_d_value_inv = 0.0
                    
                    # Hedges' g for inverted
                    if n1_inv + n2 > 2:
                        hedges_g_value_inv = cohens_d_value_inv * correction_factor
                    else:
                        hedges_g_value_inv = cohens_d_value_inv
                    
                    # Probability of Superiority for inverted
                    prob_superiority_inv = stats.norm.cdf(hedges_g_value_inv / np.sqrt(2))
                    
                    # Compute 95% CI for inverted metrics
                    if n1_inv >= 1 and n2 >= 1:
                        ci_results_inv = _compute_effect_size_ci(t_stat_inv, n1_inv, n2, alpha=0.05)
                        d_ci_lower_inv = ci_results_inv['d_ci_lower']
                        d_ci_upper_inv = ci_results_inv['d_ci_upper']
                        g_ci_lower_inv = ci_results_inv['g_ci_lower']
                        g_ci_upper_inv = ci_results_inv['g_ci_upper']
                        t_ci_lower_inv = ci_results_inv['t_ci_lower']
                        t_ci_upper_inv = ci_results_inv['t_ci_upper']
                        ps_ci_lower_inv = stats.norm.cdf(g_ci_lower_inv / np.sqrt(2)) if not np.isnan(g_ci_lower_inv) else np.nan
                        ps_ci_upper_inv = stats.norm.cdf(g_ci_upper_inv / np.sqrt(2)) if not np.isnan(g_ci_upper_inv) else np.nan
                    else:
                        d_ci_lower_inv, d_ci_upper_inv = np.nan, np.nan
                        g_ci_lower_inv, g_ci_upper_inv = np.nan, np.nan
                        t_ci_lower_inv, t_ci_upper_inv = np.nan, np.nan
                        ps_ci_lower_inv, ps_ci_upper_inv = np.nan, np.nan
                    
                    # Compute SE for inverted metrics
                    if n1_inv >= 1 and n2 >= 2 and (n1_inv + n2) > 2:
                        d_se_inv = np.sqrt((n1_inv + n2) / (n1_inv * n2) + cohens_d_value_inv**2 / (2 * (n1_inv + n2 - 2)))
                        g_se_inv = d_se_inv * correction_factor if (n1_inv + n2 > 2) else d_se_inv
                        t_se_inv = d_se_inv * np.sqrt(n1_inv * n2 / (n1_inv + n2))
                        ps_se_inv = stats.norm.pdf(hedges_g_value_inv / np.sqrt(2)) / np.sqrt(2) * g_se_inv
                    else:
                        d_se_inv, g_se_inv, t_se_inv, ps_se_inv = np.nan, np.nan, np.nan, np.nan
                    
            except Exception as e:
                t_stat = 0.0
                cohens_d_value = 0.0
                effect_size_category = "negligible"
                hedges_g_value = 0.0
                prob_superiority = 0.5
                d_ci_lower, d_ci_upper = np.nan, np.nan
                g_ci_lower, g_ci_upper = np.nan, np.nan
                t_ci_lower, t_ci_upper = np.nan, np.nan
                ps_ci_lower, ps_ci_upper = np.nan, np.nan
                d_se, g_se, t_se, ps_se = np.nan, np.nan, np.nan, np.nan
                if is_norm_dtw:
                    t_stat_inv = 0.0
                    cohens_d_value_inv = 0.0
                    hedges_g_value_inv = 0.0
                    prob_superiority_inv = 0.5
                    d_ci_lower_inv, d_ci_upper_inv = np.nan, np.nan
                    g_ci_lower_inv, g_ci_upper_inv = np.nan, np.nan
                    t_ci_lower_inv, t_ci_upper_inv = np.nan, np.nan
                    ps_ci_lower_inv, ps_ci_upper_inv = np.nan, np.nan
                    d_se_inv, g_se_inv, t_se_inv, ps_se_inv = np.nan, np.nan, np.nan, np.nan
            
            t_statistics.append(t_stat)
            cohens_d_values.append(cohens_d_value)
            effect_size_categories.append(effect_size_category)
            hedges_g_values.append(hedges_g_value)
            prob_superiority_values.append(prob_superiority)
            
            # Append CI and SE values
            cohens_d_ci_lower_values.append(d_ci_lower)
            cohens_d_ci_upper_values.append(d_ci_upper)
            cohens_d_se_values.append(d_se)
            hedges_g_ci_lower_values.append(g_ci_lower)
            hedges_g_ci_upper_values.append(g_ci_upper)
            hedges_g_se_values.append(g_se)
            t_ci_lower_values.append(t_ci_lower)
            t_ci_upper_values.append(t_ci_upper)
            t_se_values.append(t_se)
            prob_superiority_ci_lower_values.append(ps_ci_lower)
            prob_superiority_ci_upper_values.append(ps_ci_upper)
            prob_superiority_se_values.append(ps_se)
            
            if is_norm_dtw:
                t_statistics_inv.append(t_stat_inv)
                cohens_d_values_inv.append(cohens_d_value_inv)
                hedges_g_values_inv.append(hedges_g_value_inv)
                prob_superiority_values_inv.append(prob_superiority_inv)
                cohens_d_ci_lower_values_inv.append(d_ci_lower_inv)
                cohens_d_ci_upper_values_inv.append(d_ci_upper_inv)
                cohens_d_se_values_inv.append(d_se_inv)
                hedges_g_ci_lower_values_inv.append(g_ci_lower_inv)
                hedges_g_ci_upper_values_inv.append(g_ci_upper_inv)
                hedges_g_se_values_inv.append(g_se_inv)
                t_ci_lower_values_inv.append(t_ci_lower_inv)
                t_ci_upper_values_inv.append(t_ci_upper_inv)
                t_se_values_inv.append(t_se_inv)
                prob_superiority_ci_lower_values_inv.append(ps_ci_lower_inv)
                prob_superiority_ci_upper_values_inv.append(ps_ci_upper_inv)
                prob_superiority_se_values_inv.append(ps_se_inv)
        
        # Add new columns to the filtered dataframe in the specified order
        df_filtered['t'] = t_statistics
        df_filtered['t_ci_lower'] = t_ci_lower_values
        df_filtered['t_ci_upper'] = t_ci_upper_values
        df_filtered['t_se'] = t_se_values
        df_filtered['d'] = cohens_d_values
        df_filtered['d_ci_lower'] = cohens_d_ci_lower_values
        df_filtered['d_ci_upper'] = cohens_d_ci_upper_values
        df_filtered['d_se'] = cohens_d_se_values
        df_filtered['g'] = hedges_g_values
        df_filtered['g_ci_lower'] = hedges_g_ci_lower_values
        df_filtered['g_ci_upper'] = hedges_g_ci_upper_values
        df_filtered['g_se'] = hedges_g_se_values
        df_filtered['ps'] = prob_superiority_values
        df_filtered['ps_ci_lower'] = prob_superiority_ci_lower_values
        df_filtered['ps_ci_upper'] = prob_superiority_ci_upper_values
        df_filtered['ps_se'] = prob_superiority_se_values
        df_filtered['effect_size'] = effect_size_categories
        
        if is_norm_dtw:
            df_filtered['t_inv'] = t_statistics_inv
            df_filtered['t_ci_lower_inv'] = t_ci_lower_values_inv
            df_filtered['t_ci_upper_inv'] = t_ci_upper_values_inv
            df_filtered['t_se_inv'] = t_se_values_inv
            df_filtered['d_inv'] = cohens_d_values_inv
            df_filtered['d_ci_lower_inv'] = cohens_d_ci_lower_values_inv
            df_filtered['d_ci_upper_inv'] = cohens_d_ci_upper_values_inv
            df_filtered['d_se_inv'] = cohens_d_se_values_inv
            df_filtered['g_inv'] = hedges_g_values_inv
            df_filtered['g_ci_lower_inv'] = hedges_g_ci_lower_values_inv
            df_filtered['g_ci_upper_inv'] = hedges_g_ci_upper_values_inv
            df_filtered['g_se_inv'] = hedges_g_se_values_inv
            df_filtered['ps_inv'] = prob_superiority_values_inv
            df_filtered['ps_ci_lower_inv'] = prob_superiority_ci_lower_values_inv
            df_filtered['ps_ci_upper_inv'] = prob_superiority_ci_upper_values_inv
            df_filtered['ps_se_inv'] = prob_superiority_se_values_inv
        
        # For rows not in filtered data, add default values
        df_master_with_stats = df_master.copy()
        
        # Define stat columns with defaults in the specified order
        stat_defaults = [
            ('t', 0.0), ('t_ci_lower', np.nan), ('t_ci_upper', np.nan), ('t_se', np.nan),
            ('d', 0.0), ('d_ci_lower', np.nan), ('d_ci_upper', np.nan), ('d_se', np.nan),
            ('g', 0.0), ('g_ci_lower', np.nan), ('g_ci_upper', np.nan), ('g_se', np.nan),
            ('ps', 0.5), ('ps_ci_lower', np.nan), ('ps_ci_upper', np.nan), ('ps_se', np.nan),
            ('effect_size', 'negligible'),
        ]
        inv_stat_defaults = [
            ('t_inv', 0.0), ('t_ci_lower_inv', np.nan), ('t_ci_upper_inv', np.nan), ('t_se_inv', np.nan),
            ('d_inv', 0.0), ('d_ci_lower_inv', np.nan), ('d_ci_upper_inv', np.nan), ('d_se_inv', np.nan),
            ('g_inv', 0.0), ('g_ci_lower_inv', np.nan), ('g_ci_upper_inv', np.nan), ('g_se_inv', np.nan),
            ('ps_inv', 0.5), ('ps_ci_lower_inv', np.nan), ('ps_ci_upper_inv', np.nan), ('ps_se_inv', np.nan),
        ]
        
        for col, default in stat_defaults:
            df_master_with_stats[col] = default
        if is_norm_dtw:
            for col, default in inv_stat_defaults:
                df_master_with_stats[col] = default
        
        # Update the filtered rows with calculated statistics
        all_stat_cols = [col for col, _ in stat_defaults]
        if is_norm_dtw:
            all_stat_cols += [col for col, _ in inv_stat_defaults]
        
        for col in all_stat_cols:
            if col in df_filtered.columns:
                df_master_with_stats.loc[df_filtered.index, col] = df_filtered[col]
        
        # Save modified CSV
        output_filename = master_csv_filename
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df_master_with_stats.to_csv(output_filename, index=False)
        
        if mute_mode:
            print(f"✓ Data appending to CSV is done: {output_filename}")


def plot_quality_comparison_t_statistics(target_quality_indices, 
                                        output_csv_directory,
                                        input_syntheticPDF_directory,
                                        core_a_name, 
                                        core_b_name,
                                        log_columns=None,
                                        mute_mode=False, 
                                        save_fig=False, 
                                        output_figure_directory=None,
                                        fig_format=['png'],
                                        dpi=150,
                                        save_gif=False, 
                                        output_gif_directory=None, 
                                        max_frames=50,
                                        plot_real_data_histogram=False, 
                                        plot_age_removal_step_pdf=False,
                                        show_best_datum_match=True, 
                                        sequential_mappings_csv=None,
                                        skip_age_removal_graph=False,
                                        skip_pdf_graph=False,
                                        show_t_graph=False,
                                        show_d_graph=False,
                                        show_g_graph=True,
                                        linear_g_scale=False,
                                        invert_norm_dtw=True):
    """
    Plot quality index distributions comparing real data vs synthetic null hypothesis
    AND t-statistics/Cohen's d vs age constraints using pre-calculated statistics from CSV files.
    
    Brief summary: This function loads pre-calculated t-statistics from modified CSV files
    and creates distribution plots and t-statistics/Cohen's d plots for quality indices.
    Each quality index shows its distribution plot followed immediately by its statistics plots.
    
    Parameters:
    -----------
    target_quality_indices : list
        List of quality indices to process (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    output_csv_directory : str
        Directory path where output CSV files are located (should contain t-statistics columns)
    input_syntheticPDF_directory : str
        Directory path where synthetic/null hypothesis CSV files are located
    core_a_name : str
        Name of core A
    core_b_name : str
        Name of core B
    log_columns : list, optional
        List of log column names (e.g., ['hiresMS', 'CT', 'Lumin']). If provided, 
        creates subdirectory structure for file organization
    mute_mode : bool, default False
        If True, suppress detailed output messages and show only essential progress information.
        If False, show detailed output messages.
    save_fig : bool, default False
        If True, save static figures to files
    output_figure_directory : str, optional
        Directory path where output figures will be saved. Only used when save_fig=True.
        Filenames are automatically generated as {quality_index}_compare2null.{format}
    fig_format : list, default ['png']
        List of file formats for saved figures. Accepted formats: 'png', 'jpg', 'svg', 'pdf', 'tiff'.
        Only used when save_fig=True. Can specify multiple formats like ['png', 'tiff'].
    dpi : int, default 150
        Resolution for saved figures in dots per inch. Only used when save_fig=True.
    save_gif : bool, default False
        If True, create animated gif showing progressive addition of age constraints.
        When save_gif=True and save_fig=False, static figures will not be displayed (only GIFs are shown at the end).
        When save_gif=False, static figures will be displayed normally regardless of save_fig value
    output_gif_directory : str, optional
        Directory path where GIF files will be saved. Only used when save_gif=True.
        Filenames are automatically generated as {quality_index}_compare2null.gif
    max_frames : int, default 50
        Maximum number of frames for GIF animations. When there are more data points than
        available frames, the function automatically groups multiple data points per frame
        to keep the total animation length under this limit.
    plot_real_data_histogram : bool, default True
        If True, plot histograms for real data (no age constraints and all age constraints cases)
        and include them in the legend. If False, skip real data histograms and keep current legend.
    plot_age_removal_step_pdf : bool, default False
        If True, plot all PDF curves including dashed lines for partially removed age constraints
        and show the legend color bar. If False, skip dashed PDF curves and hide color bar.
    show_best_datum_match : bool, default True
        If True, plot a vertical line showing the best datum match value from sequential_mappings_csv
        where 'Ranking_datums' is 1. If False, skip plotting this vertical line.
    sequential_mappings_csv : str or dict, optional
        Path to CSV file(s) containing sequential mappings with 'Ranking_datums' column.
        Can be either:
        - str: Single CSV file path containing all quality indices as columns
        - dict: Dictionary mapping quality_index to CSV file paths for per-index files
        Only used when show_best_datum_match=True.
    skip_age_removal_graph : bool, default False
        If True, skip plotting the t-statistic vs number of age constraints remaining graph
        (both static figures and GIF animations). Only the distribution graph will be plotted.
    skip_pdf_graph : bool, default False
        If True, skip plotting the distribution graph (both static figures and GIF animations).
        Only the t-statistic vs age constraints graph will be plotted.
    show_t_graph : bool, default False
        If True, show t-statistic graph alongside other effect size graphs.
        If False, skip t-statistic graph.
    show_d_graph : bool, default False
        If True, show Cohen's d graph alongside Hedges' g graph.
        If False, skip Cohen's d graph and only show Hedges' g graph.
    show_g_graph : bool, default True
        If True, show Hedges' g graph (default behavior).
        If False, skip plotting Hedges' g graph.
    linear_g_scale : bool, default False
        If False, plot Hedges' g graph with linear g scale on left y-axis.
        If True, rescale y-axis so Probability of Superiority (PS) is linear (0-100%),
        with right y-axis showing PS at 20% intervals, and left y-axis showing
        corresponding g values at each PS tick.
    invert_norm_dtw : bool, default True
        If True, for norm_dtw and norm_dtw_sect metrics:
        - Distribution graphs plot (1 - norm_dtw) with x-axis 0 to 1, arrow pointing right (like corr_coef)
        - Effect size graphs use inverted columns (t_statistic_inv, cohens_d_inv, hedges_g_inv)
        - Improvement/deterioration calculation follows corr_coef convention (higher = better)
        If False, plot original norm_dtw values with arrow pointing left (lower = better).
        
    Returns:
    --------
    None
        Creates static plots and/or animated gifs based on parameters
    """
    
    # Normalize format names (handle jpg/jpeg)
    fig_format = [fmt.lower() for fmt in fig_format]
    fig_format = ['jpeg' if fmt == 'jpg' else fmt for fmt in fig_format]
    
    # Construct filenames from directories and parameters
    master_csv_filenames = {}
    synthetic_csv_filenames = {}
    output_figure_filenames = {} if save_fig and output_figure_directory else None
    output_gif_filenames = {} if save_gif and output_gif_directory else None
    log_cols_str = "_".join(log_columns) if log_columns else ""
    
    for quality_index in target_quality_indices:
        # Create master CSV path
        if log_cols_str:
            master_csv_filenames[quality_index] = os.path.join(
                output_csv_directory, log_cols_str, f"{quality_index}_fit_params.csv"
            )
        else:
            master_csv_filenames[quality_index] = os.path.join(
                output_csv_directory, f"{quality_index}_fit_params.csv"
            )
        
        # Create synthetic CSV path
        if log_cols_str:
            synthetic_csv_filenames[quality_index] = os.path.join(
                input_syntheticPDF_directory, f"synthetic_PDFs_{log_cols_str}_{quality_index}.csv"
            )
        else:
            synthetic_csv_filenames[quality_index] = os.path.join(
                input_syntheticPDF_directory, f"synthetic_PDFs_{quality_index}.csv"
            )
        
        # Create output figure base paths if requested (without extension)
        if save_fig and output_figure_directory:
            if log_cols_str:
                output_figure_filenames[quality_index] = os.path.join(
                    output_figure_directory, log_cols_str, f"{quality_index}_compare2null"
                )
            else:
                output_figure_filenames[quality_index] = os.path.join(
                    output_figure_directory, f"{quality_index}_compare2null"
                )
        
        # Create output GIF paths if requested
        if save_gif and output_gif_directory:
            if log_cols_str:
                output_gif_filenames[quality_index] = os.path.join(
                    output_gif_directory, log_cols_str, f"{quality_index}_compare2null.gif"
                )
            else:
                output_gif_filenames[quality_index] = os.path.join(
                    output_gif_directory, f"{quality_index}_compare2null.gif"
                )
    
    # Keep original variable names for backward compatibility with rest of function
    CORE_A = core_a_name
    CORE_B = core_b_name
    
    # Check for required statistical analysis files
    for quality_index in target_quality_indices:
        csv_to_check = master_csv_filenames[quality_index]
        try:
            temp_df = pd.read_csv(csv_to_check)
            if not {'t', 'd', 'effect_size'}.issubset(temp_df.columns):
                error_msg = (f"Required statistics columns not found in {csv_to_check}. "
                            f"Please run calculate_quality_comparison_t_statistics first.")
                if mute_mode:
                    print(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error reading {csv_to_check}: {str(e)}"
            if mute_mode:
                print(error_msg)
            raise
    
    # Use the provided master CSV filenames directly
    modified_master_csv_filenames = master_csv_filenames.copy()
    
    # Load and prepare data for all quality indices
    quality_data = load_and_prepare_quality_data(
        target_quality_indices, modified_master_csv_filenames, synthetic_csv_filenames, 
        CORE_A, CORE_B, mute_mode
    )
    
    # Load sequential mappings data for best datum match if requested
    best_datum_values = {}
    if show_best_datum_match and sequential_mappings_csv is not None:
        # Handle both dictionary (per quality index) and string (single file) inputs
        if isinstance(sequential_mappings_csv, dict):
            # Dictionary format: different CSV files for different quality indices
            for quality_index in target_quality_indices:
                if quality_index in sequential_mappings_csv:
                    csv_file = sequential_mappings_csv[quality_index]
                    try:
                        if os.path.exists(csv_file):
                            df_sequential = pd.read_csv(csv_file)
                            
                            # Check if 'Ranking_datums' column exists
                            if 'Ranking_datums' in df_sequential.columns:
                                # Find row where Ranking_datums is 1
                                best_datum_row = df_sequential[df_sequential['Ranking_datums'] == 1]
                                
                                if not best_datum_row.empty:
                                    # Extract quality index values for the best datum match
                                    if quality_index in best_datum_row.columns:
                                        best_datum_values[quality_index] = best_datum_row[quality_index].iloc[0]
                                    elif quality_index == 'corr_coef' and 'pearson_r' in best_datum_row.columns:
                                        # Handle alternative column name for correlation coefficient
                                        best_datum_values[quality_index] = best_datum_row['pearson_r'].iloc[0]
                                else:
                                    if not mute_mode:
                                        print(f"⚠️  WARNING: No row with Ranking_datums = 1 found in {csv_file} for {quality_index}.")
                            else:
                                if not mute_mode:
                                    print(f"⚠️  WARNING: 'Ranking_datums' column not found in {csv_file} for {quality_index}.")
                        else:
                            if not mute_mode:
                                print(f"⚠️  WARNING: Sequential mappings CSV file not found for {quality_index}: {csv_file}")
                    except Exception as e:
                        if not mute_mode:
                            print(f"⚠️  WARNING: Error loading sequential mappings CSV for {quality_index}: {str(e)}")
                else:
                    if not mute_mode:
                        print(f"⚠️  WARNING: No sequential mappings CSV specified for quality index: {quality_index}")
        else:
            # String format: single CSV file for all quality indices
            try:
                if os.path.exists(sequential_mappings_csv):
                    df_sequential = pd.read_csv(sequential_mappings_csv)
                    
                    # Check if 'Ranking_datums' column exists
                    if 'Ranking_datums' in df_sequential.columns:
                        # Find row where Ranking_datums is 1
                        best_datum_row = df_sequential[df_sequential['Ranking_datums'] == 1]
                        
                        if not best_datum_row.empty:
                            # Extract quality index values for the best datum match
                            for quality_index in target_quality_indices:
                                if quality_index in best_datum_row.columns:
                                    best_datum_values[quality_index] = best_datum_row[quality_index].iloc[0]
                                elif quality_index == 'corr_coef' and 'pearson_r' in best_datum_row.columns:
                                    # Handle alternative column name for correlation coefficient
                                    best_datum_values[quality_index] = best_datum_row['pearson_r'].iloc[0]
                        else:
                            if not mute_mode:
                                print("⚠️  WARNING: No row with Ranking_datums = 1 found in sequential mappings CSV.")
                    else:
                        if not mute_mode:
                            print("⚠️  WARNING: 'Ranking_datums' column not found in sequential mappings CSV.")
                else:
                    if not mute_mode:
                        print(f"⚠️  WARNING: Sequential mappings CSV file not found: {sequential_mappings_csv}")
            except Exception as e:
                if not mute_mode:
                    print(f"⚠️  WARNING: Error loading sequential mappings CSV: {str(e)}")
    elif show_best_datum_match and sequential_mappings_csv is None:
        if not mute_mode:
            print("⚠️  WARNING: show_best_datum_match=True but sequential_mappings_csv is None.")
    
    # Check for valid age constraints across all quality indices
    has_valid_age_constraints = False
    invalid_age_cores = []
    
    for quality_index in target_quality_indices:
        if quality_index not in quality_data:
            continue
            
        data = quality_data[quality_index]
        df_all_params = data['df_all_params']
        
        # Check if there are multiple constraint levels (indicating valid age analysis)
        unique_constraints = df_all_params['core_b_constraints_count'].unique()
        max_constraints = df_all_params['core_b_constraints_count'].max()
        
        # Valid age analysis requires more than just 0 constraints
        if len(unique_constraints) > 1 and max_constraints > 0:
            has_valid_age_constraints = True
            break
        else:
            # Check if only one row exists (indicating no valid age data)
            if len(df_all_params) == 1:
                # Determine which core lacks valid age data based on constraint counts
                core_a_constraints = df_all_params['core_a_constraints_count'].iloc[0]
                core_b_constraints = df_all_params['core_b_constraints_count'].iloc[0]
                
                if core_a_constraints == 0 and core_b_constraints == 0:
                    # Both cores lack age data, but we'll report the one that was expected to have it
                    # This is typically CORE_B based on the analysis pattern
                    if CORE_B not in invalid_age_cores:
                        invalid_age_cores.append(CORE_B)
                elif core_b_constraints == 0:
                    if CORE_B not in invalid_age_cores:
                        invalid_age_cores.append(CORE_B)
                elif core_a_constraints == 0:
                    if CORE_A not in invalid_age_cores:
                        invalid_age_cores.append(CORE_A)
    
    # Print warning if no valid age constraints found
    if not has_valid_age_constraints:
        if invalid_age_cores:
            for core_name in invalid_age_cores:
                print(f"⚠️  WARNING: No valid age constraints in {core_name}. Only PDFs without age consideration will be plotted.")
        else:
            print("⚠️  WARNING: No valid age constraints found. Only PDFs without age consideration will be plotted.")
        
        # Skip GIF generation when no valid age constraints
        if save_gif:
            print("⚠️  WARNING: Skipping GIF generation due to insufficient age constraint data.")
            save_gif = False
        
        print("⚠️  WARNING: Skipping t-statistics plots since they require multiple age constraint levels for meaningful analysis.")
    
    # Keep track of created GIFs for display at the end
    created_gifs = []
    
    # Determine whether to show figures:
    # - If save_gif=True and save_fig=False: suppress figure display (show only GIFs at the end)
    # - Otherwise: show figures normally (respecting mute_mode)
    suppress_figure_display = save_gif and not save_fig
    debug_param = True if suppress_figure_display else mute_mode
    
    # Process each quality index: distribution plot followed by t-statistics plot
    for quality_index in target_quality_indices:
        if quality_index not in quality_data:
            if mute_mode:
                print(f"Skipping {quality_index} - no data available")
            continue
        
        # Skip verbose header output when mute_mode=True (keep it simple)
        
        # Create static plots and optionally get plot info for gif creation
        if save_gif and output_gif_filenames and quality_index in output_gif_filenames:
            # Get plot info for gif creation while creating static plots (only if not skipping pdf graph)
            if not skip_pdf_graph:
                distribution_plot_info = plot_quality_distributions(
                    quality_data, [quality_index], output_figure_filenames if save_fig else {}, 
                    CORE_A, CORE_B, debug=debug_param, return_plot_info=True,
                    plot_real_data_histogram=plot_real_data_histogram, plot_age_removal_step_pdf=plot_age_removal_step_pdf,
                    synthetic_csv_filenames=synthetic_csv_filenames,
                    best_datum_values=best_datum_values,
                    fig_format=fig_format, dpi=dpi,
                    invert_norm_dtw=invert_norm_dtw
                )
            
            # Only create plots if valid age constraints exist and not skipping age removal graph
            if has_valid_age_constraints and not skip_age_removal_graph:
                # Create t-statistics plot only if show_t_graph is True
                if show_t_graph:
                    tstat_plot_info = plot_t_statistics_vs_constraints(
                        quality_data, [quality_index], output_figure_filenames if save_fig else {},
                        CORE_A, CORE_B, debug=debug_param, return_plot_info=True,
                        fig_format=fig_format, dpi=dpi,
                        invert_norm_dtw=invert_norm_dtw
                    )
                
                # Create Cohen's d plot only if show_d_graph is True
                if show_d_graph:
                    cohens_d_plot_info = plot_cohens_d_vs_constraints(
                        quality_data, [quality_index], output_figure_filenames if save_fig else {},
                        CORE_A, CORE_B, debug=debug_param, return_plot_info=True,
                        fig_format=fig_format, dpi=dpi,
                        invert_norm_dtw=invert_norm_dtw
                    )
                
                # Create Hedges' g plot if show_g_graph is True
                if show_g_graph:
                    hedges_g_plot_info = plot_hedges_g_vs_constraints(
                        quality_data, [quality_index], output_figure_filenames if save_fig else {},
                        CORE_A, CORE_B, debug=debug_param, return_plot_info=True,
                        fig_format=fig_format, dpi=dpi,
                        invert_norm_dtw=invert_norm_dtw, linear_g_scale=linear_g_scale
                    )
            
            # Create gifs using the plot info
            distribution_gif_filename = output_gif_filenames[quality_index]
            
            # Create distribution gif (only if not skipping pdf graph)
            if not skip_pdf_graph:
                _create_distribution_gif(distribution_plot_info[quality_index], distribution_gif_filename, mute_mode, max_frames, best_datum_values, invert_norm_dtw=invert_norm_dtw)
                created_gifs.append(distribution_gif_filename)

            # Create gifs only if valid age constraints exist and not skipping age removal graph
            if has_valid_age_constraints and not skip_age_removal_graph:
                # Create t-statistics gif only if show_t_graph is True
                if show_t_graph:
                    tstat_gif_filename = distribution_gif_filename.replace('.gif', '_tstat.gif')
                    _create_tstat_gif(tstat_plot_info[quality_index], tstat_gif_filename, mute_mode, max_frames, invert_norm_dtw=invert_norm_dtw)
                    created_gifs.append(tstat_gif_filename)
                
                # Create Cohen's d gif only if show_d_graph is True
                if show_d_graph:
                    cohens_d_gif_filename = distribution_gif_filename.replace('.gif', '_cohens_d.gif')
                    _create_cohens_d_gif(cohens_d_plot_info[quality_index], cohens_d_gif_filename, mute_mode, max_frames, invert_norm_dtw=invert_norm_dtw)
                    created_gifs.append(cohens_d_gif_filename)
                
                # Create Hedges' g gif if show_g_graph is True
                if show_g_graph:
                    hedges_g_gif_filename = distribution_gif_filename.replace('.gif', '_hedges_g.gif')
                    _create_hedges_g_gif(hedges_g_plot_info[quality_index], hedges_g_gif_filename, mute_mode, max_frames, invert_norm_dtw=invert_norm_dtw)
                    created_gifs.append(hedges_g_gif_filename)
            
        else:
            # Just create static plots (only if not skipping pdf graph)
            if not skip_pdf_graph:
                plot_quality_distributions(
                    quality_data, [quality_index], output_figure_filenames if save_fig else {}, 
                    CORE_A, CORE_B, debug=debug_param, return_plot_info=False,
                    plot_real_data_histogram=plot_real_data_histogram, plot_age_removal_step_pdf=plot_age_removal_step_pdf,
                    synthetic_csv_filenames=synthetic_csv_filenames,
                    best_datum_values=best_datum_values,
                    fig_format=fig_format, dpi=dpi,
                    invert_norm_dtw=invert_norm_dtw
                )
            
            # Only create plots if valid age constraints exist and not skipping age removal graph
            if has_valid_age_constraints and not skip_age_removal_graph:
                # Create t-statistics plot only if show_t_graph is True
                if show_t_graph:
                    plot_t_statistics_vs_constraints(
                        quality_data, [quality_index], output_figure_filenames if save_fig else {},
                        CORE_A, CORE_B, debug=debug_param, return_plot_info=False,
                        fig_format=fig_format, dpi=dpi,
                        invert_norm_dtw=invert_norm_dtw
                    )
                
                # Create Cohen's d plot only if show_d_graph is True
                if show_d_graph:
                    plot_cohens_d_vs_constraints(
                        quality_data, [quality_index], output_figure_filenames if save_fig else {},
                        CORE_A, CORE_B, debug=debug_param, return_plot_info=False,
                        fig_format=fig_format, dpi=dpi,
                        invert_norm_dtw=invert_norm_dtw
                    )
                
                # Create Hedges' g plot if show_g_graph is True
                if show_g_graph:
                    plot_hedges_g_vs_constraints(
                        quality_data, [quality_index], output_figure_filenames if save_fig else {},
                        CORE_A, CORE_B, debug=debug_param, return_plot_info=False,
                        fig_format=fig_format, dpi=dpi,
                        invert_norm_dtw=invert_norm_dtw, linear_g_scale=linear_g_scale
                    )
    
    # Display all created GIFs at the end when save_gif=True
    if save_gif and created_gifs:
        print("\n" + "="*60)
        print("DISPLAYING CREATED GIFS")
        print("="*60)
        for gif_file in created_gifs:
            if os.path.exists(gif_file):
                print(f"\nDisplaying: {os.path.basename(gif_file)}")
                try:
                    display(IPImage(filename=gif_file))
                except Exception as e:
                    print(f"Could not display {gif_file}: {e}")
            else:
                print(f"Warning: GIF file not found: {gif_file}")
    
    # Skip verbose completion message when mute_mode=True


def _create_distribution_gif(plot_info, gif_filename, mute_mode, max_frames=50, best_datum_values=None, invert_norm_dtw=True):
    """
    Create distribution comparison gif using plot info from plot_quality_distributions.
    """
    
    unique_constraints = plot_info['unique_constraints']
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    try:
        # Frame 0: Just null hypothesis
        fig, ax = plt.subplots(figsize=(10, 4.5))  # Match static plot size
        
        # Plot null hypothesis with exact same styling as static plot
        # Use synthetic_bins (actual bin edges) for consistent histogram scaling with PDF
        hist_bins = plot_info.get('synthetic_bins', plot_info['n_bins'])
        ax.hist(plot_info['combined_data'], bins=hist_bins, alpha=0.3, color='gray', 
                density=False,
                weights=np.ones(len(plot_info['combined_data'])) * 100 / len(plot_info['combined_data']), label='Synthetic data histogram')
        ax.plot(plot_info['x_synth'], plot_info['y_synth'], color='gray', linestyle=':', linewidth=2, alpha=0.8, label='Synthetic data PDF (Null Hypothesis)')
        
        # Apply exact same styling as static plot (fixed axis ranges)
        quality_index = plot_info['quality_index']
        if 'actual_plot_limits' in plot_info:
            # Use actual axis limits from static plot
            ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
            ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
        else:
            # Fallback to calculated limits
            # Sectional metrics use the same x-axis range as their non-sectional counterparts
            if quality_index in ['corr_coef', 'corr_coef_sect']:
                ax.set_xlim(0, 1.0)
            elif quality_index in ['norm_dtw', 'norm_dtw_sect']:
                ax.set_xlim(0.7, 1.0)
            else:
                ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
                ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
        # Get display name from plot_info or create it
        display_name = plot_info.get('display_name', quality_index)
        if not display_name or display_name == quality_index:
            if quality_index == 'corr_coef':
                display_name = "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                display_name = "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                display_name = "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                display_name = "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                display_name = quality_index
        
        ax.set_xlabel(f'{display_name}')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{display_name}\n{plot_info["CORE_A"]} vs {plot_info["CORE_B"]}')  # Same title as PNG
        # Create complete legend (same as static plot)
        if plot_info['legend_elements'] and plot_info['legend_labels']:
            legend = ax.legend(plot_info['legend_elements'], plot_info['legend_labels'], bbox_to_anchor=(1.02, 1), loc='upper left')
            
            # Style the group title labels (same as static plot)
            for i, label in enumerate(plot_info['legend_labels']):
                if (label.startswith('Null Hypotheses') or 
                    label.startswith('Real Data') or 
                    label.startswith('# of')):
                    legend.get_texts()[i].set_weight('bold')
                    legend.get_texts()[i].set_fontsize(10)
                    legend.get_texts()[i].set_ha('left')
                else:
                    # Make all other legend text smaller too
                    legend.get_texts()[i].set_fontsize(9)
                    
            # Add colorbar for age constraint levels (same as static plot)
            min_core_b = plot_info['min_core_b']
            max_core_b = plot_info['max_core_b']
            cmap = cm.get_cmap('Spectral_r')
            norm = colors.Normalize(vmin=min_core_b, vmax=max_core_b)
            
            # Position colorbar directly under the legend box
            # First render the plot to get accurate legend positioning
            plt.draw()
            
            # Get legend position in figure coordinates
            legend_bbox = legend.get_window_extent()
            legend_bbox_axes = legend_bbox.transformed(fig.transFigure.inverted())
            
            # Position colorbar directly below legend
            colorbar_height = 0.025
            colorbar_width = legend_bbox_axes.width * 0.9  # Slightly smaller than legend
            colorbar_x = legend_bbox_axes.x0 - legend_bbox_axes.width *.77  # Center it
            colorbar_y = legend_bbox_axes.y0 - colorbar_height + 0.13  # Below legend with gap
            
            cax = fig.add_axes([colorbar_x, colorbar_y, colorbar_width, colorbar_height])
            
            # Create a colorbar with the same normalization as the curves
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.set_label(f'# of {plot_info["CORE_B"]} Age Constraints', fontsize=9, fontweight='bold')
            
            # Set ticks to show actual constraint levels
            unique_constraints = plot_info['unique_constraints']
            cbar.set_ticks([min_core_b + (max_core_b - min_core_b) * i / (len(unique_constraints) - 1) 
                           for i in range(len(unique_constraints))])
            cbar.set_ticklabels([str(level) for level in sorted(unique_constraints)])
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Don't add best datum match line in initial frame - it will be added as final frame
        
        # Add horizontal black arrow with 'Better Correlation Quality' text (same as static plot)
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Position arrow in upper left of the plot area
        arrow_y = ax_ylim[0] + 0.92 * (ax_ylim[1] - ax_ylim[0])  # 92% up from bottom
        
        # Determine arrow direction and position based on quality index
        quality_index = plot_info['quality_index']
        # When invert_norm_dtw=True, norm_dtw metrics use right-pointing arrow (like corr_coef)
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_left_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_left_arrow:
            # For norm_dtw (non-inverted), lower values are better (arrow points left) - position in upper right
            arrow_start_x = ax_xlim[0] + 0.99 * (ax_xlim[1] - ax_xlim[0])  # Start 99% from left (far right)
            arrow_end_x = ax_xlim[0] + 0.82 * (ax_xlim[1] - ax_xlim[0])    # End 82% from left  
            text_x = ax_xlim[0] + 0.905 * (ax_xlim[1] - ax_xlim[0])        # Text centered between 82% and 99%
        else:
            # For corr_coef and other indices (or inverted norm_dtw), higher values are better (arrow points right) - position in upper left
            arrow_start_x = ax_xlim[0] + 0.07 * (ax_xlim[1] - ax_xlim[0])  # Start 7% from left
            arrow_end_x = ax_xlim[0] + 0.24 * (ax_xlim[1] - ax_xlim[0])    # End 24% from left
            text_x = ax_xlim[0] + 0.155 * (ax_xlim[1] - ax_xlim[0])        # Text centered under arrow
            
        # Add horizontal arrow
        ax.annotate('', 
                   xy=(arrow_end_x, arrow_y), 
                   xytext=(arrow_start_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2),
                   zorder=5)
        
        # Add text under the arrow
        text_y = arrow_y - 0.03 * (ax_ylim[1] - ax_ylim[0])  # 3% below arrow
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='center', va='top',
               color='black', zorder=4)
        
        frame_file = os.path.join(temp_dir, f'frame_000.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        frame_files.append(frame_file)
        
        # Progressive frames: Add individual curves one by one
        individual_curves = plot_info.get('individual_curves', [])
        
        if not individual_curves:
            # Fallback to old method if individual curves not available
            return
        
        # Calculate grouping based on max_frames limit
        total_curves = len(individual_curves)
        # Use max_frames for animation - pause frames are added after
        target_animation_frames = max_frames
        
        if total_curves <= target_animation_frames:
            # No grouping needed - one curve per frame
            curves_per_frame = 1
            num_frames = total_curves
        else:
            # Calculate how many curves per frame to stay close to max_frames
            # We want: num_frames ≈ target_animation_frames, so curves_per_frame = ceil(total_curves / target_animation_frames)
            curves_per_frame = math.ceil(total_curves / target_animation_frames)
            # Calculate actual number of frames needed
            num_frames = math.ceil(total_curves / curves_per_frame)
        
        # Skip verbose GIF frame info when mute_mode=True
        
        # Create frames with progress bar
        with tqdm(total=num_frames, desc="  Creating distribution GIF frames", disable=mute_mode) as pbar:
            for frame_idx in range(num_frames):
                # Calculate which curves to show up to this frame
                curves_to_show = min((frame_idx + 1) * curves_per_frame, total_curves)
                
                # Create frame with exact same styling as static plot
                fig, ax = plt.subplots(figsize=(10, 4.5))  # Match static plot size
            
                # Plot null hypothesis first (same as static plot)
                # Use synthetic_bins (actual bin edges) for consistent histogram scaling with PDF
                hist_bins = plot_info.get('synthetic_bins', plot_info['n_bins'])
                ax.hist(plot_info['combined_data'], bins=hist_bins, alpha=0.3, color='gray', 
                        density=False,
                        weights=np.ones(len(plot_info['combined_data'])) * 100 / len(plot_info['combined_data']), label='Synthetic data histogram')
                ax.plot(plot_info['x_synth'], plot_info['y_synth'], color='gray', linestyle=':', linewidth=2, alpha=0.8, label='Synthetic data PDF (Null Hypothesis)')
                
                # Plot all curves up to current frame (same styling as static plot)
                plotted_constraint_levels = set()
                
                for curve_idx in range(curves_to_show):
                    curve = individual_curves[curve_idx]
                    
                    # Use the exact same styling as stored from static plot
                    label = curve['label']
                    constraint_count = curve['constraint_count']
                    
                    # Only show label if this constraint level hasn't been labeled yet
                    if constraint_count in plotted_constraint_levels:
                        label = None
                    else:
                        plotted_constraint_levels.add(constraint_count)
                    
                    ax.plot(curve['x_range'], curve['y_values'], 
                           color=curve['color'], 
                           linestyle=curve['linestyle'],
                           linewidth=curve['linewidth'], 
                           alpha=curve['alpha'],
                           zorder=curve['zorder'],
                           label=label)
                
                # Apply exact same styling as static plot (fixed axis ranges)
                quality_index = plot_info['quality_index']
                if 'actual_plot_limits' in plot_info:
                    # Use actual axis limits from static plot
                    ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
                    ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
                else:
                    # Fallback to calculated limits
                    # Sectional metrics use the same x-axis range as their non-sectional counterparts
                    if quality_index in ['corr_coef', 'corr_coef_sect']:
                        ax.set_xlim(0, 1.0)
                    elif quality_index in ['norm_dtw', 'norm_dtw_sect']:
                        ax.set_xlim(0.5, 1.0)
                    else:
                        ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
                        ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
                ax.set_xlabel(f'{display_name}')
                ax.set_ylabel('Percentage (%)')
                ax.set_title(f'{display_name}\n{plot_info["CORE_A"]} vs {plot_info["CORE_B"]}')  # Same title as PNG
                # Create complete legend (same as static plot)
                if plot_info['legend_elements'] and plot_info['legend_labels']:
                    legend = ax.legend(plot_info['legend_elements'], plot_info['legend_labels'], bbox_to_anchor=(1.02, 1), loc='upper left')
                    
                    # Style the group title labels (same as static plot)
                    for i, label in enumerate(plot_info['legend_labels']):
                        if (label.startswith('Null Hypotheses') or 
                            label.startswith('Real Data') or 
                            label.startswith('# of')):
                            legend.get_texts()[i].set_weight('bold')
                            legend.get_texts()[i].set_fontsize(10)
                            legend.get_texts()[i].set_ha('left')
                        else:
                            # Make all other legend text smaller too
                            legend.get_texts()[i].set_fontsize(9)
                            
                    # Add colorbar for age constraint levels (same as static plot)
                    min_core_b = plot_info['min_core_b']
                    max_core_b = plot_info['max_core_b']
                    cmap = cm.get_cmap('Spectral_r')
                    norm = colors.Normalize(vmin=min_core_b, vmax=max_core_b)
                    
                    # Position colorbar directly under the legend box
                    # First render the plot to get accurate legend positioning
                    plt.draw()
                    
                    # Get legend position in figure coordinates
                    legend_bbox = legend.get_window_extent()
                    legend_bbox_axes = legend_bbox.transformed(fig.transFigure.inverted())
                    
                    # Position colorbar directly below legend
                    colorbar_height = 0.025
                    colorbar_width = legend_bbox_axes.width * 0.9  # Slightly smaller than legend
                    colorbar_x = legend_bbox_axes.x0 - legend_bbox_axes.width * 0.77  # Center it
                    colorbar_y = legend_bbox_axes.y0 - colorbar_height + 0.13  # Below legend with gap
                    
                    cax = fig.add_axes([colorbar_x, colorbar_y, colorbar_width, colorbar_height])
                    
                    # Create a colorbar with the same normalization as the curves
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    
                    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
                    cbar.set_label(f'# of {plot_info["CORE_B"]} Age Constraints', fontsize=9, fontweight='bold')
                    
                    # Set ticks to show actual constraint levels
                    unique_constraints = plot_info['unique_constraints']
                    cbar.set_ticks([min_core_b + (max_core_b - min_core_b) * i / (len(unique_constraints) - 1) 
                                   for i in range(len(unique_constraints))])
                    cbar.set_ticklabels([str(level) for level in sorted(unique_constraints)])
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                
                ax.grid(True, alpha=0.3)
                
                # Don't add best datum match line in progressive frames - it will be added as final frame
                
                # Add horizontal black arrow with 'Better Correlation Quality' text (same as static plot)
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                
                # Position arrow in upper left of the plot area
                arrow_y = ax_ylim[0] + 0.92 * (ax_ylim[1] - ax_ylim[0])  # 92% up from bottom
                
                # Determine arrow direction and position based on quality index
                # When invert_norm_dtw=True, norm_dtw metrics use right-pointing arrow (like corr_coef)
                quality_index = plot_info['quality_index']
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                use_left_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_left_arrow:
                    # For norm_dtw (non-inverted), lower values are better (arrow points left) - position in upper right
                    arrow_start_x = ax_xlim[0] + 0.94 * (ax_xlim[1] - ax_xlim[0])
                    arrow_end_x = ax_xlim[0] + 0.77 * (ax_xlim[1] - ax_xlim[0])
                    text_x = ax_xlim[0] + 0.855 * (ax_xlim[1] - ax_xlim[0])
                else:
                    # For other quality indices (or inverted norm_dtw), higher values are better (arrow points right) - position in upper left
                    arrow_start_x = ax_xlim[0] + 0.07 * (ax_xlim[1] - ax_xlim[0])
                    arrow_end_x = ax_xlim[0] + 0.24 * (ax_xlim[1] - ax_xlim[0])
                    text_x = ax_xlim[0] + 0.155 * (ax_xlim[1] - ax_xlim[0])
                    
                # Add horizontal arrow
                ax.annotate('', 
                           xy=(arrow_end_x, arrow_y), 
                           xytext=(arrow_start_x, arrow_y),
                           arrowprops=dict(arrowstyle='->', color='black', lw=2),
                           zorder=5)
                
                # Add text under the arrow
                text_y = arrow_y - 0.03 * (ax_ylim[1] - ax_ylim[0])  # 3% below arrow
                ax.text(text_x, text_y, 'Better Correlation Quality',
                       fontsize=8, ha='center', va='top',
                       color='black', zorder=4)
                
                # Save frame
                frame_file = os.path.join(temp_dir, f'frame_{frame_idx + 1:03d}.png')
                plt.tight_layout()
                plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                frame_files.append(frame_file)
                
                # Update progress bar
                pbar.update(1)
        
        # Create additional final frame with best datum match line if available
        if best_datum_values and plot_info['quality_index'] in best_datum_values:
            best_value = best_datum_values[plot_info['quality_index']]
            if pd.notna(best_value):  # Check if value is not NaN
                # Create final frame with all curves plus best datum match line
                fig, ax = plt.subplots(figsize=(10, 4.5))  # Match static plot size
            
                # Plot null hypothesis first (same as static plot)
                # Use synthetic_bins (actual bin edges) for consistent histogram scaling with PDF
                hist_bins = plot_info.get('synthetic_bins', plot_info['n_bins'])
                ax.hist(plot_info['combined_data'], bins=hist_bins, alpha=0.3, color='gray', 
                        density=False,
                        weights=np.ones(len(plot_info['combined_data'])) * 100 / len(plot_info['combined_data']), label='Synthetic data histogram')
                ax.plot(plot_info['x_synth'], plot_info['y_synth'], color='gray', linestyle=':', linewidth=2, alpha=0.8, label='Synthetic data PDF (Null Hypothesis)')
                
                # Plot ALL curves (same styling as static plot)
                plotted_constraint_levels = set()
                
                for curve in individual_curves:
                    # Use the exact same styling as stored from static plot
                    label = curve['label']
                    constraint_count = curve['constraint_count']
                    
                    # Only show label if this constraint level hasn't been labeled yet
                    if constraint_count in plotted_constraint_levels:
                        label = None
                    else:
                        plotted_constraint_levels.add(constraint_count)
                    
                    ax.plot(curve['x_range'], curve['y_values'], 
                           color=curve['color'], 
                           linestyle=curve['linestyle'],
                           linewidth=curve['linewidth'], 
                           alpha=curve['alpha'],
                           zorder=curve['zorder'],
                           label=label)
                
                # Invert norm_dtw value to match the x-axis (1 - norm_dtw) when invert_norm_dtw is True
                quality_index = plot_info['quality_index']
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                plot_value = (1 - best_value) if (invert_norm_dtw and is_norm_dtw_metric) else best_value
                
                # Add best datum match vertical line in dark green with long dash and highest zorder
                ax.axvline(plot_value, color='darkgreen', linestyle='--', linewidth=2, zorder=100)
                
                # Add text annotation next to the line
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                text_y = ax_ylim[0] + 0.90 * (ax_ylim[1] - ax_ylim[0])  # 90% up from bottom (higher position)
                
                # Position text based on arrow direction
                # When invert_norm_dtw=True, norm_dtw metrics use right-pointing arrows (like corr_coef)
                use_left_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_left_arrow:
                    # For left-pointing arrows, put text on left side of line
                    text_x = plot_value - 0.01 * (ax_xlim[1] - ax_xlim[0])
                    ha = 'right'
                else:
                    # For right-pointing arrows, put text on right side of line
                    text_x = plot_value + 0.01 * (ax_xlim[1] - ax_xlim[0])
                    ha = 'left'
                
                ax.text(text_x, text_y, 
                       f'Best\nDatum\nMatch\n({plot_value:.3f})', 
                       color='darkgreen', fontweight='bold', fontsize='x-small',
                       ha=ha, va='center', zorder=101)
                
                # Apply exact same styling as static plot (fixed axis ranges)
                quality_index = plot_info['quality_index']
                if 'actual_plot_limits' in plot_info:
                    # Use actual axis limits from static plot
                    ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
                    ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
                else:
                    # Fallback to calculated limits
                    # Sectional metrics use the same x-axis range as their non-sectional counterparts
                    if quality_index in ['corr_coef', 'corr_coef_sect']:
                        ax.set_xlim(0, 1.0)
                    elif quality_index in ['norm_dtw', 'norm_dtw_sect']:
                        ax.set_xlim(0.5, 1.0)
                    else:
                        ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
                        ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
                ax.set_xlabel(f'{display_name}')
                ax.set_ylabel('Percentage (%)')
                ax.set_title(f'{display_name}\n{plot_info["CORE_A"]} vs {plot_info["CORE_B"]}')  # Same title as PNG
                
                # Create complete legend (same as static plot, best datum match shown as text)
                if plot_info['legend_elements'] and plot_info['legend_labels']:
                    legend = ax.legend(plot_info['legend_elements'], plot_info['legend_labels'], bbox_to_anchor=(1.02, 1), loc='upper left')
                    
                    # Style the group title labels (same as static plot)
                    for i, label in enumerate(plot_info['legend_labels']):
                        if (label.startswith('Synthetic Data Correlations') or 
                            label.startswith('Real Data Correlations') or 
                            label.startswith('# of')):
                            legend.get_texts()[i].set_weight('bold')
                            legend.get_texts()[i].set_fontsize(10)
                            legend.get_texts()[i].set_ha('left')
                        else:
                            # Make all other legend text smaller too
                            legend.get_texts()[i].set_fontsize(9)
                            
                    # Add colorbar for age constraint levels (same as static plot)
                    min_core_b = plot_info['min_core_b']
                    max_core_b = plot_info['max_core_b']
                    cmap = cm.get_cmap('Spectral_r')
                    norm = colors.Normalize(vmin=min_core_b, vmax=max_core_b)
                    
                    # Position colorbar directly under the legend box
                    # First render the plot to get accurate legend positioning
                    plt.draw()
                    
                    # Get legend position in figure coordinates
                    legend_bbox = legend.get_window_extent()
                    legend_bbox_axes = legend_bbox.transformed(fig.transFigure.inverted())
                    
                    # Position colorbar directly below legend
                    colorbar_height = 0.025
                    colorbar_width = legend_bbox_axes.width * 0.9  # Slightly smaller than legend
                    colorbar_x = legend_bbox_axes.x0 - legend_bbox_axes.width * 0.77  # Center it
                    colorbar_y = legend_bbox_axes.y0 - colorbar_height + 0.13  # Below legend with gap
                    
                    cax = fig.add_axes([colorbar_x, colorbar_y, colorbar_width, colorbar_height])
                    
                    # Create a colorbar with the same normalization as the curves
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    
                    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
                    cbar.set_label(f'# of {plot_info["CORE_B"]} Age Constraints', fontsize=9, fontweight='bold')
                    
                    # Set ticks to show actual constraint levels
                    unique_constraints = plot_info['unique_constraints']
                    cbar.set_ticks([min_core_b + (max_core_b - min_core_b) * i / (len(unique_constraints) - 1) 
                                   for i in range(len(unique_constraints))])
                    cbar.set_ticklabels([str(level) for level in sorted(unique_constraints)])
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                
                ax.grid(True, alpha=0.3)
                
                # Add horizontal black arrow with 'Better Correlation Quality' text (same as static plot)
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                
                # Position arrow in upper left of the plot area
                arrow_y = ax_ylim[0] + 0.92 * (ax_ylim[1] - ax_ylim[0])  # 92% up from bottom
                
                # Determine arrow direction and position based on quality index
                # When invert_norm_dtw=True, norm_dtw metrics use right-pointing arrow (like corr_coef)
                quality_index = plot_info['quality_index']
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                use_left_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_left_arrow:
                    # For norm_dtw (non-inverted), lower values are better (arrow points left) - position in upper right
                    arrow_start_x = ax_xlim[0] + 0.94 * (ax_xlim[1] - ax_xlim[0])
                    arrow_end_x = ax_xlim[0] + 0.77 * (ax_xlim[1] - ax_xlim[0])
                    text_x = ax_xlim[0] + 0.855 * (ax_xlim[1] - ax_xlim[0])
                else:
                    # For other quality indices (or inverted norm_dtw), higher values are better (arrow points right) - position in upper left
                    arrow_start_x = ax_xlim[0] + 0.07 * (ax_xlim[1] - ax_xlim[0])
                    arrow_end_x = ax_xlim[0] + 0.24 * (ax_xlim[1] - ax_xlim[0])
                    text_x = ax_xlim[0] + 0.155 * (ax_xlim[1] - ax_xlim[0])
                    
                # Add horizontal arrow
                ax.annotate('', 
                           xy=(arrow_end_x, arrow_y), 
                           xytext=(arrow_start_x, arrow_y),
                           arrowprops=dict(arrowstyle='->', color='black', lw=2),
                           zorder=5)
                
                # Add text under the arrow
                text_y = arrow_y - 0.03 * (ax_ylim[1] - ax_ylim[0])  # 3% below arrow
                ax.text(text_x, text_y, 'Better Correlation Quality',
                       fontsize=8, ha='center', va='top',
                       color='black', zorder=4)
                
                # Save final frame with best datum match
                final_frame_file = os.path.join(temp_dir, f'frame_final.png')
                plt.tight_layout()
                plt.savefig(final_frame_file, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                frame_files.append(final_frame_file)
        
        # Create gif from frames
        if frame_files:
            # Skip verbose frame combining message
            
            images = []
            target_size = None
            
            for frame_file in frame_files:
                img = imageio.imread(frame_file)
                if target_size is None:
                    target_size = img.shape[:2]
                elif img.shape[:2] != target_size:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                    img = np.array(pil_img)
                images.append(img)
            
            # Add frames to make final frame stay for 2 seconds
            if images:
                final_frame = images[-1]
                fps = 10  # imageio default fps
                frames_for_2_seconds = fps * 2  # 2 seconds = fps * 2 frames
                for _ in range(frames_for_2_seconds):
                    images.append(final_frame)
            
            # Create directory if needed
            output_dir = os.path.dirname(gif_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            imageio.mimsave(gif_filename, images, fps=fps, loop=3)
            
            if mute_mode:
                print(f"✓ Distribution gif saved as: {gif_filename}")
            else:
                print(f"✓ Distribution gif saved as: {gif_filename}")
                
            # Note: GIF will be displayed at the end of the main function
        else:
            # Skip verbose error message
            pass
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _create_tstat_gif(plot_info, gif_filename, mute_mode, max_frames=50, invert_norm_dtw=True):
    """
    Create t-statistics gif using plot info from plot_t_statistics_vs_constraints.
    """
    
    unique_constraints = plot_info['unique_constraints']
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    try:
        # Frame 0: Empty t-statistics plot (match static plot exactly)
        fig, ax = plt.subplots(figsize=(9.5, 5))
        
        # Add null hypothesis line (same as static plot)
        ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        
        # Set fixed axis ranges for all frames (use actual limits from static plot)
        if 'actual_plot_limits' in plot_info:
            ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
            ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
        else:
            ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
            ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
        ax.set_xticks(range(0, int(plot_info['plot_limits']['x_max']) + 1))
        
        # Format plot (same as static plot)
        ax.set_xlabel(f'Number of {plot_info["CORE_B"]} Age Constraints')
        ax.set_ylabel('t-statistic')
        # Get display name from plot_info or create it
        quality_index = plot_info['quality_index']
        display_name = plot_info.get('display_name', quality_index)
        if not display_name or display_name == quality_index:
            if quality_index == 'corr_coef':
                display_name = "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                display_name = "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                display_name = "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                display_name = "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                display_name = quality_index
        
        ax.set_title(f'{display_name}\n{plot_info["CORE_A"]} vs {plot_info["CORE_B"]}')
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add inline text label for synthetic data line with white background box
        ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
               color='dimgray', style='italic', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Add vertical colorbar for improvement/deterioration on right side
        sm = plt.cm.ScalarMappable(cmap=plot_info['cmap'], norm=plt.Normalize(vmin=-plot_info['max_abs_score'], vmax=plot_info['max_abs_score']))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.05)
        # Set bold label on the left side of the colorbar
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel('Change in t-statistic', fontweight='bold', labelpad=5, rotation=90)
        cbar.set_ticks([-plot_info['max_abs_score'], 0, plot_info['max_abs_score']])
        cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
        
        # Add "Better Correlation Quality" arrow (same as static plot)
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Position arrow to the left of x=0
        arrow_x = -0.35  # Left of x=0
        arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2  # Center vertically
        
        # Determine arrow direction based on quality index
        # When invert_norm_dtw=True, norm_dtw metrics use upward arrow (like corr_coef)
        quality_index = plot_info['quality_index']
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_downward_arrow:
            # For norm_dtw (non-inverted), lower values are better (downward arrow)
            arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
        else:
            # For other quality indices (or inverted norm_dtw), higher values are better (upward arrow)
            arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
            
        # Create gradient arrow using LineCollection
        n_segments = 100
        y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
        x_vals = np.full_like(y_vals, arrow_x)
        
        # Create line segments for gradient effect - exclude the last segment to leave space for arrowhead
        points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create colors for each segment - full spectrum of colormap
        colors_gradient = [plot_info['cmap'](i / (n_segments - 1)) for i in range(n_segments - 1)]
        
        # Create the gradient line
        lc = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
        ax.add_collection(lc)
        
        # Add arrowhead at the end
        arrow_color = plot_info['cmap'](1.0)
        arrowhead_start_y = y_vals[-2]
        arrowhead_end_y = arrow_y_end
        
        # Add just the arrowhead
        ax.annotate('', 
                   xy=(arrow_x, arrowhead_end_y), 
                   xytext=(arrow_x, arrowhead_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                   zorder=5)
        
        # Add text next to the arrow
        text_x = arrow_x + 0.1  # Slightly to the right of arrow
        text_y = arrow_y_center
        
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='left', va='center',
               rotation=90, color='black',
               zorder=4)
        
        frame_file = os.path.join(temp_dir, f'frame_000.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        frame_files.append(frame_file)
        
        # Progressive frames: Add points by constraint level, then by individual points within level
        individual_points = plot_info.get('individual_points', [])
        individual_segments = plot_info.get('individual_segments', [])
        
        if not individual_points:
            # Fallback - no individual points available
            return
        
        # Group points by constraint level
        points_by_constraint = {}
        for i, point in enumerate(individual_points):
            constraint_level = point['x']
            if constraint_level not in points_by_constraint:
                points_by_constraint[constraint_level] = []
            points_by_constraint[constraint_level].append((i, point))
        
        # Sort constraint levels
        sorted_constraint_levels = sorted(points_by_constraint.keys())
        
        # Create ordered list of points to plot (by constraint level, then within level)
        ordered_points = []
        for constraint_level in sorted_constraint_levels:
            for original_idx, point in points_by_constraint[constraint_level]:
                ordered_points.append((original_idx, point))
        
        # Calculate grouping based on max_frames limit
        total_points = len(ordered_points)
        # Use max_frames for animation - pause frames are added after
        target_animation_frames = max_frames
        
        if total_points <= target_animation_frames:
            # No grouping needed - one point per frame
            points_per_frame = 1
            num_frames = total_points
        else:
            # Calculate how many points per frame to stay close to max_frames
            # We want: num_frames ≈ target_animation_frames, so points_per_frame = ceil(total_points / target_animation_frames)
            points_per_frame = math.ceil(total_points / target_animation_frames)
            # Calculate actual number of frames needed
            num_frames = math.ceil(total_points / points_per_frame)
        
        # Skip verbose GIF frame info when mute_mode=True
        
        # Create frames with progress bar
        with tqdm(total=num_frames, desc="  Creating t-stat GIF frames", disable=mute_mode) as pbar:
            for frame_idx in range(num_frames):
                # Calculate which points to show up to this frame
                points_to_show = min((frame_idx + 1) * points_per_frame, total_points)
                # Create frame with exact same styling as static plot
                fig, ax = plt.subplots(figsize=(9.5, 5))
                
                # Add null hypothesis line (same as static plot)
                ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                # Plot all points up to current frame with white fill
                shown_points = set()
                for i in range(points_to_show):
                    original_idx, point = ordered_points[i]
                    ax.scatter(point['x'], point['y'], 
                             color='white', 
                             edgecolor='black',
                             linewidth=point['linewidth'], 
                             s=point['size'], 
                             zorder=point['zorder'])
                    shown_points.add((point['x'], point['y']))
                
                # Plot ALL segments where BOTH endpoints have been shown
                frame_segments = []
                frame_colors = []
                
                for segment in individual_segments:
                    seg_data = segment['segment']
                    start_x, start_y = seg_data[0]
                    end_x, end_y = seg_data[1]
                    
                    # Show segment if both start and end points have been plotted
                    if (start_x, start_y) in shown_points and (end_x, end_y) in shown_points:
                        frame_segments.append(seg_data)
                        frame_colors.append(segment['color'])
                
                # Draw all segments at once using LineCollection
                if frame_segments:
                    lc = LineCollection(frame_segments, colors=frame_colors, 
                                      alpha=individual_segments[0]['alpha'] if individual_segments else 0.8, 
                                      linewidths=individual_segments[0]['linewidth'] if individual_segments else 1, 
                                      zorder=1)
                    ax.add_collection(lc)
            
                # Set fixed axis ranges for all frames (use actual limits from static plot)
                if 'actual_plot_limits' in plot_info:
                    ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
                    ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
                else:
                    ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
                    ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
                ax.set_xticks(range(0, int(plot_info['plot_limits']['x_max']) + 1))
                
                # Format plot (same as static plot)
                ax.set_xlabel(f'Number of {plot_info["CORE_B"]} Age Constraints')
                ax.set_ylabel('t-statistic')
                ax.set_title(f'{display_name}\n{plot_info["CORE_A"]} vs {plot_info["CORE_B"]}')
                ax.grid(True, alpha=0.3, zorder=0)
                
                # Add inline text label for synthetic data line with white background box
                ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
                       color='dimgray', style='italic', zorder=4,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
                
                # Add vertical colorbar for improvement/deterioration on right side
                sm = plt.cm.ScalarMappable(cmap=plot_info['cmap'], norm=plt.Normalize(vmin=-plot_info['max_abs_score'], vmax=plot_info['max_abs_score']))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.05)
                # Set bold label on the left side of the colorbar
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.set_ylabel('Change in t-statistic', fontweight='bold', labelpad=5, rotation=90)
                cbar.set_ticks([-plot_info['max_abs_score'], 0, plot_info['max_abs_score']])
                cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
                
                # Add "Better Correlation Quality" arrow (same as static plot)
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                
                # Position arrow to the left of x=0
                arrow_x = -0.35  # Left of x=0
                arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2  # Center vertically
                
                # Determine arrow direction based on quality index
                # When invert_norm_dtw=True, norm_dtw metrics use upward arrow (like corr_coef)
                quality_index = plot_info['quality_index']
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_downward_arrow:
                    # For norm_dtw (non-inverted), lower values are better (downward arrow)
                    arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
                    arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
                else:
                    # For other quality indices (or inverted norm_dtw), higher values are better (upward arrow)
                    arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
                    arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
                    
                # Create gradient arrow using LineCollection
                n_segments = 100
                y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
                x_vals = np.full_like(y_vals, arrow_x)
                
                # Create line segments for gradient effect - exclude the last segment to leave space for arrowhead
                points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create colors for each segment - full spectrum of colormap
                colors_gradient = [plot_info['cmap'](i / (n_segments - 1)) for i in range(n_segments - 1)]
                
                # Create the gradient line
                lc = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
                ax.add_collection(lc)
                
                # Add arrowhead at the end
                arrow_color = plot_info['cmap'](1.0)
                arrowhead_start_y = y_vals[-2]
                arrowhead_end_y = arrow_y_end
                
                # Add just the arrowhead
                ax.annotate('', 
                           xy=(arrow_x, arrowhead_end_y), 
                           xytext=(arrow_x, arrowhead_start_y),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                           zorder=5)
                
                # Add text next to the arrow
                text_x = arrow_x + 0.1  # Slightly to the right of arrow
                text_y = arrow_y_center
                
                ax.text(text_x, text_y, 'Better Correlation Quality',
                       fontsize=8, ha='left', va='center',
                       rotation=90, color='black',
                       zorder=4)
                
                # Save frame
                frame_file = os.path.join(temp_dir, f'frame_{frame_idx + 1:03d}.png')
                plt.tight_layout()
                plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                frame_files.append(frame_file)
                
                # Update progress bar
                pbar.update(1)
        
        # Create gif from frames
        if frame_files:
            # Skip verbose frame combining message
            
            images = []
            target_size = None
            
            for frame_file in frame_files:
                img = imageio.imread(frame_file)
                if target_size is None:
                    target_size = img.shape[:2]
                elif img.shape[:2] != target_size:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                    img = np.array(pil_img)
                images.append(img)
            
            # Add frames to make final frame stay for 2 seconds
            if images:
                final_frame = images[-1]
                fps = 10  # imageio default fps
                frames_for_2_seconds = fps * 2  # 2 seconds = fps * 2 frames
                for _ in range(frames_for_2_seconds):
                    images.append(final_frame)
            
            # Create directory if needed
            output_dir = os.path.dirname(gif_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            imageio.mimsave(gif_filename, images, fps=fps, loop=3)
            
            if mute_mode:
                print(f"✓ T-statistics gif saved as: {gif_filename}")
            else:
                print(f"✓ T-statistics gif saved as: {gif_filename}")
                
            # Note: GIF will be displayed at the end of the main function
        else:
            # Skip verbose error message
            pass
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _create_cohens_d_gif(plot_info, gif_filename, mute_mode, max_frames=50, invert_norm_dtw=True):
    """
    Create Cohen's d gif using plot info from plot_cohens_d_vs_constraints.
    """
    
    unique_constraints = plot_info['unique_constraints']
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    try:
        # Frame 0: Empty Cohen's d plot (match static plot exactly)
        fig, ax = plt.subplots(figsize=(9.5, 5))
        
        # Add null hypothesis line (d=0)
        ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        
        # Set fixed axis ranges for all frames (use actual limits from static plot)
        if 'actual_plot_limits' in plot_info:
            ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
            ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
        else:
            ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
            ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
        ax.set_xticks(range(0, int(plot_info['plot_limits']['x_max']) + 1))
        
        # Format plot (same as static plot)
        ax.set_xlabel(f'Number of {plot_info["CORE_B"]} Age Constraints')
        ax.set_ylabel("Cohen's d")
        
        # Get display name from plot_info or create it
        quality_index = plot_info['quality_index']
        display_name = plot_info.get('display_name', quality_index)
        if not display_name or display_name == quality_index:
            if quality_index == 'corr_coef':
                display_name = "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                display_name = "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                display_name = "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                display_name = "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                display_name = quality_index
        
        ax.set_title(f"{display_name}\n{plot_info['CORE_A']} vs {plot_info['CORE_B']}")
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add inline text label for synthetic data line with white background box
        x_max = int(plot_info['plot_limits']['x_max'])
        ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
               color='dimgray', style='italic', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Add vertical colorbar for improvement/deterioration on right side
        sm = plt.cm.ScalarMappable(cmap=plot_info['cmap'], norm=plt.Normalize(vmin=-plot_info['max_abs_score'], vmax=plot_info['max_abs_score']))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.05)
        # Set bold label on the left side of the colorbar
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel("Change in Cohen's d", fontweight='bold', labelpad=5, rotation=90)
        cbar.set_ticks([-plot_info['max_abs_score'], 0, plot_info['max_abs_score']])
        cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
        
        # Add "Better Correlation Quality" arrow (same as static plot)
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        arrow_x = -0.35
        arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2
        
        # When invert_norm_dtw=True, norm_dtw metrics use upward arrow (like corr_coef)
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_downward_arrow:
            arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
        else:
            arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
            
        n_segments = 100
        y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
        x_vals = np.full_like(y_vals, arrow_x)
        
        points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        colors_gradient = [plot_info['cmap'](i / (n_segments - 1)) for i in range(n_segments - 1)]
        
        lc = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
        ax.add_collection(lc)
        
        arrow_color = plot_info['cmap'](1.0)
        arrowhead_start_y = y_vals[-2]
        arrowhead_end_y = arrow_y_end
        
        ax.annotate('', 
                   xy=(arrow_x, arrowhead_end_y), 
                   xytext=(arrow_x, arrowhead_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                   zorder=5)
        
        text_x = arrow_x + 0.1
        text_y = arrow_y_center
        
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='left', va='center',
               rotation=90, color='black',
               zorder=4)
        
        frame_file = os.path.join(temp_dir, f'frame_000.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        frame_files.append(frame_file)
        
        # Progressive frames: Add points by constraint level
        individual_points = plot_info.get('individual_points', [])
        individual_segments = plot_info.get('individual_segments', [])
        
        if not individual_points:
            return
        
        # Group points by constraint level
        points_by_constraint = {}
        for i, point in enumerate(individual_points):
            constraint_level = point['x']
            if constraint_level not in points_by_constraint:
                points_by_constraint[constraint_level] = []
            points_by_constraint[constraint_level].append((i, point))
        
        sorted_constraint_levels = sorted(points_by_constraint.keys())
        
        ordered_points = []
        for constraint_level in sorted_constraint_levels:
            for original_idx, point in points_by_constraint[constraint_level]:
                ordered_points.append((original_idx, point))
        
        total_points = len(ordered_points)
        target_animation_frames = max_frames
        
        if total_points <= target_animation_frames:
            points_per_frame = 1
            num_frames = total_points
        else:
            points_per_frame = math.ceil(total_points / target_animation_frames)
            num_frames = math.ceil(total_points / points_per_frame)
        
        with tqdm(total=num_frames, desc="  Creating Cohen's d GIF frames", disable=mute_mode) as pbar:
            for frame_idx in range(num_frames):
                points_to_show = min((frame_idx + 1) * points_per_frame, total_points)
                fig, ax = plt.subplots(figsize=(9.5, 5))
                
                ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                shown_points = set()
                for i in range(points_to_show):
                    original_idx, point = ordered_points[i]
                    ax.scatter(point['x'], point['y'], 
                             color=point['color'], 
                             edgecolor=point['edgecolor'],
                             linewidth=point['linewidth'], 
                             s=point['size'], 
                             zorder=point['zorder'])
                    shown_points.add((point['x'], point['y']))
                
                frame_segments = []
                frame_colors = []
                
                for segment in individual_segments:
                    seg_data = segment['segment']
                    start_x, start_y = seg_data[0]
                    end_x, end_y = seg_data[1]
                    
                    if (start_x, start_y) in shown_points and (end_x, end_y) in shown_points:
                        frame_segments.append(seg_data)
                        frame_colors.append(segment['color'])
                
                if frame_segments:
                    lc = LineCollection(frame_segments, colors=frame_colors, 
                                      alpha=individual_segments[0]['alpha'] if individual_segments else 0.8, 
                                      linewidths=individual_segments[0]['linewidth'] if individual_segments else 1, 
                                      zorder=1)
                    ax.add_collection(lc)
            
                if 'actual_plot_limits' in plot_info:
                    ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
                    ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
                else:
                    ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
                    ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
                ax.set_xticks(range(0, int(plot_info['plot_limits']['x_max']) + 1))
                
                ax.set_xlabel(f'Number of {plot_info["CORE_B"]} Age Constraints')
                ax.set_ylabel("Cohen's d")
                ax.set_title(f"{display_name}\n{plot_info['CORE_A']} vs {plot_info['CORE_B']}")
                ax.grid(True, alpha=0.3, zorder=0)
                
                # Add inline text label for synthetic data line with white background box
                ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
                       color='dimgray', style='italic', zorder=4,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
                
                # Add vertical colorbar for improvement/deterioration on right side
                sm = plt.cm.ScalarMappable(cmap=plot_info['cmap'], norm=plt.Normalize(vmin=-plot_info['max_abs_score'], vmax=plot_info['max_abs_score']))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.05)
                # Set bold label on the left side of the colorbar
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.set_ylabel("Change in Cohen's d", fontweight='bold', labelpad=5, rotation=90)
                cbar.set_ticks([-plot_info['max_abs_score'], 0, plot_info['max_abs_score']])
                cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
                
                # Add arrow
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                
                arrow_x = -0.35
                arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2
                
                # When invert_norm_dtw=True, norm_dtw metrics use upward arrow (like corr_coef)
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_downward_arrow:
                    arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
                    arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
                else:
                    arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
                    arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
                    
                n_segments = 100
                y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
                x_vals = np.full_like(y_vals, arrow_x)
                
                points_arr = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
                segments_arr = np.concatenate([points_arr[:-1], points_arr[1:]], axis=1)
                
                colors_gradient = [plot_info['cmap'](i / (n_segments - 1)) for i in range(n_segments - 1)]
                
                lc_arrow = LineCollection(segments_arr, colors=colors_gradient, linewidths=3, zorder=4)
                ax.add_collection(lc_arrow)
                
                arrow_color = plot_info['cmap'](1.0)
                arrowhead_start_y = y_vals[-2]
                arrowhead_end_y = arrow_y_end
                
                ax.annotate('', 
                           xy=(arrow_x, arrowhead_end_y), 
                           xytext=(arrow_x, arrowhead_start_y),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                           zorder=5)
                
                text_x = arrow_x + 0.1
                text_y = arrow_y_center
                
                ax.text(text_x, text_y, 'Better Correlation Quality',
                       fontsize=8, ha='left', va='center',
                       rotation=90, color='black',
                       zorder=4)
                
                frame_file = os.path.join(temp_dir, f'frame_{frame_idx + 1:03d}.png')
                plt.tight_layout()
                plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                frame_files.append(frame_file)
                
                pbar.update(1)
        
        # Create gif from frames
        if frame_files:
            images = []
            target_size = None
            
            for frame_file in frame_files:
                img = imageio.imread(frame_file)
                if target_size is None:
                    target_size = img.shape[:2]
                elif img.shape[:2] != target_size:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                    img = np.array(pil_img)
                images.append(img)
            
            if images:
                final_frame = images[-1]
                fps = 10
                frames_for_2_seconds = fps * 2
                for _ in range(frames_for_2_seconds):
                    images.append(final_frame)
            
            output_dir = os.path.dirname(gif_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            imageio.mimsave(gif_filename, images, fps=fps, loop=3)
            
            if mute_mode:
                print(f"✓ Cohen's d gif saved as: {gif_filename}")
            else:
                print(f"✓ Cohen's d gif saved as: {gif_filename}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _create_hedges_g_gif(plot_info, gif_filename, mute_mode, max_frames=50, invert_norm_dtw=True):
    """
    Create Hedges' g gif using plot info from plot_hedges_g_vs_constraints.
    """
    
    unique_constraints = plot_info['unique_constraints']
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    try:
        # Frame 0: Empty Hedges' g plot (match static plot exactly)
        fig, ax = plt.subplots(figsize=(9.5, 5))
        
        # Add null hypothesis line (g=0)
        ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        
        # Set fixed axis ranges for all frames (use actual limits from static plot)
        if 'actual_plot_limits' in plot_info:
            ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
            ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
        else:
            ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
            ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
        ax.set_xticks(range(0, int(plot_info['plot_limits']['x_max']) + 1))
        
        # Format plot (same as static plot)
        ax.set_xlabel(f'Number of {plot_info["CORE_B"]} Age Constraints')
        ax.set_ylabel("Hedges' g")
        ax.set_ylim(-4, 4)  # Set y-axis range for Hedges' g
        ax.set_yticks(np.arange(-4, 4.5, 1.0))  # Set y-axis ticks at every 1.0 interval
        
        # Get display name from plot_info or create it
        quality_index = plot_info['quality_index']
        display_name = plot_info.get('display_name', quality_index)
        if not display_name or display_name == quality_index:
            if quality_index == 'corr_coef':
                display_name = "Pearson's r"
            elif quality_index == 'corr_coef_sect':
                display_name = "Pearson's r (Correlated Section)"
            elif quality_index == 'norm_dtw':
                display_name = "1 - Normalized DTW Cost" if invert_norm_dtw else "Normalized DTW Cost"
            elif quality_index == 'norm_dtw_sect':
                display_name = "1 - Normalized DTW Cost (Correlated Section)" if invert_norm_dtw else "Normalized DTW Cost (Correlated Section)"
            else:
                display_name = quality_index
        
        ax.set_title(f"{display_name}\n{plot_info['CORE_A']} vs {plot_info['CORE_B']}")
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add inline text label for synthetic data line
        ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
               color='dimgray', style='italic', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Add vertical colorbar
        sm = plt.cm.ScalarMappable(cmap=plot_info['cmap'], norm=plt.Normalize(vmin=-plot_info['max_abs_score'], vmax=plot_info['max_abs_score']))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.14)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel("Change in Hedges' g", fontweight='bold', labelpad=5, rotation=90)
        cbar.set_ticks([-plot_info['max_abs_score'], 0, plot_info['max_abs_score']])
        cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
        
        # Add secondary y-axis for Probability of Superiority (as %)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        primary_ticks = ax.get_yticks()
        y_min, y_max = ax.get_ylim()
        valid_ticks = [t for t in primary_ticks if y_min <= t <= y_max]
        ps_values = [stats.norm.cdf(g / np.sqrt(2)) * 100 for g in valid_ticks]
        ax2.set_yticks(valid_ticks)
        ax2.set_yticklabels([f'{ps:.0f}%' for ps in ps_values])
        ax2.set_ylabel('Probability of Superiority', fontsize=10)
        ax2.spines['right'].set_position(('outward', 0))
        
        # Add arrow
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        arrow_x = -0.35
        arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2
        
        is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
        use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
        
        if use_downward_arrow:
            arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
        else:
            arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
            arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
            
        n_segments = 100
        y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
        x_vals = np.full_like(y_vals, arrow_x)
        
        points = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        colors_gradient = [plot_info['cmap'](i / (n_segments - 1)) for i in range(n_segments - 1)]
        
        lc = LineCollection(segments, colors=colors_gradient, linewidths=3, zorder=4)
        ax.add_collection(lc)
        
        arrow_color = plot_info['cmap'](1.0)
        arrowhead_start_y = y_vals[-2]
        arrowhead_end_y = arrow_y_end
        
        ax.annotate('', 
                   xy=(arrow_x, arrowhead_end_y), 
                   xytext=(arrow_x, arrowhead_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                   zorder=5)
        
        text_x = arrow_x + 0.1
        text_y = arrow_y_center
        
        ax.text(text_x, text_y, 'Better Correlation Quality',
               fontsize=8, ha='left', va='center',
               rotation=90, color='black',
               zorder=4)
        
        frame_file = os.path.join(temp_dir, f'frame_000.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        frame_files.append(frame_file)
        
        # Progressive frames: Add points by constraint level
        individual_points = plot_info.get('individual_points', [])
        individual_segments = plot_info.get('individual_segments', [])
        
        if not individual_points:
            return
        
        # Group points by constraint level
        points_by_constraint = {}
        for i, point in enumerate(individual_points):
            constraint_level = point['x']
            if constraint_level not in points_by_constraint:
                points_by_constraint[constraint_level] = []
            points_by_constraint[constraint_level].append((i, point))
        
        sorted_constraint_levels = sorted(points_by_constraint.keys())
        
        ordered_points = []
        for constraint_level in sorted_constraint_levels:
            for original_idx, point in points_by_constraint[constraint_level]:
                ordered_points.append((original_idx, point))
        
        total_points = len(ordered_points)
        target_animation_frames = max_frames
        
        if total_points <= target_animation_frames:
            points_per_frame = 1
            num_frames = total_points
        else:
            points_per_frame = math.ceil(total_points / target_animation_frames)
            num_frames = math.ceil(total_points / points_per_frame)
        
        with tqdm(total=num_frames, desc="  Creating Hedges' g GIF frames", disable=mute_mode) as pbar:
            for frame_idx in range(num_frames):
                points_to_show = min((frame_idx + 1) * points_per_frame, total_points)
                fig, ax = plt.subplots(figsize=(9.5, 5))
                
                ax.axhline(y=0, color='darkgray', linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                shown_points = set()
                for i in range(points_to_show):
                    original_idx, point = ordered_points[i]
                    ax.scatter(point['x'], point['y'], 
                             color=point['color'], 
                             edgecolor=point['edgecolor'],
                             linewidth=point['linewidth'], 
                             s=point['size'], 
                             zorder=point['zorder'])
                    shown_points.add((point['x'], point['y']))
                
                frame_segments = []
                frame_colors = []
                
                for segment in individual_segments:
                    seg_data = segment['segment']
                    start_x, start_y = seg_data[0]
                    end_x, end_y = seg_data[1]
                    
                    if (start_x, start_y) in shown_points and (end_x, end_y) in shown_points:
                        frame_segments.append(seg_data)
                        frame_colors.append(segment['color'])
                
                if frame_segments:
                    lc = LineCollection(frame_segments, colors=frame_colors, 
                                      alpha=individual_segments[0]['alpha'] if individual_segments else 0.8, 
                                      linewidths=individual_segments[0]['linewidth'] if individual_segments else 1, 
                                      zorder=1)
                    ax.add_collection(lc)
            
                if 'actual_plot_limits' in plot_info:
                    ax.set_xlim(plot_info['actual_plot_limits']['x_min'], plot_info['actual_plot_limits']['x_max'])
                    ax.set_ylim(plot_info['actual_plot_limits']['y_min'], plot_info['actual_plot_limits']['y_max'])
                else:
                    ax.set_xlim(plot_info['plot_limits']['x_min'], plot_info['plot_limits']['x_max'])
                    ax.set_ylim(plot_info['plot_limits']['y_min'], plot_info['plot_limits']['y_max'])
                ax.set_xticks(range(0, int(plot_info['plot_limits']['x_max']) + 1))
                
                ax.set_xlabel(f'Number of {plot_info["CORE_B"]} Age Constraints')
                ax.set_ylabel("Hedges' g")
                ax.set_ylim(-4, 4)  # Set y-axis range for Hedges' g
                ax.set_yticks(np.arange(-4, 4.5, 1.0))  # Set y-axis ticks at every 1.0 interval
                ax.set_title(f"{display_name}\n{plot_info['CORE_A']} vs {plot_info['CORE_B']}")
                ax.grid(True, alpha=0.3, zorder=0)
                
                ax.text(0, 0, 'Synthetic Data', fontsize=9, ha='left', va='center',
                       color='dimgray', style='italic', zorder=4,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
                
                sm = plt.cm.ScalarMappable(cmap=plot_info['cmap'], norm=plt.Normalize(vmin=-plot_info['max_abs_score'], vmax=plot_info['max_abs_score']))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, aspect=20, pad=0.14)
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.set_ylabel("Change in Hedges' g", fontweight='bold', labelpad=5, rotation=90)
                cbar.set_ticks([-plot_info['max_abs_score'], 0, plot_info['max_abs_score']])
                cbar.set_ticklabels(['Deterioration', 'No Change', 'Improvement'])
                
                # Add secondary y-axis for Probability of Superiority (as %)
                ax2 = ax.twinx()
                ax2.set_ylim(ax.get_ylim())
                primary_ticks = ax.get_yticks()
                y_min_frame, y_max_frame = ax.get_ylim()
                valid_ticks_frame = [t for t in primary_ticks if y_min_frame <= t <= y_max_frame]
                ps_values_frame = [stats.norm.cdf(g / np.sqrt(2)) * 100 for g in valid_ticks_frame]
                ax2.set_yticks(valid_ticks_frame)
                ax2.set_yticklabels([f'{ps:.0f}%' for ps in ps_values_frame])
                ax2.set_ylabel('Probability of Superiority', fontsize=10)
                ax2.spines['right'].set_position(('outward', 0))
                
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                
                arrow_x = -0.35
                arrow_y_center = (ax_ylim[0] + ax_ylim[1]) / 2
                
                is_norm_dtw_metric = quality_index in ['norm_dtw', 'norm_dtw_sect']
                use_downward_arrow = is_norm_dtw_metric and not invert_norm_dtw
                
                if use_downward_arrow:
                    arrow_y_start = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0])
                    arrow_y_end = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])  
                else:
                    arrow_y_start = arrow_y_center - 0.2 * (ax_ylim[1] - ax_ylim[0])
                    arrow_y_end = arrow_y_center + 0.2 * (ax_ylim[1] - ax_ylim[0]) 
                    
                n_segments = 100
                y_vals = np.linspace(arrow_y_start, arrow_y_end, n_segments + 1)
                x_vals = np.full_like(y_vals, arrow_x)
                
                points_arr = np.array([x_vals[:-1], y_vals[:-1]]).T.reshape(-1, 1, 2)
                segments_arr = np.concatenate([points_arr[:-1], points_arr[1:]], axis=1)
                
                colors_gradient = [plot_info['cmap'](i / (n_segments - 1)) for i in range(n_segments - 1)]
                
                lc_arrow = LineCollection(segments_arr, colors=colors_gradient, linewidths=3, zorder=4)
                ax.add_collection(lc_arrow)
                
                arrow_color = plot_info['cmap'](1.0)
                arrowhead_start_y = y_vals[-2]
                arrowhead_end_y = arrow_y_end
                
                ax.annotate('', 
                           xy=(arrow_x, arrowhead_end_y), 
                           xytext=(arrow_x, arrowhead_start_y),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4),
                           zorder=5)
                
                text_x = arrow_x + 0.1
                text_y = arrow_y_center
                
                ax.text(text_x, text_y, 'Better Correlation Quality',
                       fontsize=8, ha='left', va='center',
                       rotation=90, color='black',
                       zorder=4)
                
                frame_file = os.path.join(temp_dir, f'frame_{frame_idx + 1:03d}.png')
                plt.tight_layout()
                plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                frame_files.append(frame_file)
                
                pbar.update(1)
        
        # Create gif from frames
        if frame_files:
            images = []
            target_size = None
            
            for frame_file in frame_files:
                img = imageio.imread(frame_file)
                if target_size is None:
                    target_size = img.shape[:2]
                elif img.shape[:2] != target_size:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                    img = np.array(pil_img)
                images.append(img)
            
            if images:
                final_frame = images[-1]
                fps = 10
                frames_for_2_seconds = fps * 2
                for _ in range(frames_for_2_seconds):
                    images.append(final_frame)
            
            output_dir = os.path.dirname(gif_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            imageio.mimsave(gif_filename, images, fps=fps, loop=3)
            
            if mute_mode:
                print(f"✓ Hedges' g gif saved as: {gif_filename}")
            else:
                print(f"✓ Hedges' g gif saved as: {gif_filename}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)