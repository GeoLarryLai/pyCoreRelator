"""
Core plotting functions for DTW correlation visualization.

Included Functions:
- plot_segment_pair_correlation: Main function for plotting correlation between log segments
- plot_multilog_segment_pair_correlation: Specialized function for multilogs with images
- visualize_combined_segments: Combine and visualize multiple segment pairs
- plot_correlation_distribution: Plot quality metric distributions from CSV results

This module provides functions for visualizing dynamic time warping (DTW) correlations
between core log data, including support for multi-segment correlations, RGB/CT images,
age constraints, and quality indicators.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats
from IPython.display import display
import os
from ..utils.path_processing import combine_segment_dtw_results
from ..visualization.matrix_plots import plot_dtw_matrix_with_paths


def plot_segment_pair_correlation(log_a, log_b, md_a, md_b, 
                                  segment_pairs=None, dtw_results=None, segments_a=None, segments_b=None,
                                  depth_boundaries_a=None, depth_boundaries_b=None,
                                  wp=None, a_start=None, a_end=None, b_start=None, b_end=None,
                                  step=5, picked_depths_a=None, picked_depths_b=None, 
                                  quality_indicators=None, combined_quality=None,
                                  age_consideration=False, ages_a=None, ages_b=None,
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
                                  color_style_map=None):
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
    picked_depths_a, picked_depths_b : array-like, optional
        Arrays of picked depth indices to display as markers
    quality_indicators : dict, optional
        Quality indicators for single pair
    combined_quality : dict, optional
        Combined quality indicators for multiple pairs
    age_consideration : bool, default=False
        Whether age data should be displayed
    ages_a, ages_b : dict, optional
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
    
    # Create figure with appropriate size
    if max_img_rows > 0:
        fig_height = 20 + (max_img_rows * 3)
        fig = plt.figure(figsize=(6, fig_height))
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
        fig = plt.figure(figsize=(6, 20))
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
    
    # Prepare logs for coloring
    log_a_inv = (np.mean(log_a, axis=1) if is_multilog_a else log_a)
    log_b_inv = (np.mean(log_b, axis=1) if is_multilog_b else log_b)
    
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
    
    def add_picked_depths(picked_depths, md_array, ages=None, is_core_a=True):
        """Add picked depth markers and age annotations to the plot."""
        if picked_depths is None or len(picked_depths) == 0:
            return
        
        xmin = 0.0 if is_core_a else 0.67
        xmax = 0.33 if is_core_a else 1.0
        text_pos = 0.5 if is_core_a else 2.5
        
        for depth in picked_depths:
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
                color = plt.cm.tab10(idx % 10)
                
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
            if picked_depths_a is not None:
                add_picked_depths(picked_depths_a, md_a, ages_a if mark_ages else None, is_core_a=True)
            elif depth_boundaries_a is not None:
                converted_depths_a = [md_a[idx] for idx in depth_boundaries_a]
                add_picked_depths(converted_depths_a, md_a, ages_a if mark_ages else None, is_core_a=True)
            
            if picked_depths_b is not None:
                add_picked_depths(picked_depths_b, md_b, ages_b if mark_ages else None, is_core_a=False)
            elif depth_boundaries_b is not None:
                converted_depths_b = [md_b[idx] for idx in depth_boundaries_b]
                add_picked_depths(converted_depths_b, md_b, ages_b if mark_ages else None, is_core_a=False)
        else:
            add_picked_depths(picked_depths_a, md_a, ages_a if mark_ages else None, is_core_a=True)
            add_picked_depths(picked_depths_b, md_b, ages_b if mark_ages else None, is_core_a=False)
    
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
    
    # Add quality indicators
    if quality_indicators is not None or combined_quality is not None:
        qi = combined_quality if multi_segment_mode else quality_indicators
        if qi:
            quality_text = (
                "DTW Quality Indicators: \n"
                f"Normalized DTW Distance: {qi.get('norm_dtw', 0):.3f} (lower is better)\n "
                f"DTW Warping Ratio: {qi.get('dtw_ratio', 0):.3f} (lower is better)\n"
                f"Warping Deviation: variance {qi.get('variance_deviation', 0):.2f} (lower is better)\n"
                f"Diagonality: {qi.get('perc_diag', 0):.1f}% (higher is better)\n"
                f"Post-warping Corr Coeff (Pearson's r): {qi.get('corr_coef', 0):.3f} (higher is better)\n"
                f"Matching Function: {qi.get('match_min', 0):.3f}; mean {qi.get('match_mean', 0):.3f} (lower is better)\n"
                f"Age Overlap: {qi.get('perc_age_overlap', 0):.1f}% (higher is better)"
            )
            plt.figtext(0.97, 0.97, quality_text, 
                       fontsize=12, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))

    # Save figure if path provided
    if save_path:
        if save_path.startswith('outputs'):
            final_save_path = save_path
        else:
            os.makedirs('outputs', exist_ok=True)
            save_filename = os.path.basename(save_path)
            final_save_path = os.path.join('outputs', save_filename)
        
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        plt.savefig(final_save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_multilog_segment_pair_correlation(log_a, log_b, md_a, md_b, 
                                  wp, a_start, a_end, b_start, b_end,
                                  step=5, quality_indicators=None, 
                                  available_columns=None,
                                  rgb_img_a=None, ct_img_a=None,
                                  rgb_img_b=None, ct_img_b=None,
                                  picked_depths_a=None, picked_depths_b=None,
                                  picked_categories_a=None, picked_categories_b=None,
                                  category_colors=None,
                                  title=None):
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
    picked_depths_a, picked_depths_b : list, optional
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
    
    # Draw correlation intervals similar to Testing9 but using avg across all dimensions
    # for coloring
    log_a_inv = np.mean(log_a, axis=1) if log_a.ndim > 1 else 1 - log_a
    log_b_inv = np.mean(log_b, axis=1) if log_b.ndim > 1 else 1 - log_b
    
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
    if picked_depths_a is not None:
        for i, depth in enumerate(picked_depths_a):
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
    
    if picked_depths_b is not None:
        for i, depth in enumerate(picked_depths_b):
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
    
    # Add quality indicators if provided
    if quality_indicators is not None:
        # Calculate combined age overlap percentage if multiple segment pairs provided
        # if (segment_pairs is not None and dtw_results is not None and 
        #     segments_a is not None and segments_b is not None):
        #     # Calculate combined age overlap percentage from multiple segment pairs
        #     age_overlap_values = []
            
        #     for a_idx, b_idx in segment_pairs:
        #         if (a_idx, b_idx) in dtw_results:
        #             paths, _, qi_list = dtw_results[(a_idx, b_idx)]
        #             if qi_list and len(qi_list) > 0:
        #                 qi = qi_list[0]
        #                 age_overlap_values.append(qi['perc_age_overlap'])
            
        #     if age_overlap_values:
        #         combined_age_overlap = sum(age_overlap_values) / len(age_overlap_values)
        #         quality_indicators['perc_age_overlap'] = combined_age_overlap
        
        quality_text = (
            "DTW Quality Indicators: \n"
            f"Normalized DTW Distance: {quality_indicators.get('norm_dtw', 0):.3f} (lower is better)\n "
            f"DTW Warping Ratio: {quality_indicators.get('dtw_ratio', 0):.3f} (lower is better)\n"
            f"Warping Deviation: variance {quality_indicators.get('variance_deviation', 0):.2f} (lower is better)\n"
            f"Diagonality: {quality_indicators.get('perc_diag', 0):.1f}% (higher is better)\n"
            f"Post-warping Corr Coeff (Pearson's r): {quality_indicators.get('corr_coef', 0):.3f} (higher is better)\n"
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


def visualize_combined_segments(log_a, log_b, md_a, md_b, dtw_results, valid_dtw_pairs, 
                              segments_a, segments_b, depth_boundaries_a, depth_boundaries_b,
                              dtw_distance_matrix_full, segment_pairs_to_combine, 
                              correlation_save_path='CombinedSegmentPairs_DTW_correlation.png',
                              matrix_save_path='CombinedSegmentPairs_DTW_matrix.png',
                              color_interval_size=None,
                              visualize_pairs=True,
                              visualize_segment_labels=True,
                              mark_depths=True, mark_ages=False,
                              ages_a=None, ages_b=None,
                              all_constraint_depths_a=None, all_constraint_depths_b=None,
                              all_constraint_ages_a=None, all_constraint_ages_b=None,
                              all_constraint_pos_errors_a=None, all_constraint_pos_errors_b=None,
                              all_constraint_neg_errors_a=None, all_constraint_neg_errors_b=None,
                              # Age constraint visualization parameters (default None)
                              age_constraint_a_source_cores=None,
                              age_constraint_b_source_cores=None,
                              core_a_name=None,
                              core_b_name=None):
    """
    Combine selected segment pairs and visualize the results.
    
    Parameters:
    -----------
    log_a, log_b : array-like
        Log data arrays
    md_a, md_b : array-like
        Measured depth arrays
    dtw_results : dict
        Dictionary containing DTW results for each segment pair
    valid_dtw_pairs : set
        Set of valid segment pairs (a_idx, b_idx)
    segments_a, segments_b : list
        Segments in log_a and log_b
    depth_boundaries_a, depth_boundaries_b : list
        Depth boundaries for log_a and log_b
    dtw_distance_matrix_full : numpy.ndarray
        The full DTW distance matrix
    segment_pairs_to_combine : list
        List of tuples (a_idx, b_idx) for segment pairs to combine
    color_interval_size : int or None, default=None
        If provided, use this value as step_size for coloring
    visualize_pairs : bool, default=True
        Whether to visualize individual segment pairs
    mark_depths : bool, default=True
        Whether to mark depth boundaries
    mark_age_constraints: bool, default=False
        Whether to display age constraint information
    ages_a, ages_b: dictionaries with age data
        Each should contain 'depths', 'ages', 'pos_uncertainties', 'neg_uncertainties'
    correlation_save_path : str, default='CombinedSegmentPairs_DTW_correlation.png'
        Path to save the correlation figure
    matrix_save_path : str, default='CombinedSegmentPairs_DTW_matrix.png'
        Path to save the matrix figure
    age_constraint_a_source_cores : list or None, default=None
        List of source core names for each age constraint in core A.
        When provided, vertical constraint lines will be drawn in the DTW matrix.
    age_constraint_b_source_cores : list or None, default=None
        List of source core names for each age constraint in core B.
        When provided, horizontal constraint lines will be drawn in the DTW matrix.
    core_a_name, core_b_name : str or None, default=None
        Core names for determining same vs adjacent core coloring.
        
    Returns:
    --------
    tuple
        (combined_wp, combined_quality, correlation_fig, matrix_fig)
    """
    
    # Helper funcition: Create a global colormap function that uses the full log data
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

    # Combine segment pairs
    combined_wp, combined_quality = combine_segment_dtw_results(
        dtw_results, segment_pairs_to_combine, segments_a, segments_b,
        depth_boundaries_a, depth_boundaries_b, log_a, log_b
    )
    
    if combined_wp is None:
        print("Failed to combine segment pairs. No visualization created.")
        return None, None, None, None
    
    # Use color_interval_size if provided, otherwise use default calculation
    if color_interval_size is not None:
        step_size = int(color_interval_size)
    else:
        # Use smaller step size for efficiency
        step_size = max(1, min(5, len(combined_wp) // 5))
    
    # Create the global colormap
    global_color_func = create_global_colormap(log_a, log_b)

    # Ensure outputs directory exists and update save paths
    os.makedirs('outputs', exist_ok=True)

    # Handle save paths - ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    if correlation_save_path:
        if not correlation_save_path.startswith('outputs'):
            correlation_save_path = os.path.join('outputs', correlation_save_path)
        
        save_dir = os.path.dirname(correlation_save_path)
        if save_dir:  # Only create if directory path exists
            os.makedirs(save_dir, exist_ok=True)

    if matrix_save_path:
        if not matrix_save_path.startswith('outputs'):
            matrix_save_path = os.path.join('outputs', matrix_save_path)
        
        save_dir = os.path.dirname(matrix_save_path)
        if save_dir:  # Only create if directory path exists
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
        ages_a=ages_a,
        ages_b=ages_b,
        all_constraint_depths_a=all_constraint_depths_a,
        all_constraint_depths_b=all_constraint_depths_b,
        all_constraint_ages_a=all_constraint_ages_a,
        all_constraint_ages_b=all_constraint_ages_b,
        all_constraint_pos_errors_a=all_constraint_pos_errors_a,
        all_constraint_pos_errors_b=all_constraint_pos_errors_b,
        all_constraint_neg_errors_a=all_constraint_neg_errors_a,
        all_constraint_neg_errors_b=all_constraint_neg_errors_b
    )

    # Create DTW matrix plot
    if mark_ages:
        matrix_age_constraint_a_depths = all_constraint_depths_a
        matrix_age_constraint_a_ages = all_constraint_ages_a
        matrix_age_constraint_a_source_cores = age_constraint_a_source_cores
        matrix_age_constraint_b_depths = all_constraint_depths_b
        matrix_age_constraint_b_ages = all_constraint_ages_b
        matrix_age_constraint_b_source_cores = age_constraint_b_source_cores
    else:
        matrix_age_constraint_a_depths = None
        matrix_age_constraint_a_ages = None
        matrix_age_constraint_a_source_cores = None
        matrix_age_constraint_b_depths = None
        matrix_age_constraint_b_ages = None
        matrix_age_constraint_b_source_cores = None

    matrix_fig = plot_dtw_matrix_with_paths(
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
        # Age constraint parameters (conditionally set)
        age_constraint_a_depths=matrix_age_constraint_a_depths,
        age_constraint_a_ages=matrix_age_constraint_a_ages,
        age_constraint_a_source_cores=matrix_age_constraint_a_source_cores,
        age_constraint_b_depths=matrix_age_constraint_b_depths,
        age_constraint_b_ages=matrix_age_constraint_b_ages,
        age_constraint_b_source_cores=matrix_age_constraint_b_source_cores,
        md_a=md_a,
        md_b=md_b,
        core_a_name=core_a_name,
        core_b_name=core_b_name
)

    return combined_wp, combined_quality, correlation_fig, matrix_fig


def plot_correlation_distribution(csv_file, target_mapping_id=None, quality_index=None, save_png=True, png_filename=None, core_a_name=None, core_b_name=None, no_bins=None, pdf_method='KDE', kde_bandwidth=0.05):
    """
    UPDATED: Handle new CSV format with different column structure.
    Plot distribution of a specified quality index.
    
    Parameters:
    - csv_file: path to the CSV file containing mapping results
    - quality_index: required parameter specifying which quality index to plot
    - target_mapping_id: optional mapping ID to highlight in the plot
    - save_png: whether to save the plot as PNG (default: True)
    - png_filename: optional custom filename for saving PNG
    - no_bins: number of bins for the histogram (default: None for automatic estimation)
    - pdf_method: str, method for probability density function overlay: 'KDE' (default), 'skew-normal', or 'normal'
    - kde_bandwidth: float, bandwidth for KDE when pdf_method='KDE' (default: 0.05)
    """
    
    # Define quality index display names and descriptions
    quality_index_mapping = {
        'norm_dtw': 'Normalized DTW Distance (lower is better)',
        'dtw_ratio': 'DTW Warping Ratio (lower is better)', 
        'variance_deviation': 'Warping Deviation variance (lower is better)',
        'perc_diag': 'Diagonality % (higher is better)',
        'corr_coef': 'Post-warping Corr Coeff (Pearson\'s r) (higher is better)',
        'match_min': 'Matching Function min (lower is better)',
        'match_mean': 'Matching Function mean (lower is better)',
        'perc_age_overlap': 'Age Overlap % (higher is better)'
    }
    
    # Check if quality_index is provided
    if quality_index is None:
        print("Error: quality_index parameter is required")
        print("Available quality indices: perc_diag, norm_dtw, dtw_ratio, corr_coef, wrapping_deviation, mean_matching_function, perc_age_overlap")
        return
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        return
    
    # Check if quality_index column exists
    if quality_index not in df.columns:
        print(f"Error: '{quality_index}' column not found in the CSV file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Convert to numpy array and remove NaN values
    quality_values = np.array(df[quality_index])
    quality_values = quality_values[~np.isnan(quality_values)]
    
    # Automatically estimate number of bins if not provided
    if no_bins is None:
        # Use multiple methods to estimate optimal bin count and take the median
        # Sturges' rule
        sturges_bins = int(np.log2(len(quality_values)) + 1)
        
        # Square root rule
        sqrt_bins = int(np.sqrt(len(quality_values)))
        
        # Scott's rule based on standard deviation
        if len(quality_values) > 1:
            data_range = quality_values.max() - quality_values.min()
            scott_width = 3.5 * quality_values.std() / (len(quality_values) ** (1/3))
            scott_bins = int(data_range / scott_width) if scott_width > 0 else sqrt_bins
        else:
            scott_bins = sqrt_bins
        
        # Freedman-Diaconis rule based on IQR
        if len(quality_values) > 1:
            q75, q25 = np.percentile(quality_values, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                fd_width = 2 * iqr / (len(quality_values) ** (1/3))
                data_range = quality_values.max() - quality_values.min()
                fd_bins = int(data_range / fd_width)
            else:
                fd_bins = sqrt_bins
        else:
            fd_bins = sqrt_bins
        
        # Take median of the estimates and constrain to reasonable range
        estimated_bins = [sturges_bins, sqrt_bins, scott_bins, fd_bins]
        no_bins = int(np.median(estimated_bins))
        no_bins = max(10, min(no_bins, 100))  # Constrain between 10 and 100 bins
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Calculate total count for percentage
    total_count = len(quality_values)
    
    # Plot histogram of quality index as percentage
    hist, bins, _ = ax.hist(quality_values, bins=no_bins, alpha=0.7, color='skyblue', 
                            edgecolor='black', weights=np.ones(total_count)*100/total_count)
    
    # Initialize fit_params dictionary
    fit_params = {}
    
    # Add probability density function curve based on method
    if len(quality_values) > 1:  # Only plot PDF if we have multiple values
        x = np.linspace(quality_values.min(), quality_values.max(), 1000)
        
        # Calculate area under the histogram in percentage terms
        bin_width = bins[1] - bins[0]
        hist_area = np.sum(hist) * bin_width
        
        if pdf_method.upper() == 'KDE':
            # Use Kernel Density Estimation
            kde = stats.gaussian_kde(quality_values, bw_method=kde_bandwidth)
            y = kde(x) * hist_area
            fit_params['method'] = 'KDE'
            fit_params['bandwidth'] = kde_bandwidth
            
            ax.plot(x, y, 'r-', linewidth=2, alpha=0.8, 
                    label=f'KDE\n(bandwidth = {kde_bandwidth})\n(n = {total_count:,})')
        
        elif pdf_method.upper() == 'SKEW-NORMAL':
            # Fit skew-normal distribution
            try:
                # Fit skew-normal distribution using maximum likelihood estimation
                shape, location, scale = stats.skewnorm.fit(quality_values)
                
                # Generate PDF
                y = stats.skewnorm.pdf(x, shape, location, scale) * hist_area
                
                fit_params['method'] = 'skew-normal'
                fit_params['shape'] = shape
                fit_params['location'] = location
                fit_params['scale'] = scale
                fit_params['skewness'] = shape / np.sqrt(1 + shape**2) * np.sqrt(2/np.pi)
                
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8, 
                        label=f'Skew-Normal Fit\n(α = {shape:.3f})\n(μ = {location:.3f})\n(σ = {scale:.3f})\n(n = {total_count:,})')
                
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Warning: Skew-normal fitting failed: {e}")
                print("Falling back to normal distribution fit")
                pdf_method = 'normal'
        
        if pdf_method.upper() == 'NORMAL':
            # Fit normal distribution
            mean_val, std_val = stats.norm.fit(quality_values)
            
            # Generate PDF
            y = stats.norm.pdf(x, mean_val, std_val) * hist_area
            
            fit_params['method'] = 'normal'
            fit_params['mean'] = mean_val
            fit_params['std'] = std_val
            
            ax.plot(x, y, 'r-', linewidth=2, alpha=0.8, 
                    label=f'Normal Fit\n(μ = {mean_val:.3f})\n(σ = {std_val:.3f})\n(n = {total_count:,})')
    
    # Add vertical line for median
    median_value = np.median(quality_values)
    ax.axvline(median_value, color='green', linestyle='dashed', linewidth=2, 
               label=f'Median: {median_value:.3f}')
    
    # If target_mapping_id is provided, highlight it
    if target_mapping_id is not None:
        target_row = df[df['mapping_id'] == target_mapping_id]
        if not target_row.empty:
            target_value = target_row[quality_index].values[0]
            percentile = (quality_values < target_value).mean() * 100
            ax.axvline(target_value, color='purple', linestyle='solid', linewidth=2,
                      label=f'Mapping {target_mapping_id}: {target_value:.3f}\n({percentile:.3f}th percentile)')
    
    # Set x-axis based on quality index
    if quality_index == 'corr_coef':
        ax.set_xlim(0, 1.0)
    
    # Get the display name for the quality index
    quality_display_name = quality_index_mapping.get(quality_index, quality_index)
    
    # Add labels and title
    ax.set_xlabel(quality_display_name)
    ax.set_ylabel('Percentage (%)')
    title = f'Distribution of {quality_index}'
    if core_a_name and core_b_name:
        title += f'\n{core_a_name} vs {core_b_name}'
    plt.title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the figure if requested
    if save_png:
        if png_filename is None:
            png_filename = f'{quality_index}_distribution_{pdf_method.lower()}.png'
        
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # Use only filename, place in outputs directory
        save_filename = os.path.basename(png_filename)
        final_png_path = os.path.join('outputs', save_filename)
        
        plt.savefig(final_png_path, dpi=150, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Summary Statistics for {quality_display_name}:")
    print(f"Median: {median_value:.3f}")
    print(f"Min: {quality_values.min():.3f}")
    print(f"Max: {quality_values.max():.3f}")
    print(f"Standard Deviation: {quality_values.std():.3f}")
    print(f"Number of data points: {total_count}")
    
    # Print distribution fitting results
    if fit_params:
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
    if target_mapping_id is not None and not target_row.empty:
        target_value = target_row[quality_index].values[0]
        percentile = (quality_values < target_value).mean() * 100
        print(f"\nMapping ID {target_mapping_id} {quality_display_name}: {target_value:.3f}")
        print(f"Percentile: {percentile:.3f}%")
    
    return fig, ax, fit_params