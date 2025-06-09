"""
DTW Matrix Visualization Module

Included Functions:
- plot_dtw_matrix_with_paths: Main function for DTW matrix visualization with multiple modes

This module provides functions for visualizing Dynamic Time Warping (DTW) distance matrices
with various path plotting options, including segment paths, combined paths, and colored paths
based on quality metrics. It also supports age constraint visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import pandas as pd
from joblib import Parallel, delayed
import warnings
from tqdm.auto import tqdm
import os


def plot_dtw_matrix_with_paths(dtw_distance_matrix_full, 
                           mode=None, 
                           valid_dtw_pairs=None, 
                           segment_pairs=None, 
                           dtw_results=None,
                           combined_wp=None, 
                           sequential_mappings_csv=None,
                           segments_a=None, 
                           segments_b=None,
                           depth_boundaries_a=None, 
                           depth_boundaries_b=None, 
                           output_filename=None,
                           visualize_pairs=True,
                           visualize_segment_labels=False,
                           n_jobs=-1,
                           color_metric=None,
                           age_constraint_a_depths=None,
                           age_constraint_a_ages=None, 
                           age_constraint_a_source_cores=None,
                           age_constraint_b_depths=None,
                           age_constraint_b_ages=None,
                           age_constraint_b_source_cores=None,
                           md_a=None,
                           md_b=None,
                           core_a_name=None,
                           core_b_name=None):
    """
    Visualize DTW distance matrices with various path plotting options and age constraints.
    
    This function creates comprehensive visualizations of DTW distance matrices with support
    for different visualization modes, path overlays, and age constraint lines. Age constraint
    lines are automatically shown when the corresponding source_cores parameters are provided.
    
    Parameters
    ----------
    dtw_distance_matrix_full : numpy.ndarray
        The DTW distance matrix to visualize as a heatmap
    mode : str
        Visualization mode. Options: 'segment_paths', 'combined_path', 'all_paths_colored'
    valid_dtw_pairs : list of tuples, optional
        List of valid (segment_a_idx, segment_b_idx) pairs for 'segment_paths' mode
    segment_pairs : list of tuples, optional
        List of segment pairs for 'combined_path' mode
    dtw_results : dict, optional
        Dictionary containing DTW results with warping paths and quality metrics
    combined_wp : numpy.ndarray, optional
        Combined warping path array for 'combined_path' mode
    sequential_mappings_csv : str, optional
        Path to CSV file containing multiple paths for 'all_paths_colored' mode
    segments_a, segments_b : list, optional
        Lists of segment definitions for cores A and B
    depth_boundaries_a, depth_boundaries_b : list, optional
        Depth boundary arrays for segment visualization
    output_filename : str, optional
        Filename to save the plot. If provided, saves to outputs/ directory
    visualize_pairs : bool, default=True
        Whether to visualize individual segment pairs
    visualize_segment_labels : bool, default=False
        Whether to show segment index labels on the plot
    n_jobs : int, default=-1
        Number of parallel jobs for processing (used in 'all_paths_colored' mode)
    color_metric : str, optional
        Metric for coloring paths in 'all_paths_colored' mode. Options: 'corr_coef',
        'norm_dtw', 'dtw_ratio', 'perc_diag', 'variance_deviation', 'match_min',
        'match_mean', 'perc_age_overlap'. If None, uses mapping_id
    age_constraint_a_depths : list, optional
        List of constraint depths for core A
    age_constraint_a_ages : list, optional
        List of constraint ages for core A
    age_constraint_a_source_cores : list, optional
        List of source core names for core A constraints. When provided,
        horizontal constraint lines will be drawn
    age_constraint_b_depths : list, optional
        List of constraint depths for core B
    age_constraint_b_ages : list, optional
        List of constraint ages for core B
    age_constraint_b_source_cores : list, optional
        List of source core names for core B constraints. When provided,
        vertical constraint lines will be drawn
    md_a, md_b : array-like, optional
        Measured depth arrays for cores A and B (needed for age constraint positioning)
    core_a_name, core_b_name : str, optional
        Core names for constraint line coloring and axis labels
        
    Returns
    -------
    str or None
        Path to saved figure if output_filename provided, None otherwise
        
    Examples
    --------
    Basic segment paths visualization:
    
    >>> import numpy as np
    >>> dtw_matrix = np.random.rand(100, 100)
    >>> valid_pairs = [(0, 0), (1, 1), (2, 2)]
    >>> dtw_results = {(0, 0): ([np.array([[0, 0], [1, 1], [2, 2]])], [], [{}])}
    >>> segments_a = [(0, 10), (10, 20), (20, 30)]
    >>> segments_b = [(0, 15), (15, 30), (30, 45)]
    >>> depth_boundaries_a = [0, 10, 20, 30]
    >>> depth_boundaries_b = [0, 15, 30, 45]
    >>> 
    >>> plot_dtw_matrix_with_paths(
    ...     dtw_matrix,
    ...     mode='segment_paths',
    ...     valid_dtw_pairs=valid_pairs,
    ...     dtw_results=dtw_results,
    ...     segments_a=segments_a,
    ...     segments_b=segments_b,
    ...     depth_boundaries_a=depth_boundaries_a,
    ...     depth_boundaries_b=depth_boundaries_b
    ... )
    
    Visualization with age constraints:
    
    >>> plot_dtw_matrix_with_paths(
    ...     dtw_matrix,
    ...     mode='combined_path',
    ...     combined_wp=np.array([[0, 0], [50, 50], [99, 99]]),
    ...     visualize_pairs=False,
    ...     age_constraint_a_depths=[25, 75],
    ...     age_constraint_a_ages=[1000, 2000],
    ...     age_constraint_a_source_cores=['CoreA', 'CoreB'],
    ...     md_a=np.linspace(0, 100, 100),
    ...     core_a_name='CoreA'
    ... )
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    import pandas as pd
    import ast
    from joblib import Parallel, delayed
    import warnings
    from tqdm.auto import tqdm

    def parse_compact_warping_path(compact_wp_str):
        """
        Parse compact warping path string format back to numpy array.
        
        Parameters
        ----------
        compact_wp_str : str
            Compact warping path in format "1,2;3,4;5,6"
            
        Returns
        -------
        numpy.ndarray
            Array of warping path coordinates
        """
        if not compact_wp_str or compact_wp_str == "":
            return np.array([])
        pairs = [list(map(int, pair.split(','))) for pair in compact_wp_str.split(';')]
        return np.array(pairs)

    def find_nearest_index(depth_array, depth_value):
        """
        Find the index in depth_array closest to the given depth_value.
        
        Parameters
        ----------
        depth_array : array-like
            Array of depth values
        depth_value : float
            Target depth value
            
        Returns
        -------
        int
            Index of closest depth value
        """
        return np.abs(np.array(depth_array) - depth_value).argmin()

    def add_age_constraint_lines(ax, constraint_depths, constraint_ages, constraint_source_cores, 
                                md_array, core_name, orientation='horizontal'):
        """
        Add age constraint lines to the DTW matrix plot.
        
        Lines are colored based on whether they come from the same core (red) or
        adjacent cores (indigo). Age labels are optionally displayed.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to draw on
        constraint_depths : list
            List of constraint depths
        constraint_ages : list
            List of constraint ages
        constraint_source_cores : list
            List of source core names for each constraint
        md_array : array-like
            Depth array to find nearest indices in the log data
        core_name : str or None
            Name of the current core for comparison
        orientation : str
            'horizontal' for core A (y-axis lines), 'vertical' for core B (x-axis lines)
        """
        if (constraint_depths is None or len(constraint_depths) == 0 or 
            constraint_source_cores is None or len(constraint_source_cores) == 0 or
            md_array is None or len(md_array) == 0):
            return
            
        # Convert inputs to numpy arrays for consistent handling
        if isinstance(constraint_depths, pd.Series):
            constraint_depths = constraint_depths.values
        else:
            constraint_depths = np.array(constraint_depths)
            
        constraint_ages = np.array(constraint_ages) if constraint_ages is not None else None
        
        for i, constraint_depth in enumerate(constraint_depths):
            # Find the matrix index for the constraint depth
            matrix_index = find_nearest_index(md_array, constraint_depth)
            
            # Determine line color based on source core
            if core_name is not None and i < len(constraint_source_cores):
                source_core = constraint_source_cores[i]
                if source_core in core_name:
                    line_color = 'red'  # Same core
                    line_alpha = 0.6
                else:
                    line_color = 'indigo'  # Adjacent core  
                    line_alpha = 0.6
            else:
                line_color = 'red'  
                line_alpha = 0.6
            
            # Draw the constraint line
            if orientation == 'horizontal':
                ax.axhline(y=matrix_index, color=line_color, linestyle='--', 
                          linewidth=1.5, alpha=line_alpha)
                
                # Add age label
                if constraint_ages is not None and i < len(constraint_ages):
                    age_label = f"{constraint_ages[i]:.0f}"
                    ax.text(ax.get_xlim()[1] * 0.95, matrix_index, age_label, 
                           rotation=0, ha='right', va='bottom', fontsize=8,
                           color=line_color, alpha=line_alpha,
                           bbox=dict(facecolor='white', alpha=0.7, pad=1))
                           
            else:  # vertical
                ax.axvline(x=matrix_index, color=line_color, linestyle='--', 
                          linewidth=1.5, alpha=line_alpha)
                
                # Add age label  
                if constraint_ages is not None and i < len(constraint_ages):
                    age_label = f"{constraint_ages[i]:.0f}"
                    ax.text(matrix_index, ax.get_ylim()[1] * 0.95, age_label, 
                           rotation=90, ha='right', va='top', fontsize=8, 
                           color=line_color, alpha=line_alpha,
                           bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # Validate parameters based on mode
    if mode == 'segment_paths':
        if valid_dtw_pairs is None or dtw_results is None or segments_a is None or segments_b is None or depth_boundaries_a is None or depth_boundaries_b is None:
            print("Error: For 'segment_paths' mode, the following parameters are required:")
            print("- valid_dtw_pairs: Set of valid segment pairs")
            print("- dtw_results: Dictionary of DTW results")
            print("- segments_a, segments_b: Lists of segments")
            print("- depth_boundaries_a, depth_boundaries_b: Depth boundaries")
            return None
    
    elif mode == 'combined_path':
        if ((segment_pairs is None or dtw_results is None or segments_a is None or segments_b is None or 
             depth_boundaries_a is None or depth_boundaries_b is None) and 
            (combined_wp is None and visualize_pairs == False)):
            print("Error: For 'combined_path' mode, the following parameters are required:")
            print("- If visualize_pairs=True: segment_pairs, dtw_results, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b")
            print("- If visualize_pairs=False: combined_wp is required")
            return None
    
    elif mode == 'all_paths_colored':
        if sequential_mappings_csv is None:
            print("Error: For 'all_paths_colored' mode, sequential_mappings_csv is required")
            return None
    
    elif mode == None:
        print(f"Error: Plotting 'mode' must be specified. Valid modes are 'segment_paths', 'combined_path', and 'all_paths_colored'")
        return None
    else:
        print(f"Error: Unknown mode '{mode}'. Valid modes are 'segment_paths', 'combined_path', and 'all_paths_colored'")
        return None
    
    # Create figure and plot DTW distance matrix heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    plt_max = np.percentile(dtw_distance_matrix_full, 95)
    im = ax.imshow(dtw_distance_matrix_full, aspect='auto', vmin=0, vmax=plt_max, 
                   cmap='gray_r', origin='lower')
    plt.colorbar(im, label='DTW distance')
    
    # Add age constraint lines after heatmap but before other plot elements
    if age_constraint_a_source_cores is not None:
        add_age_constraint_lines(ax, age_constraint_a_depths, age_constraint_a_ages, 
                               age_constraint_a_source_cores, md_a, core_a_name, 'horizontal')
    
    if age_constraint_b_source_cores is not None:
        add_age_constraint_lines(ax, age_constraint_b_depths, age_constraint_b_ages,
                               age_constraint_b_source_cores, md_b, core_b_name, 'vertical')
    
    # Set axis labels with core names when provided
    if core_b_name is not None:
        ax.set_xlabel(f'Index in {core_b_name}')
    else:
        ax.set_xlabel('Index in log_b')
        
    if core_a_name is not None:
        ax.set_ylabel(f'Index in {core_a_name}')
    else:
        ax.set_ylabel('Index in log_a')
    
    # Plot paths based on selected mode
    if mode == 'segment_paths':
        if visualize_pairs:
            # Draw all segment paths with unique colors
            for idx, (a_idx, b_idx) in enumerate(valid_dtw_pairs):
                paths, _, quality_metrics = dtw_results.get((a_idx, b_idx), ([], [], []))
                if not paths or len(paths) == 0:
                    continue
                
                # Get segment boundaries for labeling
                a_start = depth_boundaries_a[segments_a[a_idx][0]]
                a_end = depth_boundaries_a[segments_a[a_idx][1]]
                b_start = depth_boundaries_b[segments_b[b_idx][0]]
                b_end = depth_boundaries_b[segments_b[b_idx][1]]
                
                color = plt.cm.tab10(idx % 10)
                
                # Plot each path
                for wp_idx, wp in enumerate(paths):
                    ax.plot(wp[:, 1], wp[:, 0], color=color, linewidth=2, alpha=0.7)
                    
                # Add segment labels if requested
                if visualize_segment_labels:
                    ax.text(b_start + (b_end-b_start)/2, a_start + (a_end-a_start)/2, 
                        f"({a_idx+1},{b_idx+1})", ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7))
        else:
            # Draw all paths in red without labels
            for idx, (a_idx, b_idx) in enumerate(valid_dtw_pairs):
                paths, _, _ = dtw_results.get((a_idx, b_idx), ([], [], []))
                if not paths or len(paths) == 0:
                    continue
                
                for wp_idx, wp in enumerate(paths):
                    ax.plot(wp[:, 1], wp[:, 0], color='red', linewidth=2, alpha=0.7)
        
        ax.set_title('DTW Matrix with All Segment Paths')
    
    elif mode == 'combined_path':
        if visualize_pairs:
            # Highlight each segment pair with unique colors
            for idx, (a_idx, b_idx) in enumerate(segment_pairs):
                color = plt.cm.tab10(idx % 10)
                
                # Get segment boundaries
                a_start = depth_boundaries_a[segments_a[a_idx][0]]
                a_end = depth_boundaries_a[segments_a[a_idx][1]]
                b_start = depth_boundaries_b[segments_b[b_idx][0]]
                b_end = depth_boundaries_b[segments_b[b_idx][1]]
                
                # Get and filter warping path for this segment
                paths, _, _ = dtw_results.get((a_idx, b_idx), ([], [], []))
                if paths and len(paths) > 0:
                    wp_segment = paths[0]
                    if len(wp_segment) > 0:
                        mask = ((wp_segment[:, 0] >= a_start) & (wp_segment[:, 0] <= a_end) & 
                                (wp_segment[:, 1] >= b_start) & (wp_segment[:, 1] <= b_end))
                        wp_segment = wp_segment[mask]
                        
                        if len(wp_segment) > 0:
                            ax.plot(wp_segment[:, 1], wp_segment[:, 0], color=color, linewidth=2, alpha=0.7)
                
                if visualize_segment_labels:
                    ax.text(b_start + (b_end-b_start)/2, a_start + (a_end-a_start)/2, 
                            f"({a_idx+1},{b_idx+1})", ha='center', va='center', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title(f'DTW Matrix with Combined Paths', fontsize=14)
            
        else:
            # Add combined path in red
            if combined_wp is not None and len(combined_wp) > 0:
                ax.plot(combined_wp[:, 1], combined_wp[:, 0], 'r-', linewidth=2, label="DTW Path")
                ax.set_title('DTW Matrix with Combined Path')
    
    elif mode == 'all_paths_colored':
        # Read CSV and plot paths colored by specified metric
        try:
            df = pd.read_csv(sequential_mappings_csv)
            
            if 'combined_wp' not in df.columns:
                print("Error: 'combined_wp' column not found in CSV")
                return fig
            
            combined_wp_list = df['combined_wp'].tolist()
            length_list = df['length'].tolist()
            
            # Determine coloring metric and colorbar settings
            if color_metric is None:
                color_values = df['mapping_id'].tolist() if 'mapping_id' in df.columns else list(range(len(combined_wp_list)))
                colorbar_label = 'Mapping ID'
                colormap = 'tab10'
            else:
                if color_metric not in df.columns:
                    print(f"Error: '{color_metric}' column not found in CSV")
                    return fig
                
                color_values = df[color_metric].tolist()
                
                # Define colorbar labels for different metrics
                metric_labels = {
                    'corr_coef': 'Post-warping Corr Coeff (Pearson\'s r) (higher is better)',
                    'norm_dtw': 'Normalized DTW Distance (lower is better)', 
                    'dtw_ratio': 'DTW Warping Ratio (lower is better)',
                    'perc_diag': 'Diagonality % (higher is better)',
                    'variance_deviation': 'Warping Deviation variance (lower is better)',
                    'match_min': 'Matching Function min (lower is better)',
                    'match_mean': 'Matching Function mean (lower is better)',
                    'perc_age_overlap': 'Age Overlap % (higher is better)'
                }
                
                colorbar_label = metric_labels.get(color_metric, color_metric)
                colormap = 'viridis'
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return fig
        
        # Filter valid paths and sort by color values
        valid_indices = [i for i, length in enumerate(length_list) if length > 0]
        valid_wp_list = [combined_wp_list[i] for i in valid_indices]
        valid_color_values = [color_values[i] for i in valid_indices]
        
        # Sort paths by metric values for better visualization layering
        if color_metric in ['norm_dtw', 'dtw_ratio', 'variance_deviation', 'match_min', 'match_mean']:
            sorted_indices = sorted(range(len(valid_color_values)), key=lambda i: valid_color_values[i])
        else:
            sorted_indices = sorted(range(len(valid_color_values)), key=lambda i: valid_color_values[i])
        
        sorted_wp_list = [valid_wp_list[i] for i in sorted_indices]
        sorted_color_values = [valid_color_values[i] for i in sorted_indices]
        
        # Create colormap normalization
        norm = plt.Normalize(min(sorted_color_values), max(sorted_color_values))
        cmap = cm.ScalarMappable(norm=norm, cmap=colormap)
        
        def process_path(i, wp_str, color_value):
            """Process individual warping path for parallel execution."""
            try:
                wp = parse_compact_warping_path(wp_str)
                return wp, cmap.to_rgba(color_value)
            except Exception as e:
                warnings.warn(f"Error processing path {i}: {e}")
                return None, None
        
        # Process paths in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_path)(i, wp_str, color_value) 
            for i, (wp_str, color_value) in enumerate(zip(sorted_wp_list, sorted_color_values))
        )
        
        # Plot all processed paths
        for wp, color in tqdm(results, desc="Plotting paths"):
            if wp is not None and len(wp) > 0:
                ax.plot(wp[:, 1], wp[:, 0], color=color, alpha=0.7, linewidth=2)
                
        # Add colorbar legend in upper left corner
        rect_ax = fig.add_axes([0.06, 0.82, 0.28, 0.12])
        rect_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.9, 
                                    edgecolor='black', linewidth=1))
        rect_ax.axis('off')

        cax = fig.add_axes([0.075, 0.875, 0.25, 0.05])
        cbar = plt.colorbar(cmap, cax=cax, orientation='horizontal')
        cbar.set_label(colorbar_label)
        
        ax.set_title('DTW Distance Matrix with Correlation Paths')
    
    # Save figure if output filename provided
    if output_filename:
        if output_filename.startswith('outputs'):
            full_output_path = output_filename
        else:
            os.makedirs('outputs', exist_ok=True)
            save_filename = os.path.basename(output_filename)
            full_output_path = os.path.join('outputs', save_filename)
        
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(full_output_path, dpi=150, bbox_inches='tight')
        
        return full_output_path

    return None