"""
DTW matrix visualization functions
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
                           # Age constraint parameters (default None)
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
    Unified function for visualizing DTW distance matrices with various path plotting options.
    Age constraint lines are automatically shown when age_constraint_*_source_cores parameters are provided.
    
    Age Constraint Parameters:
    --------------------------
    age_constraint_a_depths : list or None
        List of mean depths for age constraints in core A
    age_constraint_a_ages : list or None
        List of ages for age constraints in core A
    age_constraint_a_source_cores : list or None
        List of source core names for each age constraint in core A.
        When provided, horizontal constraint lines will be drawn for core A.
    age_constraint_b_depths : list or None
        List of mean depths for age constraints in core B
    age_constraint_b_ages : list or None
        List of ages for age constraints in core B
    age_constraint_b_source_cores : list or None
        List of source core names for each age constraint in core B.
        When provided, vertical constraint lines will be drawn for core B.
    md_a, md_b : array-like or None
        Measured depth arrays for cores A and B (needed to find nearest indices)
    core_a_name, core_b_name : str or None
        Core names for determining same vs adjacent core coloring.
        If not provided, all constraint lines will use the same color.
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
        """Parse compact warping path "1,2;3,4" back to numpy array"""
        if not compact_wp_str or compact_wp_str == "":
            return np.array([])
        pairs = [list(map(int, pair.split(','))) for pair in compact_wp_str.split(';')]
        return np.array(pairs)

    # Helper function to find nearest index 
    def find_nearest_index(depth_array, depth_value):
        """Find the index in depth_array that has the closest depth value to the given depth_value."""
        return np.abs(np.array(depth_array) - depth_value).argmin()

    # Helper function to add age constraint lines
    def add_age_constraint_lines(ax, constraint_depths, constraint_ages, constraint_source_cores, 
                                md_array, core_name, orientation='horizontal'):
        """
        Add age constraint lines to the DTW matrix plot.
        
        Parameters:
        -----------
        ax : matplotlib axis
            The axis to draw on
        constraint_depths : list
            List of constraint depths
        constraint_ages : list  
            List of constraint ages
        constraint_source_cores : list
            List of source core names for each constraint
        md_array : array
            Depth array to find nearest indices IN THE LOG DATA
        core_name : str or None
            Name of the current core for comparison (can be None)
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
            # Find the index in the log data where depth is closest to constraint depth
            # This gives us the matrix row/column index for the DTW matrix
            matrix_index = find_nearest_index(md_array, constraint_depth)
            
            # Determine line color based on source core (only if core_name is provided)
            if core_name is not None and i < len(constraint_source_cores):
                source_core = constraint_source_cores[i]
                # Check if constraint is from same core or adjacent core
                if source_core in core_name:
                    line_color = 'red'  # Same core
                    line_alpha = 0.6
                else:
                    line_color = 'indigo'  # Adjacent core  
                    line_alpha = 0.6
            else:
                # Default color when core_name not provided
                line_color = 'red'  
                line_alpha = 0.6
            
            # Draw the constraint line at the matrix index position
            if orientation == 'horizontal':
                # Horizontal line for core A constraints (y-axis = core A indices)
                ax.axhline(y=matrix_index, color=line_color, linestyle='--', 
                          linewidth=1.5, alpha=line_alpha)
                
                # Optional: Add age label
                if constraint_ages is not None and i < len(constraint_ages):
                    age_label = f"{constraint_ages[i]:.0f}"
                    ax.text(ax.get_xlim()[1] * 0.95, matrix_index, age_label, 
                           rotation=0, ha='right', va='bottom', fontsize=8,
                           color=line_color, alpha=line_alpha,
                           bbox=dict(facecolor='white', alpha=0.7, pad=1))
                           
            else:  # vertical
                # Vertical line for core B constraints (x-axis = core B indices)
                ax.axvline(x=matrix_index, color=line_color, linestyle='--', 
                          linewidth=1.5, alpha=line_alpha)
                
                # Optional: Add age label  
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
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot the DTW distance matrix as a heatmap
    plt_max = np.percentile(dtw_distance_matrix_full, 95)
    im = ax.imshow(dtw_distance_matrix_full, aspect='auto', vmin=0, vmax=plt_max, 
                   cmap='gray_r', origin='lower')
    plt.colorbar(im, label='DTW distance')
    
    # Add age constraint lines AFTER the heatmap but BEFORE other plot elements
    # Only add lines if the corresponding source_cores parameter is provided
    
    # Add horizontal lines for core A age constraints (only if age_constraint_a_source_cores is provided)
    if age_constraint_a_source_cores is not None:
        add_age_constraint_lines(ax, age_constraint_a_depths, age_constraint_a_ages, 
                               age_constraint_a_source_cores, md_a, core_a_name, 'horizontal')
    
    # Add vertical lines for core B age constraints (only if age_constraint_b_source_cores is provided)
    if age_constraint_b_source_cores is not None:
        add_age_constraint_lines(ax, age_constraint_b_depths, age_constraint_b_ages,
                               age_constraint_b_source_cores, md_b, core_b_name, 'vertical')
    
    # Set common labels and invert y-axis
    # Update axis labels to include core names when provided
    if core_b_name is not None:
        ax.set_xlabel(f'Index in {core_b_name}')
    else:
        ax.set_xlabel('Index in log_b')
        
    if core_a_name is not None:
        ax.set_ylabel(f'Index in {core_a_name}')
    else:
        ax.set_ylabel('Index in log_a')
    
    # Plotting based on mode
    if mode == 'segment_paths':
        if visualize_pairs:
            # Draw all paths on the DTW matrix
            for idx, (a_idx, b_idx) in enumerate(valid_dtw_pairs):
                # Get paths for this segment pair
                paths, _, quality_metrics = dtw_results.get((a_idx, b_idx), ([], [], []))
                if not paths or len(paths) == 0:
                    continue
                
                # Get segment boundaries
                a_start = depth_boundaries_a[segments_a[a_idx][0]]
                a_end = depth_boundaries_a[segments_a[a_idx][1]]
                b_start = depth_boundaries_b[segments_b[b_idx][0]]
                b_end = depth_boundaries_b[segments_b[b_idx][1]]
                
                # Get diagonality for this segment if available
                diagonality = quality_metrics[0].get('perc_diag', 0) if quality_metrics and len(quality_metrics) > 0 else 0
                
                # Use a color from the tab10 colormap based on the index
                color = plt.cm.tab10(idx % 10)
                
                # Plot each path and segment boundaries
                for wp_idx, wp in enumerate(paths):
                    ax.plot(wp[:, 1], wp[:, 0], color=color, linewidth=2, alpha=0.7)
                    
                # Add segment label with only indices
                if visualize_segment_labels:
                    ax.text(b_start + (b_end-b_start)/2, a_start + (a_end-a_start)/2, 
                        f"({a_idx+1},{b_idx+1})", ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7))
        else:
            # Draw all paths on the DTW matrix in red without labels
            for idx, (a_idx, b_idx) in enumerate(valid_dtw_pairs):
                # Get paths for this segment pair
                paths, _, _ = dtw_results.get((a_idx, b_idx), ([], [], []))
                if not paths or len(paths) == 0:
                    continue
                
                # Plot each path in red
                for wp_idx, wp in enumerate(paths):
                    ax.plot(wp[:, 1], wp[:, 0], color='red', linewidth=2, alpha=0.7)
        
        ax.set_title('DTW Matrix with All Segment Paths')
    
    elif mode == 'combined_path':
        if visualize_pairs:
            # Highlight each segment pair with a unique color
            for idx, (a_idx, b_idx) in enumerate(segment_pairs):
                # Use a color based on the index
                color = plt.cm.tab10(idx % 10)
                
                # Get segment boundaries
                a_start = depth_boundaries_a[segments_a[a_idx][0]]
                a_end = depth_boundaries_a[segments_a[a_idx][1]]
                b_start = depth_boundaries_b[segments_b[b_idx][0]]
                b_end = depth_boundaries_b[segments_b[b_idx][1]]
                
                # Get warping path for this segment pair
                paths, _, _ = dtw_results.get((a_idx, b_idx), ([], [], []))
                if paths and len(paths) > 0:
                    # Extract warping path points
                    wp_segment = paths[0]
                    if len(wp_segment) > 0:
                        # Filter to points in this segment
                        mask = ((wp_segment[:, 0] >= a_start) & (wp_segment[:, 0] <= a_end) & 
                                (wp_segment[:, 1] >= b_start) & (wp_segment[:, 1] <= b_end))
                        wp_segment = wp_segment[mask]
                        
                        # Draw this segment's warping path
                        if len(wp_segment) > 0:
                            ax.plot(wp_segment[:, 1], wp_segment[:, 0], color=color, linewidth=2, alpha=0.7)
                
                if visualize_segment_labels:
                    # Add segment label with only indices
                    ax.text(b_start + (b_end-b_start)/2, a_start + (a_end-a_start)/2, 
                            f"({a_idx+1},{b_idx+1})", ha='center', va='center', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title(f'DTW Matrix with Combined Paths', fontsize=14)
            
        else:
            # Add combined path with red color
            if combined_wp is not None and len(combined_wp) > 0:
                ax.plot(combined_wp[:, 1], combined_wp[:, 0], 'r-', linewidth=2, label="DTW Path")
                ax.set_title('DTW Matrix with Combined Path')
    
    elif mode == 'all_paths_colored':
        # UPDATED: Handle new CSV format
        try:
            # Read only the columns we need
            df = pd.read_csv(sequential_mappings_csv)
            
            # UPDATED: Handle new column names and format
            if 'combined_wp' not in df.columns:
                print("Error: 'combined_wp' column not found in CSV")
                return fig
            
            combined_wp_list = df['combined_wp'].tolist()
            length_list = df['length'].tolist()
            perc_diag_list = df['perc_diag'].tolist()
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return fig
        
        # Filter for valid paths (those with non-zero path length)
        valid_indices = [i for i, length in enumerate(length_list) if length > 0]
        valid_wp_list = [combined_wp_list[i] for i in valid_indices]
        valid_perc_diag_list = [perc_diag_list[i] for i in valid_indices]
        
        # Sort by percent diagonality for visualization (higher values on top)
        sorted_indices = sorted(range(len(valid_perc_diag_list)), key=lambda i: valid_perc_diag_list[i])
        sorted_wp_list = [valid_wp_list[i] for i in sorted_indices]
        sorted_perc_diag_list = [valid_perc_diag_list[i] for i in sorted_indices]
        
        # Create a colormap for the percent diagonality
        norm = plt.Normalize(min(sorted_perc_diag_list), max(sorted_perc_diag_list))
        cmap = cm.ScalarMappable(norm=norm, cmap='viridis')
        
        # Function to process a single path
        def process_path(i, wp_str, perc_diag):
            try:
                # UPDATED: Parse compact warping path format
                wp = parse_compact_warping_path(wp_str)
                
                # Return the path data and color
                return wp, cmap.to_rgba(perc_diag)
            except Exception as e:
                warnings.warn(f"Error processing path {i}: {e}")
                return None, None
        
        # Process paths in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_path)(i, wp_str, perc_diag) 
            for i, (wp_str, perc_diag) in enumerate(zip(sorted_wp_list, sorted_perc_diag_list))
        )
        
        # Plot the processed paths
        for wp, color in tqdm(results, desc="Plotting paths"):
            if wp is not None and len(wp) > 0:
                ax.plot(wp[:, 1], wp[:, 0], color=color, alpha=0.7, linewidth=2)
                
        # Add a colorbar for the percent diagonality as a legend box in the upper left corner
        # Create a white background rectangle
        rect_ax = fig.add_axes([0.06, 0.82, 0.28, 0.12])  # Much larger to include labels
        rect_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.9, 
                                    edgecolor='black', linewidth=1))
        rect_ax.axis('off')  # Hide the axes of this rectangle

        # Then add the colorbar on top of it
        cax = fig.add_axes([0.075, 0.875, 0.25, 0.05])  # [left, bottom, width, height]
        cbar = plt.colorbar(cmap, cax=cax, orientation='horizontal')
        cbar.set_label('Percent Diagonality (%)')
        
        ax.set_title('DTW Distance Matrix with Correlation Paths')
    
    # Save figure if filename provided
    if output_filename:
        # Use the path as-is if it starts with outputs/, otherwise add outputs/
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