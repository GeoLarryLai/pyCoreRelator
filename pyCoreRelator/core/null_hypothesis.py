"""
Null hypothesis testing functions for pyCoreRelator

This module provides functions for generating synthetic core data and running
null hypothesis tests for correlation analysis. It includes segment pool management,
synthetic log generation, and visualization tools.

Functions:
- load_segment_pool: Load segment pool data from turbidite database
- plot_segment_pool: Plot all segments from the pool in a grid layout
- modify_segment_pool: Remove unwanted segments from the pool data
- create_synthetic_log_with_depths: Create synthetic log using turbidite database approach
- create_and_plot_synthetic_core_pair: Generate synthetic core pair and optionally plot the results
"""

# Data manipulation and analysis
import os
import gc
import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import from other pyCoreRelator modules
from ..utils.data_loader import load_log_data
from ..core.dtw_analysis import run_comprehensive_dtw_analysis
from ..core.path_finding import find_complete_core_paths
from ..core.age_models import calculate_interpolated_ages
from ..visualization.plotting import plot_correlation_distribution


def load_segment_pool(core_names, core_log_paths, picked_depth_paths, log_columns, 
                     depth_column, column_alternatives, boundary_category=1, neglect_topbottom=True):
    """
    Load segment pool data from turbidite database.
    
    Parameters:
    - core_names: list of core names to process
    - core_log_paths: dict mapping core names to log file paths
    - picked_depth_paths: dict mapping core names to picked depth file paths
    - log_columns: list of log column names to load
    - depth_column: name of depth column
    - column_alternatives: dict of alternative column names
    - boundary_category: category number for turbidite boundaries (default: 1)
    - neglect_topbottom: if True, skip the first and last segments of each core (default: True)
    
    Returns:
    - segment_pool_cores_data: dict containing loaded core data
    - turb_logs: list of turbidite log segments
    - depth_logs: list of turbidite depth segments
    - target_dimensions: number of dimensions in the data
    """
    
    segment_pool_cores_data = {}
    turb_logs = []
    depth_logs = []
    
    print("Loading segment pool from available cores...")
    
    for core_name in core_names:
        print(f"Processing {core_name}...")
        
        try:
            # Load data for segment pool
            log_data, md_data, available_columns, _, _ = load_log_data(
                core_log_paths[core_name],
                {},  # No images needed
                log_columns,
                depth_column=depth_column,
                normalize=True,
                column_alternatives=column_alternatives
            )
            
            # Store core data
            segment_pool_cores_data[core_name] = {
                'log_data': log_data,
                'md_data': md_data,
                'available_columns': available_columns
            }
            
            # Load turbidite boundaries for this core
            picked_file = picked_depth_paths[core_name]
            try:
                picked_df = pd.read_csv(picked_file)
                # Filter for specified category boundaries only
                category_depths = picked_df[picked_df['category'] == boundary_category]['picked_depths_cm'].values
                category_depths = np.sort(category_depths)  # Ensure sorted order
                
                # Create turbidite segments (from boundary to boundary)
                # Determine range based on neglect_topbottom parameter
                if neglect_topbottom and len(category_depths) > 2:
                    # Skip first and last segments
                    start_range = 1
                    end_range = len(category_depths) - 2
                else:
                    # Include all segments
                    start_range = 0
                    end_range = len(category_depths) - 1
                
                for i in range(start_range, end_range):
                    start_depth = category_depths[i]
                    end_depth = category_depths[i + 1]
                    
                    # Find indices corresponding to these depths
                    start_idx = np.argmin(np.abs(md_data - start_depth))
                    end_idx = np.argmin(np.abs(md_data - end_depth))
                    
                    if end_idx > start_idx:
                        # Extract turbidite segment
                        turb_segment = log_data[start_idx:end_idx]
                        turb_depth = md_data[start_idx:end_idx] - md_data[start_idx]  # Relative depths
                        
                        turb_logs.append(turb_segment)
                        depth_logs.append(turb_depth)
                
            except Exception as e:
                print(f"Warning: Could not load turbidite boundaries for {core_name}: {e}")
            
            print(f"  Loaded: {len(log_data)} points, columns: {available_columns}")
            
        except Exception as e:
            print(f"Error loading {core_name}: {e}")
    
    # Set target dimensions based on segment pool
    target_dimensions = turb_logs[0].shape[1] if len(turb_logs) > 0 and turb_logs[0].ndim > 1 else 1
    
    print(f"Segment pool created with {len(turb_logs)} turbidites")
    print(f"Total cores processed: {len(segment_pool_cores_data)}")
    print(f"Target dimensions: {target_dimensions}")
    
    return segment_pool_cores_data, turb_logs, depth_logs, target_dimensions


def plot_segment_pool(segment_logs, segment_depths, log_column_names, n_cols=8, figsize_per_row=4, 
                     plot_segments=True, save_plot=False, plot_filename=None):
    """
    Plot all segments from the pool in a grid layout.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - log_column_names: list of column names for labeling
    - n_cols: number of columns in the subplot grid
    - figsize_per_row: height per row in the figure
    - plot_segments: whether to plot the segments (default True)
    - save_plot: whether to save the plot to file (default False)
    - plot_filename: filename for saving plot (optional)
    
    Returns:
    - fig, axes: matplotlib figure and axes objects
    """
    print(f"Plotting {len(segment_logs)} segments from the pool...")
    
    if not plot_segments:
        return None, None
    
    # Create subplot grid
    n_segments = len(segment_logs)
    n_rows = int(np.ceil(n_segments / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, figsize_per_row * n_rows))
    axes = axes.flatten() if n_segments > 1 else [axes]
    
    # Define colors for different log types
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    line_styles = ['-', '-', '-', '-.', '-.', '-.', ':', ':', ':']
    
    for i, (segment, depth) in enumerate(zip(segment_logs, segment_depths)):
        ax = axes[i]
        
        # Plot segment
        if segment.ndim > 1:
            # Multi-dimensional data - plot all columns
            n_log_types = segment.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_column_names[col_idx] if col_idx < len(log_column_names) else f'Log_{col_idx}'
                
                ax.plot(segment[:, col_idx], depth, 
                       color=color, linestyle=line_style, linewidth=1, 
                       label=col_name, alpha=0.8)
            
            # Set xlabel to show all log types
            if len(log_column_names) > 1:
                ax.set_xlabel(f'Multiple Logs: {", ".join(log_column_names[:n_log_types])} (normalized)')
            else:
                ax.set_xlabel(f'{log_column_names[0]} (normalized)')
                
            # Add legend if multiple log types
            if n_log_types > 1:
                ax.legend(fontsize=8, loc='best')
                
        else:
            # 1D data
            ax.plot(segment, depth, 'b-', linewidth=1)
            ax.set_xlabel(f'{log_column_names[0]} (normalized)')
        
        ax.set_ylabel('Relative Depth (cm)')
        ax.set_title(f'Segment {i+1}\n({len(segment)} pts, {depth[-1]:.1f} cm)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Depth increases downward
    
    # Hide unused subplots
    for i in range(n_segments, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Update title to reflect multiple log types if present
    if len(log_column_names) > 1:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments, {len(log_column_names)} log types)', 
                     y=1.02, fontsize=16)
    else:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments)', y=1.02, fontsize=16)
    
    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()
    
    return fig, axes


def modify_segment_pool(segment_logs, segment_depths, remove_list=None):
    """
    Remove unwanted segments from the pool data and return the modified pool.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - remove_list: list of 1-based segment numbers to remove (optional)
                  If None or empty, no segments are removed
    
    Returns:
    - modified_segment_logs: list of remaining log data arrays
    - modified_segment_depths: list of remaining depth arrays
    """
    
    # If remove_list is None or empty, return original data
    if not remove_list:
        print("No segments to remove. Returning original pool data.")
        return segment_logs.copy(), segment_depths.copy()
    
    # Convert remove_list to 0-based indices and validate
    remove_indices = []
    for item in remove_list:
        try:
            # Convert to int (handle both string and int inputs)
            segment_num = int(item)
            if 1 <= segment_num <= len(segment_logs):
                remove_indices.append(segment_num - 1)  # Convert to 0-based
            else:
                print(f"Warning: Segment number {segment_num} is out of range (1-{len(segment_logs)}). Skipping.")
        except (ValueError, TypeError):
            print(f"Warning: Invalid segment number '{item}'. Skipping.")
    
    # Remove duplicates and sort
    remove_indices = sorted(set(remove_indices))
    
    if not remove_indices:
        print("No valid segments to remove. Returning original pool data.")
        return segment_logs.copy(), segment_depths.copy()
    
    # Create modified lists by excluding specified indices
    modified_segment_logs = []
    modified_segment_depths = []
    
    for i, (segment_log, segment_depth) in enumerate(zip(segment_logs, segment_depths)):
        if i not in remove_indices:
            modified_segment_logs.append(segment_log)
            modified_segment_depths.append(segment_depth)
    
    # Print summary of changes
    removed_segments_1based = [idx + 1 for idx in remove_indices]
    print(f"Removed segments: {removed_segments_1based}")
    print(f"Original pool size: {len(segment_logs)} segments")
    print(f"Modified pool size: {len(modified_segment_logs)} segments")
    
    return modified_segment_logs, modified_segment_depths




def create_synthetic_log_with_depths(thickness, turb_logs, depth_logs, exclude_inds=None, repetition=False):
    """Create synthetic log using turbidite database approach with picked depths at turbidite bases.
    
    Parameters:
    - thickness: target thickness for the synthetic log
    - turb_logs: list of turbidite log segments
    - depth_logs: list of corresponding depth arrays
    - exclude_inds: indices to exclude from selection (optional)
    - repetition: if True, allow reusing turbidite segments; if False, each segment can only be used once (default: False)
    - plot_results: whether to display plots (default: True)
    - save_plot: whether to save plots (default: False)
    - plot_filename: filename for saving plots (optional)
    
    Returns:
    - tuple: (log, d, inds, valid_picked_depths)
    """
    # Determine target dimensions from the first available segment
    target_dimensions = turb_logs[0].shape[1] if len(turb_logs) > 0 and turb_logs[0].ndim > 1 else 1
    
    fake_log = np.array([]).reshape(0, target_dimensions) if target_dimensions > 1 else np.array([])
    md_log = np.array([])
    max_depth = 0
    inds = []
    picked_depths = []
    
    # Initialize available indices for selection
    if repetition:
        # If repetition is allowed, always use the full range
        available_inds = list(range(len(turb_logs)))
    else:
        # If no repetition, start with all indices and remove as we use them
        available_inds = list(range(len(turb_logs)))
        if exclude_inds is not None:
            available_inds = [ind for ind in available_inds if ind not in exclude_inds]
    
    # Add initial boundary
    picked_depths.append((0, 1))
    
    while max_depth <= thickness:
        # Check if we have available indices
        if not repetition and len(available_inds) == 0:
            print("Warning: No more unique turbidite segments available. Stopping log generation.")
            break
            
        if repetition:
            # Original behavior: select from full range, excluding only exclude_inds
            potential_inds = [ind for ind in range(len(turb_logs)) if exclude_inds is None or ind not in exclude_inds]
            if not potential_inds:
                print("Warning: No available turbidite segments after exclusions. Stopping log generation.")
                break
            ind = random.choices(potential_inds, k=1)[0]
        else:
            # New behavior: select from available indices and remove after use
            ind = random.choices(available_inds, k=1)[0]
            available_inds.remove(ind)  # Remove from available list to prevent reuse
            
        inds.append(ind)
        
        # Get turbidite segment from database
        turb_segment = turb_logs[ind]
        turb_depths = depth_logs[ind]
        
        # Ensure turbidite has proper dimensions
        if turb_segment.ndim == 1:
            turb_segment = turb_segment.reshape(-1, 1)
        
        # Ensure proper dimensions match target
        if turb_segment.shape[1] < target_dimensions:
            # Pad with noise if needed
            padding = np.random.normal(0, 0.1, (len(turb_segment), target_dimensions - turb_segment.shape[1]))
            turb_segment = np.hstack([turb_segment, padding])
        elif turb_segment.shape[1] > target_dimensions:
            # Truncate if needed
            turb_segment = turb_segment[:, :target_dimensions]
        
        # Append log data
        if target_dimensions > 1:
            if len(fake_log) == 0:
                fake_log = turb_segment.copy()
            else:
                fake_log = np.vstack((fake_log, turb_segment))
        else:
            fake_log = np.hstack((fake_log, turb_segment.flatten()))
        
        # Append depth data
        if len(md_log) == 0:
            md_log = np.hstack((md_log, 1 + turb_depths))
        else:
            md_log = np.hstack((md_log, 1 + md_log[-1] + turb_depths))
            
        max_depth = md_log[-1]
        
        # Add picked depth at the base of this turbidite (current max_depth)
        if max_depth <= thickness:
            picked_depths.append((max_depth, 1))
    
    # Truncate to target thickness
    valid_indices = md_log <= thickness
    if target_dimensions > 1:
        log = fake_log[valid_indices]
    else:
        log = fake_log[valid_indices]
    d = md_log[valid_indices]
    
    # Filter picked depths to only include those within the valid range
    valid_picked_depths = [(depth, category) for depth, category in picked_depths if depth <= thickness]
    
    # Ensure we have an end boundary
    if len(valid_picked_depths) == 0 or valid_picked_depths[-1][0] != d[-1]:
        valid_picked_depths.append((d[-1], 1))
    
    return log, d, inds, valid_picked_depths


def create_and_plot_synthetic_core_pair(core_a_length, core_b_length, turb_logs, depth_logs, 
                                       log_columns, repetition=False, plot_results=True, save_plot=False, plot_filename=None):
    """
    Generate synthetic core pair and optionally plot the results.
    
    Parameters:
    - core_a_length: target length for core A
    - core_b_length: target length for core B
    - turb_logs: list of turbidite log segments
    - depth_logs: list of corresponding depth arrays
    - log_columns: list of log column names for labeling
    - repetition: if True, allow reusing turbidite segments; if False, each segment can only be used once (default: False)
    - plot_results: whether to display the plot
    - save_plot: whether to save the plot to file
    - plot_filename: filename for saving plot (if save_plot=True)
    
    Returns:
    - tuple: (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a,
              synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)
    """
    
    # Generate synthetic logs for cores A and B
    print("Generating synthetic core pair...")

    synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a_tuples = create_synthetic_log_with_depths(
        core_a_length, turb_logs, depth_logs, exclude_inds=None, repetition=repetition
    )
    synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b_tuples = create_synthetic_log_with_depths(
        core_b_length, turb_logs, depth_logs, exclude_inds=None, repetition=repetition
    )

    # Extract just the depths from the tuples
    synthetic_picked_a = [depth for depth, category in synthetic_picked_a_tuples]
    synthetic_picked_b = [depth for depth, category in synthetic_picked_b_tuples]

    # Plot synthetic core pair if requested
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 9))
        
        # Define colors for different log types
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        line_styles = ['-', '-', '-', '-.', '-.', '-.', ':', ':', ':']

        # Plot synthetic core A
        if synthetic_log_a.ndim > 1:
            n_log_types = synthetic_log_a.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_columns[col_idx] if col_idx < len(log_columns) else f'Log_{col_idx}'
                
                ax1.plot(synthetic_log_a[:, col_idx], synthetic_md_a, 
                        color=color, linestyle=line_style, linewidth=1, 
                        label=col_name, alpha=0.8)
            
            # Add legend if multiple log types
            if n_log_types > 1:
                ax1.legend(fontsize=8, loc='upper right')
                
            # Set xlabel to show all log types
            if len(log_columns) > 1:
                ax1.set_xlabel(f'Multiple Logs (normalized)')
            else:
                ax1.set_xlabel(f'{log_columns[0]} (normalized)')
        else:
            ax1.plot(synthetic_log_a, synthetic_md_a, 'b-', linewidth=1)
            ax1.set_xlabel(f'{log_columns[0]} (normalized)')

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_a:
            ax1.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

        ax1.set_ylabel('Depth (cm)')
        ax1.set_title(f'Synthetic Core A\n({len(inds_a)} turbidites)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Plot synthetic core B
        if synthetic_log_b.ndim > 1:
            n_log_types = synthetic_log_b.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_columns[col_idx] if col_idx < len(log_columns) else f'Log_{col_idx}'
                
                ax2.plot(synthetic_log_b[:, col_idx], synthetic_md_b, 
                        color=color, linestyle=line_style, linewidth=1, 
                        label=col_name, alpha=0.8)
            
            # Add legend if multiple log types
            if n_log_types > 1:
                ax2.legend(fontsize=8, loc='upper right')
                
            # Set xlabel to show all log types
            if len(log_columns) > 1:
                ax2.set_xlabel(f'Multiple Logs (normalized)')
            else:
                ax2.set_xlabel(f'{log_columns[0]} (normalized)')
        else:
            ax2.plot(synthetic_log_b, synthetic_md_b, 'g-', linewidth=1)
            ax2.set_xlabel(f'{log_columns[0]} (normalized)')

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_b:
            ax2.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

        ax2.set_ylabel('Depth (cm)')
        ax2.set_title(f'Synthetic Core B\n({len(inds_b)} turbidites)')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plot and plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.show()

    print(f"Synthetic Core A: {len(synthetic_log_a)} points, {len(inds_a)} turbidites, {len(synthetic_picked_a)} boundaries")
    print(f"Synthetic Core B: {len(synthetic_log_b)} points, {len(inds_b)} turbidites, {len(synthetic_picked_b)} boundaries")
    print(f"Turbidite indices used in A: {[int(x) for x in inds_a[:10]]}..." if len(inds_a) > 10 else f"Turbidite indices used in A: {[int(x) for x in inds_a]}")
    print(f"Turbidite indices used in B: {[int(x) for x in inds_b[:10]]}..." if len(inds_b) > 10 else f"Turbidite indices used in B: {[int(x) for x in inds_b]}")
    
    return (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a,
            synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)


def generate_constraint_subsets(n_constraints):
    """Generate all possible subsets of constraints (2^n combinations)"""
    all_subsets = []
    for r in range(n_constraints + 1):  # 0 to n_constraints
        for subset in combinations(range(n_constraints), r):
            all_subsets.append(list(subset))
    return all_subsets

def _process_single_parameter_combination(
    idx, params, 
    log_a, log_b, md_a, md_b,
    all_depths_a_cat1, all_depths_b_cat1,
    pickeddepth_ages_a, pickeddepth_ages_b,
    age_data_a, age_data_b,
    target_quality_indices,
    output_csv_filenames,
    synthetic_csv_filenames,
    pca_for_dependent_dtw,
    test_age_constraint_removal
):
    """Process a single parameter combination (exact copy of original loop body)"""
    
    # Import here to avoid circular imports in workers
    from .dtw_analysis import run_comprehensive_dtw_analysis
    from .path_finding import find_complete_core_paths
    from ..visualization.plotting import plot_correlation_distribution
    
    # Generate a random suffix for temporary files in this iteration
    random_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=8))

    # Initialize temp_mapping_file here to ensure it's always defined
    temp_mapping_file = f'temp_mappings_{random_suffix}.pkl'

    # Extract parameters
    age_consideration = params['age_consideration']
    restricted_age_correlation = params['restricted_age_correlation']
    shortest_path_search = params['shortest_path_search']
    
    # Generate parameter labels
    if age_consideration:
        if restricted_age_correlation:
            age_label = 'restricted_age'
        else:
            age_label = 'loose_age'
    else:
        age_label = 'no_age'
    
    search_label = 'optimal' if shortest_path_search else 'random'
    combo_id = f"{age_label}_{search_label}"
    
    try:        
        # Run comprehensive DTW analysis with original constraints
        dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full = run_comprehensive_dtw_analysis(
            log_a, log_b, md_a, md_b,
            picked_depths_a=all_depths_a_cat1,
            picked_depths_b=all_depths_b_cat1,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=True,
            top_depth=0.0,
            exclude_deadend=True,
            mute_mode=True,
            age_consideration=age_consideration,
            ages_a=pickeddepth_ages_a if age_consideration else None,
            ages_b=pickeddepth_ages_b if age_consideration else None,
            restricted_age_correlation=restricted_age_correlation if age_consideration else False,
            all_constraint_ages_a=age_data_a['in_sequence_ages'] if age_consideration else None,
            all_constraint_ages_b=age_data_b['in_sequence_ages'] if age_consideration else None,
            all_constraint_depths_a=age_data_a['in_sequence_depths'] if age_consideration else None,
            all_constraint_depths_b=age_data_b['in_sequence_depths'] if age_consideration else None,
            all_constraint_pos_errors_a=age_data_a['in_sequence_pos_errors'] if age_consideration else None,
            all_constraint_pos_errors_b=age_data_b['in_sequence_pos_errors'] if age_consideration else None,
            all_constraint_neg_errors_a=age_data_a['in_sequence_neg_errors'] if age_consideration else None,
            all_constraint_neg_errors_b=age_data_b['in_sequence_neg_errors'] if age_consideration else None
        )
        
        # Find complete core paths
        if shortest_path_search:
            _ = find_complete_core_paths(
                valid_dtw_pairs, segments_a, segments_b, log_a, log_b,
                depth_boundaries_a, depth_boundaries_b, dtw_results, dtw_distance_matrix_full,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=1,  # Use n_jobs=1 to avoid nested parallelism
                shortest_path_search=True, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw
            )
        else:
            _ = find_complete_core_paths(
                valid_dtw_pairs, segments_a, segments_b, log_a, log_b,
                depth_boundaries_a, depth_boundaries_b, dtw_results, dtw_distance_matrix_full,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=1,  # Use n_jobs=1 to avoid nested parallelism
                shortest_path_search=False, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw
            )
        
        # Process quality indices and collect results
        results = {}
        for quality_index in target_quality_indices:
            
            # Extract bin size information from synthetic CSV if available
            targeted_binsize = None
            if synthetic_csv_filenames and quality_index in synthetic_csv_filenames:
                synthetic_csv_file = synthetic_csv_filenames[quality_index]
                if os.path.exists(synthetic_csv_file):
                    try:
                        synthetic_df = pd.read_csv(synthetic_csv_file)
                        if not synthetic_df.empty and 'bins' in synthetic_df.columns:
                            # Parse the first row's bins to get the bin structure
                            bins_str = synthetic_df.iloc[0]['bins']
                            if pd.notna(bins_str):
                                synthetic_bins = np.fromstring(bins_str.strip('[]'), sep=' ')
                                bin_width_synthetic = np.mean(np.diff(synthetic_bins))
                                targeted_binsize = (synthetic_bins, bin_width_synthetic)
                    except Exception:
                        pass  # Use default binning if extraction fails
            
            _, _, fit_params = plot_correlation_distribution(
                csv_file=f'outputs/{temp_mapping_file}',
                quality_index=quality_index,
                no_bins=30, save_png=False, pdf_method='normal',
                kde_bandwidth=0.05, mute_mode=True, targeted_binsize=targeted_binsize
            )
            
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                fit_params_copy['combination_id'] = combo_id
                fit_params_copy['age_consideration'] = age_consideration
                fit_params_copy['restricted_age_correlation'] = restricted_age_correlation
                fit_params_copy['shortest_path_search'] = shortest_path_search
                
                # Add constraint tracking columns
                fit_params_copy['core_a_constraints_count'] = len(age_data_a['in_sequence_ages']) if age_consideration else 0
                fit_params_copy['core_b_constraints_count'] = len(age_data_b['in_sequence_ages']) if age_consideration else 0
                fit_params_copy['constraint_scenario_description'] = 'all_original_constraints_remained' if age_consideration else 'no_age_constraints_used'
                
                results[quality_index] = fit_params_copy
        
        # Clean up
        if os.path.exists(f'outputs/{temp_mapping_file}'):
            os.remove(f'outputs/{temp_mapping_file}')
        
        del dtw_results, valid_dtw_pairs, segments_a, segments_b
        del depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full
        gc.collect()
        
        return True, combo_id, results
        
    except Exception as e:
        if os.path.exists(f'outputs/{temp_mapping_file}'):
            os.remove(f'outputs/{temp_mapping_file}')
        gc.collect()
        return False, combo_id, str(e)

def _process_single_constraint_scenario(
    param_idx, params, constraint_subset, in_sequence_indices,
    log_a, log_b, md_a, md_b,
    all_depths_a_cat1, all_depths_b_cat1,
    pickeddepth_ages_a, pickeddepth_ages_b,
    age_data_a, age_data_b,
    uncertainty_method,
    target_quality_indices,
    synthetic_csv_filenames,
    pca_for_dependent_dtw
):
    """Process a single constraint scenario (exact copy of original loop body)"""
    
    # Import here to avoid circular imports in workers
    from .dtw_analysis import run_comprehensive_dtw_analysis
    from .path_finding import find_complete_core_paths
    from ..visualization.plotting import plot_correlation_distribution
    from .age_models import calculate_interpolated_ages
    
    # Generate a process-safe random suffix for temporary files using numpy
    # This avoids conflicts between parallel workers
    import os
    process_id = os.getpid()
    np.random.seed(None)  # Use current time as seed (non-deterministic)
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    random_suffix = ''.join(np.random.choice(list(chars), size=8))
    
    # Initialize temp_mapping_file here to ensure it's always defined
    temp_mapping_file = f'temp_mappings_{random_suffix}.pkl'
    
    # Reset and reload parameters correctly for each iteration
    age_consideration = params['age_consideration']
    restricted_age_correlation = params['restricted_age_correlation']
    shortest_path_search = params['shortest_path_search']
    
    # Generate parameter labels
    if restricted_age_correlation:
        age_label = 'restricted_age'
    else:
        age_label = 'loose_age'
    
    search_label = 'optimal' if shortest_path_search else 'random'
    combo_id = f"{age_label}_{search_label}"
    
    try:
        # Map subset indices to original constraint indices (in-sequence only)
        original_indices = [in_sequence_indices[i] for i in constraint_subset]
        
        # Create modified age_data_b using original indices
        age_data_b_current = {
            'depths': [age_data_b['depths'][i] for i in original_indices],
            'ages': [age_data_b['ages'][i] for i in original_indices],
            'pos_errors': [age_data_b['pos_errors'][i] for i in original_indices],
            'neg_errors': [age_data_b['neg_errors'][i] for i in original_indices],
            'in_sequence_flags': [age_data_b['in_sequence_flags'][i] for i in original_indices],
            'in_sequence_depths': [age_data_b['depths'][i] for i in original_indices],
            'in_sequence_ages': [age_data_b['ages'][i] for i in original_indices],
            'in_sequence_pos_errors': [age_data_b['pos_errors'][i] for i in original_indices],
            'in_sequence_neg_errors': [age_data_b['neg_errors'][i] for i in original_indices],
            'core': [age_data_b['core'][i] for i in original_indices]
        }
        
        # Recalculate interpolated ages for core B with reduced constraints
        pickeddepth_ages_b_current = calculate_interpolated_ages(
            picked_depths=all_depths_b_cat1,
            age_constraints_depths=age_data_b_current['depths'],
            age_constraints_ages=age_data_b_current['ages'],
            age_constraints_pos_errors=age_data_b_current['pos_errors'],
            age_constraints_neg_errors=age_data_b_current['neg_errors'],
            age_constraints_in_sequence_flags=age_data_b_current['in_sequence_flags'],
            age_constraint_source_core=age_data_b_current['core'],
            top_bottom=True,
            top_depth=0.0,
            bottom_depth=md_b[-1],
            top_age=0,
            top_age_pos_error=75,
            top_age_neg_error=75,
            uncertainty_method=uncertainty_method,
            n_monte_carlo=10000,
            show_plot=False,
            export_csv=False,
            mute_mode=True
        )
        
        # Run DTW analysis with reduced constraints
        dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full = run_comprehensive_dtw_analysis(
            log_a, log_b, md_a, md_b,
            picked_depths_a=all_depths_a_cat1,
            picked_depths_b=all_depths_b_cat1,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=True,
            top_depth=0.0,
            exclude_deadend=True,
            mute_mode=True,
            age_consideration=age_consideration,
            ages_a=pickeddepth_ages_a,  # Use original ages for core A
            ages_b=pickeddepth_ages_b_current,  # Use modified ages for core B
            restricted_age_correlation=restricted_age_correlation,
            all_constraint_ages_a=age_data_a['in_sequence_ages'],  # Original constraints for core A
            all_constraint_ages_b=age_data_b_current['in_sequence_ages'],  # Modified constraints for core B
            all_constraint_depths_a=age_data_a['in_sequence_depths'],  # Original depths for core A
            all_constraint_depths_b=age_data_b_current['in_sequence_depths'],  # Modified depths for core B
            all_constraint_pos_errors_a=age_data_a['in_sequence_pos_errors'],  # Original errors for core A
            all_constraint_pos_errors_b=age_data_b_current['in_sequence_pos_errors'],  # Modified errors for core B
            all_constraint_neg_errors_a=age_data_a['in_sequence_neg_errors'],  # Original errors for core A
            all_constraint_neg_errors_b=age_data_b_current['in_sequence_neg_errors']  # Modified errors for core B
        )
        
        # Find paths with correct parameters
        if shortest_path_search:
            _ = find_complete_core_paths(
                valid_dtw_pairs, segments_a, segments_b, log_a, log_b,
                depth_boundaries_a, depth_boundaries_b, dtw_results, dtw_distance_matrix_full,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=1,  # Use n_jobs=1 to avoid nested parallelism
                shortest_path_search=True, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw
            )
        else:
            _ = find_complete_core_paths(
                valid_dtw_pairs, segments_a, segments_b, log_a, log_b,
                depth_boundaries_a, depth_boundaries_b, dtw_results, dtw_distance_matrix_full,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=1,  # Use n_jobs=1 to avoid nested parallelism
                shortest_path_search=False, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw
            )
        


        # Process quality indices
        results = {}
        for quality_index in target_quality_indices:
            
            # Extract bin size information from synthetic CSV if available
            targeted_binsize = None
            if synthetic_csv_filenames and quality_index in synthetic_csv_filenames:
                synthetic_csv_file = synthetic_csv_filenames[quality_index]
                if os.path.exists(synthetic_csv_file):
                    try:
                        synthetic_df = pd.read_csv(synthetic_csv_file)
                        if not synthetic_df.empty and 'bins' in synthetic_df.columns:
                            # Parse the first row's bins to get the bin structure
                            bins_str = synthetic_df.iloc[0]['bins']
                            if pd.notna(bins_str):
                                synthetic_bins = np.fromstring(bins_str.strip('[]'), sep=' ')
                                bin_width_synthetic = np.mean(np.diff(synthetic_bins))
                                targeted_binsize = (synthetic_bins, bin_width_synthetic)
                    except Exception:
                        pass  # Use default binning if extraction fails
            
            _, _, fit_params = plot_correlation_distribution(
                csv_file=f'outputs/{temp_mapping_file}',
                quality_index=quality_index,
                no_bins=30, save_png=False, pdf_method='normal',
                kde_bandwidth=0.05, mute_mode=True, targeted_binsize=targeted_binsize
            )
            
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                # Store the correct parameter values
                fit_params_copy['combination_id'] = combo_id
                fit_params_copy['age_consideration'] = age_consideration
                fit_params_copy['restricted_age_correlation'] = restricted_age_correlation
                fit_params_copy['shortest_path_search'] = shortest_path_search
                
                # Add constraint tracking with correct counts
                fit_params_copy['core_a_constraints_count'] = len(age_data_a['in_sequence_ages'])  # Original count for core A
                fit_params_copy['core_b_constraints_count'] = len(constraint_subset)  # Modified count for core B
                # Convert 0-based indices to 1-based for constraint description
                remaining_indices_1based = [i + 1 for i in sorted(constraint_subset)]
                fit_params_copy['constraint_scenario_description'] = f'constraints_{remaining_indices_1based}_remained'
                
                results[quality_index] = fit_params_copy
        
        # Clean up temporary files
        if os.path.exists(f'outputs/{temp_mapping_file}'):
            os.remove(f'outputs/{temp_mapping_file}')
        
        del dtw_results, valid_dtw_pairs, segments_a, segments_b
        del depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full
        del age_data_b_current, pickeddepth_ages_b_current
        gc.collect()
        
        scenario_id = f"{combo_id}_subset_{len(constraint_subset)}"
        return True, scenario_id, results
        
    except Exception as e:
        if os.path.exists(f'outputs/{temp_mapping_file}'):
            os.remove(f'outputs/{temp_mapping_file}')
        # Clean up variables in case of error
        if 'age_data_b_current' in locals():
            del age_data_b_current
        if 'pickeddepth_ages_b_current' in locals():
            del pickeddepth_ages_b_current
        gc.collect()
        return False, f"{combo_id}_subset_error", str(e)

def run_multi_parameter_analysis(
    # Core data inputs
    log_a, log_b, md_a, md_b,
    all_depths_a_cat1, all_depths_b_cat1,
    pickeddepth_ages_a, pickeddepth_ages_b,
    age_data_a, age_data_b,
    uncertainty_method,
    
    # Analysis parameters
    parameter_combinations,
    target_quality_indices,
    test_age_constraint_removal,
    
    # Core identifiers
    core_a_name, core_b_name,
    
    # Output configuration
    output_csv_filenames,  # Dict with quality_index as key and filename as value
    
    # Optional parameters
    synthetic_csv_filenames=None,  # Dict with quality_index as key and synthetic CSV filename as value
    pca_for_dependent_dtw=False,
    n_jobs=-1,  # Number of cores used in parallel processing (-1 uses all available cores)
    max_search_per_layer=None  # Max scenarios per constraint removal layer
):
    """
    Run comprehensive multi-parameter analysis for core correlation.
    
    Brief summary: This function performs DTW analysis across multiple parameter combinations
    and optionally tests age constraint removal scenarios, generating distribution fit parameters
    for various quality indices.
    
    Parameters:
    -----------
    log_a, log_b : array-like
        Log data for cores A and B
    md_a, md_b : array-like
        Measured depth arrays for cores A and B
    all_depths_a_cat1, all_depths_b_cat1 : array-like
        Picked depths of category 1 for cores A and B
    pickeddepth_ages_a, pickeddepth_ages_b : dict
        Age interpolation results for picked depths
    age_data_a, age_data_b : dict
        Age constraint data for cores A and B
    uncertainty_method : str
        Method for uncertainty calculation
    parameter_combinations : list of dict
        List of parameter combinations to test
    target_quality_indices : list
        Quality indices to analyze (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    test_age_constraint_removal : bool
        Whether to test age constraint removal scenarios
    core_a_name, core_b_name : str
        Names of cores A and B
    output_csv_filenames : dict
        Dictionary mapping quality_index to output CSV filename
    synthetic_csv_filenames : dict or None, default=None
        Dictionary mapping quality_index to synthetic CSV filename for consistent bin sizing
    pca_for_dependent_dtw : bool
        Whether to use PCA for dependent DTW
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all available cores.
        Set to 1 for sequential processing (useful for debugging).
    max_search_per_layer : int or None, default=None
        Maximum number of scenarios to process per constraint removal layer.
        If None, processes all scenarios. A layer represents combinations with
        the same number of remaining age constraints.
    
    Returns:
    --------
    None
        Results are saved to CSV files specified in output_csv_filenames
    """
    
    # Create directories for output files if needed
    for csv_filename in output_csv_filenames.values():
        output_dir = os.path.dirname(csv_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all quality indices
    print(f"Running {len(parameter_combinations)} parameter combinations for {len(target_quality_indices)} quality indices...")
    print(f"Using {n_jobs if n_jobs > 0 else 'all available'} CPU cores for parallel processing")

    # Reset variables at the beginning
    n_constraints_b = 0
    age_enabled_params = []
    total_additional_scenarios = 0

    if test_age_constraint_removal:
        n_constraints_b = len(age_data_b['in_sequence_ages'])
        age_enabled_params = [p for p in parameter_combinations if p['age_consideration']]
        constraint_scenarios_per_param = (2 ** n_constraints_b) - 1  # Exclude empty set
        total_additional_scenarios = len(age_enabled_params) * (constraint_scenarios_per_param - 1)  # Exclude original scenario
        
        print(f"Age constraint removal testing enabled:")
        print(f"- Core B has {n_constraints_b} age constraints")
        
        if max_search_per_layer is None:
            print(f"- Additional scenarios to process: {total_additional_scenarios}")
            # Warning for large number of scenarios
            if total_additional_scenarios > 5000:
                print(f"⚠️  WARNING: Processing {total_additional_scenarios} scenarios will take very long time and use large memory.")
                print(f"   Recommend setting max_search_per_layer to about 200-300 (but >= the number of constraints in core B: {n_constraints_b})")
                print(f"   to reduce the number of scenarios per constraint removal layer to process.")
                print(f"   When max_search_per_layer is set, scenarios are randomly sampled from each layer.")
                print(f"   This provides statistical approximation while maintaining computational feasibility.")
        else:
            # Calculate how many scenarios will actually be processed
            # This is an approximation since it depends on the distribution across layers
            print(f"- Additional {total_additional_scenarios} scenarios exist")
            print(f"- As max_search_per_layer is defined: {max_search_per_layer} scenarios are randomly sampled from each constraint removal layer")
            print(f"- ⚠️  WARNING: Due to random sampling, not every run will yield identical results")
            print(f"-     This just provides statistical approximation while maintaining computational feasibility")
            # Check if max_search_per_layer is too small
            if max_search_per_layer < n_constraints_b:
                print(f"- ⚠️  WARNING: max_search_per_layer ({max_search_per_layer}) is less than the number of constraints ({n_constraints_b})")
                print(f"-     Recommend setting max_search_per_layer >= {n_constraints_b} to ensure all age constraints are evaluated")

    # PHASE 1: Run original parameter combinations
    if test_age_constraint_removal:
        print("\n=== PHASE 1: Running original parameter combinations ===")
    else:
        print("\nRunning parameter combinations...")

    # Prepare data for parallel processing
    phase1_args = [
        (idx, params, 
         log_a, log_b, md_a, md_b,
         all_depths_a_cat1, all_depths_b_cat1,
         pickeddepth_ages_a, pickeddepth_ages_b,
         age_data_a, age_data_b,
         target_quality_indices,
         output_csv_filenames,
         synthetic_csv_filenames,
         pca_for_dependent_dtw,
         test_age_constraint_removal) 
        for idx, params in enumerate(parameter_combinations)
    ]
    
    # Run Phase 1 in parallel
    phase1_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_single_parameter_combination)(*args) 
        for args in tqdm(phase1_args, desc="Parameter combinations" if not test_age_constraint_removal else "Original parameter combinations")
    )
    
    # Process Phase 1 results and write to CSV
    for idx, (success, combo_id, results) in enumerate(phase1_results):
        if success:
            # Write results to CSV files
            for quality_index in target_quality_indices:
                if quality_index in results:
                    fit_params = results[quality_index]
                    master_csv_filename = output_csv_filenames[quality_index]
                    
                    df_single = pd.DataFrame([fit_params])
                    if idx == 0:
                        df_single.to_csv(master_csv_filename, mode='w', index=False, header=True)
                    else:
                        df_single.to_csv(master_csv_filename, mode='a', index=False, header=False)
                    del df_single
        else:
            print(f"✗ Error in {combo_id}: {results}")

    print("✓ All original parameter combinations processed" if test_age_constraint_removal else "✓ All parameter combinations processed")

    # PHASE 2: Run age constraint removal scenarios (if enabled)
    # Warning for large number of scenarios
    if test_age_constraint_removal:
        print("\n=== PHASE 2: Running age constraint removal scenarios ===")
        
        # Calculate additional scenarios
        n_constraints_b = len(age_data_b['in_sequence_ages'])
        age_enabled_params = [p for p in parameter_combinations if p['age_consideration']]
        constraint_scenarios_per_param = (2 ** n_constraints_b) - 1  # Exclude empty set
        total_additional_scenarios = len(age_enabled_params) * (constraint_scenarios_per_param - 1)  # Exclude original scenario
       
        print(f"- Core B has {n_constraints_b} age constraints")
        print(f"- Processing {total_additional_scenarios} additional constraint removal scenarios")
        
        # Get indices of only in-sequence constraints from the original data
        in_sequence_indices = [i for i, flag in enumerate(age_data_b['in_sequence_flags']) if flag == 1]
        n_constraints_b = len(in_sequence_indices)  # Count of in-sequence constraints only

        # Generate subsets from in-sequence constraint indices only
        all_subsets = generate_constraint_subsets(n_constraints_b)
        constraint_subsets = [subset for subset in all_subsets if 0 < len(subset) < n_constraints_b]
        
        # Apply max_search_per_layer limitation if specified
        if max_search_per_layer is not None:
            # Group constraint subsets by layer (number of remaining constraints)
            print(f"max_search_per_layer is defined: randomly sampling up to {max_search_per_layer} scenarios per layer of search")
            layers = {}
            for subset in constraint_subsets:
                layer_size = len(subset)
                if layer_size not in layers:
                    layers[layer_size] = []
                layers[layer_size].append(subset)
            
            # Sample from each layer if it exceeds max_search_per_layer
            limited_constraint_subsets = []
            for layer_size in sorted(layers.keys()):
                layer_subsets = layers[layer_size]
                if len(layer_subsets) > max_search_per_layer:
                    # Use numpy random to avoid conflicts and ensure no repeats within layer
                    layer_indices = np.arange(len(layer_subsets))
                    sampled_indices = np.random.choice(layer_indices, size=max_search_per_layer, replace=False)
                    sampled_subsets = [layer_subsets[i] for i in sampled_indices]
                    
                    limited_constraint_subsets.extend(sampled_subsets)
                    print(f"- Layer {layer_size} constraints: {len(layer_subsets)} scenarios → {max_search_per_layer} sampled")
                else:
                    limited_constraint_subsets.extend(layer_subsets)
                    print(f"- Layer {layer_size} constraints: {len(layer_subsets)} scenarios (all processed)")
            
            constraint_subsets = limited_constraint_subsets
        
        # Prepare data for Phase 2 parallel processing
        phase2_args = []
        for param_idx, params in enumerate(age_enabled_params):
            for constraint_subset in constraint_subsets:
                phase2_args.append((
                    param_idx, params, constraint_subset, in_sequence_indices,
                    log_a, log_b, md_a, md_b,
                    all_depths_a_cat1, all_depths_b_cat1,
                    pickeddepth_ages_a, pickeddepth_ages_b,
                    age_data_a, age_data_b,
                    uncertainty_method,
                    target_quality_indices,
                    synthetic_csv_filenames,
                    pca_for_dependent_dtw
                ))
        
        # Run Phase 2 in parallel
        phase2_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_single_constraint_scenario)(*args) 
            for args in tqdm(phase2_args, desc="Age constraint removal scenarios")
        )
        
        # Process Phase 2 results and append to CSV
        for success, scenario_id, results in phase2_results:
            if success:
                # Append results to CSV files
                for quality_index in target_quality_indices:
                    if quality_index in results:
                        fit_params = results[quality_index]
                        master_csv_filename = output_csv_filenames[quality_index]
                        
                        df_single = pd.DataFrame([fit_params])
                        df_single.to_csv(master_csv_filename, mode='a', index=False, header=False)
                        del df_single
            else:
                print(f"✗ Error in {scenario_id}: {results}")

        print("✓ Phase 2 completed: All age constraint removal scenarios processed")

    # Final summary and CSV sorting
    print(f"\n✓ All processing completed")
    
    # Sort CSV files to match original sequential order
    print("Sorting CSV files for consistent ordering...")
    for quality_index in target_quality_indices:
        filename = output_csv_filenames[quality_index]
        
        try:
            # Read the CSV file
            df = pd.read_csv(filename)
            
            # Check if we have enough rows and the required columns
            if len(df) >= 3 and 'core_b_constraints_count' in df.columns and 'constraint_scenario_description' in df.columns:
                # Separate header rows (first 2 rows) from data rows (3rd row onwards)
                header_rows = df.iloc[:2].copy()
                data_rows = df.iloc[2:].copy()
                
                # Sort data rows by core_b_constraints_count, then by constraint_scenario_description
                data_rows_sorted = data_rows.sort_values(
                    by=['core_b_constraints_count', 'constraint_scenario_description'],
                    ascending=[True, True]
                )
                
                # Combine header and sorted data
                df_sorted = pd.concat([header_rows, data_rows_sorted], ignore_index=True)
                
                # Write back to CSV
                df_sorted.to_csv(filename, index=False)
                print(f"✓ Sorted {filename}")
            else:
                print(f"⚠ Skipped sorting {filename} (insufficient rows or missing columns)")
                
        except Exception as e:
            print(f"⚠ Error sorting {filename}: {str(e)}")
    
    for quality_index in target_quality_indices:
        filename = output_csv_filenames[quality_index]
        print(f"✓ {quality_index} fit_params saved to: {filename}")