"""
Null hypothesis testing functions for pyCoreRelator

This module provides functions for generating synthetic core data and running
null hypothesis tests for correlation analysis. It includes segment pool management,
synthetic log generation, and visualization tools.

Functions:
- load_segment_pool: Load segment pool data from turbidite database
- plot_segment_pool: Plot all segments from the pool in a grid layout
- print_segment_pool_summary: Print summary statistics for the segment pool
- create_synthetic_log_with_depths: Create synthetic log using turbidite database approach
- create_and_plot_synthetic_core_pair: Generate synthetic core pair and optionally plot the results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import warnings
warnings.filterwarnings('ignore')

# Import from other pyCoreRelator modules
from ..utils.data_loader import load_log_data


def load_segment_pool(core_names, core_log_paths, picked_depth_paths, log_columns, 
                     depth_column, column_alternatives, boundary_category=1):
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
                for i in range(len(category_depths) - 1):
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
    
    for i, (segment, depth) in enumerate(zip(segment_logs, segment_depths)):
        ax = axes[i]
        
        # Plot segment
        if segment.ndim > 1:
            # Multi-dimensional data - plot first column
            ax.plot(segment[:, 0], depth, 'b-', linewidth=1)
            ax.set_xlabel(f'{log_column_names[0]} (normalized)')
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
    plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments)', y=1.02, fontsize=16)
    
    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()
    
    return fig, axes


def print_segment_pool_summary(segment_logs, segment_depths, target_dimensions):
    """
    Print summary statistics for the segment pool.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - target_dimensions: number of dimensions in the data
    """
    segment_lengths = [len(seg) for seg in segment_logs]
    segment_depths_max = [depth[-1] for depth in segment_depths]
    
    print(f"\nSegment Pool Summary:")
    print(f"  Total segments: {len(segment_logs)}")
    print(f"  Length range: {min(segment_lengths)}-{max(segment_lengths)} points")
    print(f"  Depth range: {min(segment_depths_max):.1f}-{max(segment_depths_max):.1f} cm")
    print(f"  Mean depth: {np.mean(segment_depths_max):.1f} cm")
    print(f"  Target dimensions: {target_dimensions}")


def create_synthetic_log_with_depths(thickness, turb_logs, depth_logs, exclude_inds=None, plot_results=True, save_plot=False, plot_filename=None):
    """Create synthetic log using turbidite database approach with picked depths at turbidite bases."""
    # Determine target dimensions from the first available segment
    target_dimensions = turb_logs[0].shape[1] if len(turb_logs) > 0 and turb_logs[0].ndim > 1 else 1
    
    fake_log = np.array([]).reshape(0, target_dimensions) if target_dimensions > 1 else np.array([])
    md_log = np.array([])
    max_depth = 0
    inds = []
    picked_depths = []
    
    # Add initial boundary
    picked_depths.append((0, 1))
    
    while max_depth <= thickness:
        ind = random.choices(np.arange(len(turb_logs)), k=1)[0]
        
        # Skip if this index should be excluded
        if exclude_inds is not None and ind in exclude_inds:
            continue
            
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
                                       log_columns, plot_results=True, save_plot=False, plot_filename=None):
    """
    Generate synthetic core pair and optionally plot the results.
    
    Parameters:
    - core_a_length: target length for core A
    - core_b_length: target length for core B
    - turb_logs: list of turbidite log segments
    - depth_logs: list of corresponding depth arrays
    - log_columns: list of log column names for labeling
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
        core_a_length, turb_logs, depth_logs, exclude_inds=None
    )
    synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b_tuples = create_synthetic_log_with_depths(
        core_b_length, turb_logs, depth_logs, exclude_inds=None
    )

    # Extract just the depths from the tuples
    synthetic_picked_a = [depth for depth, category in synthetic_picked_a_tuples]
    synthetic_picked_b = [depth for depth, category in synthetic_picked_b_tuples]

    # Plot synthetic core pair if requested
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 8))

        # Plot synthetic core A
        if synthetic_log_a.ndim > 1:
            ax1.plot(synthetic_log_a[:, 0], synthetic_md_a, 'b-', linewidth=1)
        else:
            ax1.plot(synthetic_log_a, synthetic_md_a, 'b-', linewidth=1)

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_a:
            ax1.axhline(y=depth, color='red', linestyle='--', alpha=0.7, linewidth=1)

        ax1.set_xlabel(f'{log_columns[0]}\n(normalized)')
        ax1.set_ylabel('Depth (cm)')
        ax1.set_title(f'Synthetic Core A\n({len(inds_a)} turbidites)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Plot synthetic core B
        if synthetic_log_b.ndim > 1:
            ax2.plot(synthetic_log_b[:, 0], synthetic_md_b, 'g-', linewidth=1)
        else:
            ax2.plot(synthetic_log_b, synthetic_md_b, 'g-', linewidth=1)

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_b:
            ax2.axhline(y=depth, color='red', linestyle='--', alpha=0.7, linewidth=1)

        ax2.set_xlabel(f'{log_columns[0]}\n(normalized)')
        ax2.set_ylabel('Depth (cm)')
        ax2.set_title(f'Synthetic Core B\n({len(inds_b)} turbidites)')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout()
        
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