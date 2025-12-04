"""
Synthetic stratigraphy plotting functions for pyCoreRelator

This module provides functions for generating synthetic core data and running
synthetic stratigraphy plotting for correlation analysis. It includes segment pool management,
synthetic log generation, and visualization tools.

Functions:
 data from turbidite database
- plot_segment_pool: Plot all segments from the pool in a grid layout
 from the pool data
 using turbidite database approach
- plot_segment_pool: Plot all segments from the pool
- plot_synthetic_core_pair: Plot synthetic core pair
- create_and_plot_synthetic_core_pair: Generate and plot synthetic core pair the results
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
from .dtw_core import run_comprehensive_dtw_analysis
from .path_finding import find_complete_core_paths
from .age_models import calculate_interpolated_ages

from .syn_strat import create_synthetic_log

def plot_segment_pool(segment_logs, segment_depths, log_data_type, n_cols=8, figsize_per_row=4, 
                     plot_segments=True, save_plot=False, plot_filename=None):
    """
    Plot all segments from the pool in a grid layout.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - log_data_type: list of column names for labeling
    - n_cols: number of columns in the subplot grid
    - figsize_per_row: height per row in the figure
    - plot_segments: whether to plot the segments (default True)
    - save_plot: whether to save the plot to file (default False)
    - plot_filename: filename for saving plot (optional)
    
    Returns:
    - None
    """
    print(f"Plotting {len(segment_logs)} segments from the pool...")
    
    if not plot_segments:
        return
    
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
                col_name = log_data_type[col_idx] if col_idx < len(log_data_type) else f'Log_{col_idx}'
                
                ax.plot(segment[:, col_idx], depth, 
                       color=color, linestyle=line_style, linewidth=1, 
                       label=col_name, alpha=0.8)
            
            # Set xlabel to show all log types
            if len(log_data_type) > 1:
                ax.set_xlabel(f'Multiple Logs: {", ".join(log_data_type[:n_log_types])} (normalized)')
            else:
                ax.set_xlabel(f'{log_data_type[0]} (normalized)')
                
            # Add legend if multiple log types
            if n_log_types > 1:
                ax.legend(fontsize=8, loc='best')
                
        else:
            # 1D data
            ax.plot(segment, depth, 'b-', linewidth=1)
            ax.set_xlabel(f'{log_data_type[0]} (normalized)')
        
        ax.set_ylabel('Relative Depth (cm)')
        ax.set_title(f'Segment {i+1}\n({len(segment)} pts, {depth[-1]:.1f} cm)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Depth increases downward
    
    # Hide unused subplots
    for i in range(n_segments, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Update title to reflect multiple log types if present
    if len(log_data_type) > 1:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments, {len(log_data_type)} log types)', 
                     y=1.02, fontsize=16)
    else:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments)', y=1.02, fontsize=16)
    
    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()



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

    synthetic_log_a, synthetic_md_a, synthetic_picked_a_tuples, inds_a = create_synthetic_log(
        core_a_length, turb_logs, depth_logs, exclude_inds=None, repetition=repetition
    )
    synthetic_log_b, synthetic_md_b, synthetic_picked_b_tuples, inds_b = create_synthetic_log(
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


