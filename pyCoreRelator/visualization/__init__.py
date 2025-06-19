"""
Visualization functions for pyCoreRelator

This module contains plotting, animation, and matrix visualization functions
for DTW correlation results and core data display.
"""

# Basic core plotting
from .core_plots import plot_core_data

# Advanced DTW plotting
from .plotting import (
    plot_segment_pair_correlation,
    plot_multilog_segment_pair_correlation,
    visualize_combined_segments,
    plot_correlation_distribution
)

# Matrix plots
from .matrix_plots import (
    plot_dtw_matrix_with_paths
)

# Animation and interactive visualization
from .animation import (
    create_segment_dtw_animation,
    visualize_dtw_results_from_csv,
    create_gif
)

__all__ = [
    # Basic core plotting
    'plot_core_data',
    
    # Advanced DTW plotting
    'plot_segment_pair_correlation',
    'plot_multilog_segment_pair_correlation',
    'visualize_combined_segments',
    'plot_correlation_distribution',
    
    # Matrix plots
    'plot_dtw_matrix_with_paths',
    
    # Animation and interactive visualization
    'create_segment_dtw_animation',
    'visualize_dtw_results_from_csv',
    'create_gif'
]