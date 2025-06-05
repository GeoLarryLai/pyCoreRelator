"""
Visualization functions for pyCoreRelator

This module contains plotting, animation, and matrix visualization functions
for DTW correlation results.
"""

from .plotting import (
    plot_segment_pair_correlation,
    plot_multilog_segment_pair_correlation,
    visualize_combined_segments,
    plot_correlation_distribution
)

from .matrix_plots import (
    plot_dtw_matrix_with_paths
)

from .animation import (
    create_segment_dtw_animation,
    visualize_dtw_results_from_csv
)

__all__ = [
    'plot_segment_pair_correlation',
    'plot_multilog_segment_pair_correlation',
    'visualize_combined_segments',
    'plot_correlation_distribution',
    'plot_dtw_matrix_with_paths',
    'create_segment_dtw_animation',
    'visualize_dtw_results_from_csv'
]