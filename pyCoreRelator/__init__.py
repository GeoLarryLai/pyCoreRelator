"""
pyCoreRelator: A Python package for core-to-core correlation using Dynamic Time Warping (DTW)

Main user functions for core correlation analysis with age constraints and quality metrics.
"""

from .core.dtw_analysis import run_comprehensive_dtw_analysis
from .core.segment_analysis import find_complete_core_paths, diagnose_chain_breaks
from .core.age_models import calculate_interpolated_ages
from .visualization.plotting import visualize_combined_segments, plot_correlation_distribution
from .visualization.matrix_plots import plot_dtw_matrix_with_paths
from .visualization.animation import visualize_dtw_results_from_csv
from .utils.data_loader import load_log_data, plot_core_data
from .core.null_hypothesis import (
    create_segment_pool_from_available_cores,
    generate_synthetic_core_pair,
    compute_pycorerelator_null_hypothesis,
    create_synthetic_picked_depths
)

__version__ = "0.1.0"
__author__ = "Larry Syu-Heng lai"

# Main user workflow functions
__all__ = [
    'run_comprehensive_dtw_analysis',
    'find_complete_core_paths', 
    'diagnose_chain_breaks',
    'calculate_interpolated_ages',
    'visualize_combined_segments',
    'visualize_dtw_results_from_csv',
    'plot_correlation_distribution',
    'load_log_data',
    'plot_core_data',
    'plot_dtw_matrix_with_paths',
    'create_segment_pool_from_available_cores',
    'generate_synthetic_core_pair',
    'compute_pycorerelator_null_hypothesis',
    'create_synthetic_picked_depths'
]