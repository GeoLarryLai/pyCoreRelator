"""
Utility functions for pyCoreRelator

This module contains data loading, path processing utilities, and visualization functions.
"""

# Data loading
from .data_loader import (
    load_log_data,
    resample_datasets,
    load_age_constraints_from_csv,
    combine_age_constraints,
    load_core_age_constraints,
    load_pickeddepth_ages_from_csv,
    load_and_prepare_quality_data,
    reconstruct_raw_data_from_histogram,
    load_sequential_mappings
)

# Path processing utilities
from .path_processing import (
    is_subset_or_superset,
    filter_against_existing
)

# Visualization functions
from .plotting import (
    plot_segment_pair_correlation,
    plot_multilog_segment_pair_correlation,
    visualize_combined_segments,
    plot_correlation_distribution,
    plot_quality_comparison_t_statistics,
    calculate_quality_comparison_t_statistics,
    plot_t_statistics_vs_constraints,
    plot_quality_distributions
)

from .matrix_plots import (
    plot_dtw_matrix_with_paths
)

from .animation import (
    visualize_dtw_results_from_csv
)

__all__ = [
    # Data loading
    'load_log_data',
    'resample_datasets',
    'load_age_constraints_from_csv',
    'combine_age_constraints',
    'load_core_age_constraints',
    'load_pickeddepth_ages_from_csv',
    'load_and_prepare_quality_data',
    'reconstruct_raw_data_from_histogram',
    'load_sequential_mappings',
    # Path processing
    'is_subset_or_superset',
    'filter_against_existing',
    # Visualization
    'plot_segment_pair_correlation',
    'plot_multilog_segment_pair_correlation',
    'visualize_combined_segments',
    'plot_correlation_distribution',
    'plot_quality_comparison_t_statistics',
    'calculate_quality_comparison_t_statistics',
    'plot_t_statistics_vs_constraints',
    'plot_quality_distributions',
    'plot_dtw_matrix_with_paths',
    'visualize_dtw_results_from_csv'
]
