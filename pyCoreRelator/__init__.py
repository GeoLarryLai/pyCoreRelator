"""
pyCoreRelator: Python package for geological core correlation using Dynamic Time Warping.

This package provides comprehensive tools for correlating geological core data using
advanced dynamic time warping algorithms, segment analysis, and quality assessment.
"""

__version__ = "0.1.1"

# Core functionality - Data loading and basic operations
from .utils.data_loader import load_log_data, resample_datasets
from .utils.helpers import find_nearest_index
from .utils.path_processing import combine_segment_dtw_results

# Core analysis functions - Segment operations
from .core.segment_operations import (
    find_all_segments,
    build_connectivity_graph, 
    identify_special_segments,
    filter_dead_end_pairs
)

# Core analysis functions - Path finding  
from .core.path_finding import (
    compute_total_complete_paths,
    find_complete_core_paths
)

# Core analysis functions - Diagnostics
from .core.diagnostics import diagnose_chain_breaks

# Core analysis functions - Null hypothesis testing
from .core.null_hypothesis import (
    load_segment_pool,
    plot_segment_pool, 
    print_segment_pool_summary,
    create_synthetic_log_with_depths,
    create_and_plot_synthetic_core_pair
)

# DTW and quality analysis
from .core.dtw_analysis import run_comprehensive_dtw_analysis
from .core.quality_metrics import (
    compute_quality_indicators,
    calculate_age_overlap_percentage,
    find_best_mappings
)
from .core.age_models import calculate_interpolated_ages

# Visualization functions - Basic plotting
from .visualization.core_plots import plot_core_data

# Visualization functions - Advanced DTW plotting
from .visualization.plotting import (
    plot_segment_pair_correlation,
    plot_multilog_segment_pair_correlation,
    visualize_combined_segments,
    plot_correlation_distribution
)

# Visualization functions - Matrix and advanced plots
from .visualization.matrix_plots import plot_dtw_matrix_with_paths
from .visualization.animation import visualize_dtw_results_from_csv

# Make commonly used functions available at package level
__all__ = [
    # Version
    '__version__',
    
    # Core data operations
    'load_log_data',
    'plot_core_data',
    
    # Main analysis functions  
    'run_comprehensive_dtw_analysis',
    'find_complete_core_paths',
    'diagnose_chain_breaks',
    'calculate_interpolated_ages',
    
    # Null hypothesis testing functions
    'load_segment_pool',
    'plot_segment_pool', 
    'print_segment_pool_summary',
    'create_synthetic_log_with_depths',
    'create_and_plot_synthetic_core_pair',
    
    # Visualization functions
    'visualize_combined_segments',
    'visualize_dtw_results_from_csv',
    'plot_dtw_matrix_with_paths',
    'plot_correlation_distribution',
    'find_best_mappings',
    
    # Segment operations
    'find_all_segments',
    'compute_total_complete_paths',
    
    # Quality metrics
    'compute_quality_indicators',
    'calculate_age_overlap_percentage',
    
    # Utilities
    'find_nearest_index',
    'combine_segment_dtw_results'
]