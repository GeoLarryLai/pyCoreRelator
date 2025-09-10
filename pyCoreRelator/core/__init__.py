"""
Core analysis modules for pyCoreRelator.

This package contains the core analytical functions for geological core correlation,
including DTW analysis, segment operations, path finding, quality metrics, and diagnostics.
"""

# Segment operations and analysis
from .segment_operations import (
    find_all_segments,
    build_connectivity_graph,
    identify_special_segments,
    filter_dead_end_pairs
)

# Path finding algorithms
from .path_finding import (
    compute_total_complete_paths,
    find_complete_core_paths
)

# Diagnostic functions
from .diagnostics import diagnose_chain_breaks

# DTW analysis
from .dtw_analysis import run_comprehensive_dtw_analysis

# Quality metrics and assessment
from .quality_metrics import (
    compute_quality_indicators,
    calculate_age_overlap_percentage
)

# Age modeling
from .age_models import calculate_interpolated_ages

# Null hypothesis testing
from .null_hypothesis import (
    load_segment_pool,
    plot_segment_pool,
    modify_segment_pool,
    create_synthetic_log_with_depths,
    create_and_plot_synthetic_core_pair,
    generate_constraint_subsets,
    run_multi_parameter_analysis
)

__all__ = [
    # Segment operations
    'find_all_segments',
    'build_connectivity_graph',
    'identify_special_segments',
    'filter_dead_end_pairs',
    
    # Path finding
    'compute_total_complete_paths',
    'find_complete_core_paths',
    
    # Diagnostics
    'diagnose_chain_breaks',
    
    # DTW analysis
    'run_comprehensive_dtw_analysis',
    
    # Quality metrics
    'compute_quality_indicators',
    'calculate_age_overlap_percentage',
    
    # Age modeling
    'calculate_interpolated_ages',
    
    # Null hypothesis testing
    'load_segment_pool',
    'plot_segment_pool',
    'modify_segment_pool',
    'create_synthetic_log_with_depths',
    'create_and_plot_synthetic_core_pair',
    'generate_constraint_subsets',
    'run_multi_parameter_analysis'
]