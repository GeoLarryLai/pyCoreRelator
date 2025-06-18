"""
Core DTW analysis functions for pyCoreRelator

This module contains the main DTW computation, quality assessment, 
age modeling, and segment analysis functions.
"""

from .dtw_analysis import (
    custom_dtw,
    run_comprehensive_dtw_analysis
)

from .quality_metrics import (
    compute_quality_indicators,
    calculate_age_overlap_percentage
)

from .age_models import (
    calculate_interpolated_ages,
    check_age_constraint_compatibility
)

from .segment_analysis import (
    find_all_segments,
    find_complete_core_paths,
    diagnose_chain_breaks,
    compute_total_complete_paths
)

from .null_hypothesis import (
    load_segment_pool,
    plot_segment_pool,
    print_segment_pool_summary,
    create_synthetic_log_with_depths,
    create_and_plot_synthetic_core_pair
)

__all__ = [
    'custom_dtw',
    'run_comprehensive_dtw_analysis',
    'compute_quality_indicators',
    'calculate_age_overlap_percentage',
    'calculate_interpolated_ages',
    'check_age_constraint_compatibility',
    'find_all_segments',
    'find_complete_core_paths',
    'diagnose_chain_breaks',
    'compute_total_complete_paths',
    'load_segment_pool',
    'plot_segment_pool',
    'print_segment_pool_summary',
    'create_synthetic_log_with_depths',
    'create_and_plot_synthetic_core_pair'
]