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
    create_segment_pool_from_available_cores,
    generate_synthetic_core_pair,
    compute_pycorerelator_null_hypothesis
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
    'create_segment_pool_from_available_cores',
    'generate_synthetic_core_pair',
    'compute_pycorerelator_null_hypothesis'
]