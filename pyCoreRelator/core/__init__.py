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
    calculate_age_overlap_percentage,
    find_best_mappings
)

# Age modeling
from .age_models import calculate_interpolated_ages

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
    'find_best_mappings',
    
    # Age modeling
    'calculate_interpolated_ages'
]