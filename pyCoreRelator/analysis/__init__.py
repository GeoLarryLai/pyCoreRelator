"""
Analysis functions for pyCoreRelator.

This module contains core analysis functions for geological core correlation,
including DTW analysis, segment operations, path finding, quality metrics, age modeling,
and synthetic stratigraphy.
"""

# Segment operations
from .segments import (
    find_all_segments,
    build_connectivity_graph,
    identify_special_segments,
    filter_dead_end_pairs
)

# Path finding
from .path_finding import (
    compute_total_complete_paths,
    find_complete_core_paths
)

# Path combining (DTW)
from .path_combining import (
    combine_segment_dtw_results,
    compute_combined_path_metrics
)

# Diagnostics
from .diagnostics import (
    diagnose_chain_breaks
)

# DTW analysis
from .dtw_core import (
    run_comprehensive_dtw_analysis
)

# Quality metrics
from .quality import (
    compute_quality_indicators,
    calculate_age_overlap_percentage,
    find_best_mappings,
    find_nearest_index,
    cohens_d
)

# Age modeling
from .age_models import (
    calculate_interpolated_ages
)

# Synthetic stratigraphy
from .syn_strat import (
    load_segment_pool,
    modify_segment_pool,
    create_synthetic_log,
    create_synthetic_core_pair,
    plot_synthetic_log,
    synthetic_correlation_quality,
    plot_synthetic_correlation_quality,
    generate_constraint_subsets,
    run_multi_parameter_analysis
)

from .syn_strat_plot import (
    plot_segment_pool,
    create_and_plot_synthetic_core_pair
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
    # Path combining
    'combine_segment_dtw_results',
    'compute_combined_path_metrics',
    # Diagnostics
    'diagnose_chain_breaks',
    # DTW analysis
    'run_comprehensive_dtw_analysis',
    # Quality metrics
    'compute_quality_indicators',
    'calculate_age_overlap_percentage',
    'find_best_mappings',
    'find_nearest_index',
    'cohens_d',
    # Age modeling
    'calculate_interpolated_ages',
    # Synthetic stratigraphy
    'load_segment_pool',
    'modify_segment_pool',
    'create_synthetic_log',
    'create_synthetic_core_pair',
    'plot_segment_pool',
    'plot_synthetic_log',
    'synthetic_correlation_quality',
    'plot_synthetic_correlation_quality',
    'generate_constraint_subsets',
    'run_multi_parameter_analysis',
    'create_and_plot_synthetic_core_pair'
]

