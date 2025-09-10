"""
pyCoreRelator: Python package for geological core correlation using Dynamic Time Warping.

This package provides comprehensive tools for correlating geological core data using
advanced dynamic time warping algorithms, segment analysis, and quality assessment.
"""

__version__ = "0.1.2"

# Core functionality - Data loading and basic operations
from .utils.data_loader import (
    load_log_data, 
    resample_datasets,
    load_age_constraints_from_csv,
    combine_age_constraints,
    load_core_age_constraints,
    load_pickeddepth_ages_from_csv,
    load_and_prepare_quality_data,
    reconstruct_raw_data_from_histogram
)
from .utils.helpers import find_nearest_index, cohens_d
from .utils.path_processing import (
    combine_segment_dtw_results,
    find_best_mappings
)

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
    create_and_plot_synthetic_core_pair,
    generate_constraint_subsets,
    run_multi_parameter_analysis
)

# DTW and quality analysis
from .core.dtw_analysis import run_comprehensive_dtw_analysis
from .core.quality_metrics import (
    compute_quality_indicators,
    calculate_age_overlap_percentage
)
from .core.age_models import calculate_interpolated_ages

# Visualization functions - Basic plotting
from .visualization.core_plots import plot_core_data

# Visualization functions - Advanced DTW plotting
from .visualization.plotting import (
    plot_segment_pair_correlation,
    plot_multilog_segment_pair_correlation,
    visualize_combined_segments,
    plot_correlation_distribution,
    plot_quality_comparison_t_statistics,         
    calculate_quality_comparison_t_statistics,    
    plot_t_statistics_vs_constraints,
    plot_quality_distributions     
)

# Visualization functions - Matrix and advanced plots
from .visualization.matrix_plots import plot_dtw_matrix_with_paths
from .visualization.animation import visualize_dtw_results_from_csv

# Log processing functions - RGB image analysis
from .log.rgb_image2log import (
    trim_image,
    extract_rgb_profile,
    plot_rgb_profile,
    stitch_core_sections
)

# Log processing functions - CT image analysis
from .log.ct_image2log import (
    load_dicom_files,
    get_slice,
    trim_slice,
    get_brightness_trace,
    get_brightness_stats,
    display_slice,
    display_slice_bt_std,
    process_brightness_data,
    find_best_overlap,
    stitch_curves,
    plot_stitched_curves,
    create_stitched_slice,
    process_single_scan,
    process_two_scans,
    process_and_stitch_segments
)

# Log processing functions - Core datum picking
from .log.core_datum_picker import (
    onclick_boundary,
    get_category_color,
    onkey_boundary,
    create_interactive_figure,
    pick_stratigraphic_levels
)

# Log processing functions - Machine learning data imputation
from .log.ml_log_data_imputation import (
    preprocess_core_data,
    plot_core_logs,
    plot_filled_data,
    prepare_feature_data,
    apply_feature_weights,
    adjust_gap_predictions,
    fill_gaps_with_ml,
    process_and_fill_logs
)

# Make commonly used functions available at package level
__all__ = [
    # Version
    '__version__',
    
    # Core data operations
    'load_log_data',
    'load_age_constraints_from_csv',
    'combine_age_constraints',
    'load_core_age_constraints',
    'load_pickeddepth_ages_from_csv',
    'load_and_prepare_quality_data',
    'reconstruct_raw_data_from_histogram',
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
    'generate_constraint_subsets',
    'run_multi_parameter_analysis',
    
    # Visualization functions
    'visualize_combined_segments',
    'visualize_dtw_results_from_csv',
    'plot_dtw_matrix_with_paths',
    'plot_correlation_distribution',
    'calculate_quality_comparison_t_statistics',  
    'plot_quality_comparison_t_statistics',   
    'plot_t_statistics_vs_constraints',
    'plot_quality_distributions',
    'find_best_mappings',
    
    # Segment operations
    'find_all_segments',
    'compute_total_complete_paths',
    
    # Quality metrics
    'compute_quality_indicators',
    'calculate_age_overlap_percentage',
    
    # Utilities
    'find_nearest_index',
    'combine_segment_dtw_results',
    'cohens_d',
    
    # RGB image processing functions
    'trim_image',
    'extract_rgb_profile',
    'plot_rgb_profile',
    'stitch_core_sections',
    
    # CT image processing functions
    'load_dicom_files',
    'get_slice',
    'trim_slice',
    'get_brightness_trace',
    'get_brightness_stats',
    'display_slice',
    'display_slice_bt_std',
    'process_brightness_data',
    'find_best_overlap',
    'stitch_curves',
    'plot_stitched_curves',
    'create_stitched_slice',
    'process_single_scan',
    'process_two_scans',
    'process_and_stitch_segments',
    
    # Core datum picking functions
    'onclick_boundary',
    'get_category_color',
    'onkey_boundary',
    'create_interactive_figure',
    'pick_stratigraphic_levels',
    
    # Machine learning log data imputation functions
    'preprocess_core_data',
    'plot_core_logs',
    'plot_filled_data',
    'prepare_feature_data',
    'apply_feature_weights',
    'adjust_gap_predictions',
    'fill_gaps_with_ml',
    'process_and_fill_logs'
]