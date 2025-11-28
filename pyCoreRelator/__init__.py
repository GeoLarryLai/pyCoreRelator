"""
pyCoreRelator: Python package for geological core correlation using Dynamic Time Warping.

This package provides comprehensive tools for correlating geological core data using
advanced dynamic time warping algorithms, segment analysis, and quality assessment.
"""

__version__ = "0.1.2"

# Core functionality - Data loading and basic operations
from .utils.data_loader import (
    load_core_log_data,
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

# Utilities (merged from helpers)
from .analysis.quality import find_nearest_index, cohens_d

# Path processing utilities
from .utils.path_processing import (
    is_subset_or_superset,
    filter_against_existing
)

# Analysis functions - Segment operations
from .analysis.segments import (
    find_all_segments,
    build_connectivity_graph, 
    identify_special_segments,
    filter_dead_end_pairs
)

# Analysis functions - Path finding  
from .analysis.path_finding import (
    compute_total_complete_paths,
    find_complete_core_paths
)

# Analysis functions - Path combining
from .analysis.path_combining import (
    combine_segment_dtw_results
)

# Analysis functions - Diagnostics
from .analysis.diagnostics import diagnose_chain_breaks

# Analysis functions - Synthetic stratigraphy
from .analysis.syn_strat import (
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

from .analysis.syn_strat_plot import (
    plot_segment_pool,
    create_and_plot_synthetic_core_pair
)

# DTW and quality analysis
from .analysis.dtw_core import run_comprehensive_dtw_analysis
from .analysis.quality import (
    compute_quality_indicators,
    calculate_age_overlap_percentage,
    find_best_mappings
)
from .analysis.age_models import calculate_interpolated_ages

# Visualization functions - Core plotting and DTW
from .utils.plotting import (
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
from .utils.matrix_plots import plot_dtw_matrix_with_paths
from .utils.animation import visualize_dtw_results_from_csv

# Preprocessing functions - RGB image analysis
from .preprocessing.rgb_processing import (
    trim_image,
    extract_rgb_profile,
    rgb_process_and_stitch
)

from .preprocessing.rgb_plotting import (
    plot_rgbimg_curves
)

# Preprocessing functions - CT image analysis
from .preprocessing.ct_processing import (
    load_dicom_files,
    get_slice,
    trim_slice,
    get_brightness_trace,
    get_brightness_stats,
    process_brightness_data,
    find_best_overlap,
    stitch_curves,
    create_stitched_slice,
    process_single_scan,
    process_two_scans,
    ct_process_and_stitch
)

from .preprocessing.ct_plotting import (
    display_slice,
    plot_ctimg_curves,
    plot_stitched_curves
)

# Preprocessing functions - Core datum picking
from .preprocessing.datum_picker import (
    onclick_boundary,
    get_category_color,
    onkey_boundary,
    create_interactive_figure,
    pick_stratigraphic_levels,
    interpret_bed_names
)

# Preprocessing functions - Machine learning data imputation
from .preprocessing.gap_filling import (
    preprocess_core_data,
    prepare_feature_data,
    apply_feature_weights,
    adjust_gap_predictions,
    train_model,
    fill_gaps_with_ml,
    process_and_fill_logs
)

from .preprocessing.gap_filling_plots import (
    plot_core_logs,
    plot_filled_data
)

# Make commonly used functions available at package level
__all__ = [
    # Version
    '__version__',
    
    # Core data operations
    'load_core_log_data',
    'load_log_data',
    'load_age_constraints_from_csv',
    'combine_age_constraints',
    'load_core_age_constraints',
    'load_pickeddepth_ages_from_csv',
    'load_and_prepare_quality_data',
    'reconstruct_raw_data_from_histogram',
    'load_sequential_mappings',
    
    # Main analysis functions  
    'run_comprehensive_dtw_analysis',
    'find_complete_core_paths',
    'diagnose_chain_breaks',
    'calculate_interpolated_ages',
    
    # Synthetic stratigraphy functions
    'load_segment_pool',
    'plot_segment_pool', 
    'modify_segment_pool',
    'create_synthetic_log',
    'create_synthetic_core_pair',
    'create_and_plot_synthetic_core_pair',
    'plot_synthetic_log',
    'synthetic_correlation_quality',
    'plot_synthetic_correlation_quality',
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
    
    # Segment operations
    'find_all_segments',
    'compute_total_complete_paths',
    'combine_segment_dtw_results',
    
    # Quality metrics
    'compute_quality_indicators',
    'calculate_age_overlap_percentage',
    'find_best_mappings',
    
    # Utilities
    'find_nearest_index',
    'cohens_d',
    'is_subset_or_superset',
    'filter_against_existing',
    
    # RGB image processing functions
    'trim_image',
    'extract_rgb_profile',
    'plot_rgbimg_curves',
    'rgb_process_and_stitch',
    
    # CT image processing functions
    'load_dicom_files',
    'get_slice',
    'trim_slice',
    'get_brightness_trace',
    'get_brightness_stats',
    'display_slice',
    'plot_ctimg_curves',
    'process_brightness_data',
    'find_best_overlap',
    'stitch_curves',
    'plot_stitched_curves',
    'create_stitched_slice',
    'process_single_scan',
    'process_two_scans',
    'ct_process_and_stitch',
    
    # Core datum picking functions
    'onclick_boundary',
    'get_category_color',
    'onkey_boundary',
    'create_interactive_figure',
    'pick_stratigraphic_levels',
    'interpret_bed_names',
    
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
