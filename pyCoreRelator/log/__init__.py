"""
Log processing functions for pyCoreRelator.

This module contains functions for processing various types of log data
including RGB image analysis, CT image processing, interactive core 
datum picking, and machine learning-based data gap filling.
"""

from .rgb_image2log import (
    trim_image,
    extract_rgb_profile, 
    plot_rgb_profile,
    stitch_core_sections
)

from .ct_image2log import (
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

from .core_datum_picker import (
    onclick_boundary,
    get_category_color,
    onkey_boundary,
    create_interactive_figure,
    pick_stratigraphic_levels
)

from .ml_log_data_imputation import (
    preprocess_core_data,
    plot_core_logs,
    plot_filled_data,
    prepare_feature_data,
    apply_feature_weights,
    adjust_gap_predictions,
    train_model,
    fill_gaps_with_ml,
    process_and_fill_logs
)

__all__ = [
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
    'train_model',
    'fill_gaps_with_ml',
    'process_and_fill_logs'
] 