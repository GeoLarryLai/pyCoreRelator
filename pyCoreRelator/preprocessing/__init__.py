"""
Preprocessing functions for pyCoreRelator.

This module contains functions for preprocessing core data including:
- CT image processing and visualization
- RGB image processing and visualization
- Machine learning-based gap filling
- Interactive datum picking

Note: plot_core_data has been moved to pyCoreRelator.utils.plotting
"""

# CT image processing
from .ct_processing import (
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

from .ct_plotting import (
    display_slice,
    plot_ctimg_curves,
    plot_stitched_curves
)

# RGB image processing
from .rgb_processing import (
    trim_image,
    extract_rgb_profile,
    rgb_process_and_stitch
)

from .rgb_plotting import (
    plot_rgbimg_curves
)

# Gap filling
from .gap_filling import (
    preprocess_core_data,
    prepare_feature_data,
    apply_feature_weights,
    adjust_gap_predictions,
    train_model,
    fill_gaps_with_ml,
    process_and_fill_logs
)

from .gap_filling_plots import (
    plot_core_logs,
    plot_filled_data
)

# Datum picking
from .datum_picker import (
    onclick_boundary,
    get_category_color,
    onkey_boundary,
    create_interactive_figure,
    pick_stratigraphic_levels,
    interpret_bed_names
)

__all__ = [
    # CT processing
    'load_dicom_files',
    'get_slice',
    'trim_slice',
    'get_brightness_trace',
    'get_brightness_stats',
    'process_brightness_data',
    'find_best_overlap',
    'stitch_curves',
    'create_stitched_slice',
    'process_single_scan',
    'process_two_scans',
    'ct_process_and_stitch',
    # CT plotting
    'display_slice',
    'plot_ctimg_curves',
    'plot_stitched_curves',
    # RGB processing
    'trim_image',
    'extract_rgb_profile',
    'rgb_process_and_stitch',
    # RGB plotting
    'plot_rgbimg_curves',
    # Gap filling
    'preprocess_core_data',
    'prepare_feature_data',
    'apply_feature_weights',
    'adjust_gap_predictions',
    'train_model',
    'fill_gaps_with_ml',
    'process_and_fill_logs',
    # Gap filling plotting
    'plot_core_logs',
    'plot_filled_data',
    # Datum picking
    'onclick_boundary',
    'get_category_color',
    'onkey_boundary',
    'create_interactive_figure',
    'pick_stratigraphic_levels',
    'interpret_bed_names'
]

