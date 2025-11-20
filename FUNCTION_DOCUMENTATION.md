# pyCoreRelator Function Documentation

This document provides detailed documentation for all functions in the pyCoreRelator package, organized by the current module structure for geological core correlation analysis.

## Analysis Module (`pyCoreRelator.analysis`)

### DTW Analysis (`dtw_core.py`)

#### `run_comprehensive_dtw_analysis(log_a, log_b, md_a, md_b, **kwargs)`

**ENHANCED** - Main function for segment-based DTW analysis with comprehensive age constraint integration, visualization capabilities, and performance optimizations.

**Parameters:**
- `log_a, log_b` (array-like): Well log data for cores A and B (1D or multidimensional)
- `md_a, md_b` (array-like): Measured depth arrays corresponding to the logs
- `picked_depths_a, picked_depths_b` (list, optional): User-picked depth boundaries for segmentation
- `core_a_name, core_b_name` (str, optional): Core identifiers for output files
- `top_bottom` (bool, default=True): Whether to include top and bottom boundaries automatically
- `top_depth` (float, default=0.0): Depth value for top boundary
- `independent_dtw` (bool, default=False): Whether to process multidimensional logs independently
- `exclude_deadend` (bool, default=True): Whether to filter out dead-end segment pairs
- `age_consideration` (bool, default=False): Whether to apply age-based filtering
- `ages_a, ages_b` (dict, optional): Age constraint dictionaries with interpolated ages
- `restricted_age_correlation` (bool, default=True): Whether to use strict age correlation filtering
- `all_constraint_ages_a, all_constraint_ages_b` (list, optional): All age constraints for filtering
- `all_constraint_depths_a, all_constraint_depths_b` (list, optional): Corresponding depth constraints
- `all_constraint_pos_errors_a, all_constraint_pos_errors_b` (list, optional): Positive age uncertainties
- `all_constraint_neg_errors_a, all_constraint_neg_errors_b` (list, optional): Negative age uncertainties
- `visualize_pairs` (bool, default=True): Whether to create segment pair visualizations
- `create_dtw_matrix` (bool, default=True): Whether to generate DTW matrix visualization
- `creategif` (bool, default=False): Whether to create animated GIF sequences
- `mute_mode` (bool, default=False): Whether to suppress print output for batch processing

**Returns:**
- `dtw_results` (dict): Comprehensive DTW results for each valid segment pair
- `valid_dtw_pairs` (set): Set of valid segment pair indices after all filtering
- `segments_a, segments_b` (list): Lists of segment boundaries for each core
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `dtw_distance_matrix_full` (np.ndarray): Full DTW distance matrix for visualization

#### `handle_single_point_dtw(log1, log2, exponent=1, QualityIndex=False)`

Handles DTW computation when the first log contains only a single data point by creating a custom warping path that maps the single point to all points in the second log.

**Parameters:**
- `log1` (array-like): First well log data with a single point
- `log2` (array-like): Second well log data with multiple points
- `exponent` (float, default=1): Exponent applied to cost calculation for distance weighting
- `QualityIndex` (bool, default=False): Whether to compute and return quality indicators

**Returns:**
- `D` (np.ndarray): Accumulated cost matrix with shape (1, len(log2))
- `wp` (np.ndarray): Warping path as sequence of index pairs mapping single point to all log2 points
- `QIdx` (dict, optional): Quality indicators if QualityIndex=True

#### `handle_single_point_log2_dtw(log1, log2, exponent=1, QualityIndex=False)`

Handles DTW computation when the second log contains only a single data point by creating a custom warping path that maps all points in the first log to the single point.

**Parameters:**
- `log1` (array-like): First well log data with multiple points
- `log2` (array-like): Second well log data with a single point
- `exponent` (float, default=1): Exponent applied to cost calculation for distance weighting
- `QualityIndex` (bool, default=False): Whether to compute and return quality indicators

**Returns:**
- `D` (np.ndarray): Accumulated cost matrix with shape (len(log1), 1)
- `wp` (np.ndarray): Warping path as sequence of index pairs mapping all log1 points to single point
- `QIdx` (dict, optional): Quality indicators if QualityIndex=True

#### `handle_two_single_points(log1, log2, exponent=1, QualityIndex=False)`

Handles DTW computation when both logs contain only a single data point each.

**Parameters:**
- `log1, log2` (array-like): Single-point log data arrays
- `exponent` (float, default=1): Exponent applied to cost calculation
- `QualityIndex` (bool, default=False): Whether to compute and return quality indicators

**Returns:**
- `D` (np.ndarray): Accumulated cost matrix (1x1)
- `wp` (np.ndarray): Warping path as single index pair (0,0)
- `QIdx` (dict, optional): Quality indicators if QualityIndex=True

#### `custom_dtw(log1, log2, subseq=False, exponent=1, QualityIndex=False, independent_dtw=False, available_columns=None)`

Custom implementation of Dynamic Time Warping for well log correlation that handles all edge cases including single points, multidimensional data, and independent processing of each dimension.

**Parameters:**
- `log1, log2` (array-like): Well log data arrays to be compared
- `subseq` (bool, default=False): Whether to perform subsequence DTW allowing partial matching
- `exponent` (float, default=1): Exponent applied to distance calculation for cost weighting
- `QualityIndex` (bool, default=False): Whether to compute and return quality indicators
- `independent_dtw` (bool, default=False): Whether to process each dimension separately for multidimensional data
- `available_columns` (list, default=None): Column names for logging when independent_dtw=True

**Returns:**
- `D` (np.ndarray): Accumulated cost matrix where D[i,j] contains minimum cumulative cost to reach point (i,j)
- `wp` (np.ndarray): Optimal warping path as sequence of index pairs
- `QIdx` (dict, optional): Quality indicators including correlation coefficient, diagonality, and DTW metrics if QualityIndex=True

### Quality Metrics (`quality.py`)

#### `compute_quality_indicators(log1, log2, p, q, D)`

Computes comprehensive quality indicators for DTW alignment including normalized distance, correlation, and path characteristics.

**Parameters:**
- `log1, log2` (array-like): Original log arrays being compared
- `p, q` (array-like): Warping path indices for log1 and log2 respectively
- `D` (np.ndarray): Accumulated cost matrix from DTW computation

**Returns:**
- `dict`: Quality indicators including:
  - `norm_dtw`: Normalized DTW distance (cost per path step)
  - `dtw_ratio`: Ratio of DTW distance to Euclidean distance
  - `variance_deviation`: Variance of index differences along warping path
  - `perc_diag`: Percentage diagonality of warping path
  - `corr_coef`: Correlation coefficient between aligned sequences
  - `dtw_warp_eff`: DTW warping efficiency measure
  - `match_min, match_mean`: Minimum and mean matching function values

#### `calculate_age_overlap_percentage(a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound)`

Calculates the percentage of age range overlap between two age intervals relative to their union.

**Parameters:**
- `a_lower_bound, a_upper_bound` (float): Lower and upper bounds of first age range
- `b_lower_bound, b_upper_bound` (float): Lower and upper bounds of second age range

**Returns:**
- `float`: Percentage overlap (0-100) of age ranges relative to their combined span

### Age Models (`age_models.py`)

#### `calculate_interpolated_ages(picked_depths, age_constraints_depths, age_constraints_ages, age_constraints_pos_errors, age_constraints_neg_errors, **kwargs)`

**ENHANCED** - Calculates interpolated or extrapolated ages for picked depths based on age constraints using various uncertainty propagation methods.

**Parameters:**
- `picked_depths` (list): List of picked depths in cm requiring age estimates
- `age_constraints_depths` (list): Mean depths for age constraint points
- `age_constraints_ages` (list): Calibrated ages for constraint points
- `age_constraints_pos_errors, age_constraints_neg_errors` (list): Positive and negative age uncertainties
- `age_constraints_in_sequence_flags` (list, optional): Boolean flags indicating which constraints are stratigraphically in-sequence
- `uncertainty_method` (str, default='MonteCarlo'): Method for uncertainty propagation ('Linear', 'MonteCarlo', 'Gaussian')
- `n_monte_carlo` (int, default=10000): Number of Monte Carlo iterations for uncertainty estimation
- `top_age, top_age_pos_error, top_age_neg_error` (float): Age and uncertainties at top depth
- `show_plot` (bool, default=False): Whether to display age-depth model plot
- `export_csv` (bool, default=True): Whether to export results to CSV file

**Returns:**
- `dict`: Dictionary containing interpolated ages and uncertainties for each depth with keys 'depths', 'ages', 'pos_uncertainties', 'neg_uncertainties'

#### `check_age_constraint_compatibility(a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound, constraint_ages_a, constraint_ages_b, constraint_pos_errors_a, constraint_pos_errors_b, constraint_neg_errors_a, constraint_neg_errors_b, ages_a=None, ages_b=None)`

Checks compatibility between two segment pairs based on their age ranges and constraint overlaps.

**Parameters:**
- `a_lower_bound, a_upper_bound` (float): Depth bounds for segment A
- `b_lower_bound, b_upper_bound` (float): Depth bounds for segment B  
- `constraint_ages_a, constraint_ages_b` (list): Age constraint values for each core
- `constraint_pos_errors_a, constraint_pos_errors_b` (list): Positive age uncertainties
- `constraint_neg_errors_a, constraint_neg_errors_b` (list): Negative age uncertainties
- `ages_a, ages_b` (dict, optional): Complete age model dictionaries for picked depths

**Returns:**
- `tuple`: (is_compatible, age_overlap_percentage) indicating whether segments are age-compatible and their overlap percentage

### Segment Operations (`segments.py`)

#### `find_all_segments(log_a, log_b, md_a, md_b, picked_depths_a=None, picked_depths_b=None, top_bottom=True, top_depth=0.0, mute_mode=False)`

Identifies segments in two logs using picked depths, creating consecutive boundary segments and single point segments for correlation analysis.

**Parameters:**
- `log_a, log_b` (array-like): Log data arrays for cores A and B
- `md_a, md_b` (array-like): Measured depth arrays corresponding to logs
- `picked_depths_a, picked_depths_b` (list, optional): User-picked depth values (not indices)
- `top_bottom` (bool, default=True): Whether to include top and bottom boundaries
- `top_depth` (float, default=0.0): Depth value for top boundary

**Returns:**
- `segments_a, segments_b` (list): Lists of segment boundary index pairs for each core
- `depth_boundaries_a, depth_boundaries_b` (list): Lists of depth boundary indices
- `depth_values_a, depth_values_b` (list): Lists of actual depth values used as boundaries

#### `find_complete_core_paths(valid_dtw_pairs, segments_a, segments_b, log_a, log_b, depth_boundaries_a, depth_boundaries_b, dtw_results, **kwargs)`

Finds complete correlation paths spanning entire cores by connecting valid segment pairs from top to bottom.

**Parameters:**
- `valid_dtw_pairs` (set): Set of valid segment pair indices
- `segments_a, segments_b` (list): Segment definitions for each core
- `log_a, log_b` (array-like): Log data for path quality assessment
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `dtw_results` (dict): DTW results for segment pairs
- `output_csv` (str, default="complete_core_paths.csv"): Output file for complete paths
- `debug` (bool, default=False): Whether to print detailed progress information
- `start_from_top_only` (bool, default=True): Whether to only consider paths starting from top segments
- `shortest_path_search` (bool, default=True): Whether to prioritize shorter paths
- `max_search_path` (int, default=5000): Maximum number of paths to explore

**Returns:**
- `str`: Path to output CSV file containing complete correlation paths with quality metrics

#### `diagnose_chain_breaks(valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b)`

Diagnoses connectivity issues in segment chains by identifying missing connections and isolated segments.

**Parameters:**
- `valid_dtw_pairs` (set): Set of valid segment pairs
- `segments_a, segments_b` (list): Segment definitions
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices

**Returns:**
- `dict`: Diagnostic information including missing connections, isolated segments, and connectivity statistics

#### `compute_total_complete_paths(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b)`

Computes the total number of complete paths connecting top and bottom segments for pathway complexity assessment.

**Parameters:**
- `valid_dtw_pairs` (set): Set of valid segment pairs
- `detailed_pairs` (dict): Dictionary containing segment depth details
- `max_depth_a, max_depth_b` (float): Maximum depths for cores A and B

**Returns:**
- `int`: Total count of complete paths from top to bottom segments

#### `build_connectivity_graph(valid_dtw_pairs, detailed_pairs)`

Builds predecessor and successor relationships between valid segment pairs for advanced path finding algorithms.

**Parameters:**
- `valid_dtw_pairs` (set): Valid segment pairs from DTW analysis
- `detailed_pairs` (dict): Dictionary mapping segment pairs to their depth details

**Returns:**
- `tuple`: (successors, predecessors) dictionaries mapping segments to connected segments

#### `identify_special_segments(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b)`

Identifies special types of segments: tops, bottoms, dead ends, and orphans for path analysis.

**Parameters:**
- `valid_dtw_pairs` (set): Valid segment pairs
- `detailed_pairs` (dict): Segment depth details
- `max_depth_a, max_depth_b` (float): Maximum depths for cores A and B

**Returns:**
- `tuple`: (top_segments, bottom_segments, dead_ends, orphans, successors, predecessors)

#### `filter_dead_end_pairs(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=False)`

Removes dead end and orphan segment pairs from the valid set to improve path connectivity.

**Parameters:**
- `valid_dtw_pairs` (set): Valid segment pairs
- `detailed_pairs` (dict): Segment depth details
- `max_depth_a, max_depth_b` (float): Maximum depths for cores A and B
- `debug` (bool, default=False): Whether to print debugging information

**Returns:**
- `set`: Filtered set of valid segment pairs excluding dead ends and orphans

### Path Combining (`path_combining.py`)

#### `combine_segment_dtw_results(dtw_results, segment_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, log_a, log_b)`

Combines DTW results from multiple segment pairs into a unified correlation result by concatenating warping paths and averaging quality metrics.

**Parameters:**
- `dtw_results` (dict): Dictionary containing DTW results for each segment pair
- `segment_pairs` (list): List of (a_idx, b_idx) tuples for segments to combine
- `segments_a, segments_b` (list): Segment definitions for each core
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `log_a, log_b` (array-like): Original log data for quality assessment

**Returns:**
- `combined_wp` (np.ndarray): Combined warping path spanning all selected segments
- `combined_quality` (dict): Averaged quality metrics across all segments

#### `compute_combined_path_metrics(combined_wp, log_a, log_b, segment_quality_indicators, age_overlap_values=None)`

Computes quality metrics for combined correlation paths by aggregating segment-level indicators.

**Parameters:**
- `combined_wp` (np.ndarray): Combined warping path
- `log_a, log_b` (array-like): Original log data
- `segment_quality_indicators` (list): List of quality indicator dictionaries from segments
- `age_overlap_values` (list, optional): Age overlap percentages for each segment

**Returns:**
- `dict`: Combined quality metrics including normalized DTW, correlation, and age overlap

## Preprocessing Module (`pyCoreRelator.preprocessing`)

### RGB Image Processing (`rgb_processing.py`)

#### `extract_rgb_profile(image_path, upper_rgb_threshold=100, lower_rgb_threshold=0, buffer=20, top_trim=0, bottom_trim=0, target_luminance=130, bin_size=10, width_start_pct=0.25, width_end_pct=0.75)`

Extracts RGB color profiles along the y-axis of an image file, analyzing the center strip and calculating statistics for binned data with normalization.

**Parameters:**
- `image_path` (str): Path to the image file (BMP, JPEG, PNG, TIFF formats supported)
- `upper_rgb_threshold` (float, default=100): Upper RGB threshold for filtering bright artifacts
- `lower_rgb_threshold` (float, default=0): Lower RGB threshold for excluding dark regions
- `buffer` (int, default=20): Buffer pixels above and below filtered regions
- `top_trim, bottom_trim` (int, default=0): Pixels to trim from image edges
- `target_luminance` (float, default=130): Target mean luminance for scaling
- `bin_size` (int, default=10): Bin size in pixels for depth averaging
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Width analysis strip boundaries

**Returns:**
- `tuple`: (depths_pixels, widths_pixels, r_means, g_means, b_means, r_stds, g_stds, b_stds, lum_means, lum_stds, img_array) containing depth positions, RGB statistics, and processed image

#### `trim_image(img_array, top_trim=0, bottom_trim=0)`

Removes specified pixels from top and bottom edges of image array to eliminate borders or artifacts.

**Parameters:**
- `img_array` (array-like): Input image array with shape (height, width, channels)
- `top_trim, bottom_trim` (int, default=0): Pixels to trim from edges

**Returns:**
- `array-like`: Trimmed image array with reduced height

### RGB Image Plotting (`rgb_plotting.py`)

#### `plot_rgb_profile(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img, core_name=None, save_figs=False, output_dir=None)`

Creates comprehensive three-panel visualization of RGB analysis results with image, color profiles, and standard deviation plots.

**Parameters:**
- `depths` (array-like): Depth positions in pixels
- `r, g, b` (array-like): RGB color intensity values
- `r_std, g_std, b_std` (array-like): RGB standard deviations
- `lum, lum_std` (array-like): Luminance values and standard deviations
- `img` (array-like): Core image array for display
- `core_name` (str, optional): Core identifier for titles and file naming
- `save_figs` (bool, default=False): Whether to save plots as files
- `output_dir` (str, optional): Directory for saved files

**Returns:**
- None (displays plot and optionally saves files)

#### `stitch_core_sections(core_structure, mother_dir, stitchbuffer=10, width_start_pct=0.25, width_end_pct=0.75)`

Stitches multiple core section images by processing RGB profiles with section-specific parameters and combining results into continuous arrays.

**Parameters:**
- `core_structure` (dict): Dictionary with filenames as keys and processing parameters as values
- `mother_dir` (str): Base directory path containing image files
- `stitchbuffer` (int, default=10): Bin rows to remove at stitching edges
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis strip boundaries

**Returns:**
- `tuple`: (all_depths, all_r, all_g, all_b, all_r_std, all_g_std, all_b_std, all_lum, all_lum_std, stitched_image) containing continuous RGB data and combined image

### CT Image Processing (`ct_processing.py`)

#### `load_dicom_files(dir_path, force=True)`

Loads DICOM files from directory and creates 3D volume data with pixel spacing and slice thickness information.

**Parameters:**
- `dir_path` (str): Directory path containing DICOM files
- `force` (bool, default=True): Whether to ignore problematic files and continue processing

**Returns:**
- `tuple`: (volume_data, pixel_spacing_x, pixel_spacing_y, slice_thickness) containing 3D array and spatial metadata

#### `get_slice(volume, index, axis=0)`

Extracts 2D slice from 3D volume along specified axis for viewing different orientations of CT data.

**Parameters:**
- `volume` (array-like): 3D numpy array with shape (height, width, depth)
- `index` (int): Slice index along specified axis
- `axis` (int, default=0): Axis for slice extraction (0=sagittal, 1=coronal, 2=axial)

**Returns:**
- `array-like`: 2D numpy array of extracted slice

#### `get_brightness_trace(slice_data, axis=0, width_start_pct=0.25, width_end_pct=0.75)`

Calculates brightness trace along specified axis within central strip of slice, excluding edge artifacts.

**Parameters:**
- `slice_data` (array-like): 2D slice data array
- `axis` (int, default=0): Direction for trace calculation (0=vertical, 1=horizontal)
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Central strip boundaries

**Returns:**
- `array-like`: Brightness values along specified axis

#### `get_brightness_stats(slice_data, axis=1, width_start_pct=0.25, width_end_pct=0.75)`

Computes mean and standard deviation of brightness along specified axis within central analysis strip.

**Parameters:**
- `slice_data` (array-like): 2D slice data array
- `axis` (int, default=1): Statistical calculation direction
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis strip boundaries

**Returns:**
- `tuple`: (mean_values, std_values) containing brightness statistics along axis

#### `process_brightness_data(slice_data, px_spacing_y, trim_top, trim_bottom, min_brightness=400, buffer=5, width_start_pct=0.25, width_end_pct=0.75)`

Processes CT slice data with trimming, filtering, and masking operations to extract clean brightness statistics.

**Parameters:**
- `slice_data` (array-like): 2D CT scan slice data
- `px_spacing_y` (float): Pixel spacing in y direction (mm/pixel)
- `trim_top, trim_bottom` (int): Pixels to trim from slice edges
- `min_brightness` (float, default=400): Minimum threshold for masking
- `buffer` (float, default=5): Buffer size in mm around masked values
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis strip boundaries

**Returns:**
- `tuple`: (brightness, stddev, trimmed_slice) containing masked brightness values and processed slice

#### `find_best_overlap(curve1, curve2, min_overlap=20, max_overlap=450)`

Determines optimal overlap between two brightness curves by maximizing correlation and peak matching scores.

**Parameters:**
- `curve1, curve2` (array-like): Brightness curves to be overlapped
- `min_overlap, max_overlap` (int, default=20, 450): Overlap length constraints in pixels

**Returns:**
- `tuple`: (best_overlap, max_score) containing optimal overlap length and correlation score

#### `stitch_curves(bright1, bright2, std1, std2, px_spacing_y1, px_spacing_y2, min_overlap=20, max_overlap=450)`

Stitches two brightness curves by finding optimal overlap and creating continuous depth-aligned profiles.

**Parameters:**
- `bright1, bright2` (array-like): Brightness curves from two scans
- `std1, std2` (array-like): Standard deviation curves from two scans
- `px_spacing_y1, px_spacing_y2` (float): Pixel spacing for each scan
- `min_overlap, max_overlap` (int, default=20, 450): Overlap constraints

**Returns:**
- `tuple`: (final_overlap, od1, od2, st_bright, st_std, st_depth, bright2_shifted, std2_shifted) containing stitching parameters and merged data

#### `process_single_scan(data_dir, params, segment, scan_name, width_start_pct=0.25, width_end_pct=0.75, max_value_side_trim=1200)`

Processes complete workflow for single CT scan from DICOM loading through brightness extraction and visualization.

**Parameters:**
- `data_dir` (str): Directory containing DICOM files
- `params` (dict): Processing parameters (trim_top, trim_bottom, min_brightness, buffer)
- `segment, scan_name` (str): Identifiers for core segment and scan
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis strip boundaries
- `max_value_side_trim` (float, default=1200): Threshold for automatic side trimming

**Returns:**
- `tuple`: (brightness, stddev, trimmed_slice, px_spacing_x, px_spacing_y) containing processed data and metadata

#### `process_two_scans(segment_data, segment, ct_data_dir, width_start_pct=0.25, width_end_pct=0.75, max_value_side_trim=1200, min_overlap=20, max_overlap=450)`

Processes and stitches two CT scans for single core segment with complete workflow from processing through visualization.

**Parameters:**
- `segment_data` (dict): Dictionary containing scan names and processing parameters
- `segment` (str): Core segment identifier
- `ct_data_dir` (str): Base directory containing scan subdirectories
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis boundaries
- `max_value_side_trim` (float, default=1200): Automatic trimming threshold
- `min_overlap, max_overlap` (int, default=20, 450): Stitching overlap constraints

**Returns:**
- `tuple`: (st_bright_re, st_std_re, st_depth_re, st_slice, pixel_spacing) containing stitched data and combined slice

#### `ct_process_and_stitch(data_reading_structure, ct_data_dir, width_start_pct=0.15, width_end_pct=0.85, max_value_side_trim=1300, min_overlap=20, max_overlap=400, vmin=None, vmax=None, save_csv=True, output_csv=None, total_length_cm=None)`

Orchestrates complete processing workflow for multi-segment core with rescaling to match RGB dimensions, final stitching, and optional CSV export.

**Parameters:**
- `data_reading_structure` (dict or list): Core structure definition with segment parameters including RGB target dimensions
- `ct_data_dir` (str): Base directory path containing all segment subdirectories
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis strip boundaries
- `max_value_side_trim` (float, default=1200): Automatic trimming threshold
- `min_overlap, max_overlap` (int, default=20, 450): Stitching overlap constraints
- `vmin, vmax` (float, optional): Colormap scaling values for display
- `save_csv` (bool, default=True): Whether to save results to CSV file
- `output_csv` (str, optional): Full path for output CSV file (required if save_csv=True)
- `total_length_cm` (float, optional): Total core length in centimeters for depth conversion (required if save_csv=True)

**Returns:**
- `tuple`: (final_stitched_slice, final_stitched_brightness, final_stitched_stddev, final_stitched_depth, px_spacing_x, px_spacing_y) containing complete core data

**Raises:**
- `ValueError`: If save_csv is True but output_csv or total_length_cm is not specified

### CT Image Plotting (`ct_plotting.py`)

#### `display_slice(slice_data, px_spacing_x, px_spacing_y, core_name='', depth_units='mm')`

Display CT slice with proper aspect ratio and axis labels.

**Parameters:**
- `slice_data` (array-like): 2D CT scan slice data
- `px_spacing_x, px_spacing_y` (float): Pixel spacing values
- `core_name` (str, default=''): Core identifier for title
- `depth_units` (str, default='mm'): Units for depth axis

**Returns:**
- None (displays plot)

#### `plot_ctimg_curves(slice_data, brightness, stddev, depths, px_spacing_x, px_spacing_y, core_name='', depth_units='mm')`

Display CT slices with brightness traces and standard deviation plots.

**Parameters:**
- `slice_data` (array-like): 2D CT scan slice data
- `brightness` (array-like): Brightness values
- `stddev` (array-like): Standard deviation values
- `depths` (array-like): Depth values
- `px_spacing_x, px_spacing_y` (float): Pixel spacing values
- `core_name` (str, default=''): Core identifier for title
- `depth_units` (str, default='mm'): Units for depth axis

**Returns:**
- None (displays plot)

#### `plot_stitched_curves(st_bright, st_std, st_depth, core_name='', depth_units='cm')`

Plot stitched brightness curves with standard deviation shading.

**Parameters:**
- `st_bright` (array-like): Stitched brightness values
- `st_std` (array-like): Stitched standard deviation values
- `st_depth` (array-like): Stitched depth values
- `core_name` (str, default=''): Core identifier for title
- `depth_units` (str, default='cm'): Units for depth axis

**Returns:**
- None (displays plot)

### Machine Learning Gap Filling (`gap_filling.py`)

#### `preprocess_core_data(data_config)`

Preprocesses core data by cleaning and scaling depth values using configurable parameters. All processing actions are driven by the data_config content.

**Parameters:**
- `data_config` (dict): Configuration dictionary containing:
  - `depth_column`: Primary depth column name
  - `column_configs`: Dictionary of data type configurations with thresholds
  - `mother_dir`: Base directory path
  - `clean_output_folder`: Output folder for cleaned data
  - `input_file_paths`: Dictionary of input file paths by data type
  - `clean_file_paths`: Dictionary of output file paths by data type
  - `core_length`: Target core length for scaling

**Returns:**
- None (saves cleaned data files to the specified output directory)

#### `plot_core_logs(data_config, file_type='clean', title=None)`

Plot core logs using fully configurable parameters from data_config. Creates subplot panels for different types of core data (images and logs) based on the configuration provided.

**Parameters:**
- `data_config` (dict): Configuration dictionary containing plotting parameters
- `file_type` (str, default='clean'): Type of data files to plot ('clean' or 'filled')
- `title` (str, optional): Custom title for the plot. If None, generates default title

**Returns:**
- `tuple`: (fig, axes) - matplotlib figure and axes objects

#### `plot_filled_data(target_log, original_data, filled_data, data_config, ML_type='ML')`

Plot original and ML-filled data for a given log using configurable parameters. Creates a horizontal plot showing the original data overlaid with ML-filled gaps, including uncertainty shading if available.

**Parameters:**
- `target_log` (str): Name of the log to plot
- `original_data` (pandas.DataFrame): Original data containing the log with gaps
- `filled_data` (pandas.DataFrame): Data with ML-filled gaps
- `data_config` (dict): Configuration containing all parameters including depth column, plot labels, etc.
- `ML_type` (str, default='ML'): Type of ML method used for title

**Returns:**
- None (displays the plot directly)

#### `fill_gaps_with_ml(target_log, All_logs, data_config, output_csv=True, merge_tolerance=3.0, ml_method='xgblgbm')`

Fill gaps in target data using specified ML method. Prepares feature data, applies the specified machine learning method, and fills gaps in the target log data.

**Parameters:**
- `target_log` (str): Name of the target column to fill gaps in
- `All_logs` (dict): Dictionary of dataframes containing feature data and target data
- `data_config` (dict): Configuration containing all parameters including file paths, core info, etc.
- `output_csv` (bool, default=True): Whether to output filled data to CSV file
- `merge_tolerance` (float, default=3.0): Maximum allowed difference in depth for merging rows
- `ml_method` (str, default='xgblgbm'): ML method to use - 'rf', 'rftc', 'xgb', 'xgblgbm'

**Returns:**
- `tuple`: (target_data_filled, gap_mask) containing filled data and gap locations

#### `process_and_fill_logs(data_config, ml_method='xgblgbm')`

Process and fill gaps in log data using ML methods with fully configurable parameters. Orchestrates the complete ML-based gap filling process for all configured log data types.

**Parameters:**
- `data_config` (dict): Configuration containing all parameters including data paths and column configurations
- `ml_method` (str, default='xgblgbm'): ML method to use - 'rf', 'rftc', 'xgb', 'xgblgbm'
  - 'rf': Random Forest
  - 'rftc': Random Forest with Trend Constraints
  - 'xgb': XGBoost
  - 'xgblgbm': XGBoost + LightGBM ensemble

**Returns:**
- None (saves filled data files and displays progress information)

#### Helper Functions

**`prepare_feature_data(target_log, All_logs, merge_tolerance, data_config)`**
- Prepares merged feature data for ML training using configurable parameters

**`apply_feature_weights(X, data_config)`**
- Applies feature weights using configurable parameters from data_config

**`adjust_gap_predictions(df, gap_mask, ml_preds, target_log, data_config)`**
- Adjusts ML predictions for gap rows to blend with linear interpolation between boundaries

**`train_model(model)`**
- Helper function for parallel model training

### Gap Filling Plots (`gap_filling_plots.py`)

#### `plot_core_logs(data_config, file_type='clean', title=None)`

Plot core logs using fully configurable parameters from data_config.

**Parameters:**
- `data_config` (dict): Configuration dictionary containing plotting parameters
- `file_type` (str, default='clean'): Type of data files to plot ('clean' or 'filled')
- `title` (str, optional): Custom title for the plot

**Returns:**
- `tuple`: (fig, axes) - matplotlib figure and axes objects

#### `plot_filled_data(target_log, original_data, filled_data, data_config, ML_type='ML')`

Plot original and ML-filled data for a given log.

**Parameters:**
- `target_log` (str): Name of the log to plot
- `original_data` (pandas.DataFrame): Original data containing the log with gaps
- `filled_data` (pandas.DataFrame): Data with ML-filled gaps
- `data_config` (dict): Configuration containing all parameters
- `ML_type` (str, default='ML'): Type of ML method used for title

**Returns:**
- None (displays the plot)

### Datum Picker (`datum_picker.py`)

#### `pick_stratigraphic_levels(md, log, core_img_1=None, core_img_2=None, core_name="", csv_filename=None)`

Creates an interactive matplotlib environment for manually picking stratigraphic boundaries and datum levels with real-time visualization and CSV export.

**Parameters:**
- `md` (array-like): Depth values for x-axis data
- `log` (array-like): Log data for y-axis data (typically normalized 0-1)
- `core_img_1` (numpy.ndarray, optional): First core image data (e.g., RGB image)
- `core_img_2` (numpy.ndarray, optional): Second core image data (e.g., CT image)
- `core_name` (str, default=""): Name of the core for display in plot title
- `csv_filename` (str, optional): Full path/filename for the output CSV file

**Interactive Controls:**
- Left-click: Add depth point
- Number keys (0-9): Change current category
- Delete/Backspace: Remove last point
- Enter: Finish selection and save
- Pan/Zoom tools: Temporarily disable point selection

**Returns:**
- `tuple`: (picked_depths, categories) - Lists of picked depth values and their categories

#### `create_interactive_figure(md, log, core_img_1=None, core_img_2=None, miny=0, maxy=1)`

Creates a matplotlib figure with subplots for core images and log data visualization, optimized for interactive boundary picking.

**Parameters:**
- `md` (array-like): Depth values for x-axis data
- `log` (array-like): Log data for y-axis data
- `core_img_1` (numpy.ndarray, optional): First core image data
- `core_img_2` (numpy.ndarray, optional): Second core image data
- `miny` (float, default=0): Minimum y-axis limit for log plot
- `maxy` (float, default=1): Maximum y-axis limit for log plot

**Returns:**
- `tuple`: (figure, axes) - Matplotlib figure and the interactive axes object

#### `onclick_boundary(event, xs, lines, ax, toolbar, categories, current_category, status_text=None)`

Handles mouse click events for interactive boundary picking.

**Parameters:**
- `event` (matplotlib event object): Mouse click event
- `xs` (list): List to store x-coordinate values
- `lines` (list): List to store matplotlib line objects
- `ax` (matplotlib.axes.Axes): The axes object where clicking occurs
- `toolbar` (matplotlib toolbar object): Navigation toolbar
- `categories` (list): List to store category values
- `current_category` (list): Single-element list containing current category
- `status_text` (matplotlib.text.Text, optional): Text object for status messages

**Returns:**
- None (modifies input lists and plot in place)

#### `onkey_boundary(event, xs, lines, ax, cid, toolbar, categories, current_category, csv_filename=None, status_text=None)`

Handles keyboard events for interactive boundary picking.

**Parameters:**
- `event` (matplotlib event object): Keyboard event
- `xs` (list): List storing x-coordinate values
- `lines` (list): List storing matplotlib line objects
- `ax` (matplotlib.axes.Axes): The axes object
- `cid` (list): List containing connection IDs
- `toolbar` (matplotlib toolbar object): Navigation toolbar
- `categories` (list): List storing category values
- `current_category` (list): Single-element list with current category
- `csv_filename` (str, optional): Output CSV file path
- `status_text` (matplotlib.text.Text, optional): Text for status messages

**Returns:**
- None (modifies input lists and saves data when 'enter' is pressed)

#### `get_category_color(category)`

Maps category identifiers to specific colors for visualization.

**Parameters:**
- `category` (str or int): Category identifier

**Returns:**
- `str`: Color string compatible with matplotlib

### Core Display (`core_display.py`)

#### `plot_core_data(md, log, title, rgb_img=None, ct_img=None, figsize=(20, 4), label_name=None, available_columns=None, is_multilog=False, picked_depths=None, picked_categories=None, picked_uncertainties=None, show_category=None, show_bed_number=False)`

Plots core data with optional RGB and CT images and support for multiple log types with category visualization.

**Parameters:**
- `md` (array-like): Array of depth values
- `log` (array-like): Array of log values (1D or 2D)
- `title` (str): Title for the plot
- `rgb_img, ct_img` (array-like, optional): RGB and CT image arrays
- `figsize` (tuple, default=(20, 4)): Figure size
- `label_name` (str, optional): Name for log curve label
- `available_columns` (list, optional): Names of log columns for multilogs
- `is_multilog` (bool, default=False): Whether log contains multiple columns
- `picked_depths` (list, optional): List of picked depths for category visualization
- `picked_categories` (list, optional): Categories for picked_depths
- `picked_uncertainties` (list, optional): Uncertainties for each picked depth
- `show_category` (list, optional): Specific categories to show
- `show_bed_number` (bool, default=False): Display bed numbers

**Returns:**
- `tuple`: (fig, plot_ax) containing the figure and main plotting axis

## Utils Module (`pyCoreRelator.utils`)

### Data Loader (`data_loader.py`)

#### `load_log_data(log_paths, img_paths, log_columns, depth_column='SB_DEPTH_cm', normalize=True, column_alternatives=None)`

Loads and preprocesses well log data and core images from multiple file sources with automatic normalization and resampling.

**Parameters:**
- `log_paths` (dict): Dictionary mapping log names to file paths
- `img_paths` (dict): Dictionary mapping image types ('rgb', 'ct') to file paths
- `log_columns` (list): List of log column names to load from files
- `depth_column` (str, default='SB_DEPTH_cm'): Name of depth column in data files
- `normalize` (bool, default=True): Whether to normalize log data to [0,1] range
- `column_alternatives` (dict, optional): Alternative column names to try if primary names not found

**Returns:**
- `tuple`: (log, md, available_columns, rgb_img, ct_img) containing loaded data and images

#### `load_core_age_constraints(core_name, age_base_path, consider_adjacent_core=False, data_columns=None, mute_mode=False)`

Loads age constraint data from CSV files with support for adjacent cores and flexible column mapping.

**Parameters:**
- `core_name` (str): Name of the core to load age constraints for
- `age_base_path` (str): Base directory path containing age constraint CSV files
- `consider_adjacent_core` (bool, default=False): Whether to also search for similar core names
- `data_columns` (list, optional): Specific column names to load from CSV files
- `mute_mode` (bool, default=False): If True, suppress print output

**Returns:**
- `dict`: Age constraint data with keys: 'depths', 'ages', 'pos_errors', 'neg_errors', 'core', 'in_sequence_flags'

#### `resample_datasets(datasets, target_resolution_factor=2)`

Resamples multiple datasets to a common depth scale with improved resolution for consistent analysis.

**Parameters:**
- `datasets` (list): List of dictionaries containing depth and data arrays
- `target_resolution_factor` (float, default=2): Factor to improve resolution by dividing lowest resolution

**Returns:**
- `dict`: Dictionary with resampled data arrays and common depth scale

#### `load_sequential_mappings(csv_path)`

Loads sequential correlation mappings from CSV file in compact format for visualization and analysis.

**Parameters:**
- `csv_path` (str): Path to CSV file containing sequential mappings

**Returns:**
- `list`: List of correlation paths as lists of (index_a, index_b) tuples

### Path Processing (`path_processing.py`)

#### `is_subset_or_superset(path_info, other_path_info, early_terminate=True)`

Checks if one correlation path is a subset or superset of another for path filtering and deduplication.

**Parameters:**
- `path_info, other_path_info` (dict): Path information dictionaries containing length and segment data
- `early_terminate` (bool, default=True): Whether to use early termination for efficiency

**Returns:**
- `tuple`: (is_subset, is_superset) boolean flags indicating relationship between paths

#### `filter_against_existing(new_path, filtered_paths, group_writer)`

Filters new correlation path against existing paths to remove duplicates and maintain unique path collection.

**Parameters:**
- `new_path` (dict): New path information to be filtered
- `filtered_paths` (list): Existing collection of filtered paths
- `group_writer`: Writer object for outputting filtered results

**Returns:**
- `tuple`: (is_valid, paths_to_remove, updated_count) indicating filter results

### Plotting (`plotting.py`)

#### `plot_segment_pair_correlation(log_a, log_b, md_a, md_b, **kwargs)`

Creates comprehensive visualization of DTW correlation between log segments with support for single and multiple segment pairs.

**Parameters:**
- `log_a, log_b` (array-like): Full log data arrays (single or multidimensional)
- `md_a, md_b` (array-like): Measured depth arrays for full logs
- `segment_pairs` (list, optional): List of (a_idx, b_idx) tuples for multi-segment mode
- `dtw_results` (dict, optional): DTW results dictionary for multi-segment mode
- `wp` (np.ndarray, optional): Warping path for single segment mode
- `a_start, a_end, b_start, b_end` (int, optional): Segment boundaries for single segment mode
- `visualize_pairs` (bool, default=True): Whether to color-code segment pairs
- `age_consideration` (bool, default=False): Whether to display age information
- `rgb_img_a, ct_img_a, rgb_img_b, ct_img_b` (array-like, optional): Core images
- `available_columns_a, available_columns_b` (list, optional): Column names for multilogs

**Returns:**
- `matplotlib.figure.Figure`: Complete correlation visualization figure

#### `visualize_combined_segments(log_a, log_b, md_a, md_b, segment_pairs, dtw_results, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, **kwargs)`

Display segment correlations overlaid on log plots.

**Parameters:**
- `log_a, log_b` (array-like): Log data arrays
- `md_a, md_b` (array-like): Measured depth arrays
- `segment_pairs` (list): List of segment pair tuples to visualize
- `dtw_results` (dict): DTW results dictionary
- `segments_a, segments_b` (list): Segment definitions
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- Additional kwargs for images and visualization options

**Returns:**
- `matplotlib.figure.Figure`: Visualization figure

#### `plot_correlation_distribution(quality_dict, save_path=None)`

Visualize and statistically analyze the distributions of the correlation quality metrics.

**Parameters:**
- `quality_dict` (dict): Dictionary of quality metrics
- `save_path` (str, optional): Path to save the figure

**Returns:**
- None (displays plot and optionally saves)

#### `calculate_quality_comparison_t_statistics(real_data, syn_data, quality_indices)`

Calculate t-statistics for quality metric comparisons.

**Parameters:**
- `real_data` (dict): Real correlation quality data
- `syn_data` (dict): Synthetic correlation quality data
- `quality_indices` (list): List of quality metrics to compare

**Returns:**
- `dict`: T-statistics and p-values for each quality metric

#### `plot_quality_comparison_t_statistics(t_stats_dict, save_path=None)`

Plot quality metric comparison results with statistical analysis.

**Parameters:**
- `t_stats_dict` (dict): Dictionary of t-statistics
- `save_path` (str, optional): Path to save the figure

**Returns:**
- None (displays plot and optionally saves)

### Matrix Plots (`matrix_plots.py`)

#### `plot_dtw_matrix_with_paths(D, paths, log1=None, log2=None, save_path=None, **kwargs)`

Visualize DTW cost matrices with correlation paths.

**Parameters:**
- `D` (np.ndarray): DTW cost matrix
- `paths` (list): List of warping paths to overlay
- `log1, log2` (array-like, optional): Log data for additional context
- `save_path` (str, optional): Path to save the figure
- Additional kwargs for plot customization

**Returns:**
- `matplotlib.figure.Figure`: DTW matrix visualization

### Animation (`animation.py`)

#### `visualize_dtw_results_from_csv(csv_path, log_a, log_b, md_a, md_b, segments_a, segments_b, dtw_results, output_gif=None, **kwargs)`

Generate animated correlation sequences from results.

**Parameters:**
- `csv_path` (str): Path to CSV containing correlation results
- `log_a, log_b` (array-like): Log data arrays
- `md_a, md_b` (array-like): Measured depth arrays
- `segments_a, segments_b` (list): Segment definitions
- `dtw_results` (dict): DTW results dictionary
- `output_gif` (str, optional): Path to save animated GIF
- Additional kwargs for animation options

**Returns:**
- None (creates and saves animation)

### Helpers (`helpers.py`)

#### `find_nearest_index(depth_array, depth_value)`

Finds the index in a depth array that corresponds to the closest depth value to a target depth.

**Parameters:**
- `depth_array` (array-like): Array of depth values to search
- `depth_value` (float): Target depth value to find

**Returns:**
- `int`: Index in depth_array with closest value to target depth