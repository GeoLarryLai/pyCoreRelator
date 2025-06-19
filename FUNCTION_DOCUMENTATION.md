# pyCoreRelator Function Documentation

This document provides detailed documentation for all functions in the pyCoreRelator package.

## Core Module (`pyCoreRelator.core`)

### DTW Analysis (`dtw_analysis.py`)

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

#### `run_comprehensive_dtw_analysis(log_a, log_b, md_a, md_b, **kwargs)`

Runs comprehensive DTW analysis with integrated age correlation functionality, processing all possible segment pairs and optionally filtering based on age constraints.

**Parameters:**
- `log_a, log_b` (array-like): Well log data for cores A and B
- `md_a, md_b` (array-like): Measured depth arrays corresponding to the logs
- `picked_depths_a, picked_depths_b` (list, optional): User-picked depth boundaries
- `top_bottom` (bool, default=True): Whether to include top and bottom boundaries
- `top_depth` (float, default=0.0): Depth value for top boundary
- `independent_dtw` (bool, default=False): Whether to process multidimensional logs independently
- `create_dtw_matrix` (bool, default=True): Whether to generate DTW matrix visualization
- `visualize_pairs` (bool, default=True): Whether to create segment pair visualizations
- `age_consideration` (bool, default=False): Whether to apply age-based filtering
- `ages_a, ages_b` (dict, optional): Age constraint dictionaries with keys: depths, ages, pos_uncertainties, neg_uncertainties
- `restricted_age_correlation` (bool, default=True): Whether to use strict age correlation filtering
- `exclude_deadend` (bool, default=True): Whether to filter out dead-end segment pairs

**Returns:**
- `dtw_results` (dict): Dictionary containing DTW results for each valid segment pair
- `valid_dtw_pairs` (set): Set of valid segment pair indices after all filtering
- `segments_a, segments_b` (list): Lists of segment boundaries for each core
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `dtw_distance_matrix_full` (np.ndarray): Full DTW distance matrix for reference

### Quality Metrics (`quality_metrics.py`)

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

Calculates interpolated or extrapolated ages for picked depths based on age constraints using various uncertainty propagation methods.

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

### Segment Analysis (`segment_analysis.py`)

#### `find_all_segments(log_a, log_b, md_a, md_b, picked_depths_a=None, picked_depths_b=None, top_bottom=True, top_depth=0.0)`

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

## Utilities Module (`pyCoreRelator.utils`)

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
- `log` (np.ndarray): Log data with shape (n_samples, n_logs) or (n_samples,) for single log
- `md` (np.ndarray): Measured depths corresponding to log data
- `available_columns` (list): Names of successfully loaded log columns
- `rgb_img, ct_img` (np.ndarray or None): RGB and CT images if available

#### `resample_datasets(datasets, target_resolution_factor=2)`

Resamples multiple datasets to a common depth scale with improved resolution for consistent analysis.

**Parameters:**
- `datasets` (list): List of dictionaries containing depth and data arrays
- `target_resolution_factor` (float, default=2): Factor to improve resolution by dividing lowest resolution

**Returns:**
- `dict`: Dictionary with resampled data arrays and common depth scale

#### `plot_core_data(md, log, title, rgb_img=None, ct_img=None, boundaries=None, figsize=(20, 4), label_name=None, available_columns=None, is_multilog=False)`

Creates comprehensive visualization of core data including logs, images, and boundaries with support for both single and multiple log types.

**Parameters:**
- `md` (array-like): Depth values for plotting
- `log` (array-like): Log values (single log or multidimensional)
- `title` (str): Plot title
- `rgb_img, ct_img` (array-like, optional): RGB and CT image arrays
- `boundaries` (array-like, optional): Depth points for boundary markers
- `figsize` (tuple, default=(20, 4)): Figure size (width, height)
- `label_name` (str, optional): Label for single log curve
- `available_columns` (list, optional): Column names for multidimensional logs
- `is_multilog` (bool, default=False): Whether log contains multiple columns

**Returns:**
- `fig`: Matplotlib figure object containing the visualization

### Path Processing (`path_processing.py`)

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

#### `load_sequential_mappings(csv_path)`

Loads sequential correlation mappings from CSV file in compact format for visualization and analysis.

**Parameters:**
- `csv_path` (str): Path to CSV file containing sequential mappings

**Returns:**
- `list`: List of correlation paths as lists of (index_a, index_b) tuples

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
- `bool`: Whether the new path was accepted (True) or filtered out (False)

### Helpers (`helpers.py`)

#### `find_nearest_index(depth_array, depth_value)`

Finds the index in a depth array that corresponds to the closest depth value to a target depth.

**Parameters:**
- `depth_array` (array-like): Array of depth values to search
- `depth_value` (float): Target depth value to find

**Returns:**
- `int`: Index in depth_array with closest value to target depth

## Visualization Module (`pyCoreRelator.visualization`)

### Plotting (`plotting.py`)

#### `plot_segment_pair_correlation(log_a, log_b, md_a, md_b, **kwargs)`

Creates comprehensive visualization of DTW correlation between log segments with support for single and multiple segment pairs, including RGB/CT images and age constraints.

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

#### `visualize_combined_segments(log_a, log_b, md_a, md_b, dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full, segment_pairs_to_combine, **kwargs)`

Creates combined visualization of multiple segment correlations with both correlation plot and DTW matrix display.

**Parameters:**
- `log_a, log_b` (array-like): Full log data arrays
- `md_a, md_b` (array-like): Measured depth arrays  
- `dtw_results` (dict): DTW results for all segment pairs
- `valid_dtw_pairs` (set): Set of valid segment pair indices
- `segments_a, segments_b` (list): Segment definitions
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `dtw_distance_matrix_full` (np.ndarray): Full DTW distance matrix
- `segment_pairs_to_combine` (list): Specific segment pairs to visualize together
- `correlation_save_path, matrix_save_path` (str): Paths for saving output figures

**Returns:**
- `tuple`: (correlation_figure, matrix_figure) matplotlib figure objects

#### `plot_correlation_distribution(csv_file, target_mapping_id=None, quality_index=None, save_png=True, png_filename=None, core_a_name=None, core_b_name=None, no_bins=50)`

Creates histogram distribution plot of correlation quality metrics from complete path analysis results.

**Parameters:**
- `csv_file` (str): Path to CSV file containing correlation path results
- `target_mapping_id` (int, optional): Specific mapping ID to highlight in distribution
- `quality_index` (str, optional): Specific quality metric to plot ('corr_coef', 'norm_dtw', etc.)
- `save_png` (bool, default=True): Whether to save plot as PNG file
- `png_filename` (str, optional): Custom filename for saved plot
- `core_a_name, core_b_name` (str, optional): Core names for plot labeling
- `no_bins` (int, default=50): Number of histogram bins

**Returns:**
- `matplotlib.figure.Figure`: Distribution plot figure

### Matrix Plots (`matrix_plots.py`)

#### `plot_dtw_matrix_with_paths(dtw_distance_matrix_full, mode=None, **kwargs)`

Creates comprehensive DTW distance matrix visualization with various path overlay options and age constraint lines.

**Parameters:**
- `dtw_distance_matrix_full` (np.ndarray): Full DTW distance matrix to visualize
- `mode` (str): Visualization mode - 'segment_paths', 'combined_path', or 'all_paths_colored'
- `valid_dtw_pairs` (set, optional): Valid segment pairs for 'segment_paths' mode
- `combined_wp` (np.ndarray, optional): Combined warping path for 'combined_path' mode
- `sequential_mappings_csv` (str, optional): CSV file for 'all_paths_colored' mode
- `age_constraint_a_depths, age_constraint_b_depths` (list, optional): Age constraint depths
- `age_constraint_a_source_cores, age_constraint_b_source_cores` (list, optional): Source core names for constraints
- `color_metric` (str, optional): Quality metric for path coloring
- `output_filename` (str, optional): Path to save output figure

**Returns:**
- `matplotlib.figure.Figure`: DTW matrix visualization with overlaid paths and constraints

### Animation (`animation.py`)

#### `create_segment_dtw_animation(log_a, log_b, md_a, md_b, dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, **kwargs)`

Creates animated GIF showing DTW correlations between segment pairs with optional age information display.

**Parameters:**
- `log_a, log_b` (array-like): Full log data arrays
- `md_a, md_b` (array-like): Measured depth arrays
- `dtw_results` (dict): DTW results for all segment pairs
- `valid_dtw_pairs` (set): Set of valid segment pairs to animate
- `segments_a, segments_b` (list): Segment definitions
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `output_filename` (str, default='SegmentPair_DTW_animation.gif'): Output GIF filename
- `max_frames` (int, default=100): Maximum number of animation frames
- `age_consideration` (bool, default=False): Whether to include age information
- `keep_frames` (bool, default=True): Whether to preserve individual frame files

**Returns:**
- `str`: Message indicating animation creation status and location

#### `visualize_dtw_results_from_csv(csv_path, log_a, log_b, md_a, md_b, dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full, **kwargs)`

Creates animated visualizations from CSV file containing complete correlation paths with both correlation and matrix views.

**Parameters:**
- `csv_path` (str): Path to CSV file containing correlation mapping results
- `log_a, log_b` (array-like): Full log data arrays
- `md_a, md_b` (array-like): Measured depth arrays
- `dtw_results` (dict): DTW results for quality assessment
- `valid_dtw_pairs` (set): Set of valid segment pairs
- `segments_a, segments_b` (list): Segment definitions
- `depth_boundaries_a, depth_boundaries_b` (list): Depth boundary indices
- `dtw_distance_matrix_full` (np.ndarray): Full DTW distance matrix
- `max_frames` (int, default=150): Maximum number of animation frames
- `creategif` (bool, default=True): Whether to create GIF animations
- `correlation_gif_output_filename, matrix_gif_output_filename` (str): Output GIF filenames

**Returns:**
- `tuple`: (correlation_status, matrix_status) indicating animation creation results

#### `create_gif(frame_folder, output_filename, duration=300)`

Creates GIF animation from a folder of PNG frame images with memory-efficient processing.

**Parameters:**
- `frame_folder` (str): Path to folder containing PNG frame images
- `output_filename` (str): Path and filename for output GIF
- `duration` (int, default=300): Duration in milliseconds for each frame

**Returns:**
- `str`: Status message indicating GIF creation success and frame count 