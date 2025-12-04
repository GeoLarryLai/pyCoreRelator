# pyCoreRelator Function Documentation

This document provides detailed documentation for all functions in the pyCoreRelator package, organized by the current module structure for geological core correlation analysis.

## Analysis Module (`pyCoreRelator.analysis`)

### DTW Analysis (`dtw_core.py`)

#### `run_comprehensive_dtw_analysis(log_a, log_b, md_a, md_b, **kwargs)`

**ENHANCED** - Main function for segment-based DTW analysis with comprehensive age constraint integration, visualization capabilities, and performance optimizations.

**Parameters:**
- `log_a, log_b` (array-like): Well log data for cores A and B (1D or multidimensional)
- `md_a, md_b` (array-like): Measured depth arrays corresponding to the logs
- `picked_datum_a, picked_datum_b` (list, optional): User-picked depth boundaries for segmentation
- `core_a_name, core_b_name` (str, optional): Core identifiers for output files
- `top_bottom` (bool, default=True): Whether to include top and bottom boundaries automatically
- `top_depth` (float, default=0.0): Depth value for top boundary
- `independent_dtw` (bool, default=False): Whether to process multidimensional logs independently
- `create_dtw_matrix` (bool, default=False): Whether to generate DTW matrix visualization
- `visualize_pairs` (bool, default=True): Whether to create segment pair visualizations
- `visualize_segment_labels` (bool, default=False): Whether to show segment labels in visualizations
- `dtwmatrix_output_filename` (str, default='SegmentPair_DTW_matrix.png'): Filename for DTW matrix output
- `creategif` (bool, default=False): Whether to create animated GIF sequences
- `gif_output_filename` (str, default='SegmentPair_DTW_animation.gif'): Filename for animation output
- `max_frames` (int, default=100): Maximum number of frames in animation
- `debug` (bool, default=False): Enable debug output
- `color_interval_size` (float, default=10): Color interval size for visualizations
- `keep_frames` (bool, default=True): Keep individual animation frames
- `age_consideration` (bool, default=False): Whether to apply age-based filtering
- `ages_a, ages_b` (dict, optional): Age constraint dictionaries with interpolated ages for picked depths
- `restricted_age_correlation` (bool, default=True): Whether to use strict age correlation filtering
- `core_a_age_data, core_b_age_data` (dict, optional): Complete age constraint data from `load_core_age_constraints()`. Expected keys: 'in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors', 'core'. Required when `age_consideration=True`
- `dtw_distance_threshold` (float, default=None): Maximum allowed DTW distance for segment acceptance
- `exclude_deadend` (bool, default=True): Whether to filter out dead-end segment pairs
- `mute_mode` (bool, default=False): Whether to suppress print output for batch processing
- `pca_for_dependent_dtw` (bool, default=False): Use PCA for dependent multidimensional DTW (if False, uses conventional multidimensional DTW)
- `dpi` (int, default=None): Resolution for saved figures and GIF frames in dots per inch. If None, uses default (150)

**Returns:**
- `dict`: Dictionary containing all DTW analysis results with the following keys:
  - `dtw_correlation` (dict): DTW results for valid segment pairs (renamed from dtw_results)
  - `valid_dtw_pairs` (set): Set of valid segment pair indices after all filtering
  - `segments_a` (list): Segment definitions for log_a
  - `segments_b` (list): Segment definitions for log_b
  - `depth_boundaries_a` (list): Depth boundaries for log_a segments
  - `depth_boundaries_b` (list): Depth boundaries for log_b segments
  - `dtw_distance_matrix_full` (np.ndarray): Full DTW distance matrix for visualization

**Example:**
```python
# Load age constraint data
age_data_a = load_core_age_constraints('M9907-25PC', 'example_data/raw_data/C14age_data', data_columns)
age_data_b = load_core_age_constraints('M9907-23PC', 'example_data/raw_data/C14age_data', data_columns)

# Run DTW analysis with simplified age data parameters
dtw_result = run_comprehensive_dtw_analysis(
    log_a, log_b, md_a, md_b,
    picked_datum_a=picked_depths_a,
    picked_datum_b=picked_depths_b,
    age_consideration=True,
    ages_a=estimated_datum_ages_a,
    ages_b=estimated_datum_ages_b,
    core_a_age_data=age_data_a,  # Simplified parameter
    core_b_age_data=age_data_b,  # Simplified parameter
    core_a_name='M9907-25PC',
    core_b_name='M9907-23PC'
)
```

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

#### `find_best_mappings(input_mapping_csv, top_n=10, filter_shortest_dtw=True, metric_weight=None, core_a_picked_datums=None, core_b_picked_datums=None, core_a_interpreted_beds=None, core_b_interpreted_beds=None, dtw_result=None)`

Finds the best DTW mappings based on multiple quality metrics with configurable scoring. Supports two modes: standard best mappings finder and boundary correlation filtering mode.

**Parameters:**
- `input_mapping_csv` (str): Path to CSV file containing DTW mapping results
- `top_n` (int, default=10): Number of top mappings to return
- `filter_shortest_dtw` (bool, default=True): Whether to filter for shortest DTW path length first
- `metric_weight` (dict, optional): Dictionary of weights for different quality metrics. If None, uses default weights
- `core_a_picked_datums` (list, optional): Picked depth values for core A. Required for boundary correlation mode
- `core_b_picked_datums` (list, optional): Picked depth values for core B. Required for boundary correlation mode
- `core_a_interpreted_beds` (list, optional): Interpreted bed names for core A boundaries. Required for boundary correlation mode
- `core_b_interpreted_beds` (list, optional): Interpreted bed names for core B boundaries. Required for boundary correlation mode
- `dtw_result` (dict, optional): Dictionary containing DTW analysis results from `run_comprehensive_dtw_analysis()`. Expected keys: 'valid_dtw_pairs', 'segments_a', 'segments_b'. Required only for boundary correlation mode

**Returns:**
- `tuple`: (top_mapping_ids, top_mapping_pairs, top_mappings_df)
  - `top_mapping_ids` (list): List of best mapping IDs
  - `top_mapping_pairs` (list): List of segment pair combinations for each mapping
  - `top_mappings_df` (pandas.DataFrame): DataFrame containing top mappings with scores

**Example:**
```python
# Standard mode - find best mappings by quality metrics
top_ids, top_pairs, top_df = find_best_mappings(
    input_mapping_csv='mappings.csv',
    top_n=10,
    filter_shortest_dtw=True,
    metric_weight={'corr_coef': 1.0, 'norm_dtw': 1.0}
)

# Boundary correlation mode - find mappings that match interpreted bed correlations
dtw_result = run_comprehensive_dtw_analysis(...)
top_ids, top_pairs, top_df = find_best_mappings(
    input_mapping_csv='mappings.csv',
    core_a_picked_datums=picked_depths_a,
    core_b_picked_datums=picked_depths_b,
    core_a_interpreted_beds=interpreted_bed_a,
    core_b_interpreted_beds=interpreted_bed_b,
    dtw_result=dtw_result
)
```

### Age Models (`age_models.py`)

#### `calculate_interpolated_ages(picked_datum, age_data=None, **kwargs)`

**ENHANCED** - Calculates interpolated or extrapolated ages for picked depths based on age constraints using various uncertainty propagation methods. Supports both simplified `age_data` parameter and legacy individual parameters.

**Parameters:**
- `picked_datum` (list): List of picked depths in cm requiring age estimates
- `age_data` (dict, optional): Dictionary containing age constraint data from `load_core_age_constraints()`. If provided, this will be used instead of individual age constraint parameters. Expected keys: 'depths', 'ages', 'pos_errors', 'neg_errors', 'in_sequence_flags', 'core'
- `age_constraints_depths` (list, optional): Mean depths for age constraint points (not required if `age_data` is provided)
- `age_constraints_ages` (list, optional): Calibrated ages for constraint points (not required if `age_data` is provided)
- `age_constraints_pos_errors, age_constraints_neg_errors` (list, optional): Positive and negative age uncertainties (not required if `age_data` is provided)
- `age_constraints_in_sequence_flags` (list, optional): Boolean flags indicating which constraints are stratigraphically in-sequence (not required if `age_data` is provided)
- `age_constraint_source_core` (list, optional): Source core names for each age constraint (not required if `age_data` is provided)
- `uncertainty_method` (str, default='MonteCarlo'): Method for uncertainty propagation ('Linear', 'MonteCarlo', 'Gaussian')
- `n_monte_carlo` (int, default=10000): Number of Monte Carlo iterations for uncertainty estimation
- `top_age, top_age_pos_error, top_age_neg_error` (float): Age and uncertainties at top depth
- `bottom_depth` (float, optional): Depth at the bottom of the core in cm
- `show_plot` (bool, default=False): Whether to display age-depth model plot
- `export_csv` (bool, default=True): Whether to export results to CSV file
- `csv_filename` (str, optional): Full path for output CSV file
- `print_ages` (bool, default=True): If True, print age constraint data and estimated ages information. If False, suppress printing
- `core_name` (str, optional): Name of the core for plot title and file naming
- `mute_mode` (bool, default=False): If True, suppress all print outputs

**Returns:**
- `dict`: Dictionary containing interpolated ages and uncertainties for each depth with keys 'depths', 'ages', 'pos_uncertainties', 'neg_uncertainties', 'uncertainty_method'

**Example:**
```python
# Recommended method using age_data
age_data = load_core_age_constraints('M9907-25PC', 'example_data/raw_data/C14age_data', data_columns)
result = calculate_interpolated_ages(
    picked_datum=[10, 20, 30],
    age_data=age_data,
    uncertainty_method='MonteCarlo',
    core_name='M9907-25PC'
)
```

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

#### `find_all_segments(log_a, log_b, md_a, md_b, picked_datum_a=None, picked_datum_b=None, top_bottom=True, top_depth=0.0, mute_mode=False)`

Identifies segments in two logs using picked depths, creating consecutive boundary segments and single point segments for correlation analysis.

**Parameters:**
- `log_a, log_b` (array-like): Log data arrays for cores A and B
- `md_a, md_b` (array-like): Measured depth arrays corresponding to logs
- `picked_datum_a, picked_datum_b` (list, optional): User-picked depth values (not indices)
- `top_bottom` (bool, default=True): Whether to include top and bottom boundaries
- `top_depth` (float, default=0.0): Depth value for top boundary

**Returns:**
- `segments_a, segments_b` (list): Lists of segment boundary index pairs for each core
- `depth_boundaries_a, depth_boundaries_b` (list): Lists of depth boundary indices
- `depth_values_a, depth_values_b` (list): Lists of actual depth values used as boundaries

#### `find_complete_core_paths(dtw_result, log_a, log_b, **kwargs)`

Finds complete correlation paths spanning entire cores by connecting valid segment pairs from top to bottom.

**Parameters:**
- `dtw_result` (dict): Dictionary containing DTW analysis results from `run_comprehensive_dtw_analysis()`. Expected keys: 'dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full'
- `log_a, log_b` (array-like): Log data for path quality assessment
- `output_csv` (str, default="complete_core_paths.csv"): Output file for complete paths
- `debug` (bool, default=False): Whether to print detailed progress information
- `start_from_top_only` (bool, default=True): Whether to only consider paths starting from top segments
- `batch_size` (int, default=1000): Processing batch size for memory management
- `n_jobs` (int, default=-1): Number of parallel jobs (-1 uses all CPU cores)
- `shortest_path_search` (bool, default=True): Whether to prioritize shorter paths
- `shortest_path_level` (int, default=2): Number of shortest unique lengths to keep (higher = more segments)
- `max_search_path` (int, default=5000): Maximum number of paths to explore per segment pair to prevent memory overflow
- `output_metric_only` (bool, default=False): If True, only output quality metrics without full path details
- `mute_mode` (bool, default=False): Suppress all print output
- `pca_for_dependent_dtw` (bool, default=False): Use PCA for dependent DTW quality calculations

**Returns:**
- `str`: Path to output CSV file containing complete correlation paths with quality metrics

#### `diagnose_chain_breaks(dtw_result)`

Diagnoses connectivity issues in segment chains by identifying missing connections and isolated segments.

**Parameters:**
- `dtw_result` (dict): Dictionary containing DTW analysis results from `run_comprehensive_dtw_analysis()`. Expected keys: 'valid_dtw_pairs', 'segments_a', 'segments_b', 'depth_boundaries_a', 'depth_boundaries_b'

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

### Synthetic Stratigraphy (`syn_strat.py`)

#### `load_segment_pool(core_names, log_data_csv, log_data_type, picked_datum, depth_column, alternative_column_names=None, boundary_category=None, neglect_topbottom=True)`

Load segment pool data from turbidite database for synthetic core generation.

**Parameters:**
- `core_names` (list): List of core names to process
- `log_data_csv` (dict): Dictionary mapping core names to log file paths
- `log_data_type` (list): List of log column names to load
- `picked_datum` (dict): Dictionary mapping core names to picked depth file paths
- `depth_column` (str): Name of depth column
- `alternative_column_names` (dict, optional): Dictionary of alternative column names
- `boundary_category` (int, default=None): Category number for turbidite boundaries. If None, uses category 1 if available, otherwise uses the lowest available category
- `neglect_topbottom` (bool, default=True): If True, skip the first and last segments of each core

**Returns:**
- `tuple`: (seg_logs, seg_depths, seg_pool_metadata) containing turbidite log segments, depth segments, and loaded core metadata

#### `modify_segment_pool(segment_logs, segment_depths, remove_list=None)`

Remove unwanted segments from the pool data and return the modified pool.

**Parameters:**
- `segment_logs` (list): List of log data arrays (segments)
- `segment_depths` (list): List of depth arrays corresponding to each segment
- `remove_list` (list, optional): List of 1-based segment numbers to remove. If None or empty, no segments are removed

**Returns:**
- `tuple`: (modified_segment_logs, modified_segment_depths) containing remaining log and depth arrays

#### `create_synthetic_log(target_thickness, segment_logs, segment_depths, exclude_inds=None, repetition=False)`

Create synthetic log using turbidite database approach with picked depths at turbidite bases.

**Parameters:**
- `target_thickness` (float): Target thickness for the synthetic log
- `segment_logs` (list): List of turbidite log segments
- `segment_depths` (list): List of corresponding depth arrays
- `exclude_inds` (list, optional): Indices to exclude from selection
- `repetition` (bool, default=False): If True, allow reusing turbidite segments; if False, each segment can only be used once

**Returns:**
- `tuple`: (log, d, valid_picked_depths, inds)
  - `log` (numpy.ndarray): Synthetic log data array
  - `d` (numpy.ndarray): Depth values for the synthetic log
  - `valid_picked_depths` (list): List of boundary depth values
  - `inds` (list): Indices of segments used from the segment pool

**Example:**
```python
syn_log_a, syn_md_a, syn_depth_a, inds_a = create_synthetic_log(
    target_thickness=600,
    segment_logs=mod_seg_logs,
    segment_depths=mod_seg_depths,
    repetition=False
)
# syn_depth_a is a list of depth values (e.g., [0, 59.48, 92.21, ...])
```

#### `create_synthetic_core_pair(core_a_length, core_b_length, seg_logs, seg_depths, log_columns, repetition=False, plot_results=True, save_plot=False, plot_filename=None)`

Generate synthetic core pair (computation only) with optional plotting.

**Parameters:**
- `core_a_length` (float): Target length for core A
- `core_b_length` (float): Target length for core B
- `seg_logs` (list): List of turbidite log segments
- `seg_depths` (list): List of corresponding depth arrays
- `log_columns` (list): List of log column names for labeling
- `repetition` (bool, default=False): If True, allow reusing turbidite segments
- `plot_results` (bool, default=True): Whether to display the plot
- `save_plot` (bool, default=False): Whether to save the plot to file
- `plot_filename` (str, optional): Filename for saving plot (if save_plot=True)

**Returns:**
- `tuple`: (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a, synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)
  - `synthetic_log_a, synthetic_log_b` (numpy.ndarray): Synthetic log data arrays
  - `synthetic_md_a, synthetic_md_b` (numpy.ndarray): Depth values for synthetic logs
  - `inds_a, inds_b` (list): Indices of segments used from the segment pool
  - `synthetic_picked_a, synthetic_picked_b` (list): Lists of boundary depth values

#### `plot_synthetic_log(synthetic_log, synthetic_md, synthetic_picked_datum, log_data_type, title="Synthetic Log", save_plot=False, plot_filename=None)`

Plot a single synthetic log with turbidite boundaries.

**Parameters:**
- `synthetic_log` (array-like): Numpy array of log values (can be 1D or 2D for multiple log types)
- `synthetic_md` (array-like): Numpy array of depth values
- `synthetic_picked_datum` (list): List of turbidite boundary depths
- `log_data_type` (str or list): Name(s) of the log column(s) for labeling
- `title` (str, default="Synthetic Log"): Title for the plot
- `save_plot` (bool, default=False): Whether to save the plot to file
- `plot_filename` (str, optional): Filename for saving plot (if save_plot=True)

**Returns:**
- `tuple`: (fig, ax) matplotlib figure and axis objects

#### `synthetic_correlation_quality(segment_logs, segment_depths, log_data_type, quality_indices=['corr_coef', 'norm_dtw'], number_of_iterations=20, core_a_length=600, core_b_length=600, repetition=False, pca_for_dependent_dtw=False, output_csv_dir=None, mute_mode=True)`

Generate DTW correlation quality analysis for synthetic core pairs with multiple iterations. This function saves distribution parameters for each correlation quality metric across all iterations.

**Parameters:**
- `segment_logs` (list): List of turbidite log segments from `load_segment_pool()` or `modify_segment_pool()`
- `segment_depths` (list): List of turbidite depth segments from `load_segment_pool()` or `modify_segment_pool()`
- `log_data_type` (list): List of log column names
- `quality_indices` (list, default=['corr_coef', 'norm_dtw']): List of quality indices to analyze
- `number_of_iterations` (int, default=20): Number of synthetic pairs to generate
- `core_a_length` (float, default=600): Target length for synthetic core A in cm
- `core_b_length` (float, default=600): Target length for synthetic core B in cm
- `repetition` (bool, default=False): Allow reselecting turbidite segments
- `pca_for_dependent_dtw` (bool, default=False): Use PCA for dependent DTW analysis
- `output_csv_dir` (str, optional): Directory path for output CSV files. If None, saves files in current directory
- `mute_mode` (bool, default=True): Suppress detailed output messages

**Returns:**
- `dict`: Mapping quality indices to their output CSV filenames

#### `plot_synthetic_correlation_quality(input_csv, quality_indices=['corr_coef', 'norm_dtw'], bin_width=None, plot_individual_pdf=False, save_plot=False, plot_filename=None)`

Plot synthetic correlation quality distributions from saved CSV files.

**Parameters:**
- `input_csv` (str): Path to the CSV file containing fit parameters. Can include {quality_index} placeholder
- `quality_indices` (list, default=['corr_coef', 'norm_dtw']): List of quality indices to plot
- `bin_width` (float, optional): Bin width for histogram. If None, uses quality-specific defaults
- `plot_individual_pdf` (bool, default=False): If True, plots all individual iteration PDFs overlaid; if False, plots combined distribution
- `save_plot` (bool, default=False): Whether to save the plot to file
- `plot_filename` (str, optional): Filename for saving plot. Can include {quality_index} placeholder

**Returns:**
- None (displays and optionally saves plots)

#### `generate_constraint_subsets(n_constraints)`

Generate all possible subsets of constraints (2^n combinations) for age constraint removal testing.

**Parameters:**
- `n_constraints` (int): Number of constraints

**Returns:**
- `list`: List of all possible constraint subsets

#### `run_multi_parameter_analysis(log_a, log_b, md_a, md_b, all_depths_a_cat1, all_depths_b_cat1, pickeddepth_ages_a, pickeddepth_ages_b, age_data_a, age_data_b, uncertainty_method, parameter_combinations, target_quality_indices, test_age_constraint_removal, core_a_name, core_b_name, output_csv_filenames, synthetic_csv_filenames=None, pca_for_dependent_dtw=False, n_jobs=-1, max_search_per_layer=None)`

Run comprehensive multi-parameter analysis for core correlation with optional age constraint removal testing.

**Parameters:**
- `log_a, log_b` (array-like): Log data for cores A and B
- `md_a, md_b` (array-like): Measured depth arrays for cores A and B
- `all_depths_a_cat1, all_depths_b_cat1` (array-like): Picked depths of category 1 for cores A and B
- `pickeddepth_ages_a, pickeddepth_ages_b` (dict): Age interpolation results for picked depths
- `age_data_a, age_data_b` (dict): Age constraint data for cores A and B
- `uncertainty_method` (str): Method for uncertainty calculation
- `parameter_combinations` (list): List of parameter combinations to test
- `target_quality_indices` (list): Quality indices to analyze (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
- `test_age_constraint_removal` (bool): Whether to test age constraint removal scenarios
- `core_a_name, core_b_name` (str): Names of cores A and B
- `output_csv_filenames` (dict): Dictionary mapping quality_index to output CSV filename
- `synthetic_csv_filenames` (dict, optional): Dictionary mapping quality_index to synthetic CSV filename for consistent bin sizing
- `pca_for_dependent_dtw` (bool, default=False): Whether to use PCA for dependent DTW
- `n_jobs` (int, default=-1): Number of parallel jobs to run. -1 means using all available cores
- `max_search_per_layer` (int, optional): Maximum number of scenarios to process per constraint removal layer

**Returns:**
- None (Results are saved to CSV files specified in output_csv_filenames)

### Synthetic Stratigraphy Plotting (`syn_strat_plot.py`)

#### `plot_segment_pool(segment_logs, segment_depths, log_data_type, n_cols=8, figsize_per_row=4, plot_segments=True, save_plot=False, plot_filename=None)`

Plot all segments from the pool in a grid layout.

**Parameters:**
- `segment_logs` (list): List of log data arrays (segments)
- `segment_depths` (list): List of depth arrays corresponding to each segment
- `log_data_type` (list): List of column names for labeling
- `n_cols` (int, default=8): Number of columns in the subplot grid
- `figsize_per_row` (float, default=4): Height per row in the figure
- `plot_segments` (bool, default=True): Whether to plot the segments
- `save_plot` (bool, default=False): Whether to save the plot to file
- `plot_filename` (str, optional): Filename for saving plot

**Returns:**
- None

#### `create_and_plot_synthetic_core_pair(core_a_length, core_b_length, turb_logs, depth_logs, log_columns, repetition=False, plot_results=True, save_plot=False, plot_filename=None)`

Generate synthetic core pair and optionally plot the results.

**Parameters:**
- `core_a_length` (float): Target length for core A
- `core_b_length` (float): Target length for core B
- `turb_logs` (list): List of turbidite log segments
- `depth_logs` (list): List of corresponding depth arrays
- `log_columns` (list): List of log column names for labeling
- `repetition` (bool, default=False): If True, allow reusing turbidite segments
- `plot_results` (bool, default=True): Whether to display the plot
- `save_plot` (bool, default=False): Whether to save the plot to file
- `plot_filename` (str, optional): Filename for saving plot (if save_plot=True)

**Returns:**
- `tuple`: (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a, synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)

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

#### `plot_rgbimg_curves(depths=None, r=None, g=None, b=None, r_std=None, g_std=None, b_std=None, lum=None, lum_std=None, img=None, rgb_metadata=None, core_name=None, save_figs=False, output_dir=None, fig_format=['png'], dpi=150)`

Creates comprehensive three-panel visualization of RGB analysis results with image, color profiles, and standard deviation plots, supporting multiple output formats.

**Parameters:**
- `depths` (array-like, optional): Depth positions in pixels (not required if rgb_metadata is provided)
- `r, g, b` (array-like, optional): RGB color intensity values (not required if rgb_metadata is provided)
- `r_std, g_std, b_std` (array-like, optional): RGB standard deviations (not required if rgb_metadata is provided)
- `lum, lum_std` (array-like, optional): Luminance values and standard deviations (not required if rgb_metadata is provided)
- `img` (array-like, optional): Core image array for display (not required if rgb_metadata is provided)
- `rgb_metadata` (dict, optional): Dictionary from `rgb_process_and_stitch()` containing all RGB data. Expected keys: 'depths', 'r', 'g', 'b', 'r_std', 'g_std', 'b_std', 'lum', 'lum_std', 'image'. If provided, individual parameters are ignored
- `core_name` (str, optional): Core identifier for titles and file naming
- `save_figs` (bool, default=False): Whether to save plots as files
- `output_dir` (str, optional): Directory for saved files
- `fig_format` (list, default=['png', 'tiff']): List of file formats to save. Acceptable formats: 'png', 'jpg'/'jpeg', 'svg', 'tiff', 'pdf'
- `dpi` (int, default=150): Resolution in dots per inch for saved figures

**Returns:**
- None (displays plot and optionally saves files)

**Raises:**
- `ValueError`: If output_dir is not provided when save_figs is True

**Example:**
```python
# Using individual parameters
plot_rgbimg_curves(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img,
                   core_name='M9907-23PC_RGB', save_figs=True, output_dir='output/')

# Using metadata from rgb_process_and_stitch
rgb_metadata = rgb_process_and_stitch(...)
plot_rgbimg_curves(rgb_metadata=rgb_metadata, core_name='M9907-23PC_RGB', 
                   save_figs=True, output_dir='output/')
```

### CT Image Plotting (`ct_plotting.py`)

#### `plot_ctimg_curves(slice_data=None, brightness=None, stddev=None, pixel_spacing=None, ct_metadata=None, core_name="", save_figs=False, output_dir=None, vmin=400, vmax=2400, fig_format=['png', 'tiff'], dpi=150)`

Creates comprehensive three-panel visualization of CT analysis results with CT slice, brightness trace, and standard deviation plots, supporting multiple output formats.

**Parameters:**
- `slice_data` (numpy.ndarray, optional): 2D CT slice data to display (not required if ct_metadata is provided)
- `brightness` (numpy.ndarray, optional): 1D array of mean brightness values along depth (not required if ct_metadata is provided)
- `stddev` (numpy.ndarray, optional): 1D array of standard deviation values along depth (not required if ct_metadata is provided)
- `pixel_spacing` (tuple of float, optional): Tuple of (x, y) pixel spacing in mm/pixel for physical scaling (not required if ct_metadata is provided)
- `ct_metadata` (dict, optional): Dictionary from `ct_process_and_stitch()` containing all CT data. Expected keys: 'slice', 'brightness', 'stddev', 'px_spacing_x', 'px_spacing_y'. If provided, individual parameters are ignored
- `core_name` (str, default=""): Name of the core for title and filenames
- `save_figs` (bool, default=False): Whether to save figures to files
- `output_dir` (str, optional): Directory to save figures (required if save_figs=True)
- `vmin` (float, default=400): Minimum value for colormap scaling
- `vmax` (float, default=2400): Maximum value for colormap scaling
- `fig_format` (list, default=['png', 'tiff']): List of file formats to save. Acceptable formats: 'png', 'jpg'/'jpeg', 'svg', 'pdf', 'tiff'
- `dpi` (int, default=150): Resolution in dots per inch for saved figures

**Returns:**
- None (displays plot and optionally saves files)

**Raises:**
- `ValueError`: If output_dir is not provided when save_figs is True

**Example:**
```python
# Using individual parameters
plot_ctimg_curves(slice_data, brightness, stddev, pixel_spacing=(1, 1),
                  core_name='M9907-23PC_CT', save_figs=True, output_dir='output/')

# Using metadata from ct_process_and_stitch
ct_metadata = ct_process_and_stitch(...)
plot_ctimg_curves(ct_metadata=ct_metadata, core_name='M9907-23PC_CT', 
                  save_figs=True, output_dir='output/')
```

### RGB Image Processing (`rgb_processing.py`)

#### `rgb_process_and_stitch(core_structure, mother_dir, stitchbuffer=10, width_start_pct=0.25, width_end_pct=0.75, save_csv=True, output_csv=None, total_length_cm=None)`

Stitches multiple core section images by processing RGB profiles with section-specific parameters, combining results into continuous arrays, and optionally exporting to CSV.

**Parameters:**
- `core_structure` (dict or list): Core structure definition with filenames as keys and processing parameters as values
- `mother_dir` (str): Base directory path containing image files
- `stitchbuffer` (int, default=10): Bin rows to remove at stitching edges
- `width_start_pct, width_end_pct` (float, default=0.25, 0.75): Analysis strip boundaries
- `save_csv` (bool, default=True): Whether to save results to CSV file
- `output_csv` (str, optional): Full path for output CSV file (required if save_csv=True)
- `total_length_cm` (float, optional): Total core length in centimeters for depth conversion (required if save_csv=True)

**Returns:**
- `stitched_rgb_metadata` (dict): Dictionary containing all stitched RGB data with keys:
  - `depths` (numpy.ndarray): Depth values in pixels
  - `r` (numpy.ndarray): Red channel intensities
  - `g` (numpy.ndarray): Green channel intensities
  - `b` (numpy.ndarray): Blue channel intensities
  - `r_std` (numpy.ndarray): Red channel standard deviations
  - `g_std` (numpy.ndarray): Green channel standard deviations
  - `b_std` (numpy.ndarray): Blue channel standard deviations
  - `lum` (numpy.ndarray): Relative luminance values
  - `lum_std` (numpy.ndarray): Luminance standard deviations
  - `image` (numpy.ndarray): Complete stitched RGB core image

**Raises:**
- `ValueError`: If save_csv is True but output_csv or total_length_cm is not specified

**Example:**
```python
rgb_metadata = rgb_process_and_stitch(
    data_reading_structure, 
    rgb_data_dir='example_data/raw_data/Image_data',
    save_csv=True,
    output_csv='output/M9907-23PC_RGB.csv',
    total_length_cm=783
)
# Access individual components
depths = rgb_metadata['depths']
stitched_image = rgb_metadata['image']
```

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
- `stitched_ct_metadata` (dict): Dictionary containing all stitched CT data with keys:
  - `slice` (numpy.ndarray): Complete stitched CT slice
  - `brightness` (numpy.ndarray): Complete brightness profile
  - `stddev` (numpy.ndarray): Complete standard deviation profile
  - `depths` (numpy.ndarray): Depth coordinates in pixels
  - `px_spacing_x` (float): Final pixel spacing in x direction (always 1.0)
  - `px_spacing_y` (float): Final pixel spacing in y direction (always 1.0)

**Raises:**
- `ValueError`: If save_csv is True but output_csv or total_length_cm is not specified

**Example:**
```python
ct_metadata = ct_process_and_stitch(
    data_reading_structure, 
    ct_data_dir='example_data/raw_data/CT_data',
    save_csv=True,
    output_csv='output/M9907-23PC_CT.csv',
    total_length_cm=783
)
# Access individual components
stitched_slice = ct_metadata['slice']
brightness = ct_metadata['brightness']
```

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

#### `preprocess_core_data(data_config, resample_resolution=1)`

Preprocesses core data by cleaning and scaling depth values using configurable parameters. All processing actions are driven by the data_config content.

**Parameters:**
- `data_config` (dict): Configuration dictionary containing:
  - `column_configs`: Dictionary of data type configurations with thresholds and depth_col
  - `mother_dir`: Base directory path
  - `clean_output_folder`: Output folder for cleaned data
  - `input_file_paths`: Dictionary of input file paths by data type
  - `clean_file_paths`: Dictionary of output file paths by data type
  - `core_length`: Target core length for scaling
- `resample_resolution` (float, default=1): Target depth resolution for resampling (spacing between depth values)

**Returns:**
- None (saves cleaned data files to the specified output directory)

#### `plot_core_logs(data_config, file_type='clean', title=None, pickeddepth_csv=None, save_fig=False, output_dir=None, fig_format=['png'], dpi=None)`

Plot core logs using fully configurable parameters from data_config. Creates subplot panels for different types of core data (images and logs) based on the configuration provided.

**Parameters:**
- `data_config` (dict): Configuration dictionary containing:
  - `column_configs`: Dictionary of data type configurations with depth_col
  - `clean_file_paths` or `filled_file_paths`: Dictionary of file paths by data type
  - `core_length`: Core length for y-axis limits
  - `core_name`: Core name for title
- `file_type` (str, default='clean'): Type of data files to plot ('clean' or 'filled')
- `title` (str, optional): Custom title for the plot. If None, generates default title
- `pickeddepth_csv` (str, optional): Path to CSV file containing picked datum depths for visualization
- `save_fig` (bool, default=False): Whether to save the figure to disk
- `output_dir` (str, optional): Directory to save figures (required if save_fig=True)
- `fig_format` (list, default=['png']): List of file formats to save (options: 'png', 'jpg', 'svg', 'pdf')
- `dpi` (int, optional): Resolution in dots per inch for saved figures

**Returns:**
- None (displays the plot and optionally saves figures)

#### `plot_filled_data(target_log, original_data, filled_data, data_config, ML_type='ML')`

Plot original and ML-filled data for a given log using configurable parameters. Creates a horizontal plot showing the original data overlaid with ML-filled gaps, including uncertainty shading if available.

**Parameters:**
- `target_log` (str): Name of the log to plot
- `original_data` (pandas.DataFrame): Original data containing the log with gaps
- `filled_data` (pandas.DataFrame): Data with ML-filled gaps
- `data_config` (dict): Configuration containing:
  - `column_configs`: Dictionary of data type configurations with depth_col and plot labels
  - `core_length`: Core length for x-axis limits
  - `core_name`: Core name for title
- `ML_type` (str, default='ML'): Type of ML method used for title

**Returns:**
- None (displays the plot directly)

#### `fill_gaps_with_ml(target_log, All_logs, data_config, output_csv=True, merge_tolerance=3.0, ml_method='xgblgbm')`

Fill gaps in target data using specified ML method. Prepares feature data, applies the specified machine learning method, and fills gaps in the target log data.

**Parameters:**
- `target_log` (str): Name of the target column to fill gaps in
- `All_logs` (dict): Dictionary of dataframes containing feature data and target data
- `data_config` (dict): Configuration containing:
  - `column_configs`: Dictionary of data type configurations with depth_col
  - `filled_file_paths`: Dictionary of output file paths by data type
  - Other parameters including core info, etc.
- `output_csv` (bool, default=True): Whether to output filled data to CSV file
- `merge_tolerance` (float, default=3.0): Maximum allowed difference in depth for merging rows
- `ml_method` (str, default='xgblgbm'): ML method to use - 'rf', 'rftc', 'xgb', 'xgblgbm'

**Returns:**
- `tuple`: (target_data_filled, gap_mask) containing filled data and gap locations

#### `process_and_fill_logs(data_config, ml_method='xgblgbm', n_jobs=-1, show_plots=True)`

Process and fill gaps in log data using ML methods with fully configurable parameters. Orchestrates the complete ML-based gap filling process for all configured log data types. Supports parallel processing of multiple target logs with simple progress messages.

**Parameters:**
- `data_config` (dict): Configuration containing:
  - `column_configs`: Dictionary of data type configurations with depth_col
  - `clean_file_paths`: Dictionary of input file directories by data type
  - `filled_file_paths`: Dictionary of output file directories by data type
- `ml_method` (str, default='xgblgbm'): ML method to use - 'rf', 'rftc', 'xgb', 'xgblgbm'
  - 'rf': Random Forest
  - 'rftc': Random Forest with Trend Constraints
  - 'xgb': XGBoost
  - 'xgblgbm': XGBoost + LightGBM ensemble
- `n_jobs` (int, default=-1): Number of parallel jobs for processing multiple target logs
  - -1: Use all available CPU cores (fastest, plots disabled)
  - 1: Sequential processing with plots enabled (recommended for interactive use)
  - n: Use n CPU cores (plots disabled)
- `show_plots` (bool, default=True): Whether to generate and display plots during processing
  - Works in both sequential and parallel modes using appropriate matplotlib backend
  - Set to False to disable plotting for faster processing

**Returns:**
- None (saves filled data files and displays progress information)

**Notes:**
- Parallel processing is applied at the target log level, allowing multiple logs to be processed simultaneously
- Each parallel job uses its own memory space
- Progress is tracked with simple print messages: "[1/6] Processing Lumin...", "[2/6] Processing CT...", etc.
- Plots work in both sequential and parallel modes using the Agg backend for parallel workers
- For interactive use with plots: use n_jobs=1 or n_jobs=-1 with show_plots=True
- For fastest batch processing without plots: use n_jobs=-1 and show_plots=False

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

#### `pick_stratigraphic_levels(md=None, log=None, core_img_1=None, core_img_2=None, core_name="", csv_filename=None, sort_csv=True, core_log_paths=None, log_columns=None, depth_column='SB_DEPTH_cm', rgb_img_path=None, ct_img_path=None)`

Creates an interactive matplotlib environment for manually picking stratigraphic boundaries and datum levels with real-time visualization and CSV export. Supports two modes: direct data input or file-based loading.

**Parameters:**
- `md` (array-like, optional): Depth values for x-axis data. If None, will load from core_log_paths.
- `log` (array-like, optional): Log data for y-axis data (typically normalized 0-1). If None, will load from core_log_paths.
- `core_img_1` (numpy.ndarray, optional): First core image data (e.g., RGB image). If None, will load from rgb_img_path.
- `core_img_2` (numpy.ndarray, optional): Second core image data (e.g., CT image). If None, will load from ct_img_path.
- `core_name` (str, default=""): Name of the core for display in plot title
- `csv_filename` (str, optional): Full path/filename for the output CSV file. If None, defaults to `{core_name}_pickeddepth.csv` or `pickeddepth.csv`
- `sort_csv` (bool, default=True): Whether to sort CSV data by category then by picked_depths_cm when saving
- `core_log_paths` (dict, optional): Dictionary mapping log names to their file paths
- `log_columns` (list, optional): List of log column names to load from core_log_paths
- `depth_column` (str, default='SB_DEPTH_cm'): Name of the depth column in the log files
- `rgb_img_path` (str, optional): Path to RGB image file to load
- `ct_img_path` (str, optional): Path to CT image file to load

**Interactive Controls:**
- Left-click: Add depth point
- Number keys (0-9): Change current category
- Delete/Backspace: Remove last point
- Enter: Finish selection and save
- Esc: Exit without saving any changes
- Pan/Zoom tools: Temporarily disable point selection

**Returns:**
- `tuple`: (picked_depths, categories) - Lists of picked depth values and their categories

#### `interpret_bed_names(csv_filename, core_name="", core_log_paths=None, log_columns=None, depth_column='SB_DEPTH_cm', rgb_img_path=None, ct_img_path=None)`

Interactive Jupyter widget interface for naming picked stratigraphic beds. Loads picked depths from CSV file, displays core data with marked boundaries, and allows users to interactively assign names to each bed.

**Parameters:**
- `csv_filename` (str, **required**): Path to CSV file containing picked depths (e.g., 'example_data/picked_datum/M9907-23PC_pickeddepth.csv')
- `core_name` (str, default=""): Name of the core for display
- `core_log_paths` (dict, optional): Dictionary mapping log names to their file paths
- `log_columns` (list, optional): List of log column names to load from core_log_paths
- `depth_column` (str, default='SB_DEPTH_cm'): Name of the depth column in the log files
- `rgb_img_path` (str, optional): Path to RGB image file to load
- `ct_img_path` (str, optional): Path to CT image file to load

**Interactive Features:**
- Dropdown selector for choosing rows by depth and category
- Text input for entering bed names
- "Update Name" button to update individual rows
- "Save All Changes" button to save and display final plot with names

**Returns:**
- None (updates CSV file in place with interpreted bed names)

**Notes:**
- Requires Jupyter environment with ipywidgets installed
- Works seamlessly with output from `pick_stratigraphic_levels()`
- Creates interactive widgets for bed naming workflow
- Displays visualization with color-coded categories before and after naming

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

## Utils Module (`pyCoreRelator.utils`)

### Data Loader (`data_loader.py`)

#### `load_core_log_data(log_paths, core_name, log_columns=None, depth_column='SB_DEPTH_cm', normalize=True, column_alternatives=None, core_img_1=None, core_img_2=None, figsize=(20, 4), picked_datum=None, categories=None, show_bed_number=False, cluster_data=None, core_img_1_cmap_range=None, core_img_2_cmap_range=None, show_fig=True)`

**ENHANCED** - Load core log data from CSV files, optionally load picked depths from CSV, and create visualization with optional core images.

**Parameters:**
- `log_paths` (dict): Dictionary mapping log names to file paths
- `core_name` (str): Core name for plot title and identification (e.g., "M9907-25PC")
- `log_columns` (list, optional): List of log column names to load from the CSV files. If None, uses all keys from log_paths
- `depth_column` (str, default='SB_DEPTH_cm'): Name of the depth column in the CSV files
- `normalize` (bool, default=True): Whether to normalize each log to the range [0, 1]
- `column_alternatives` (dict, optional): Alternative column names to try if primary names not found
- `core_img_1` (str or array_like, optional): First core image (file path or array). If None, no image displayed
- `core_img_2` (str or array_like, optional): Second core image (file path or array). If None, no image displayed
- `figsize` (tuple, default=(20, 4)): Figure size (width, height)
- `picked_datum` (str, optional): Path to CSV file containing picked depths. CSV should have columns 'picked_depths_cm', 'category', and optionally 'interpreted_bed'
- `categories` (int, list, tuple, or set, optional): Category or categories to filter and display. Can be a single category (e.g., 1), multiple categories (e.g., [1, 2, 3]), or None to load all
- `show_bed_number` (bool, default=False): Whether to display bed numbers
- `cluster_data` (dict, optional): Cluster data for visualization
- `core_img_1_cmap_range, core_img_2_cmap_range` (tuple, optional): Color map ranges
- `show_fig` (bool, default=True): If True, displays the figure. If False, closes without displaying

**Returns:**
- `tuple`: (log, md, picked_depths, interpreted_bed)
  - `log` (numpy.ndarray): Log data array with shape (n_samples, n_logs) for multiple logs or (n_samples,) for single log
  - `md` (numpy.ndarray): Measured depths array
  - `picked_depths` (list): List of picked depth values (empty list if no picked_datum provided)
  - `interpreted_bed` (list): List of interpreted bed names corresponding to picked_depths (empty list if not available)

**Example:**
```python
>>> log, md, picked_depths, interpreted_bed = load_core_log_data(
...     log_paths={'MS': 'data/ms.csv', 'Lumin': 'data/lumin.csv'},
...     core_name="M9907-25PC",
...     log_columns=['MS', 'Lumin'],
...     core_img_1='data/rgb_image.png',
...     core_img_2='data/ct_image.png',
...     picked_datum='pickeddepth/M9907-25PC_pickeddepth.csv',
...     categories=[1],
...     show_fig=True
... )
```

#### `load_log_data(log_paths, img_paths=None, log_columns=None, depth_column='SB_DEPTH_cm', normalize=True, column_alternatives=None)`

**ENHANCED** - Loads and preprocesses well log data and core images from multiple file sources with automatic normalization and resampling. Now supports loading images from both file paths and directories.

**Parameters:**
- `log_paths` (dict): Dictionary mapping log names to file paths
- `img_paths` (dict, optional): Dictionary mapping image types ('rgb', 'ct') to file paths or directories. If a directory is provided, loads the first valid image file found. Supports .jpg, .jpeg, .png, .tiff, .tif, .bmp formats
- `log_columns` (list): List of log column names to load from files
- `depth_column` (str, default='SB_DEPTH_cm'): Name of depth column in data files
- `normalize` (bool, default=True): Whether to normalize log data to [0,1] range
- `column_alternatives` (dict, optional): Alternative column names to try if primary names not found

**Returns:**
- `tuple`: (log, md, available_columns, rgb_img, ct_img) containing loaded data and images

#### `load_core_age_constraints(core_name, age_base_path, data_columns=None, mute_mode=False)`

Loads age constraint data from CSV files with flexible column mapping.

**Parameters:**
- `core_name` (str): Name of the core to load age constraints for
- `age_base_path` (str): Base directory path containing age constraint CSV files
- `data_columns` (dict, optional): Dictionary mapping standard column names to actual CSV column names. Expected keys: 'age', 'pos_error', 'neg_error', 'min_depth', 'max_depth', 'in_sequence', 'core', 'interpreted_bed'
- `mute_mode` (bool, default=False): If True, suppress print output

**Returns:**
- `dict`: Age constraint data with keys: 'depths', 'ages', 'pos_errors', 'neg_errors', 'core', 'in_sequence_flags', 'interpreted_bed'

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

#### `visualize_combined_segments(dtw_result, log_a, log_b, md_a, md_b, segment_pairs_to_combine, **kwargs)`

Display segment correlations overlaid on log plots with optional age constraint visualization.

**Parameters:**
- `dtw_result` (dict): Dictionary containing DTW analysis results from `run_comprehensive_dtw_analysis()`. Expected keys: 'dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full'
- `log_a, log_b` (array-like): Log data arrays
- `md_a, md_b` (array-like): Measured depth arrays
- `segment_pairs_to_combine` (list): List of segment pairs to combine and visualize
- `ages_a, ages_b` (dict, optional): Age data dictionaries for picked depths
- `core_a_age_data, core_b_age_data` (dict, optional): Complete age constraint data from `load_core_age_constraints()`. Expected keys: 'in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors', 'core'
- `mark_ages` (bool, optional): Whether to mark age constraints in visualization
- `correlation_save_path, matrix_save_path` (str, optional): Paths to save output figures
- `core_a_interpreted_beds, core_b_interpreted_beds` (list, optional): Interpreted bed names for bed correlation
- `dpi` (int, default=None): Resolution for saved figures in dots per inch. If None, uses default (150)
- Additional kwargs for visualization options

**Returns:**
- `tuple`: (combined_wp, combined_quality)
  - `combined_wp` (numpy.ndarray): Combined warping path spanning all selected segments
  - `combined_quality` (dict): Aggregated quality metrics for the combined correlation

**Example:**
```python
dtw_result = run_comprehensive_dtw_analysis(...)
combined_wp, combined_quality = visualize_combined_segments(
    dtw_result,
    log_a, log_b, md_a, md_b,
    segment_pairs_to_combine=top_mapping_pairs[0],
    correlation_save_path='outputs/correlation.png',
    matrix_save_path='outputs/matrix.png',
    mark_ages=True,
    ages_a=estimated_datum_ages_a,
    ages_b=estimated_datum_ages_b,
    core_a_age_data=age_data_a,
    core_b_age_data=age_data_b
)
# Figures are automatically saved to the specified paths
```

#### `plot_correlation_distribution(mapping_csv, target_mapping_id=None, quality_index=None, save_png=True, png_filename=None, core_a_name=None, core_b_name=None, bin_width=None, pdf_method='normal', kde_bandwidth=0.05, mute_mode=False, targeted_binsize=None, dpi=None)`

Visualize and statistically analyze the distributions of the correlation quality metrics.

**Parameters:**
- `mapping_csv` (str): Path to the CSV/Parquet file containing mapping results
- `target_mapping_id` (int, optional): Optional mapping ID to highlight in the plot
- `quality_index` (str, **required**): Quality metric to plot ('corr_coef', 'norm_dtw', 'dtw_ratio', 'variance_deviation', 'perc_diag', 'match_min', 'match_mean', 'perc_age_overlap')
- `save_png` (bool, default=True): Whether to save plot as PNG
- `png_filename` (str, optional): Output PNG filename
- `core_a_name, core_b_name` (str, optional): Core names for plot title
- `bin_width` (float, optional): Histogram bin width (auto if None, based on quality_index)
- `pdf_method` (str, default='normal'): PDF fitting method ('KDE', 'skew-normal', or 'normal')
- `kde_bandwidth` (float, default=0.05): Bandwidth for KDE when pdf_method='KDE'
- `mute_mode` (bool, default=False): If True, suppress all print statements
- `targeted_binsize` (tuple, optional): (synthetic_bins, bin_width) for consistent bin sizing with synthetic data
- `dpi` (int, default=None): Resolution for saved figures in dots per inch. If None, uses default (150)

**Returns:**
- `fit_params` (dict): Dictionary containing distribution statistics including histogram data, PDF parameters, bin information, and percentile data when target_mapping_id is specified

#### `calculate_quality_comparison_t_statistics(real_data, syn_data, quality_indices)`

Calculate t-statistics for quality metric comparisons.

**Parameters:**
- `real_data` (dict): Real correlation quality data
- `syn_data` (dict): Synthetic correlation quality data
- `quality_indices` (list): List of quality metrics to compare

**Returns:**
- `dict`: T-statistics and p-values for each quality metric

#### `plot_quality_comparison_t_statistics(target_quality_indices, master_csv_filenames, synthetic_csv_filenames, CORE_A, CORE_B, mute_mode=False, save_fig=True, output_figure_filenames=None, save_gif=False, output_gif_filenames=None, max_frames=50, plot_real_data_histogram=False, plot_age_removal_step_pdf=True, show_best_datum_match=True, sequential_mappings_csv=None)`

Plot quality index distributions comparing real data vs synthetic null hypothesis with t-statistics analysis.

**Parameters:**
- `target_quality_indices` (list): Quality metrics to plot (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
- `master_csv_filenames` (dict): Dictionary mapping quality indices to master CSV file paths (should contain t-statistics columns)
- `synthetic_csv_filenames` (dict): Dictionary mapping quality indices to synthetic CSV file paths
- `CORE_A` (str): Name of core A for plot titles
- `CORE_B` (str): Name of core B for plot titles
- `mute_mode` (bool, default=False): If True, suppress detailed output messages and show only essential progress information
- `save_fig` (bool, default=True): If True, save static figures to files
- `output_figure_filenames` (dict, optional): Dictionary mapping quality indices to output figure file paths (only used when save_fig=True)
- `save_gif` (bool, default=False): If True, create animated GIF showing progressive addition of age constraints. When save_gif=True and save_fig=False, static figures will not be displayed (only GIFs are shown at the end). When save_gif=False (default), static figures will be displayed normally regardless of save_fig value
- `output_gif_filenames` (dict, optional): Dictionary mapping quality indices to GIF file paths (only used when save_gif=True)
- `max_frames` (int, default=50): Maximum number of frames for GIF animations
- `plot_real_data_histogram` (bool, default=False): If True, plot histograms for real data (no age and all age constraint cases)
- `plot_age_removal_step_pdf` (bool, default=True): If True, plot all PDF curves including dashed lines for partially removed constraints
- `show_best_datum_match` (bool, default=True): If True, plot vertical line showing best datum match value from sequential_mappings_csv
- `sequential_mappings_csv` (str or dict, optional): Path to CSV file(s) containing sequential mappings with 'Ranking_datums' column. Can be a single CSV path (str) or dictionary mapping quality indices to CSV paths

**Returns:**
- None (creates static plots and/or animated GIFs based on parameters)

### Matrix Plots (`matrix_plots.py`)

#### `plot_dtw_matrix_with_paths(dtw_distance_matrix_full, mode=None, **kwargs)`

Visualize DTW cost matrices with correlation paths and optional age constraint masking.

**Parameters:**
- `dtw_distance_matrix_full` (np.ndarray): DTW cost matrix to visualize
- `mode` (str): Visualization mode ('segment_paths', 'combined_path', 'all_paths_colored')
- `valid_dtw_pairs` (list, optional): List of valid segment pairs
- `dtw_results` (dict, optional): DTW results dictionary
- `sequential_mappings_csv` (str, optional): Path to CSV with multiple paths
- `segments_a, segments_b` (list, optional): Segment definitions
- `depth_boundaries_a, depth_boundaries_b` (list, optional): Depth boundaries
- `core_a_age_data, core_b_age_data` (dict, optional): Complete age constraint data from `load_core_age_constraints()`. Expected keys: 'in_sequence_ages', 'in_sequence_depths', 'core'. When 'core' key is provided, constraint lines will be drawn
- `md_a, md_b` (array-like, optional): Measured depth arrays
- `core_a_name, core_b_name` (str, optional): Core names for labels
- `output_filename` (str, optional): Path to save the figure
- `color_metric` (str, optional): Metric for coloring paths ('corr_coef', 'norm_dtw', etc.)
- `dpi` (int, default=None): Resolution for saved figures in dots per inch. If None, uses default (150)
- Additional kwargs for plot customization

**Returns:**
- `str or None`: Path to saved figure if output_filename provided

**Example:**
```python
plot_dtw_matrix_with_paths(
    dtw_distance_matrix_full,
    mode='all_paths_colored',
    sequential_mappings_csv='mappings.csv',
    color_metric='norm_dtw',
    core_a_age_data=age_data_a,
    core_b_age_data=age_data_b,
    md_a=md_a, md_b=md_b,
    core_a_name='M9907-25PC',
    core_b_name='M9907-23PC'
)
```

### Animation (`animation.py`)

#### `visualize_dtw_results_from_csv(input_mapping_csv, dtw_result, log_a, log_b, md_a, md_b, **kwargs)`

Generate animated correlation sequences from results with optional age constraint visualization. When `creategif=True`, automatically displays the generated GIFs in Jupyter/IPython environments.

**Parameters:**
- `input_mapping_csv` (str): Path to CSV file containing correlation mapping results
- `dtw_result` (dict): Dictionary containing DTW analysis results from `run_comprehensive_dtw_analysis()`. Expected keys: 'dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full'
- `log_a, log_b` (array-like): Log data arrays
- `md_a, md_b` (array-like): Measured depth arrays
- `color_interval_size` (int, default=10): Step size for warping path visualization
- `max_frames` (int, default=100): Maximum number of frames to generate
- `debug` (bool, default=False): Enable debug output
- `creategif` (bool, default=True): Whether to create and display GIF files
- `keep_frames` (bool, default=False): Whether to preserve individual frame files
- `correlation_gif_output_filename` (str, default='CombinedDTW_correlation_mappings.gif'): Path to save correlation GIF
- `matrix_gif_output_filename` (str, default='CombinedDTW_matrix_mappings.gif'): Path to save matrix GIF
- `visualize_pairs` (bool, default=False): Whether to visualize segment pairs
- `visualize_segment_labels` (bool, default=False): Whether to show segment labels
- `mark_depths` (bool, default=True): Whether to mark depth boundaries
- `mark_ages` (bool, default=True): Whether to mark age constraints in visualization
- `ages_a, ages_b` (dict, optional): Age data dictionaries for picked depths
- `core_a_age_data, core_b_age_data` (dict, optional): Complete age constraint data from `load_core_age_constraints()`. Expected keys: 'in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors', 'core'
- `core_a_name, core_b_name` (str, optional): Core names for labels
- `core_a_interpreted_beds, core_b_interpreted_beds` (dict, optional): Interpreted bed names for cores
- `dpi` (int, default=None): Resolution for saved frames and GIFs in dots per inch. If None, uses default (150)

**Returns:**
- None (creates, saves, and displays animations when creategif=True)

**Example:**
```python
dtw_result = run_comprehensive_dtw_analysis(...)
visualize_dtw_results_from_csv(
    input_mapping_csv=f'example_data/analytical_outputs/{CORE_A}_{CORE_B}/mappings.csv',
    dtw_result=dtw_result,
    log_a=log_a, log_b=log_b, md_a=md_a, md_b=md_b,
    correlation_gif_output_filename=f'outputs/correlation_{CORE_A}_{CORE_B}.gif',
    matrix_gif_output_filename=f'outputs/matrix_{CORE_A}_{CORE_B}.gif',
    mark_ages=True,
    ages_a=estimated_datum_ages_a,
    ages_b=estimated_datum_ages_b,
    core_a_age_data=age_data_a,
    core_b_age_data=age_data_b
)
# GIFs are automatically displayed after creation
```

### Helpers (`helpers.py`)

#### `find_nearest_index(depth_array, depth_value)`

Finds the index in a depth array that corresponds to the closest depth value to a target depth.

**Parameters:**
- `depth_array` (array-like): Array of depth values to search
- `depth_value` (float): Target depth value to find

**Returns:**
- `int`: Index in depth_array with closest value to target depth