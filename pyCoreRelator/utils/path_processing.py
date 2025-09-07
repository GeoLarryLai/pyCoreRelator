"""
Path processing and manipulation functions for pyCoreRelator.

Included Functions:
- combine_segment_dtw_results: Combine DTW results from multiple segment pairs
- compute_combined_path_metrics: Compute quality metrics from combined warping paths
- load_sequential_mappings: Load sequential mappings from CSV files
- is_subset_or_superset: Check subset/superset relationships between paths
- filter_against_existing: Filter new paths against existing filtered paths
- find_best_mappings: Find the best DTW mappings based on multiple quality metrics
- find_target_mappings: Find mappings that comply with boundary correlations between cores

This module provides utilities for combining DTW segment results, computing combined
path metrics, loading sequential mappings from CSV files, filtering paths based
on subset/superset relationships, and identifying optimal correlation mappings.
These functions are essential for post-processing DTW analysis results and managing
path data in geological core correlation workflows.
"""

import numpy as np
import pandas as pd
import csv


def combine_segment_dtw_results(dtw_results, segment_pairs, segments_a, segments_b, 
                               depth_boundaries_a, depth_boundaries_b, log_a, log_b, dtw_distance_matrix_full, pca_for_dependent_dtw=False):
    """
    Combine DTW results from multiple segment pairs into a unified result.
    
    This function takes DTW analysis results from individual segment pairs and combines
    them into a single warping path and quality metric set. It handles sorting, duplicate
    removal, and quality metric aggregation across segments.
    
    Parameters
    ----------
    dtw_results : dict
        Dictionary containing DTW results for each segment pair from run_comprehensive_dtw_analysis
    segment_pairs : list
        List of tuples (a_idx, b_idx) for segment pairs to combine
    segments_a : list
        Segments in log_a
    segments_b : list
        Segments in log_b
    depth_boundaries_a : list
        Depth boundaries for log_a
    depth_boundaries_b : list
        Depth boundaries for log_b
    log_a : array-like
        Original log data for core A
    log_b : array-like
        Original log data for core B
    
    Returns
    -------
    tuple
        (combined_wp, combined_quality) where:
        - combined_wp: numpy.ndarray of combined warping path coordinates
        - combined_quality: dict of averaged quality metrics
    
    Example
    -------
    >>> dtw_results = {(0, 0): (paths, matrices, quality), (1, 1): (paths, matrices, quality)}
    >>> segment_pairs = [(0, 0), (1, 1)]
    >>> combined_wp, combined_quality = combine_segment_dtw_results(
    ...     dtw_results, segment_pairs, segments_a, segments_b,
    ...     depth_boundaries_a, depth_boundaries_b, log_a, log_b
    ... )
    """
    all_warping_paths = []
    all_quality_indicators = []
    
    # Check if segment_pairs is empty
    if not segment_pairs or len(segment_pairs) == 0:
        print("No segment pairs provided to combine.")
        return None, None, None
    
    # Process each segment pair and collect valid paths
    for a_idx, b_idx in segment_pairs:
        if (a_idx, b_idx) not in dtw_results:
            print(f"Warning: Segment pair ({a_idx+1}, {b_idx+1}) not found in DTW results. Skipping.")
            continue
        
        paths, cost_matrices, quality_indicators = dtw_results[(a_idx, b_idx)]
        
        if not paths or len(paths) == 0:
            print(f"Warning: No valid path for segment pair ({a_idx+1}, {b_idx+1}). Skipping.")
            continue
        
        # Add the best path (first one) and its quality indicators
        all_warping_paths.append(paths[0])
        
        if quality_indicators and len(quality_indicators) > 0:
            all_quality_indicators.append(quality_indicators[0])
    
    # Return None if no valid paths found
    if not all_warping_paths:
        print("No valid warping paths found in the selected segment pairs.")
        return None, None, None
    
    # Sort paths by their starting coordinates and combine
    all_warping_paths.sort(key=lambda wp: (wp[0, 0], wp[0, 1]))
    combined_wp = np.vstack(all_warping_paths)
    
    # Remove duplicate points at segment boundaries
    combined_wp = np.unique(combined_wp, axis=0)
    combined_wp = combined_wp[combined_wp[:, 0].argsort()]
    
    # Calculate combined quality metrics
    if all_quality_indicators:
        age_overlap_values = []
        for qi in all_quality_indicators:
            if 'perc_age_overlap' in qi:
                age_overlap_values.append(float(qi['perc_age_overlap']))
        
        combined_quality = compute_combined_path_metrics(
            combined_wp, log_a, log_b, all_quality_indicators, dtw_distance_matrix_full, age_overlap_values, 
            pca_for_dependent_dtw=pca_for_dependent_dtw
        )
    else:
        combined_quality = None
        
    return combined_wp, combined_quality


def compute_combined_path_metrics(combined_wp, log_a, log_b, segment_quality_indicators, dtw_distance_matrix_full, age_overlap_values=None, pca_for_dependent_dtw=False):
    """
    Compute quality metrics from combined warping path and log data.
    
    This function calculates comprehensive quality metrics for a combined warping path
    using the original continuous log data to maintain geological coherence. All metrics
    are computed from the complete combined path for consistency.
    
    Parameters
    ----------
    combined_wp : numpy.ndarray
        Combined warping path with indices referencing original continuous logs
    log_a : numpy.ndarray
        Original continuous log data array for core A
    log_b : numpy.ndarray
        Original continuous log data array for core B
    segment_quality_indicators : list
        Quality indicators from individual segments (used only for age overlap)
    age_overlap_values : list, optional
        Age overlap values for averaging
    
    Returns
    -------
    dict
        Combined quality metrics including normalized DTW distance, correlation
        coefficient, path characteristics, and age overlap percentage
    """
    from ..core.quality_metrics import compute_quality_indicators
    
    # Initialize metrics dictionary
    metrics = {
        'norm_dtw': 0.0,
        'dtw_ratio': 0.0,
        'perc_diag': 0.0,
        'dtw_warp_eff': 0.0,
        'corr_coef': 0.0,
        'perc_age_overlap': 0.0
    }
    
    # Compute all metrics using the combined warping path
    if combined_wp is not None and len(combined_wp) > 1:
        # Extract and validate indices from combined warping path
        p_indices = combined_wp[:, 0].astype(int)
        q_indices = combined_wp[:, 1].astype(int)
        
        p_indices = np.clip(p_indices, 0, len(log_a) - 1)
        q_indices = np.clip(q_indices, 0, len(log_b) - 1)
        
        # Calculate DTW step costs along the specific combined path
        def get_path_dtw_cost_efficient(combined_wp, dtw_matrix):
            """Extract step costs only at path coordinates"""
            if dtw_matrix is None:
                return 0.0
                
            total_cost = 0.0
            
            for i in range(len(combined_wp)):
                a_idx = int(combined_wp[i, 0])
                b_idx = int(combined_wp[i, 1])
                
                # Calculate step cost for this specific point
                if a_idx == 0 and b_idx == 0:
                    step_cost = dtw_matrix[0, 0]
                elif a_idx == 0:
                    step_cost = dtw_matrix[0, b_idx] - dtw_matrix[0, b_idx-1]
                elif b_idx == 0:
                    step_cost = dtw_matrix[a_idx, 0] - dtw_matrix[a_idx-1, 0]
                else:
                    min_pred = min(dtw_matrix[a_idx-1, b_idx], 
                                  dtw_matrix[a_idx, b_idx-1], 
                                  dtw_matrix[a_idx-1, b_idx-1])
                    step_cost = dtw_matrix[a_idx, b_idx] - min_pred
                
                total_cost += step_cost
            
            return total_cost
        
        path_cost = get_path_dtw_cost_efficient(combined_wp, dtw_distance_matrix_full)
        
        # Calculate norm_dtw directly
        metrics['norm_dtw'] = path_cost / (dtw_distance_matrix_full.shape[0] + dtw_distance_matrix_full.shape[1])
        
        # Create dummy cost matrix for other metrics computation
        dummy_D = np.array([[path_cost]])
        combined_metrics = compute_quality_indicators(log_a, log_b, p_indices, q_indices, dummy_D, pca_for_dependent_dtw=pca_for_dependent_dtw)
        
        # Update other metrics from compute_quality_indicators (excluding norm_dtw)
        metrics['dtw_ratio'] = float(combined_metrics.get('dtw_ratio', 0.0))
        metrics['dtw_warp_eff'] = float(combined_metrics.get('dtw_warp_eff', 0.0))
        metrics['corr_coef'] = float(combined_metrics.get('corr_coef', 0.0))
        metrics['perc_diag'] = float(combined_metrics.get('perc_diag', 0.0))
    
    # Average age overlap values across segments
    if age_overlap_values:
        metrics['perc_age_overlap'] = float(sum(age_overlap_values) / len(age_overlap_values))
    
    return metrics


def load_sequential_mappings(csv_path):
    """
    Load sequential mappings from a CSV file.
    
    This function reads warping path data stored in compact format from CSV files.
    It handles the parsing of compressed path strings back into lists of coordinate tuples.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing sequential mappings
    
    Returns
    -------
    list
        List of warping paths, where each path is a list of (x, y) coordinate tuples
    
    Example
    -------
    >>> mappings = load_sequential_mappings('path_data.csv')
    >>> print(f"Loaded {len(mappings)} paths")
    >>> print(f"First path: {mappings[0][:3]}...")  # Show first 3 points
    """
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    mappings = []
    try:
        # Try pandas for efficient CSV reading
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            try:
                path = parse_compact_path(row['path'])
                mappings.append(path)
            except:
                continue
                
    except ImportError:
        # Fallback to csv module if pandas not available
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    path = parse_compact_path(row['path'])
                    mappings.append(path)
                except:
                    continue
    
    return mappings


def is_subset_or_superset(path_info, other_path_info, early_terminate=True):
    """
    Check if one path is a subset or superset of another with early termination.
    
    This function efficiently determines subset/superset relationships between two
    warping paths using set operations with optional early termination for performance.
    
    Parameters
    ----------
    path_info : dict
        First path information dictionary containing 'length' and 'path_set' keys
    other_path_info : dict
        Second path information dictionary containing 'length' and 'path_set' keys
    early_terminate : bool, default=True
        Whether to use early termination checks based on path lengths
    
    Returns
    -------
    tuple
        (is_subset, is_superset) boolean flags indicating the relationship
    
    Example
    -------
    >>> path1_info = {'length': 3, 'path_set': {(0,0), (1,1), (2,2)}}
    >>> path2_info = {'length': 5, 'path_set': {(0,0), (1,1), (2,2), (3,3), (4,4)}}
    >>> is_subset, is_superset = is_subset_or_superset(path1_info, path2_info)
    >>> print(f"Path1 is subset: {is_subset}, superset: {is_superset}")
    """
    # Early termination based on length comparisons
    if early_terminate:
        if path_info['length'] < other_path_info['length']:
            return (False, False)
        
        if path_info['length'] > other_path_info['length']:
            return (False, False)
    
    # Perform full set comparison
    path_set = path_info['path_set']
    other_path_set = other_path_info['path_set']
    
    is_subset = path_set.issubset(other_path_set)
    is_superset = path_set.issuperset(other_path_set)
    
    return (is_subset, is_superset)


def filter_against_existing(new_path, filtered_paths, group_writer):
    """
    Filter a new path against existing filtered paths with optimized checking.
    
    This function determines if a new warping path should be added to the filtered
    set by checking for subset/superset relationships with existing paths. It uses
    length-based grouping for efficient filtering.
    
    Parameters
    ----------
    new_path : dict
        New path information dictionary containing path data and metadata
    filtered_paths : list
        List of existing filtered path information dictionaries
    group_writer : csv.writer
        CSV writer object to write accepted paths
    
    Returns
    -------
    tuple
        (is_valid, paths_to_remove, updated_count) where:
        - is_valid: bool indicating if the new path should be added
        - paths_to_remove: list of indices of paths to remove from filtered_paths
        - updated_count: int (0 or 1) indicating if count should be incremented
    
    Example
    -------
    >>> new_path = {'length': 10, 'path_set': set(...), 'row_data': [...]}
    >>> filtered_paths = [existing_path1, existing_path2]
    >>> is_valid, to_remove, count = filter_against_existing(
    ...     new_path, filtered_paths, csv_writer
    ... )
    """
    is_valid = True
    paths_to_remove = []
    
    # Group existing paths by length for efficient comparison
    length_groups = {}
    for i, existing_path in enumerate(filtered_paths):
        length = existing_path['length']
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append((i, existing_path))
    
    # Check if any existing path contains this path (making it invalid)
    for length in sorted(length_groups.keys(), reverse=True):
        if length < new_path['length']:
            break  # No need to check shorter paths
            
        for i, existing_path in length_groups[length]:
            _, is_superset = is_subset_or_superset(existing_path, new_path)
            if is_superset:
                is_valid = False
                return (is_valid, [], 0)  # Early exit if contained by existing path
    
    # If valid, check if it contains any existing paths for removal
    if is_valid:
        for length in sorted(length_groups.keys()):
            if length > new_path['length']:
                break  # No need to check longer paths
                
            for i, existing_path in length_groups[length]:
                is_subset, _ = is_subset_or_superset(existing_path, new_path)
                if is_subset:
                    paths_to_remove.append(i)
    
    # Write valid path to output and return results
    if is_valid:
        group_writer.writerow(new_path['row_data'])
        return (is_valid, paths_to_remove, 1)
    
    return (is_valid, paths_to_remove, 0)


def find_best_mappings(csv_file_path, 
                       top_n=5, 
                       filter_shortest_dtw=True,
                       metric_weight=None):
    """
    Find the best DTW mappings based on multiple quality metrics with configurable scoring.
    
    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing DTW results
    top_n : int, optional
        Number of top mappings to return (default: 5)
    filter_shortest_dtw : bool, optional
        If True, only consider mappings with shortest DTW path length (default: True)
    metric_weight : dict, optional
        Dictionary defining metric weights for scoring
        Format: {metric_name: weight_value}
        If None, uses default weights
    
    Returns
    -------
    tuple
        - top_mapping_id : list
            List of top mapping IDs in order
        - top_mapping_pairs : list
            List of valid_pairs_to_combine for each top mapping ID
        - top_mapping_df : pandas.DataFrame
            DataFrame containing the top N mappings sorted by combined score
    
    Examples
    --------
    >>> # Using default weights
    >>> top_ids, top_pairs, top_df = find_best_mappings('dtw_results.csv', top_n=3)
    
    >>> # Using custom weights
    >>> custom_weights = {
    ...     'corr_coef': 5.0,
    ...     'perc_diag': 2.0,
    ...     'norm_dtw': 1.0
    ... }
    >>> top_ids, top_pairs, top_df = find_best_mappings('dtw_results.csv', 
    ...                                       metric_weight=custom_weights)
    """
    
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    # Default metrics configuration with fixed higher_is_better values
    default_weights = {
        'perc_diag': 0.0,
        'norm_dtw': 1.0,
        'dtw_ratio': 0.0,
        'corr_coef': 1.0,
        'dtw_warp_eff': 0.0,
        'perc_age_overlap': 0.0
    }
    
    # Fixed higher_is_better configuration (cannot be changed)
    higher_is_better_config = {
        'perc_diag': True,
        'norm_dtw': False,
        'dtw_ratio': False,
        'corr_coef': True,
        'dtw_warp_eff': True,
        'perc_age_overlap': True,
    }
    
    # Use provided weights or default weights
    weights = metric_weight if metric_weight is not None else default_weights
    
    # Load and clean the data
    dtw_results_df = pd.read_csv(csv_file_path)
    required_cols = ['corr_coef', 'perc_diag', 'perc_age_overlap']
    dtw_results_df = dtw_results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)
    
    print(f"=== Top {top_n} Overall Best Mappings ===")
    
    # Filter for shortest DTW path length if requested
    if filter_shortest_dtw and 'length' in dtw_results_df.columns:
        dtw_results_df['dtw_path_length'] = dtw_results_df['length']
        min_length = dtw_results_df['dtw_path_length'].min()
        shortest_mappings = dtw_results_df[dtw_results_df['dtw_path_length'] == min_length]
        
        print(f"Filtering for shortest DTW path length: {min_length}")
        print(f"Number of mappings found: {len(shortest_mappings)} out of {len(dtw_results_df)}")
    else:
        shortest_mappings = dtw_results_df
        if filter_shortest_dtw:
            print("Warning: No 'length' column found. Using all mappings.")
    
    # Create a copy for scoring calculations
    df_for_ranking = shortest_mappings.copy()
    
    # Initialize the combined score column
    df_for_ranking['combined_score'] = 0.0
    total_weight = 0.0
    
    # Calculate scores for each metric and add to combined score
    for metric, weight in weights.items():
        if metric in df_for_ranking.columns:
            # Make sure we have valid data to work with
            valid_data = df_for_ranking[~df_for_ranking[metric].isna()]
            
            if len(valid_data) > 0:
                min_val = valid_data[metric].min()
                max_val = valid_data[metric].max()
                
                # Only normalize if there's a range
                if max_val > min_val:
                    if higher_is_better_config.get(metric, True):
                        # For metrics where higher values are better
                        df_for_ranking[f'{metric}_score'] = (df_for_ranking[metric] - min_val) / (max_val - min_val)
                    else:
                        # For metrics where lower values are better
                        df_for_ranking[f'{metric}_score'] = 1 - ((df_for_ranking[metric] - min_val) / (max_val - min_val))
                else:
                    # If all values are the same, assign a score of 1
                    df_for_ranking[f'{metric}_score'] = 1.0
                    
                # Add to weighted sum
                df_for_ranking['combined_score'] += df_for_ranking[f'{metric}_score'].fillna(0) * weight
                total_weight += weight
    
    # Normalize the combined score by the total weight
    if total_weight > 0:
        df_for_ranking['combined_score'] = df_for_ranking['combined_score'] / total_weight
    else:
        print("Warning: No valid metrics found for scoring.")
        df_for_ranking['combined_score'] = 0.0
    
    # Handle NaN values in combined score
    if df_for_ranking['combined_score'].isna().any():
        print("Warning: NaN values detected in combined scores. Replacing with zeros.")
        df_for_ranking['combined_score'] = df_for_ranking['combined_score'].fillna(0)
    
    # Get top N mappings by combined score
    top_mappings = df_for_ranking.sort_values(by='combined_score', ascending=False).head(top_n)
    
    # Parse paths and create valid_pairs_to_combine for each mapping
    top_mapping_ids = []
    top_mapping_pairs = []
    
    # Print detailed results
    print(f"\nTop {top_n} mappings by combined score:")
    for idx, row in top_mappings.iterrows():
        if 'mapping_id' in row:
            mapping_id = int(row['mapping_id'])
            top_mapping_ids.append(mapping_id)
            print(f"Mapping ID {mapping_id}: Combined Score={row['combined_score']:.3f}")
            
            # Parse the path and convert to valid_pairs_to_combine
            if 'path' in row:
                target_data_row = parse_compact_path(row['path'])
                # Convert 1-based to 0-based indices for python
                valid_pairs_to_combine = [(a-1, b-1) for a, b in target_data_row]
                top_mapping_pairs.append(valid_pairs_to_combine)
            else:
                # If no path column, append empty list
                top_mapping_pairs.append([])
        
        # Print all available metrics
        metric_outputs = []
        if 'dtw_path_length' in row and weights.get('dtw_path_length', 0) != 0:
            metric_outputs.append(f"dtw_path_length={row['dtw_path_length']:.1f}")
        if 'corr_coef' in row and weights.get('corr_coef', 0) != 0:
            metric_outputs.append(f"correlation coefficient r={row['corr_coef']:.3f}")
        if 'perc_diag' in row and weights.get('perc_diag', 0) != 0:
            metric_outputs.append(f"perc_diag={row['perc_diag']:.1f}%")
        if 'norm_dtw' in row and weights.get('norm_dtw', 0) != 0:
            metric_outputs.append(f"norm_dtw={row['norm_dtw']:.3f}")
        if 'dtw_ratio' in row and weights.get('dtw_ratio', 0) != 0:
            metric_outputs.append(f"dtw_ratio={row['dtw_ratio']:.3f}")
        if 'dtw_warp_eff' in row and weights.get('dtw_warp_eff', 0) != 0:
            metric_outputs.append(f"dtw_warp_eff={row['dtw_warp_eff']:.1f}%")
        if 'perc_age_overlap' in row and weights.get('perc_age_overlap', 0) != 0:
            metric_outputs.append(f"perc_age_overlap={row['perc_age_overlap']:.1f}%")
        
        # Print metrics with proper indentation
        for metric_output in metric_outputs:
            print(f"  {metric_output}")
        
        # Print valid_pairs_to_combine at the end of each mapping with 1-based numbering
        if len(top_mapping_pairs) > 0 and len(top_mapping_pairs[-1]) > 0:
            # Convert back to 1-based for display
            pairs_display = [(a+1, b+1) for a, b in top_mapping_pairs[-1]]
            print(f"  valid_pairs_to_combine={pairs_display}")
        else:
            print(f"  valid_pairs_to_combine=[]")
        print("")
    
    # Handle case where no valid mappings found
    if not top_mappings.empty:
        print(f"Best mapping ID: {top_mapping_ids[0] if top_mapping_ids else 'None'}")
    else:
        print("Warning: No valid mappings found")
    
    return top_mapping_ids, top_mapping_pairs, top_mappings


def find_target_mappings(picked_depths_a_cat1, picked_depths_b_cat1, 
                         interpreted_bed_a, interpreted_bed_b,
                         valid_dtw_pairs, sequential_mappings_csv,
                         segments_a, segments_b,
                         top_n=5,
                         metric_weight=None):
    """
    Find mappings that comply with boundary correlations between two cores.
    
    For DTW consecutive segments, checks if at least one boundary has matching 
    interpreted_bed names between cores (ignoring "?" and empty names).
    
    Parameters
    ----------
    picked_depths_a_cat1 : array-like
        Picked depths for core A category 1
    picked_depths_b_cat1 : array-like  
        Picked depths for core B category 1
    interpreted_bed_a : array-like
        Interpreted bed names for core A
    interpreted_bed_b : array-like
        Interpreted bed names for core B
    valid_dtw_pairs : list
        List of valid DTW pairs
    sequential_mappings_csv : str or DataFrame
        Path to CSV file or DataFrame containing mappings
    segments_a : list
        Segments for core A
    segments_b : list
        Segments for core B
    top_n : int, optional
        Number of top mappings to return (default: 5)
    metric_weight : dict, optional
        Dictionary defining metric weights for scoring
        If None, uses default weights from find_best_mappings
        
    Returns
    -------
    tuple
        - top_target_mapping_ids : list
            List of top mapping IDs in order
        - top_target_mapping_pairs : list
            List of valid_pairs_to_combine for each top mapping ID
        - target_mappings_df : pandas.DataFrame
            DataFrame containing all target mappings
    """
    
    # Helper function to clean bed names (remove "?")
    def clean_name(name):
        if pd.isna(name) or name == '':
            return ''
        return name.replace('?', '').strip()
    
    # Helper function to parse compact path format
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    # Step 1: Find valid consecutive DTW pairs that have at least one matching boundary
    matching_pairs = []
    matching_boundary_names = set()
    matching_details = []  # Store detailed information for better output
    
    for pair in valid_dtw_pairs:
        seg_a_idx, seg_b_idx = pair
        
        # Get the actual segment tuples
        seg_a = segments_a[seg_a_idx]
        seg_b = segments_b[seg_b_idx]
        
        # Only check consecutive segments (i, i+1) - these represent intervals between boundaries
        if seg_a[1] == seg_a[0] + 1 and seg_b[1] == seg_b[0] + 1:
            start_idx_a, end_idx_a = seg_a
            start_idx_b, end_idx_b = seg_b
            
            # Check if both start and end boundaries exist in interpreted_bed arrays
            if (start_idx_a < len(interpreted_bed_a) and end_idx_a < len(interpreted_bed_a) and
                start_idx_b < len(interpreted_bed_b) and end_idx_b < len(interpreted_bed_b)):
                
                # Get boundary names for this segment
                start_name_a = clean_name(interpreted_bed_a[start_idx_a])
                end_name_a = clean_name(interpreted_bed_a[end_idx_a])
                start_name_b = clean_name(interpreted_bed_b[start_idx_b])
                end_name_b = clean_name(interpreted_bed_b[end_idx_b])
                
                # Check if at least one boundary matches (top OR bottom)
                top_match = start_name_a and start_name_b and start_name_a == start_name_b
                bottom_match = end_name_a and end_name_b and end_name_a == end_name_b
                
                if top_match or bottom_match:
                    matching_pairs.append(pair)
                    
                    # Create detailed description with 1-based pair numbers
                    match_type = []
                    if top_match:
                        matching_boundary_names.add(start_name_a)
                        match_type.append(f"top:{start_name_a}")
                    if bottom_match:
                        matching_boundary_names.add(end_name_a)
                        match_type.append(f"bottom:{end_name_a}")
                    
                    # Store with sorting key (use start boundary index for depth order)
                    matching_details.append({
                        'sort_key': start_idx_a,  # Use for sorting from top to bottom
                        'description': f"Segment pair ({seg_a_idx+1},{seg_b_idx+1}): [{','.join(match_type)}]"
                    })
    
    # Sort matching details from top to bottom (by boundary index)
    matching_details.sort(key=lambda x: x['sort_key'])
    
    # Step 2: Filter mappings that cover all required boundary names
    target_mappings = []
    
    # Load the CSV if it's a filename string
    if isinstance(sequential_mappings_csv, str):
        mappings_df = pd.read_csv(sequential_mappings_csv)
    else:
        mappings_df = sequential_mappings_csv
    
    for _, mapping_row in mappings_df.iterrows():
        # The segment pairs are stored in the 'path' column
        mapping_pairs = mapping_row['path']
        
        # Parse pairs if they're stored as string
        if isinstance(mapping_pairs, str):
            # Handle different string formats
            if ';' in mapping_pairs:
                try:
                    pairs_list = []
                    for pair_str in mapping_pairs.split(';'):
                        a, b = pair_str.split(',')
                        # Convert from 1-based to 0-based indexing
                        pairs_list.append((int(a) - 1, int(b) - 1))
                    mapping_pairs = pairs_list
                except:
                    continue
            else:
                try:
                    import ast
                    mapping_pairs = ast.literal_eval(mapping_pairs)
                    # Convert from 1-based to 0-based indexing
                    mapping_pairs = [(a - 1, b - 1) for a, b in mapping_pairs]
                except:
                    continue
        else:
            # Convert from 1-based to 0-based indexing if not string
            mapping_pairs = [(a - 1, b - 1) for a, b in mapping_pairs]
        
        # Check what boundary names are covered by this mapping
        covered_boundaries = set()
        
        for seg_a_idx, seg_b_idx in mapping_pairs:
            # Check if these segment indices are valid
            if seg_a_idx < len(segments_a) and seg_b_idx < len(segments_b):
                seg_a = segments_a[seg_a_idx]
                seg_b = segments_b[seg_b_idx]
                
                # Only check consecutive segments
                if seg_a[1] == seg_a[0] + 1 and seg_b[1] == seg_b[0] + 1:
                    start_idx_a, end_idx_a = seg_a
                    start_idx_b, end_idx_b = seg_b
                    
                    if (start_idx_a < len(interpreted_bed_a) and end_idx_a < len(interpreted_bed_a) and
                        start_idx_b < len(interpreted_bed_b) and end_idx_b < len(interpreted_bed_b)):
                        
                        start_name_a = clean_name(interpreted_bed_a[start_idx_a])
                        end_name_a = clean_name(interpreted_bed_a[end_idx_a])
                        start_name_b = clean_name(interpreted_bed_b[start_idx_b])
                        end_name_b = clean_name(interpreted_bed_b[end_idx_b])
                        
                        # Check if at least one boundary matches
                        top_match = start_name_a and start_name_b and start_name_a == start_name_b
                        bottom_match = end_name_a and end_name_b and end_name_a == end_name_b
                        
                        if top_match:
                            covered_boundaries.add(start_name_a)
                        if bottom_match:
                            covered_boundaries.add(end_name_a)
        
        # Check if this mapping covers ALL required boundary names
        if matching_boundary_names.issubset(covered_boundaries):
            target_mappings.append(mapping_row)
    
    # Convert to DataFrame if mappings found
    if target_mappings:
        target_mappings_df = pd.DataFrame(target_mappings).reset_index(drop=True)
        
        # Initialize return variables
        top_target_mapping_ids = []
        top_target_mapping_pairs = []
        
        # Apply ranking if metric weights are provided
        if metric_weight is not None:
            # Default metrics configuration
            default_weights = {
                'perc_diag': 0.0,
                'norm_dtw': 1.0,
                'dtw_ratio': 0.0,
                'corr_coef': 1.0,
                'dtw_warp_eff': 0.0,
                'perc_age_overlap': 0.0
            }
            
            # Fixed higher_is_better configuration
            higher_is_better_config = {
                'perc_diag': True,
                'norm_dtw': False,
                'dtw_ratio': False,
                'corr_coef': True,
                'dtw_warp_eff': True,
                'perc_age_overlap': True,
            }
            
            # Use provided weights or default weights
            weights = metric_weight
            
            # Clean the data
            required_cols = ['corr_coef', 'perc_diag', 'perc_age_overlap']
            existing_cols = [col for col in required_cols if col in target_mappings_df.columns]
            if existing_cols:
                target_mappings_df = target_mappings_df.replace([np.inf, -np.inf], np.nan).dropna(subset=existing_cols)
            
            # Calculate combined score
            target_mappings_df['combined_score'] = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in target_mappings_df.columns:
                    # Make sure we have valid data to work with
                    valid_data = target_mappings_df[~target_mappings_df[metric].isna()]
                    
                    if len(valid_data) > 0:
                        min_val = valid_data[metric].min()
                        max_val = valid_data[metric].max()
                        
                        # Only normalize if there's a range
                        if max_val > min_val:
                            if higher_is_better_config.get(metric, True):
                                # For metrics where higher values are better
                                target_mappings_df[f'{metric}_score'] = (target_mappings_df[metric] - min_val) / (max_val - min_val)
                            else:
                                # For metrics where lower values are better
                                target_mappings_df[f'{metric}_score'] = 1 - ((target_mappings_df[metric] - min_val) / (max_val - min_val))
                        else:
                            # If all values are the same, assign a score of 1
                            target_mappings_df[f'{metric}_score'] = 1.0
                            
                        # Add to weighted sum
                        target_mappings_df['combined_score'] += target_mappings_df[f'{metric}_score'].fillna(0) * weight
                        total_weight += weight
            
            # Normalize the combined score by the total weight
            if total_weight > 0:
                target_mappings_df['combined_score'] = target_mappings_df['combined_score'] / total_weight
            else:
                target_mappings_df['combined_score'] = 0.0
            
            # Handle NaN values in combined score
            target_mappings_df['combined_score'] = target_mappings_df['combined_score'].fillna(0)
            
            # Sort by combined score
            target_mappings_df = target_mappings_df.sort_values(by='combined_score', ascending=False)
            
            # Get top N for display and return
            top_mappings = target_mappings_df.head(top_n)
            
            print(f"Number of mappings found: {len(target_mappings_df)} out of {len(mappings_df)}")
            print(f"Required boundary names: {sorted(matching_boundary_names)}")
            print(f"Matching boundary correlations (top to bottom):")
            for detail in matching_details:
                print(f"  {detail['description']}")
            
            print(f"\nTop {min(top_n, len(target_mappings_df))} mappings by combined score:")
            for idx, row in top_mappings.iterrows():
                if 'mapping_id' in row:
                    mapping_id = int(row['mapping_id'])
                    top_target_mapping_ids.append(mapping_id)
                    print(f"Mapping ID {mapping_id}: Combined Score={row['combined_score']:.3f}")
                    
                    # Parse the path and convert to valid_pairs_to_combine
                    if 'path' in row:
                        target_data_row = parse_compact_path(row['path'])
                        # Convert 1-based to 0-based indices for python
                        valid_pairs_to_combine = [(a-1, b-1) for a, b in target_data_row]
                        top_target_mapping_pairs.append(valid_pairs_to_combine)
                    else:
                        # If no path column, append empty list
                        top_target_mapping_pairs.append([])
                    
                    # Print all available metrics
                    metric_outputs = []
                    if 'length' in row and weights.get('length', 0) != 0:
                        metric_outputs.append(f"dtw_path_length={row['length']:.1f}")
                    if 'corr_coef' in row and weights.get('corr_coef', 0) != 0:
                        metric_outputs.append(f"correlation coefficient r={row['corr_coef']:.3f}")
                    if 'perc_diag' in row and weights.get('perc_diag', 0) != 0:
                        metric_outputs.append(f"perc_diag={row['perc_diag']:.1f}%")
                    if 'norm_dtw' in row and weights.get('norm_dtw', 0) != 0:
                        metric_outputs.append(f"norm_dtw={row['norm_dtw']:.3f}")
                    if 'dtw_ratio' in row and weights.get('dtw_ratio', 0) != 0:
                        metric_outputs.append(f"dtw_ratio={row['dtw_ratio']:.3f}")
                    if 'dtw_warp_eff' in row and weights.get('dtw_warp_eff', 0) != 0:
                        metric_outputs.append(f"dtw_warp_eff={row['dtw_warp_eff']:.1f}%")
                    if 'perc_age_overlap' in row and weights.get('perc_age_overlap', 0) != 0:
                        metric_outputs.append(f"perc_age_overlap={row['perc_age_overlap']:.1f}%")
                    
                    # Print metrics with proper indentation
                    for metric_output in metric_outputs:
                        print(f"  {metric_output}")
                    
                    # Print valid_pairs_to_combine at the end of each mapping with 1-based numbering
                    if len(top_target_mapping_pairs) > 0 and len(top_target_mapping_pairs[-1]) > 0:
                        # Convert back to 1-based for display
                        pairs_display = [(a+1, b+1) for a, b in top_target_mapping_pairs[-1]]
                        print(f"  valid_pairs_to_combine={pairs_display}")
                    else:
                        print(f"  valid_pairs_to_combine=[]")
                    print("")
            
            # Handle case where no valid mappings found
            if not top_mappings.empty:
                best_mapping_id = top_target_mapping_ids[0] if top_target_mapping_ids else 'None'
                print(f"Best mapping ID: {best_mapping_id}")
            else:
                print("Warning: No valid mappings found")
            
        else:
            print(f"Found {len(target_mappings_df)} target mappings")
            print(f"Required boundary names: {sorted(matching_boundary_names)}")
            print(f"Matching boundary correlations (top to bottom):")
            for detail in matching_details:
                print(f"  {detail['description']}")
            
            # If no metric weights, just return top N by original order
            for idx, row in target_mappings_df.head(top_n).iterrows():
                if 'mapping_id' in row:
                    mapping_id = int(row['mapping_id'])
                    top_target_mapping_ids.append(mapping_id)
                    
                    # Parse the path and convert to valid_pairs_to_combine
                    if 'path' in row:
                        target_data_row = parse_compact_path(row['path'])
                        # Convert 1-based to 0-based indices for python
                        valid_pairs_to_combine = [(a-1, b-1) for a, b in target_data_row]
                        top_target_mapping_pairs.append(valid_pairs_to_combine)
                    else:
                        # If no path column, append empty list
                        top_target_mapping_pairs.append([])
        
        # Return tuple with IDs, pairs, and full dataframe
        return top_target_mapping_ids, top_target_mapping_pairs, target_mappings_df
    else:
        print("No target mappings found")
        print(f"Required boundary names: {sorted(matching_boundary_names)}")
        print(f"Matching boundary correlations (top to bottom):")
        for detail in matching_details:
            print(f"  {detail['description']}")
        return [], [], pd.DataFrame()