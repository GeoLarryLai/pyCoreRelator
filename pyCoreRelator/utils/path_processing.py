"""
Path processing and manipulation functions for pyCoreRelator.

Included Functions:
- combine_segment_dtw_results: Combine DTW results from multiple segment pairs
- compute_combined_path_metrics: Compute quality metrics from combined warping paths
- load_sequential_mappings: Load sequential mappings from CSV files
- is_subset_or_superset: Check subset/superset relationships between paths
- filter_against_existing: Filter new paths against existing filtered paths
- find_best_mappings: Find the best DTW mappings based on multiple quality metrics
  (supports both standard best mappings and boundary correlation filtering modes)

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
                       top_n=10, 
                       filter_shortest_dtw=True,
                       metric_weight=None,
                       picked_depths_a_cat1=None,
                       picked_depths_b_cat1=None,
                       interpreted_bed_a=None,
                       interpreted_bed_b=None,
                       valid_dtw_pairs=None,
                       segments_a=None,
                       segments_b=None):
    """
    Find the best DTW mappings based on multiple quality metrics with configurable scoring.
    
    If boundary correlation parameters are provided, filters mappings that comply with 
    boundary correlations between cores. If those parameters are not provided or no 
    matching boundaries are found, behaves as standard best mappings finder.
    """
    
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    def clean_name(name):
        """Helper function to clean bed names (remove "?")"""
        if pd.isna(name) or name == '':
            return ''
        return name.replace('?', '').strip()
    
    def _calculate_combined_scores(df_input, weights, higher_is_better_config):
        """Calculate combined scores for each mapping"""
        df_for_ranking = df_input.copy()
        
        # Initialize combined score
        df_for_ranking['combined_score'] = 0.0
        
        # Calculate weighted combined score
        total_weight = 0
        for metric, weight in weights.items():
            if weight != 0 and metric in df_for_ranking.columns:
                # Normalize metric values to 0-1 range
                metric_values = df_for_ranking[metric].copy()
                metric_min = metric_values.min()
                metric_max = metric_values.max()
                
                if metric_max != metric_min:
                    normalized_values = (metric_values - metric_min) / (metric_max - metric_min)
                    
                    # Flip values if lower is better
                    if not higher_is_better_config.get(metric, True):
                        normalized_values = 1 - normalized_values
                    
                    df_for_ranking['combined_score'] += weight * normalized_values
                    total_weight += weight
        
        # Normalize by total weight if weights were applied
        if total_weight > 0:
            df_for_ranking['combined_score'] /= total_weight
        
        # Handle NaN values in combined score
        if df_for_ranking['combined_score'].isna().any():
            print("Warning: NaN values detected in combined scores. Replacing with zeros.")
            df_for_ranking['combined_score'] = df_for_ranking['combined_score'].fillna(0)
        
        return df_for_ranking
    
    def _print_results(top_mappings, weights, mode_title):
        """Print detailed results for mappings"""
        top_mapping_ids = []
        top_mapping_pairs = []
        
        print(f"\nTop {len(top_mappings)} mappings by combined score:")
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
            
            # Print valid_pairs_to_combine with display numbering (export values + 1)
            if len(top_mapping_pairs) > 0 and len(top_mapping_pairs[-1]) > 0:
                # Convert export values to display format (add 1 more)
                pairs_display = [(a+1, b+1) for a, b in top_mapping_pairs[-1]]
                print(f"  valid_pairs_to_combine={pairs_display}")
            else:
                print(f"  valid_pairs_to_combine=[]")
            
            # Print matched datums if available
            if 'matched_datums' in row and row['matched_datums']:
                datums_list = row['matched_datums'].split(',')
                print(f"  matched_datums={datums_list}")
            
            print("")
        
        # Handle case where no valid mappings found
        if not top_mappings.empty:
            best_mapping_id = top_mapping_ids[0] if top_mapping_ids else 'None'
            print(f"Best mapping ID: {best_mapping_id}")
        else:
            print("Warning: No valid mappings found")
        
        return top_mapping_ids, top_mapping_pairs
    
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
    if isinstance(csv_file_path, str):
        dtw_results_df = pd.read_csv(csv_file_path)
    else:
        dtw_results_df = csv_file_path
    
    # Always add ranking columns after loading CSV
    if 'Ranking' not in dtw_results_df.columns:
        dtw_results_df['Ranking'] = ''
    if 'Ranking_datums' not in dtw_results_df.columns:
        dtw_results_df['Ranking_datums'] = ''
   
    # Check if we should use boundary correlation mode
    use_boundary_mode = all(param is not None for param in [
        picked_depths_a_cat1, picked_depths_b_cat1, interpreted_bed_a, 
        interpreted_bed_b, valid_dtw_pairs, segments_a, segments_b
    ])
    
    target_mappings_df = None
    matching_boundary_names = set()
    matching_details = []
    
    if use_boundary_mode:
        # Check if all interpreted bed names are empty
        all_beds_empty = (
            all(pd.isna(name) or clean_name(name) == '' for name in interpreted_bed_a) and
            all(pd.isna(name) or clean_name(name) == '' for name in interpreted_bed_b)
        )
        
        if not all_beds_empty:
            # Step 1: Find all bed names that appear in both interpreted_bed_a and interpreted_bed_b
            bed_names_a = set()
            bed_names_b = set()
            
            for name in interpreted_bed_a:
                cleaned = clean_name(name)
                if cleaned:
                    bed_names_a.add(cleaned)
            
            for name in interpreted_bed_b:
                cleaned = clean_name(name)
                if cleaned:
                    bed_names_b.add(cleaned)
            
            # Find common bed names
            common_bed_names = bed_names_a.intersection(bed_names_b)
            
            # Step 2: The common bed names ARE our matching datums
            matching_boundary_names = common_bed_names.copy()
            
            # Find segment pairs that contain any of these common bed names
            # (for finding correlations, but the matching datums are already identified)
            matching_pairs = []
            
            if common_bed_names:
                for pair in valid_dtw_pairs:
                    seg_a_idx, seg_b_idx = pair
                    
                    # Get the actual segment tuples
                    seg_a = segments_a[seg_a_idx]
                    seg_b = segments_b[seg_b_idx]
                    
                    # Only check consecutive segments (i, i+1) - these represent intervals between boundaries
                    if seg_a[1] == seg_a[0] + 1 and seg_b[1] == seg_b[0] + 1:
                        start_idx_a, end_idx_a = seg_a
                        start_idx_b, end_idx_b = seg_b
                        
                        # Check if boundaries exist in interpreted_bed arrays
                        if (start_idx_a < len(interpreted_bed_a) and end_idx_a < len(interpreted_bed_a) and
                            start_idx_b < len(interpreted_bed_b) and end_idx_b < len(interpreted_bed_b)):
                            
                            # Get boundary names for this segment
                            start_name_a = clean_name(interpreted_bed_a[start_idx_a])
                            end_name_a = clean_name(interpreted_bed_a[end_idx_a])
                            start_name_b = clean_name(interpreted_bed_b[start_idx_b])
                            end_name_b = clean_name(interpreted_bed_b[end_idx_b])
                            
                            # Check for matching boundaries (top-to-top or bottom-to-bottom)
                            top_match = False
                            bottom_match = False
                            matched_names = []
                            
                            # Check if top boundaries match
                            if start_name_a and start_name_b and start_name_a == start_name_b and start_name_a in common_bed_names:
                                top_match = True
                                matched_names.append(f"top:{start_name_a}")
                            
                            # Check if bottom boundaries match
                            if end_name_a and end_name_b and end_name_a == end_name_b and end_name_a in common_bed_names:
                                bottom_match = True
                                matched_names.append(f"bottom:{end_name_a}")
                            
                            # Only include if at least one boundary matches
                            if top_match or bottom_match:
                                matching_pairs.append(pair)
                                
                                matching_details.append({
                                    'sort_key': start_idx_a,
                                    'description': f"Segment pair ({seg_a_idx+3},{seg_b_idx+3}): [{','.join(matched_names)}]"
                                })
                
                # Sort matching details from top to bottom (by boundary index)
                matching_details.sort(key=lambda x: x['sort_key'])

            # Convert matching_pairs to set for faster lookup
            if matching_pairs:
                valid_pairs_set = set(matching_pairs)

        # Print matching datums and segment pairs
        print(f"Matching datums: {sorted(matching_boundary_names)}")
        if matching_boundary_names:
            print(f"Datum matching correlations:")
            for detail in matching_details:
                print(f"  {detail['description']}")

    # Step 2: Find DTW results that cover all matching datums
    if matching_pairs:
        target_mappings = []
        
        # Convert boundary pairs to 1-based format for comparison with CSV path values
        boundary_pairs_1based = set((seg_a_idx + 3, seg_b_idx + 3) for seg_a_idx, seg_b_idx in valid_pairs_set)
        
        # Extract all datum names from matching_details to check coverage
        import re
        all_datums_in_pairs = set()
        for detail in matching_details:
            description = detail['description']
            bed_names = re.findall(r'(?:top|bottom):(\w+)', description)
            all_datums_in_pairs.update(bed_names)
        
        for _, mapping_row in dtw_results_df.iterrows():
            path_str = mapping_row.get('path')
            
            if pd.isna(path_str) or path_str == '':
                continue
            
            # Parse path: "2,2;4,4;6,6" -> {(2,2), (4,4), (6,6)}
            try:
                path_pairs = set()
                for pair_str in path_str.split(';'):
                    a, b = pair_str.split(',')
                    path_pairs.add((int(a), int(b)))
            except:
                continue  # Skip if parsing fails
            
            # Check which segment pairs from our list are in this path
            matched_pairs = boundary_pairs_1based.intersection(path_pairs)
            
            # Extract datums covered by the matched segment pairs
            covered_datums = set()
            for i, detail in enumerate(matching_details):
                seg_pair_display = detail['description'].split(':')[0].strip().replace('Segment pair ', '')
                # Check if this segment pair is in the matched set
                if matching_pairs[i] in valid_pairs_set:
                    seg_a_idx, seg_b_idx = matching_pairs[i]
                    if (seg_a_idx + 3, seg_b_idx + 3) in matched_pairs:
                        # Extract datums from this matched pair
                        bed_names = re.findall(r'(?:top|bottom):(\w+)', detail['description'])
                        covered_datums.update(bed_names)
            
            # Check if ALL datums are covered
            if all_datums_in_pairs.issubset(covered_datums):
                # Store the matched datums with this mapping
                mapping_dict = mapping_row.to_dict()
                mapping_dict['matched_datums'] = ','.join(sorted(covered_datums))
                target_mappings.append(mapping_dict)
        
        # Convert to DataFrame if mappings found
        if target_mappings:
            target_mappings_df = pd.DataFrame(target_mappings).reset_index(drop=True)
    
    # Determine which dataframe to use for ranking
    if target_mappings_df is not None and not target_mappings_df.empty:
        working_df = target_mappings_df
        mode_title = "Target Mappings (Boundary Correlation)"
        print(f"\n{len(target_mappings_df)}/{len(dtw_results_df)} mappings found with all {len(matching_boundary_names)} matched datums ({len(matching_pairs)} segment pairs)")
    elif use_boundary_mode and matching_pairs:
        # No mappings contain ALL segment pairs, but we have segment pairs
        # Find mappings that contain the MOST segment pairs from our list
        print("No mappings found with matched datums. Searching for mappings with most matched segment pairs.")
        
        # Convert boundary pairs to 1-based format for comparison with CSV path values
        boundary_pairs_1based = set((seg_a_idx + 3, seg_b_idx + 3) for seg_a_idx, seg_b_idx in valid_pairs_set)
        
        # Extract all datum names for counting
        import re
        all_datums_in_pairs = set()
        for detail in matching_details:
            description = detail['description']
            bed_names = re.findall(r'(?:top|bottom):(\w+)', description)
            all_datums_in_pairs.update(bed_names)
        
        # Count how many matching segment pairs and datums each mapping contains
        mapping_scores = []
        for _, mapping_row in dtw_results_df.iterrows():
            path_str = mapping_row.get('path')
            
            if pd.isna(path_str) or path_str == '':
                continue
            
            # Parse path: "2,2;4,4;6,6" -> {(2,2), (4,4), (6,6)}
            try:
                path_pairs = set()
                for pair_str in path_str.split(';'):
                    a, b = pair_str.split(',')
                    path_pairs.add((int(a), int(b)))
            except:
                continue  # Skip if parsing fails
            
            # Count how many of our target segment pairs are in this mapping
            matched_pairs = boundary_pairs_1based.intersection(path_pairs)
            matching_count = len(matched_pairs)
            
            # Count how many datums are covered
            covered_datums = set()
            for i, detail in enumerate(matching_details):
                if matching_pairs[i] in valid_pairs_set:
                    seg_a_idx, seg_b_idx = matching_pairs[i]
                    if (seg_a_idx + 3, seg_b_idx + 3) in matched_pairs:
                        bed_names = re.findall(r'(?:top|bottom):(\w+)', detail['description'])
                        covered_datums.update(bed_names)
            datum_count = len(covered_datums)
            
            if matching_count > 0:
                # Add the mapping with its matching count
                mapping_dict = mapping_row.to_dict()
                mapping_dict['matching_segment_count'] = matching_count
                mapping_dict['matching_datum_count'] = datum_count
                mapping_dict['matched_datums'] = ','.join(sorted(covered_datums))
                mapping_scores.append(mapping_dict)
        
        if mapping_scores:
            # Convert to DataFrame and find mappings with the highest datum count first, then segment pairs
            scored_df = pd.DataFrame(mapping_scores)
            max_datum_count = scored_df['matching_datum_count'].max()
            best_by_datum = scored_df[scored_df['matching_datum_count'] == max_datum_count]
            max_segment_count = best_by_datum['matching_segment_count'].max()
            best_partial_mappings = best_by_datum[best_by_datum['matching_segment_count'] == max_segment_count]
            
            working_df = best_partial_mappings
            mode_title = f"Best Partial Mappings ({max_datum_count}/{len(all_datums_in_pairs)} datums, {max_segment_count}/{len(boundary_pairs_1based)} segment pairs)"
            print(f"\n{len(best_partial_mappings)}/{len(dtw_results_df)} mappings found with {max_datum_count}/{len(all_datums_in_pairs)} matched datums ({max_segment_count}/{len(boundary_pairs_1based)} segment pairs)")
        else:
            # No mappings contain any of our segment pairs - fall back to standard
            working_df = dtw_results_df
            mode_title = "Overall Best Mappings"
            print(f"=== Top {top_n} Overall Best Mappings ===")
            print("No mappings found with any matched segment pairs. Falling back to standard best mappings mode.")
    else:
        working_df = dtw_results_df
        mode_title = "Overall Best Mappings"
        print(f"=== Top {top_n} Overall Best Mappings ===")
        
        if use_boundary_mode and not matching_boundary_names:
            print("No matching datums found. Falling back to standard best mappings mode.")
        elif use_boundary_mode:
            print("No segment pairs found. Falling back to standard best mappings mode.")
    
    # Step 1: ALWAYS compute standard mode ranking for all mappings first
    # Clean the data for standard ranking
    standard_working_df = dtw_results_df.copy()
    required_cols = ['corr_coef', 'perc_diag', 'perc_age_overlap']
    existing_cols = [col for col in required_cols if col in standard_working_df.columns]
    if existing_cols:
        standard_working_df = standard_working_df.replace([np.inf, -np.inf], np.nan).dropna(subset=existing_cols)
    
    # Filter for shortest DTW path length for standard mode (only when not in boundary mode)
    if filter_shortest_dtw and not use_boundary_mode and 'length' in standard_working_df.columns:
        standard_working_df['dtw_path_length'] = standard_working_df['length']
        min_length = standard_working_df['dtw_path_length'].min()
        standard_shortest = standard_working_df[standard_working_df['dtw_path_length'] == min_length]
        print(f"Filtering for shortest DTW path length: {min_length}")
    else:
        standard_shortest = standard_working_df
        if filter_shortest_dtw and not use_boundary_mode and 'length' not in standard_working_df.columns:
            print("Warning: No 'length' column found. Using all mappings.")
    
    # Calculate combined scores for standard mode
    standard_df_for_ranking = _calculate_combined_scores(standard_shortest.copy(), weights, higher_is_better_config)
    
    # Always calculate and append standard mode ranking to 'Ranking' column
    standard_ranked_df = standard_df_for_ranking.sort_values(by='combined_score', ascending=False)
    top_n_standard = standard_ranked_df.head(top_n)
    for i, (idx, row) in enumerate(top_n_standard.iterrows(), 1):
        if 'mapping_id' in row:
            mapping_id = int(row['mapping_id'])
            # Update the original dtw_results_df with standard ranking
            dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'Ranking'] = i
    
    # Step 2: Check if we should proceed with boundary mode
    if not use_boundary_mode or not matching_boundary_names:
        # No boundary mode or no matching names - return standard mode results
        print(f"=== Top {top_n} Overall Best Mappings ===")
        if use_boundary_mode and not matching_boundary_names:
            print("No matching datums found. Using standard best mappings mode.")
        
        top_mapping_ids, top_mapping_pairs = _print_results(top_n_standard, weights, "Overall Best Mappings")
        
        # Save the updated CSV if it was loaded from a file path
        if isinstance(csv_file_path, str):
            dtw_results_df.to_csv(csv_file_path, index=False)
        
        return top_mapping_ids, top_mapping_pairs, standard_ranked_df
    
    # Step 3: Proceed with boundary mode processing since we have matching names
    # The working_df and mode_title were already set above (lines 740-800)
    # Do NOT overwrite them here - they contain the filtered mappings we want to use
    
    # Clean the working dataframe for boundary mode
    existing_cols = [col for col in required_cols if col in working_df.columns]
    if existing_cols:
        working_df = working_df.replace([np.inf, -np.inf], np.nan).dropna(subset=existing_cols)
    
    # In boundary mode, don't filter for shortest DTW path length
    shortest_mappings = working_df
    
    # Create a copy for scoring calculations
    df_for_ranking = shortest_mappings.copy()
    
    # Calculate combined scores for boundary mode
    df_for_ranking = _calculate_combined_scores(df_for_ranking, weights, higher_is_better_config)
    
    # Get top N mappings by combined score for boundary mode
    top_mappings_df = df_for_ranking.sort_values(by='combined_score', ascending=False)
    
    # Add datums ranking if we have target mappings (matched boundary names) or partial mappings
    if target_mappings_df is not None and not target_mappings_df.empty:
        # Calculate combined scores for target mappings if not already done
        if 'combined_score' not in target_mappings_df.columns:
            target_mappings_df = _calculate_combined_scores(target_mappings_df, weights, higher_is_better_config)
        
        # Rank the target mappings (those with matched datums) for 'Ranking_datums' column
        target_ranked = target_mappings_df.sort_values(by='combined_score', ascending=False)
        for i, (idx, row) in enumerate(target_ranked.iterrows(), 1):
            if 'mapping_id' in row:
                mapping_id = int(row['mapping_id'])
                # Update the original dtw_results_df with datums ranking
                dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'Ranking_datums'] = i
    elif use_boundary_mode and matching_pairs and "Best Partial Mappings" in mode_title:
        # Rank partial mappings (those with some matched segment pairs) for 'Ranking_datums' column
        partial_ranked = top_mappings_df.sort_values(by='combined_score', ascending=False)
        for i, (idx, row) in enumerate(partial_ranked.iterrows(), 1):
            if 'mapping_id' in row:
                mapping_id = int(row['mapping_id'])
                # Update the original dtw_results_df with datums ranking
                dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'Ranking_datums'] = i
    
    # Print detailed results and return boundary mode results
    top_mapping_ids, top_mapping_pairs = _print_results(top_mappings_df.head(top_n), weights, mode_title)
    
    # Save the updated CSV if it was loaded from a file path
    if isinstance(csv_file_path, str):
        dtw_results_df.to_csv(csv_file_path, index=False)
    
    return top_mapping_ids, top_mapping_pairs, top_mappings_df