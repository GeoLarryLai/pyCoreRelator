"""
Path processing and manipulation functions
"""

import numpy as np
import pandas as pd
import csv


def combine_segment_dtw_results(dtw_results, segment_pairs, segments_a, segments_b, 
                               depth_boundaries_a, depth_boundaries_b, log_a, log_b):
    """
    Combine DTW results from multiple segment pairs into a unified result.
    
    Parameters:
    -----------
    dtw_results : dict
        Dictionary containing DTW results for each segment pair from run_comprehensive_dtw_analysis
    segment_pairs : list
        List of tuples (a_idx, b_idx) for segment pairs to combine
    segments_a, segments_b : list
        Segments in log_a and log_b
    depth_boundaries_a, depth_boundaries_b : list
        Depth boundaries for log_a and log_b
    
    Returns:
    --------
    tuple
        (combined_wp, combined_D, combined_quality)
    """
    
    # Initialize lists to store all warping paths and quality indicators
    all_warping_paths = []
    all_quality_indicators = []
    
    # Check if segment_pairs is empty
    if not segment_pairs or len(segment_pairs) == 0:
        print("No segment pairs provided to combine.")
        return None, None, None
    
    # Process each segment pair
    for a_idx, b_idx in segment_pairs:
        # Check if this pair exists in dtw_results
        if (a_idx, b_idx) not in dtw_results:
            print(f"Warning: Segment pair ({a_idx+1}, {b_idx+1}) not found in DTW results. Skipping.")
            continue
        
        # Get paths, cost matrices, and quality indicators for this pair
        paths, cost_matrices, quality_indicators = dtw_results[(a_idx, b_idx)]
        
        # Skip if no valid paths exist
        if not paths or len(paths) == 0:
            print(f"Warning: No valid path for segment pair ({a_idx+1}, {b_idx+1}). Skipping.")
            continue
        
        # Add the best path (first one) to the list of all paths
        all_warping_paths.append(paths[0])
        
        # Add the quality indicators to the list
        if quality_indicators and len(quality_indicators) > 0:
            all_quality_indicators.append(quality_indicators[0])
    
    # If no valid paths were found, return None
    if not all_warping_paths:
        print("No valid warping paths found in the selected segment pairs.")
        return None, None, None
    
    # Combine warping paths
    # First, sort all paths by their first point's coordinates
    all_warping_paths.sort(key=lambda wp: (wp[0, 0], wp[0, 1]))
    
    # Concatenate all warping paths
    combined_wp = np.vstack(all_warping_paths)
    
    # Remove duplicate points (which can occur at boundaries between segments)
    combined_wp = np.unique(combined_wp, axis=0)
    
    # Sort the combined path by the first coordinate
    combined_wp = combined_wp[combined_wp[:, 0].argsort()]
    
    # Calculate average quality indicators
    if all_quality_indicators:
        # Collect age overlap values
        age_overlap_values = []
        for qi in all_quality_indicators:
            if 'perc_age_overlap' in qi:
                age_overlap_values.append(float(qi['perc_age_overlap']))
        
        combined_quality = compute_combined_path_metrics(
            combined_wp, log_a, log_b, all_quality_indicators, age_overlap_values
        )
    else:
        combined_quality = None
        
    return combined_wp, combined_quality


def compute_combined_path_metrics(combined_wp, log_a, log_b, segment_quality_indicators, age_overlap_values=None):
    """
    Compute quality metrics from combined warping path and log data.
    
    Parameters:
    -----------
    combined_wp : np.ndarray
        Combined warping path
    log_a, log_b : np.ndarray
        Log data arrays
    segment_quality_indicators : list
        Quality indicators from individual segments
    age_overlap_values : list, optional
        Age overlap values for averaging
    
    Returns:
    --------
    dict : Combined quality metrics
    """
    from ..core.quality_metrics import compute_quality_indicators
    
    # Initialize metrics
    metrics = {
        'norm_dtw': 0.0,
        'dtw_ratio': 0.0,
        'variance_deviation': 0.0,
        'perc_diag': 0.0,
        'corr_coef': 0.0,
        'match_min': 0.0,
        'match_mean': 0.0,
        'perc_age_overlap': 0.0
    }
    
    # Collect distance metrics for summing
    metric_values = {metric: [] for metric in metrics}
    for qi in segment_quality_indicators:
        for metric in ['norm_dtw', 'match_min', 'match_mean']:
            if metric in qi:
                metric_values[metric].append(float(qi[metric]))
    
    # Sum distance metrics
    for metric in ['norm_dtw', 'match_min', 'match_mean']:
        values = metric_values[metric]
        if values:
            metrics[metric] = float(sum(values))
    
    # Compute from combined warping path if available
    if combined_wp is not None and len(combined_wp) > 1:
        # Calculate diagonality
        def calculate_diagonality(wp):
            if len(wp) < 2:
                return 1.0
            a_indices = wp[:, 0]
            b_indices = wp[:, 1]
            a_range = np.max(a_indices) - np.min(a_indices)
            b_range = np.max(b_indices) - np.min(b_indices)
            if a_range == 0 or b_range == 0:
                return 0.0
            a_norm = (a_indices - np.min(a_indices)) / a_range
            b_norm = (b_indices - np.min(b_indices)) / b_range
            distances = np.abs(a_norm - b_norm)
            return float(1.0 - np.mean(distances))
        
        metrics['perc_diag'] = float(calculate_diagonality(combined_wp) * 100)
        
        # Extract log values at warping path points
        p_indices = combined_wp[:, 0]
        q_indices = combined_wp[:, 1]
        
        if log_a.ndim > 1:
            aligned_log_a = log_a[p_indices].mean(axis=1)
        else:
            aligned_log_a = log_a[p_indices]
            
        if log_b.ndim > 1:
            aligned_log_b = log_b[q_indices].mean(axis=1)
        else:
            aligned_log_b = log_b[q_indices]
        
        # Create dummy cost matrix
        dummy_D = np.array([[np.linalg.norm(aligned_log_a - aligned_log_b)]])
        
        # Compute combined metrics
        combined_metrics = compute_quality_indicators(aligned_log_a, aligned_log_b, p_indices, q_indices, dummy_D)
        
        # Use combined calculations for these three metrics
        metrics['dtw_ratio'] = float(combined_metrics.get('dtw_ratio', 0.0))
        metrics['variance_deviation'] = float(combined_metrics.get('variance_deviation', 0.0))
        metrics['corr_coef'] = float(combined_metrics.get('corr_coef', 0.0))
    
    # Average age overlap
    if age_overlap_values:
        metrics['perc_age_overlap'] = float(sum(age_overlap_values) / len(age_overlap_values))
    
    return metrics


def load_sequential_mappings(csv_path):
    """
    Load sequential mappings from a CSV file.
    UPDATED to handle new compact format.
    """
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    mappings = []
    try:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            try:
                # UPDATED: Parse compact format
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
    
    Parameters:
    -----------
    path_info : dict
        First path information dictionary
    other_path_info : dict
        Second path information dictionary
    early_terminate : bool
        Whether to use early termination checks
    
    Returns:
    --------
    tuple
        (is_subset, is_superset) boolean flags
    """
    # Quick length check for early termination
    if early_terminate:
        # Cannot be a superset if shorter
        if path_info['length'] < other_path_info['length']:
            return (False, False)
        
        # Cannot be a subset if longer
        if path_info['length'] > other_path_info['length']:
            return (False, False)
    
    # Full set comparison
    path_set = path_info['path_set']
    other_path_set = other_path_info['path_set']
    
    is_subset = path_set.issubset(other_path_set)
    is_superset = path_set.issuperset(other_path_set)
    
    return (is_subset, is_superset)


def filter_against_existing(new_path, filtered_paths, group_writer):
    """
    Filter a new path against existing filtered paths with optimized checking.
    
    Parameters:
    -----------
    new_path : dict
        New path information dictionary
    filtered_paths : list
        List of existing filtered path information dictionaries
    group_writer : csv.writer
        CSV writer to write accepted paths
    
    Returns:
    --------
    tuple
        (is_valid, paths_to_remove, updated_count) where:
        - is_valid: bool indicating if the new path should be added
        - paths_to_remove: list of indices of paths to remove
        - updated_count: int indicating if count should be incremented
    """
    is_valid = True
    paths_to_remove = []
    
    # Group existing paths by length for efficient filtering
    length_groups = {}
    for i, existing_path in enumerate(filtered_paths):
        length = existing_path['length']
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append((i, existing_path))
    
    # Check if any existing path contains this path (making it invalid)
    # Only check paths with length >= new_path length
    for length in sorted(length_groups.keys(), reverse=True):
        if length < new_path['length']:
            break  # No need to check shorter paths
            
        for i, existing_path in length_groups[length]:
            _, is_superset = is_subset_or_superset(existing_path, new_path)
            if is_superset:
                is_valid = False
                return (is_valid, [], 0)  # Early exit
    
    # If valid, check if it contains any existing paths
    # Only check paths with length <= new_path length
    if is_valid:
        for length in sorted(length_groups.keys()):
            if length > new_path['length']:
                break  # No need to check longer paths
                
            for i, existing_path in length_groups[length]:
                is_subset, _ = is_subset_or_superset(existing_path, new_path)
                if is_subset:
                    paths_to_remove.append(i)
    
    # If valid, write to output and increment count
    if is_valid:
        group_writer.writerow(new_path['row_data'])
        return (is_valid, paths_to_remove, 1)
    
    return (is_valid, paths_to_remove, 0)