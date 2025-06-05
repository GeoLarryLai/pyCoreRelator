"""
Path processing and manipulation functions
"""

import numpy as np
import pandas as pd
import csv


def combine_segment_dtw_results(dtw_results, segment_pairs, segments_a, segments_b, 
                               depth_boundaries_a, depth_boundaries_b):
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
        combined_quality = {}
        for key in all_quality_indicators[0].keys():
            # Calculate the average for each quality indicator
            combined_quality[key] = np.mean([qi[key] for qi in all_quality_indicators if key in qi])
    else:
        combined_quality = None
    
    return combined_wp, combined_quality


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