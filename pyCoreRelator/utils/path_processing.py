"""
Path utility functions for pyCoreRelator.

Included Functions:
 from multiple segment pairs
 from combined warping paths
 from CSV files
- is_subset_or_superset: Check subset/superset relationships between paths
- filter_against_existing: Filter new paths against existing filtered paths
 based on multiple quality metrics
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


