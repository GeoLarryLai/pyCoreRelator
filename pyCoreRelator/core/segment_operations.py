"""
Basic segment operations for core correlation analysis.

Included Functions:
- find_all_segments: Find segments in two logs using depth boundaries.
- build_connectivity_graph: Build predecessor and successor relationships between valid segment pairs.
- identify_special_segments: Identify special types of segments: tops, bottoms, dead ends, and orphans.
- filter_dead_end_pairs: Remove dead end and orphan segment pairs from the valid set.

This module provides fundamental segment analysis functionality for geological core 
correlation workflows, including segment decomposition, connectivity analysis, and 
filtering operations.
"""

import numpy as np
from collections import defaultdict


def find_all_segments(log_a, log_b, md_a, md_b, picked_depths_a=None, picked_depths_b=None, top_bottom=True, top_depth=0.0, mute_mode=False):
    """
    Find segments in two logs using depth boundaries to create consecutive and single-point segments.
    
    Converts user-picked depth values to indices in the log arrays and generates all possible
    segment combinations for DTW analysis.
    
    Args:
        log_a (array): Log data for core A
        log_b (array): Log data for core B
        md_a (array): Measured depth values corresponding to log_a
        md_b (array): Measured depth values corresponding to log_b
        picked_depths_a (list, optional): User-selected depth values for core A boundaries
        picked_depths_b (list, optional): User-selected depth values for core B boundaries
        top_bottom (bool): Whether to add top and bottom boundaries automatically
        top_depth (float): Depth value to use for top boundary
        mute_mode (bool, default=False): If True, suppress all print output
        
    Returns:
        tuple: (segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, depth_values_a, depth_values_b)
            - segments_a/b: List of (start_idx, end_idx) tuples for each segment
            - depth_boundaries_a/b: List of indices corresponding to depth values
            - depth_values_a/b: List of actual depth values used
    
    Example:
        >>> segments_a, segments_b, bounds_a, bounds_b, depths_a, depths_b = find_all_segments(
        ...     log_a, log_b, md_a, md_b, 
        ...     picked_depths_a=[0, 100, 200], 
        ...     picked_depths_b=[0, 150, 300]
        ... )
        >>> print(f"Core A has {len(segments_a)} segments")
        >>> print(f"First segment A spans indices {segments_a[0]}")
    """
    
    # Initialize depth lists
    if picked_depths_a is None:
        picked_depths_a = []
    if picked_depths_b is None:
        picked_depths_b = []
    
    # Ensure picked_depths are Python lists
    if isinstance(picked_depths_a, np.ndarray):
        depth_values_a = picked_depths_a.tolist()
    else:
        depth_values_a = list(picked_depths_a)
        
    if isinstance(picked_depths_b, np.ndarray):
        depth_values_b = picked_depths_b.tolist()
    else:
        depth_values_b = list(picked_depths_b)
    
    # Add top and bottom boundaries if requested
    if top_bottom:
        if top_depth not in depth_values_a:
            depth_values_a.append(top_depth)
        if md_a[-1] not in depth_values_a:
            depth_values_a.append(md_a[-1])
            
        if top_depth not in depth_values_b:
            depth_values_b.append(top_depth)
        if md_b[-1] not in depth_values_b:
            depth_values_b.append(md_b[-1])
    
    # Sort and remove duplicates
    depth_values_a = sorted(list(set(depth_values_a)))
    depth_values_b = sorted(list(set(depth_values_b)))
    
    # Create default segments if no depths specified
    if len(depth_values_a) == 0:
        if not mute_mode:
            print("Warning: No depth boundaries specified for log A. Using evenly spaced boundaries.")
        depth_values_a = [top_depth, md_a[len(log_a) // 3], md_a[2 * len(log_a) // 3], md_a[-1]]
    
    if len(depth_values_b) == 0:
        if not mute_mode:
            print("Warning: No depth boundaries specified for log B. Using evenly spaced boundaries.")
        depth_values_b = [top_depth, md_b[len(log_b) // 3], md_b[2 * len(log_b) // 3], md_b[-1]]
    
    def find_nearest_index(depth_array, depth_value):
        """Find the index in depth_array closest to the given depth_value."""
        return np.abs(np.array(depth_array) - depth_value).argmin()
    
    # Convert depth values to array indices
    depth_boundaries_a = [find_nearest_index(md_a, depth) for depth in depth_values_a]
    depth_boundaries_b = [find_nearest_index(md_b, depth) for depth in depth_values_b]
    
    # Generate consecutive and single-point segments
    segments_a = []
    for i in range(len(depth_boundaries_a)):
        segments_a.append((i, i))  # Single point segment
        if i < len(depth_boundaries_a) - 1:
            segments_a.append((i, i+1))  # Consecutive segment
    
    segments_b = []
    for i in range(len(depth_boundaries_b)):
        segments_b.append((i, i))  # Single point segment
        if i < len(depth_boundaries_b) - 1:
            segments_b.append((i, i+1))  # Consecutive segment

    # Print summary information
    if not mute_mode:
        print(f"\nLog A depth values: {[float(d) for d in depth_values_a]}")
        print(f"Log A depth boundaries: {[int(i) for i in depth_boundaries_a]}")
        print(f"\nLog B depth values: {[float(d) for d in depth_values_b]}")
        print(f"Log B depth boundaries: {[int(i) for i in depth_boundaries_b]}")
        print(f"Generated {len(segments_a)} possible segments for log A")
        print(f"Generated {len(segments_b)} possible segments for log B")
    
    return segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, depth_values_a, depth_values_b


def build_connectivity_graph(valid_dtw_pairs, detailed_pairs):
    """
    Build predecessor and successor relationships between valid segment pairs.
    
    Two segments are connected if the end depth of one segment matches the start depth
    of another segment for both cores A and B.
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs from DTW analysis
        detailed_pairs (dict): Dictionary mapping segment pairs to their depth details
        
    Returns:
        tuple: (successors, predecessors) dictionaries mapping segments to connected segments
    
    Example:
        >>> successors, predecessors = build_connectivity_graph(valid_pairs, details)
        >>> # Check what follows segment (1,2)
        >>> next_segments = successors.get((1,2), [])
        >>> print(f"Segment (1,2) connects to: {next_segments}")
    """
    
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    
    # Build connectivity by comparing end/start depths
    for a_idx, b_idx in valid_dtw_pairs:
        pair_details = detailed_pairs[(a_idx, b_idx)]
        a_end = pair_details['a_end']
        b_end = pair_details['b_end']
        
        for next_a_idx, next_b_idx in valid_dtw_pairs:
            if (a_idx, b_idx) != (next_a_idx, next_b_idx):
                next_details = detailed_pairs[(next_a_idx, next_b_idx)]
                next_a_start = next_details['a_start']
                next_b_start = next_details['b_start']
                
                # Check if segments connect exactly
                if (abs(next_a_start - a_end) < 1e-6 and 
                    abs(next_b_start - b_end) < 1e-6):
                    successors[(a_idx, b_idx)].append((next_a_idx, next_b_idx))
                    predecessors[(next_a_idx, next_b_idx)].append((a_idx, b_idx))
    
    return dict(successors), dict(predecessors)


def identify_special_segments(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b):
    """
    Identify special types of segments: tops, bottoms, dead ends, and orphans.
    
    - Top segments: Start at depth 0 for both cores
    - Bottom segments: End at maximum depth for both cores  
    - Dead ends: Have no successors (but aren't bottom segments)
    - Orphans: Have no predecessors (but aren't top segments)
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs
        detailed_pairs (dict): Segment depth details
        max_depth_a (float): Maximum depth for core A
        max_depth_b (float): Maximum depth for core B
        
    Returns:
        tuple: (top_segments, bottom_segments, dead_ends, orphans, successors, predecessors)
    
    Example:
        >>> tops, bottoms, dead, orphans, succ, pred = identify_special_segments(
        ...     valid_pairs, details, 1000.0, 1200.0
        ... )
        >>> print(f"Found {len(tops)} top segments and {len(bottoms)} bottom segments")
        >>> print(f"Warning: {len(dead)} dead ends and {len(orphans)} orphans detected")
    """
    
    # Build connectivity graph first
    successors, predecessors = build_connectivity_graph(valid_dtw_pairs, detailed_pairs)
    
    # Classify segments based on position and connectivity
    top_segments = []
    bottom_segments = []
    dead_ends = []
    orphans = []
    
    for a_idx, b_idx in valid_dtw_pairs:
        details = detailed_pairs[(a_idx, b_idx)]
        
        # Top segments start at depth 0 for both cores
        if abs(details['a_start']) < 1e-6 and abs(details['b_start']) < 1e-6:
            top_segments.append((a_idx, b_idx))
        
        # Bottom segments end at maximum depth for both cores
        if (abs(details['a_end'] - max_depth_a) < 1e-6 and 
            abs(details['b_end'] - max_depth_b) < 1e-6):
            bottom_segments.append((a_idx, b_idx))
        
        # Dead ends have no successors but aren't bottom segments
        if len(successors.get((a_idx, b_idx), [])) == 0 and (a_idx, b_idx) not in bottom_segments:
            dead_ends.append((a_idx, b_idx))
        
        # Orphans have no predecessors but aren't top segments
        if len(predecessors.get((a_idx, b_idx), [])) == 0 and (a_idx, b_idx) not in top_segments:
            orphans.append((a_idx, b_idx))
    
    return top_segments, bottom_segments, dead_ends, orphans, successors, predecessors


def filter_dead_end_pairs(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=False):
    """
    Remove dead end and orphan segment pairs from the valid set.
    
    This filtering improves path finding by removing segments that cannot be part
    of complete paths from top to bottom.
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs to filter
        detailed_pairs (dict): Segment depth details  
        max_depth_a (float): Maximum depth for core A
        max_depth_b (float): Maximum depth for core B
        debug (bool): Whether to print filtering statistics
        
    Returns:
        set: Filtered segment pairs without dead ends and orphans
    
    Example:
        >>> filtered_pairs = filter_dead_end_pairs(valid_pairs, details, 1000, 1200, debug=True)
        >>> print(f"Retained {len(filtered_pairs)} viable segments")
    """
    
    # Get special segment classifications
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    # Combine problematic segments
    dead_end_pairs = set(dead_ends + orphans)
    
    # Filter out problematic segments
    filtered_pairs = set(valid_dtw_pairs) - dead_end_pairs
    
    if debug:
        print(f"Dead end filtering: {len(filtered_pairs)}/{len(valid_dtw_pairs)} segments retained")
        print(f"Removed {len(dead_end_pairs)} dead-end pairs:")
        print(f"  - {len(dead_ends)} dead ends (no successors)")
        print(f"  - {len(orphans)} orphans (no predecessors)")
    
    return filtered_pairs 