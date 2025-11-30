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


def find_all_segments(log_a, log_b, md_a, md_b, picked_datum_a=None, picked_datum_b=None, top_bottom=True, top_depth=0.0, mute_mode=False):
    """
    Find segments in two logs using depth boundaries to create consecutive and single-point segments.
    
    Converts user-picked depth values to indices in the log arrays and generates all possible
    segment combinations for DTW analysis.
    
    Args:
        log_a (array): Log data for core A
        log_b (array): Log data for core B
        md_a (array): Measured depth values corresponding to log_a
        md_b (array): Measured depth values corresponding to log_b
        picked_datum_a (list, optional): User-selected depth values for core A boundaries
        picked_datum_b (list, optional): User-selected depth values for core B boundaries
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
        ...     picked_datum_a=[0, 100, 200], 
        ...     picked_datum_b=[0, 150, 300]
        ... )
        >>> print(f"Core A has {len(segments_a)} segments")
        >>> print(f"First segment A spans indices {segments_a[0]}")
    """
    
    # Initialize depth lists
    if picked_datum_a is None:
        picked_datum_a = []
    if picked_datum_b is None:
        picked_datum_b = []
    
    # Ensure picked_datum are Python lists
    if isinstance(picked_datum_a, np.ndarray):
        depth_values_a = picked_datum_a.tolist()
    else:
        depth_values_a = list(picked_datum_a)
        
    if isinstance(picked_datum_b, np.ndarray):
        depth_values_b = picked_datum_b.tolist()
    else:
        depth_values_b = list(picked_datum_b)
    
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


def filter_dead_end_pairs(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=False):
    """
    Enhanced dead-end filtering that removes ALL segments that cannot be part of complete paths.
    
    Uses forward/backward reachability analysis to ensure every retained segment can participate
    in at least one complete path from top to bottom.
    """
    from collections import deque
    
    # Build connectivity graph
    successors, predecessors = build_connectivity_graph(valid_dtw_pairs, detailed_pairs)
    
    # Find top and bottom segments
    top_segments = []
    bottom_segments = []
    
    for a_idx, b_idx in valid_dtw_pairs:
        details = detailed_pairs[(a_idx, b_idx)]
        
        # Top segments start at depth 0 for both cores
        if abs(details['a_start']) < 1e-6 and abs(details['b_start']) < 1e-6:
            top_segments.append((a_idx, b_idx))
        
        # Bottom segments end at maximum depth for both cores
        if (abs(details['a_end'] - max_depth_a) < 1e-6 and 
            abs(details['b_end'] - max_depth_b) < 1e-6):
            bottom_segments.append((a_idx, b_idx))
    
    # Forward reachability: segments reachable from tops
    forward_reachable = set()
    
    def forward_search_from_tops():
        """Find all segments reachable from top segments."""
        queue = deque(top_segments)
        visited = set(top_segments)
        
        while queue:
            current = queue.popleft()
            forward_reachable.add(current)
            
            # Add all successors to queue
            for successor in successors.get(current, []):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
    
    # Backward reachability: segments that can reach bottoms
    backward_reachable = set()
    
    def backward_search_from_bottoms():
        """Find all segments that can reach bottom segments."""
        queue = deque(bottom_segments)
        visited = set(bottom_segments)
        
        while queue:
            current = queue.popleft()
            backward_reachable.add(current)
            
            # Add all predecessors to queue
            for predecessor in predecessors.get(current, []):
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)
    
    # Perform reachability analysis
    if top_segments:
        forward_search_from_tops()
    if bottom_segments:
        backward_search_from_bottoms()
    
    # Viable segments must be reachable from tops AND able to reach bottoms
    viable_segments = forward_reachable & backward_reachable
    
    # If no tops or bottoms, keep all segments (prevent over-filtering)
    if not top_segments or not bottom_segments:
        if debug:
            print("Warning: No complete start/end points found - keeping all segments")
        viable_segments = set(valid_dtw_pairs)
    
    # Statistics
    removed_count = len(valid_dtw_pairs) - len(viable_segments)
    
    if debug:
        print(f"Enhanced dead-end filtering results:")
        print(f"  Original segments: {len(valid_dtw_pairs)}")
        print(f"  Top segments: {len(top_segments)}")
        print(f"  Bottom segments: {len(bottom_segments)}")
        print(f"  Forward reachable: {len(forward_reachable)}")
        print(f"  Backward reachable: {len(backward_reachable)}")
        print(f"  Viable segments: {len(viable_segments)}")
        print(f"  Removed: {removed_count}")
        
        if removed_count > 0:
            removed_segments = valid_dtw_pairs - viable_segments
            sample_removed = list(removed_segments)[:5]
            print(f"  Sample removed: {[(a+1, b+1) for a, b in sample_removed]}")
    
    return viable_segments


def identify_special_segments(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b):
    """
    Enhanced segment classification that also identifies isolation issues.
    """
    # Build connectivity graph
    successors, predecessors = build_connectivity_graph(valid_dtw_pairs, detailed_pairs)
    
    # Initialize lists
    top_segments = []
    bottom_segments = []
    dead_ends = []
    orphans = []
    isolated = []  # New category: segments with no connections
    
    for a_idx, b_idx in valid_dtw_pairs:
        details = detailed_pairs[(a_idx, b_idx)]
        
        # Top segments start at depth 0 for both cores
        if abs(details['a_start']) < 1e-6 and abs(details['b_start']) < 1e-6:
            top_segments.append((a_idx, b_idx))
        
        # Bottom segments end at maximum depth for both cores
        if (abs(details['a_end'] - max_depth_a) < 1e-6 and 
            abs(details['b_end'] - max_depth_b) < 1e-6):
            bottom_segments.append((a_idx, b_idx))
        
        # Check connectivity
        has_successors = len(successors.get((a_idx, b_idx), [])) > 0
        has_predecessors = len(predecessors.get((a_idx, b_idx), [])) > 0
        
        # Dead ends: no successors but not bottom segments
        if not has_successors and (a_idx, b_idx) not in bottom_segments:
            dead_ends.append((a_idx, b_idx))
        
        # Orphans: no predecessors but not top segments
        if not has_predecessors and (a_idx, b_idx) not in top_segments:
            orphans.append((a_idx, b_idx))
        
        # Isolated: no connections at all
        if not has_successors and not has_predecessors:
            isolated.append((a_idx, b_idx))
    
    return top_segments, bottom_segments, dead_ends, orphans, successors, predecessors