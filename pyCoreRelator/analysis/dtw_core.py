"""
Core DTW analysis functions for pyCoreRelator

Included Functions:
- has_complete_paths: Quick check if complete paths exist without verbose diagnostics
- find_depth_gaps: Identify gaps in depth coverage from valid segment pairs
- custom_dtw: Core DTW implementation with edge case handling
- handle_single_point: Handle DTW for single-point segments
- handle_identical_segments: Handle DTW for identical log segments
- handle_dtw_edge_cases: Wrapper for all DTW edge case scenarios
- process_segment_pair: Process a single segment pair for DTW analysis
- run_comprehensive_dtw_analysis: Complete workflow for segment-based DTW analysis
- force_bottom_segment_pairing: Force pairing of unpaired bottom segments
- handle_single_point_dtw: Handle DTW for single-point first log
- handle_single_point_log2_dtw: Handle DTW for single-point second log
- handle_two_single_points: Handle DTW for two single-point logs

This module provides comprehensive Dynamic Time Warping (DTW) analysis functionality
for geological well log correlation. It includes custom DTW implementation with
special case handling, quality metrics computation, and integrated age constraint
compatibility checking for segment-based correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import LinearSegmentedColormap
import librosa
import librosa.display
from librosa.sequence import dtw
import os
from PIL import Image as PILImage
from IPython.display import Image, display
import time
from tqdm import tqdm
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from joblib import Parallel, delayed
import itertools
from scipy import stats
from IPython.display import Image as IPImage

# Import from other modules
from .quality import compute_quality_indicators, calculate_age_overlap_percentage
from .age_models import check_age_constraint_compatibility
from .segments import find_all_segments, filter_dead_end_pairs, build_connectivity_graph, identify_special_segments
from .quality import find_nearest_index
from .diagnostics import diagnose_chain_breaks
# Note: plotting imports (plot_dtw_matrix_with_paths, create_segment_dtw_animation) moved to function level to avoid circular imports

def force_bottom_segment_pairing(valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, 
                                log_a, log_b, final_dtw_results, independent_dtw=False, mute_mode=False, pca_for_dependent_dtw=False):
    """Force pairing of unpaired segments below the deepest valid pairs to ensure bottom completion."""
    
    # Find deepest valid pair depths
    max_valid_a = 0
    max_valid_b = 0
    for a_idx, b_idx in valid_dtw_pairs:
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        max_valid_a = max(max_valid_a, a_end)
        max_valid_b = max(max_valid_b, b_end)
    
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    # Find unpaired segments below deepest valid pairs
    unpaired_a = []
    unpaired_b = []
    
    for a_idx, (a_start, a_end) in enumerate(segments_a):
        a_start_depth = depth_boundaries_a[a_start]
        if a_start_depth > max_valid_a and (a_idx, None) not in [(pair[0], None) for pair in valid_dtw_pairs]:
            # Check if this segment is actually unpaired
            is_paired = any(pair[0] == a_idx for pair in valid_dtw_pairs)
            if not is_paired:
                unpaired_a.append(a_idx)
    
    for b_idx, (b_start, b_end) in enumerate(segments_b):
        b_start_depth = depth_boundaries_b[b_start]
        if b_start_depth > max_valid_b and (None, b_idx) not in [(None, pair[1]) for pair in valid_dtw_pairs]:
            # Check if this segment is actually unpaired
            is_paired = any(pair[1] == b_idx for pair in valid_dtw_pairs)
            if not is_paired:
                unpaired_b.append(b_idx)
    
    # Create force pairs (avoid single-point to single-point)
    force_candidates = []
    
    for a_idx in unpaired_a:
        a_start, a_end = segments_a[a_idx]
        is_single_a = (a_start == a_end)
        
        for b_idx in unpaired_b:
            b_start, b_end = segments_b[b_idx]
            is_single_b = (b_start == b_end)
            
            # Skip single-point to single-point pairs
            if is_single_a and is_single_b:
                continue
                
            force_candidates.append((a_idx, b_idx))
    
    # Also pair deepest A segments with bottom B single-point
    bottom_single_b = [i for i, (start, end) in enumerate(segments_b) 
                      if start == end and depth_boundaries_b[start] == max_depth_b]
    
    for a_idx in unpaired_a:
        a_start, a_end = segments_a[a_idx]
        if a_start != a_end:  # Not single-point
            for b_idx in bottom_single_b:
                if (a_idx, b_idx) not in force_candidates:
                    force_candidates.append((a_idx, b_idx))
    
    # Also pair deepest B segments with bottom A single-point
    bottom_single_a = [i for i, (start, end) in enumerate(segments_a) 
                      if start == end and depth_boundaries_a[start] == max_depth_a]
    
    for b_idx in unpaired_b:
        b_start, b_end = segments_b[b_idx]
        if b_start != b_end:  # Not single-point
            for a_idx in bottom_single_a:
                if (a_idx, b_idx) not in force_candidates:
                    force_candidates.append((a_idx, b_idx))
    
    # Add force pairs
    added = 0
    for a_idx, b_idx in force_candidates:
        if (a_idx, b_idx) in valid_dtw_pairs:
            continue
            
        try:
            a_start = depth_boundaries_a[segments_a[a_idx][0]]
            a_end = depth_boundaries_a[segments_a[a_idx][1]]
            b_start = depth_boundaries_b[segments_b[b_idx][0]]
            b_end = depth_boundaries_b[segments_b[b_idx][1]]
            
            log_a_segment = log_a[a_start:a_end+1]
            log_b_segment = log_b[b_start:b_end+1]
            
            D_sub, wp, QIdx = custom_dtw(log_a_segment, log_b_segment, subseq=False, 
                                       exponent=1, QualityIndex=True, independent_dtw=independent_dtw,
                                       pca_for_dependent_dtw=pca_for_dependent_dtw, sakoe_chiba=True)
            
            adjusted_wp = wp.copy()
            adjusted_wp[:, 0] += a_start
            adjusted_wp[:, 1] += b_start
            QIdx['perc_age_overlap'] = 0.0
            
            valid_dtw_pairs.add((a_idx, b_idx))
            final_dtw_results[(a_idx, b_idx)] = ([adjusted_wp], [], [QIdx])
            added += 1
            
        except Exception:
            continue
    
    if not mute_mode and added > 0:
        print(f"Force-paired {added} unpaired bottom segments")
    
    return added

def has_complete_paths(valid_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b):
    """
    Proper connectivity check using the same approach as diagnose_chain_breaks.
    
    Uses existing functions to identify special segments and check for complete paths.
    
    Parameters
    ----------
    valid_pairs : set
        Set of valid segment pair indices (a_idx, b_idx)
    segments_a : list
        List of segment definitions for log_a
    segments_b : list
        List of segment definitions for log_b
    depth_boundaries_a : list
        Depth boundaries for log_a segments
    depth_boundaries_b : list
        Depth boundaries for log_b segments
        
    Returns
    -------
    bool
        True if complete paths exist, False otherwise
    """
    if not valid_pairs:
        return False
    
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    # Create detailed pairs using same format as diagnose_chain_breaks
    detailed_pairs = {}
    for a_idx, b_idx in valid_pairs:
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        detailed_pairs[(a_idx, b_idx)] = {
            'a_start': a_start, 'a_end': a_end,
            'b_start': b_start, 'b_end': b_end,
            'a_len': a_end - a_start + 1,
            'b_len': b_end - b_start + 1
        }
    
    # Import the existing function to reuse the same logic
    from .segments import identify_special_segments
    
    # Use the same identification logic as diagnose_chain_breaks
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    # If no tops or bottoms, no complete paths possible
    if not top_segments or not bottom_segments:
        return False
    
    # Check if any top segment can reach any bottom segment using BFS
    def find_reachable_from_top(start_segment):
        """Find all segments reachable from a top segment using BFS."""
        visited = set()
        queue = [start_segment]
        reachable = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            reachable.add(current)
            
            # Add all successors to queue
            for successor in successors.get(current, []):
                if successor not in visited:
                    queue.append(successor)
        
        return reachable
    
    # Check if any top can reach any bottom
    for top_segment in top_segments:
        reachable = find_reachable_from_top(top_segment)
        # Check if any bottom segment is reachable
        if any(bottom in reachable for bottom in bottom_segments):
            return True
    
    return False


def find_depth_gaps(valid_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b):
    """
    Find uncovered depth ranges needing gap-filling.
    
    Parameters
    ----------
    valid_pairs : set
        Set of valid segment pair indices (a_idx, b_idx)
    segments_a : list
        List of segment definitions for log_a
    segments_b : list
        List of segment definitions for log_b
    depth_boundaries_a : list
        Depth boundaries for log_a segments
    depth_boundaries_b : list
        Depth boundaries for log_b segments
        
    Returns
    -------
    tuple
        (gaps_a, gaps_b) where each is a list of (start, end) depth ranges
    """
    if not valid_pairs:
        max_depth_a = max(depth_boundaries_a)
        max_depth_b = max(depth_boundaries_b)
        return [(0, max_depth_a)], [(0, max_depth_b)]
    
    # Get covered ranges
    covered_a = []
    covered_b = []
    
    for a_idx, b_idx in valid_pairs:
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        covered_a.append((a_start, a_end))
        covered_b.append((b_start, b_end))
    
    def find_gaps(covered_ranges, max_depth):
        if not covered_ranges:
            return [(0, max_depth)]
        
        sorted_ranges = sorted(covered_ranges)
        gaps = []
        
        # Gap at start
        if sorted_ranges[0][0] > 0:
            gaps.append((0, sorted_ranges[0][0]))
        
        # Gaps between ranges
        for i in range(len(sorted_ranges) - 1):
            current_end = sorted_ranges[i][1]
            next_start = sorted_ranges[i + 1][0]
            if next_start > current_end:
                gaps.append((current_end, next_start))
        
        # Gap at end
        if sorted_ranges[-1][1] < max_depth:
            gaps.append((sorted_ranges[-1][1], max_depth))
        
        return gaps
    
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    gaps_a = find_gaps(covered_a, max_depth_a)
    gaps_b = find_gaps(covered_b, max_depth_b)
    
    return gaps_a, gaps_b


def handle_single_point_dtw(log1, log2, exponent=1, QualityIndex=False):
    """
    Handle DTW for the case where log1 contains only a single data point.
    
    This function creates a custom warping path that maps the single point in log1
    to all points in log2, providing a meaningful correlation when one sequence
    has only one data point.
    
    Parameters
    ----------
    log1 : array-like
        First well log data with a single point
    log2 : array-like
        Second well log data with multiple points
    exponent : float, default=1
        Exponent for cost calculation in distance computation
    QualityIndex : bool, default=False
        If True, computes and returns quality indicators
        
    Returns
    -------
    D : numpy.ndarray
        The accumulated cost matrix (1 x len(log2))
    wp : numpy.ndarray
        The warping path as a sequence of index pairs
    QIdx : dict, optional
        Quality indicators dictionary (returned if QualityIndex=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> log1 = [2.5]
    >>> log2 = [1.0, 2.0, 3.0, 4.0]
    >>> D, wp = handle_single_point_dtw(log1, log2)
    >>> print(f"Warping path shape: {wp.shape}")
    Warping path shape: (4, 2)
    """
    
    log1 = np.array(log1)
    log2 = np.array(log2)
    c = len(log2)
    
    # Create cost matrix (1 x len(log2))
    log1_value = log1[0]
    if log1.ndim > 1 and log1.shape[1] > 1:
        # Handle multidimensional log1
        if log2.ndim > 1 and log2.shape[1] > 1:
            # Both logs are multidimensional
            sm = np.array([np.linalg.norm(log1_value - log2[j]) for j in range(c)]) ** exponent
        else:
            # log1 is multidimensional, log2 is not
            sm = np.array([np.linalg.norm(log1_value - log2[j]) for j in range(c)]) ** exponent
    else:
        # log1 is 1D
        if log2.ndim > 1 and log2.shape[1] > 1:
            # log2 is multidimensional
            sm = np.array([np.linalg.norm(log1_value - log2[j]) for j in range(c)]) ** exponent
        else:
            # Both logs are 1D
            sm = (np.abs(log2 - log1_value)) ** exponent
    
    # Reshape to 2D matrix
    D = sm.reshape(1, -1)
    
    # Create warping path: single point in log1 maps to all points in log2
    wp = np.zeros((c, 2), dtype=int)
    wp[:, 0] = 0  # All points map to index 0 in log1
    wp[:, 1] = np.arange(c)  # Sequential indices in log2
    
    if QualityIndex:
        # Create quality indicators
        p = wp[:, 0]
        q = wp[:, 1]
        QIdx = compute_quality_indicators(log1, log2, p, q, D)
        return D, wp, QIdx
    else:
        return D, wp


def handle_single_point_log2_dtw(log1, log2, exponent=1, QualityIndex=False):
    """
    Handle DTW for the case where log2 contains only a single data point.
    
    This function creates a custom warping path that maps all points in log1
    to the single point in log2, providing meaningful correlation when the
    second sequence has only one data point.
    
    Parameters
    ----------
    log1 : array-like
        First well log data with multiple points
    log2 : array-like
        Second well log data with a single point
    exponent : float, default=1
        Exponent for cost calculation in distance computation
    QualityIndex : bool, default=False
        If True, computes and returns quality indicators
        
    Returns
    -------
    D : numpy.ndarray
        The accumulated cost matrix (len(log1) x 1)
    wp : numpy.ndarray
        The warping path as a sequence of index pairs
    QIdx : dict, optional
        Quality indicators dictionary (returned if QualityIndex=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> log1 = [1.0, 2.0, 3.0, 4.0]
    >>> log2 = [2.5]
    >>> D, wp = handle_single_point_log2_dtw(log1, log2)
    >>> print(f"Cost matrix shape: {D.shape}")
    Cost matrix shape: (4, 1)
    """
    
    log1 = np.array(log1)
    log2 = np.array(log2)
    r = len(log1)
    
    # Create cost matrix (len(log1) x 1)
    log2_value = log2[0]
    sm = np.zeros(r)
    
    # Handle multidimensional logs
    log1_is_multidim = (log1.ndim > 1 and log1.shape[1] > 1)
    log2_is_multidim = (log2.ndim > 1 and log2.shape[1] > 1)
    
    if log1_is_multidim or log2_is_multidim:
        log1_array = np.atleast_2d(log1)
        log2_array = np.atleast_2d(log2)
        
        for i in range(r):
            if log1_is_multidim and log2_is_multidim:
                sm[i] = np.linalg.norm(log1_array[i] - log2_array[0]) ** exponent
            elif log1_is_multidim:
                sm[i] = np.linalg.norm(log1_array[i] - log2_value) ** exponent
            else:
                sm[i] = np.linalg.norm(log1[i] - log2_array[0]) ** exponent
    else:
        # For 1D data
        sm = (np.abs(log1 - log2_value)) ** exponent
    
    # Reshape to 2D matrix
    D = sm.reshape(-1, 1)
    
    # Create warping path: all points in log1 map to index 0 in log2
    wp = np.zeros((r, 2), dtype=int)
    wp[:, 0] = np.arange(r)  # Sequential indices in log1
    wp[:, 1] = 0  # All points map to index 0 in log2
    
    if QualityIndex:
        # Create quality indicators
        p = wp[:, 0]
        q = wp[:, 1]
        QIdx = compute_quality_indicators(log1, log2, p, q, D)
        return D, wp, QIdx
    else:
        return D, wp


def handle_two_single_points(log1, log2, exponent=1, QualityIndex=False):
    """
    Handle DTW when both logs contain only a single data point.
    
    This function handles the degenerate case where both sequences contain
    only one point each, creating a trivial warping path.
    
    Parameters
    ----------
    log1 : array-like
        Single-point log data
    log2 : array-like
        Single-point log data
    exponent : float, default=1
        Exponent for cost calculation in distance computation
    QualityIndex : bool, default=False
        If True, computes and returns quality indicators
    
    Returns
    -------
    D : numpy.ndarray
        The accumulated cost matrix (1x1)
    wp : numpy.ndarray
        The warping path as a single index pair
    QIdx : dict, optional
        Quality indicators dictionary (returned if QualityIndex=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> log1 = [2.5]
    >>> log2 = [3.0]
    >>> D, wp = handle_two_single_points(log1, log2)
    >>> print(f"Distance: {D[0,0]:.2f}")
    Distance: 0.50
    """
    
    log1_value = log1[0]
    log2_value = log2[0]
    
    # Calculate the distance between the two points
    if hasattr(log1_value, '__len__') and hasattr(log2_value, '__len__'):
        # Both are multidimensional points
        diff = np.linalg.norm(log1_value - log2_value)
    else:
        # At least one is a scalar
        diff = abs(log1_value - log2_value)
    
    # Create the 1x1 cost matrix
    D = np.array([[diff ** exponent]])
    
    # Create the warping path - just a single pair (0,0)
    wp = np.array([[0, 0]])
    
    if QualityIndex:
        # Create quality indicators via the compute_quality_indicators function
        p = wp[:, 0]
        q = wp[:, 1]
        QIdx = compute_quality_indicators(log1, log2, p, q, D)
        return D, wp, QIdx
    else:
        return D, wp


def custom_dtw(log1, log2, subseq=False, exponent=1, QualityIndex=False, independent_dtw=False, available_columns=None, pca_for_dependent_dtw=False, sakoe_chiba=False):
    """
    Custom implementation of Dynamic Time Warping for well log correlation.
    
    This function creates a similarity matrix between two well logs and applies DTW
    to find the optimal alignment, handling all edge cases including single-point
    sequences and multidimensional data with independent or dependent analysis modes.
    
    Parameters
    ----------
    log1 : array-like
        First well log data to be compared
    log2 : array-like
        Second well log data to be compared
    subseq : bool, default=False
        If True, performs subsequence DTW
    exponent : float, default=1
        Exponent for cost calculation in distance computation
    QualityIndex : bool, default=False
        If True, computes and returns quality indicators
    independent_dtw : bool, default=False
        If True, performs independent DTW on each dimension separately
    available_columns : list, default=None
        Column names for logging when independent_dtw=True
    pca_for_dependent_dtw : bool, default=True
        Whether to use PCA for dependent multidimensional DTW.
        If False, uses conventional multidimensional DTW with librosa directly without PCA projection.
    sakoe_chiba : bool, default=True
        If True, applies Sakoe-Chiba band constraint with band_rad=0.5.
        If False, performs unconstrained DTW.
        
    Returns
    -------
    D : numpy.ndarray
        The accumulated cost matrix where D[i,j] contains the minimum cumulative cost
        to reach point (i,j) from the starting point
    wp : numpy.ndarray
        The warping path as a sequence of index pairs
    QIdx : dict, optional
        Quality indicators dictionary (returned if QualityIndex=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> log1 = np.array([1.0, 2.0, 3.0, 4.0])
    >>> log2 = np.array([1.1, 2.1, 3.1, 4.1])
    >>> D, wp, quality = custom_dtw(log1, log2, QualityIndex=True)
    >>> print(f"DTW distance: {D[-1,-1]:.3f}")
    DTW distance: 0.400
    """

    # Convert inputs to float32 if not already
    log1 = log1.astype(np.float32)
    log2 = log2.astype(np.float32)

    # Check for empty logs
    if log1 is None or len(log1) == 0:
        print("Error: log1 is empty or None. Cannot perform valid DTW.")
        if QualityIndex:
            return np.array([[0]]), np.array([[0, 0]]), {'norm_dtw': 0, 'dtw_ratio': 0, 'perc_diag': 0, 'dtw_warp_eff': 0, 'corr_coef': 0}
        return np.array([[0]]), np.array([[0, 0]])
    
    if log2 is None or len(log2) == 0:
        print("Error: log2 is empty or None. Cannot perform valid DTW.")
        if QualityIndex:
            return np.array([[0]]), np.array([[0, 0]]), {'norm_dtw': 0, 'dtw_ratio': 0, 'perc_diag': 0, 'dtw_warp_eff': 0, 'corr_coef': 0}
        return np.array([[0]]), np.array([[0, 0]])

    # Convert logs to numpy arrays
    log1 = np.array(log1)
    log2 = np.array(log2)
    r = len(log1)
    c = len(log2)
    
    # Handle special case: log1 is a single point
    if r == 1:
        return handle_single_point_dtw(log1, log2, exponent, QualityIndex)
    
    # Handle special case: log2 is a single point
    if c == 1:
        return handle_single_point_log2_dtw(log1, log2, exponent, QualityIndex)
    
    # Handle special case: both logs are single points
    if r == 1 and c == 1:
        print("WARNING: A DTW between two single points is being performed. This is not recommended as it may not provide a meaningful correlation.")
        return handle_two_single_points(log1, log2, exponent, QualityIndex)
    
    # Check if we should use independent DTW
    if independent_dtw and log1.ndim > 1 and log2.ndim > 1 and log1.shape[1] > 1 and log2.shape[1] > 1:
        print("WARNING: Using Independent DTW mode - processing each dimension separately")
        
        # Initialize arrays to store results
        all_D = []
        all_wp = []
        all_QIdx = [] if QualityIndex else None
        
        # Process each dimension separately
        for i in range(log1.shape[1]):
            # Extract single dimension
            dim_name = f"{available_columns[i]}" if available_columns and i < len(available_columns) else f"Dimension {i+1}"
            dim_log1 = log1[:, i].reshape(-1)  # Flatten
            dim_log2 = log2[:, i].reshape(-1)  # Flatten
            
            # Skip independent_dtw flag to avoid recursion
            if QualityIndex:
                D_dim, wp_dim, QIdx_dim = custom_dtw(dim_log1, dim_log2, subseq=subseq, 
                                                   exponent=exponent, QualityIndex=True, 
                                                   independent_dtw=False, pca_for_dependent_dtw=pca_for_dependent_dtw,
                                                   sakoe_chiba=sakoe_chiba)
                all_QIdx.append(QIdx_dim)
            else:
                D_dim, wp_dim = custom_dtw(dim_log1, dim_log2, subseq=subseq, 
                                         exponent=exponent, QualityIndex=False, 
                                         independent_dtw=False, pca_for_dependent_dtw=pca_for_dependent_dtw,
                                         sakoe_chiba=sakoe_chiba)
            
            all_D.append(D_dim)
            all_wp.append(wp_dim)
        
        # Combine results - use mean for distance matrix
        combined_D = np.mean(np.array(all_D), axis=0)
        
        # Use warping path from first dimension for visualization
        wp = all_wp[0]
        
        # Compute combined quality indicators if requested
        if QualityIndex:
            # Create a combined quality index by averaging across dimensions
            combined_QIdx = {}
            for key in all_QIdx[0].keys():
                combined_QIdx[key] = np.mean([qi[key] for qi in all_QIdx])
            
            return combined_D, wp, combined_QIdx
        else:
            return combined_D, wp
    
    # Normal case - standard dependent DTW
    sm = np.zeros((r, c))

    # Check if data are multidimensional
    log1_is_multidim = (log1.ndim > 1 and log1.shape[1] > 1)
    log2_is_multidim = (log2.ndim > 1 and log2.shape[1] > 1)

    if log1_is_multidim and log2_is_multidim:
        # MULTIDIMENSIONAL DEPENDENT DTW
        try:
            # Check if both logs have the same number of dimensions
            if log1.shape[1] != log2.shape[1]:
                # Fallback to Euclidean norm if dimension mismatch
                log1_array = np.atleast_2d(log1)
                log2_array = np.atleast_2d(log2)
                for i in range(r):
                    diffs = np.array([np.linalg.norm(log1_array[i] - log2_array[j]) for j in range(c)])
                    sm[i, :] = diffs ** exponent
            else:
                if pca_for_dependent_dtw:
                    # Use PCA-based distance calculation
                    # Combine both logs for consistent PCA transformation
                    combined_logs = np.vstack([log1, log2])
                    
                    # Check if combined data has enough variation for PCA
                    if np.var(combined_logs, axis=0).sum() < 1e-10:
                        # Fallback to mean if no variation
                        mean_log1 = np.mean(log1, axis=1)
                        mean_log2 = np.mean(log2, axis=1)
                        for i in range(r):
                            sm[i, :] = (np.abs(mean_log2 - mean_log1[i])) ** exponent
                    else:
                        # Perform PCA to find main trend direction
                        # Center the data
                        mean_combined = np.mean(combined_logs, axis=0)
                        centered_combined = combined_logs - mean_combined
                        
                        # Calculate covariance matrix and eigenvectors
                        cov_matrix = np.cov(centered_combined.T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                        pc1_direction = eigenvectors[:, -1]  # Largest eigenvalue = main trend
                        
                        # Transform both logs to PC1 space
                        centered_log1 = log1 - mean_combined
                        centered_log2 = log2 - mean_combined
                        
                        pc1_log1 = np.dot(centered_log1, pc1_direction)  # Shape: (r,)
                        pc1_log2 = np.dot(centered_log2, pc1_direction)  # Shape: (c,)
                        
                        # Calculate DTW distance matrix in PC1 space (now 1D)
                        for i in range(r):
                            sm[i, :] = (np.abs(pc1_log2 - pc1_log1[i])) ** exponent
                else:
                    # CONVENTIONAL MULTIDIMENSIONAL DTW - compute similarity matrix using Euclidean distances
                    for i in range(r):
                        diffs = np.array([np.linalg.norm(log1[i] - log2[j]) for j in range(c)])
                        sm[i, :] = diffs ** exponent
                        
        except Exception:
            # Fallback to Euclidean norm if PCA fails
            log1_array = np.atleast_2d(log1)
            log2_array = np.atleast_2d(log2)
            for i in range(r):
                diffs = np.array([np.linalg.norm(log1_array[i] - log2_array[j]) for j in range(c)])
                sm[i, :] = diffs ** exponent

    elif log1_is_multidim or log2_is_multidim:
        # Mixed dimensionality - use Euclidean norm
        log1_array = np.atleast_2d(log1)
        log2_array = np.atleast_2d(log2)
        for i in range(r):
            if log1_is_multidim and log2_is_multidim:
                diffs = np.array([np.linalg.norm(log1_array[i] - log2_array[j]) for j in range(c)])
            elif log1_is_multidim:
                diffs = np.array([np.linalg.norm(log1_array[i] - log2_array[j, 0]) for j in range(c)])
            else:
                diffs = np.array([np.linalg.norm(log1_array[i, 0] - log2_array[j]) for j in range(c)])
            sm[i, :] = diffs ** exponent
    else:
        # For 1D data - use original approach
        for i in range(r):
            sm[i, :] = (np.abs(log2 - log1[i])) ** exponent

    # Compute the accumulated cost matrix D and the warping path wp
    if sakoe_chiba:
        D, wp = dtw(C=sm, subseq=subseq, global_constraints=True, band_rad=0.25)
    else:
        D, wp = dtw(C=sm, subseq=subseq)

    # Adjust warping path indices to match input arrays (librosa may use 0-based indexing)
    p = wp[:, 0]  # indices for log1
    q = wp[:, 1]  # indices for log2

    if QualityIndex:
        # Compute quality indicators
        QIdx = compute_quality_indicators(log1, log2, p, q, D, pca_for_dependent_dtw)
        return D, wp, QIdx
    else:
        return D, wp


def _process_segment_pair_worker(a_idx, b_idx, pair_info, log_a, log_b, 
                                  segments_a, segments_b, 
                                  depth_boundaries_a, depth_boundaries_b,
                                  independent_dtw, pca_for_dependent_dtw,
                                  dtw_distance_threshold, debug, mute_mode):
    """
    Worker function for parallel DTW computation of a single segment pair.
    This is a module-level function to enable pickling by joblib.
    """
    # Extract segments
    a_start = depth_boundaries_a[segments_a[a_idx][0]]
    a_end = depth_boundaries_a[segments_a[a_idx][1]]
    b_start = depth_boundaries_b[segments_b[b_idx][0]]
    b_end = depth_boundaries_b[segments_b[b_idx][1]]
    
    segment_a = log_a[a_start:a_end+1]
    segment_b = log_b[b_start:b_end+1]
    
    # Perform DTW
    try:
        D_sub, wp, QIdx = custom_dtw(segment_a, segment_b, subseq=False, exponent=1, 
                                      QualityIndex=True, independent_dtw=independent_dtw, 
                                      pca_for_dependent_dtw=pca_for_dependent_dtw, sakoe_chiba=True)
        
        # Adjust warping path coordinates
        adjusted_wp = wp.copy()
        adjusted_wp[:, 0] += a_start
        adjusted_wp[:, 1] += b_start
        
        final_dist = D_sub[-1, -1]
        
        # Add age overlap percentage to quality indicators
        if 'perc_age_overlap' in pair_info:
            QIdx['perc_age_overlap'] = pair_info['perc_age_overlap']
        
        # Flexible DTW distance filtering
        if dtw_distance_threshold is None:
            passes_distance = True
        else:
            passes_distance = final_dist < dtw_distance_threshold or len(segment_a) == 1 or len(segment_b) == 1
        
        return (a_idx, b_idx, {
            'dtw_results': ([adjusted_wp], [], [QIdx]),
            'dtw_distance': final_dist,
            'passes_distance': passes_distance,
        })
    except Exception as e:
        return (a_idx, b_idx, {
            'dtw_results': ([], [], []),
            'dtw_distance': float('inf'),
            'passes_distance': False,
        })


def run_comprehensive_dtw_analysis(log_a, log_b, md_a, md_b, picked_datum_a=None, picked_datum_b=None, 
                              top_bottom=True, top_depth=0.0,
                              independent_dtw=False, 
                              create_dtw_matrix=False,
                              visualize_pairs=True, 
                              visualize_segment_labels=False,
                              dtwmatrix_output_filename='SegmentPair_DTW_matrix.png',
                              creategif=False, 
                              gif_output_filename='SegmentPair_DTW_animation.gif', max_frames=100, 
                              debug=False, color_interval_size=10,
                              keep_frames=True, age_consideration=False, datum_ages_a=None, datum_ages_b=None,
                              restricted_age_correlation=True, 
                              core_a_age_data=None, core_b_age_data=None,
                              dtw_distance_threshold=None,
                              exclude_deadend=True,
                              core_a_name=None,
                              core_b_name=None,
                              mute_mode=False,
                              pca_for_dependent_dtw=False,
                              dpi=None,
                              n_jobs=-1):
    """
    Run comprehensive DTW analysis with integrated age correlation functionality.
    
    This function performs a complete segment-based DTW analysis between two well logs,
    including age constraint compatibility checking, dead-end filtering, and visualization
    generation. It identifies valid correlation segments, applies age constraints if
    specified, and generates comprehensive output including DTW matrices and animations.
    
    Parameters
    ----------
    log_a : array-like
        First well log data
    log_b : array-like
        Second well log data
    md_a : array-like
        Measured depth values for log_a
    md_b : array-like
        Measured depth values for log_b
    picked_datum_a : list, optional
        Specific depths to analyze in log_a
    picked_datum_b : list, optional
        Specific depths to analyze in log_b
    top_bottom : bool, default=True
        If True, include top and bottom boundaries in analysis
    top_depth : float, default=0.0
        Top depth value for analysis
    independent_dtw : bool, default=False
        If True, process each log dimension independently
    create_dtw_matrix : bool, default=True
        If True, generate DTW matrix visualization
    visualize_pairs : bool, default=True
        If True, visualize segment pairs in matrix plot
    visualize_segment_labels : bool, default=False
        If True, show segment labels in visualizations
    dtwmatrix_output_filename : str, default='SegmentPair_DTW_matrix.png'
        Filename for DTW matrix output
    creategif : bool, default=True
        If True, create animated GIF of segment correlations
    gif_output_filename : str, default='SegmentPair_DTW_animation.gif'
        Filename for animation output
    max_frames : int, default=100
        Maximum number of frames in animation
    debug : bool, default=False
        If True, enable debug output
    color_interval_size : float, optional
        Color interval size for visualizations
    keep_frames : bool, default=True
        If True, keep individual animation frames
    age_consideration : bool, default=False
        If True, apply age constraint analysis
    datum_ages_a : dict, optional
        Age data for log_a with keys: 'depths', 'ages', 'pos_uncertainties', 'neg_uncertainties'
    datum_ages_b : dict, optional
        Age data for log_b with keys: 'depths', 'ages', 'pos_uncertainties', 'neg_uncertainties'
    restricted_age_correlation : bool, default=True
        If True, use strict age overlap requirements
    core_a_age_data : dict, optional
        Complete age constraint data for core A from load_core_age_constraints(). Expected keys: 
        'in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors', 'core'
        Not required if age_consideration is False
    core_b_age_data : dict, optional
        Complete age constraint data for core B from load_core_age_constraints(). Expected keys: 
        'in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors', 'core'
        Not required if age_consideration is False
    dtw_distance_threshold : float, optional
        Maximum allowed DTW distance for segment acceptance
    exclude_deadend : bool, default=True
        If True, filter out dead-end segment pairs
    core_a_name : str, optional
        Name of first core for labeling
    core_b_name : str, optional
        Name of second core for labeling
    mute_mode : bool, default=False
        If True, suppress all print output
    pca_for_dependent_dtw : bool, default=True
        Whether to use PCA for dependent multidimensional DTW. If False, uses conventional 
        multidimensional DTW with librosa directly without PCA projection.
    n_jobs : int, default=-1
        Number of parallel jobs for DTW computation across segment pairs.
        -1 means using all available cores. Set to 1 for sequential processing.
    
    Returns
    -------
    dict
        Dictionary containing all DTW analysis results with the following keys:
        
        - dtw_correlation : dict
            DTW results for valid segment pairs (renamed from dtw_results)
        - valid_dtw_pairs : set
            Set of valid segment pair indices
        - segments_a : list
            Segment definitions for log_a
        - segments_b : list
            Segment definitions for log_b
        - depth_boundaries_a : list
            Depth boundaries for log_a segments
        - depth_boundaries_b : list
            Depth boundaries for log_b segments
        - dtw_distance_matrix_full : numpy.ndarray
            Full DTW distance matrix between logs
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyCoreRelator.utils import load_core_age_constraints
    >>> log_a = np.random.randn(100)
    >>> log_b = np.random.randn(120)
    >>> md_a = np.arange(100)
    >>> md_b = np.arange(120)
    >>> age_data_a = load_core_age_constraints('CoreA', 'path/to/age_data')
    >>> age_data_b = load_core_age_constraints('CoreB', 'path/to/age_data')
    >>> dtw_result = run_comprehensive_dtw_analysis(
    ...     log_a, log_b, md_a, md_b,
    ...     picked_datum_a=[20, 50, 80],
    ...     picked_datum_b=[25, 60, 95],
    ...     age_consideration=True,
    ...     core_a_age_data=age_data_a,
    ...     core_b_age_data=age_data_b
    ... )
    >>> print(f"Found {len(dtw_result['valid_dtw_pairs'])} valid segment pairs")
    Found 3 valid segment pairs
    """
    
    if not mute_mode:
        print("Starting comprehensive DTW analysis with integrated age correlation...")
    
    # Find all segments
    segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dated_picked_depths_a, dated_picked_depths_b = find_all_segments(
        log_a, log_b, md_a, md_b, 
        picked_datum_a, picked_datum_b,
        top_bottom=top_bottom, 
        top_depth=top_depth,
        mute_mode=mute_mode
    )

    # Check if age consideration is enabled and validate age data
    if age_consideration:
        if not mute_mode:
            print(f"\nAge consideration enabled - {'restricted' if restricted_age_correlation else 'flexible'} age correlation")
        
        if datum_ages_a is None or datum_ages_b is None:
            raise ValueError("Both datum_ages_a and datum_ages_b must be provided when age_consideration is True")
        
        # Check if age data dictionaries have the required keys and non-empty values
        required_keys = ['depths', 'ages', 'pos_uncertainties', 'neg_uncertainties']
        for key in required_keys:
            if key not in datum_ages_a or not datum_ages_a[key] or key not in datum_ages_b or not datum_ages_b[key]:
                raise ValueError(f"Missing or empty required key '{key}' in datum_ages_a or datum_ages_b")
        
        # Check if depths match picked_datum
        if (picked_datum_a is not None and len(dated_picked_depths_a) != len(datum_ages_a['depths'])) or \
           (picked_datum_b is not None and len(dated_picked_depths_b) != len(datum_ages_b['depths'])):
            raise ValueError("The number of depths in datum_ages_a/datum_ages_b must match the number of dated picked depths")
        
        # Extract age constraint data from core_a_age_data and core_b_age_data
        if core_a_age_data is None or core_b_age_data is None:
            raise ValueError("Both core_a_age_data and core_b_age_data must be provided when age_consideration is True")
        
        # Extract constraint data from the age_data dictionaries
        required_constraint_keys = ['in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors', 'core']
        for key in required_constraint_keys:
            if key not in core_a_age_data or key not in core_b_age_data:
                raise ValueError(f"Missing required key '{key}' in core_a_age_data or core_b_age_data")
        
        all_constraint_ages_a = core_a_age_data['in_sequence_ages']
        all_constraint_depths_a = core_a_age_data['in_sequence_depths']
        all_constraint_pos_errors_a = core_a_age_data['in_sequence_pos_errors']
        all_constraint_neg_errors_a = core_a_age_data['in_sequence_neg_errors']
        age_constraint_a_source_cores = core_a_age_data['core']
        
        all_constraint_ages_b = core_b_age_data['in_sequence_ages']
        all_constraint_depths_b = core_b_age_data['in_sequence_depths']
        all_constraint_pos_errors_b = core_b_age_data['in_sequence_pos_errors']
        all_constraint_neg_errors_b = core_b_age_data['in_sequence_neg_errors']
        age_constraint_b_source_cores = core_b_age_data['core']
        
        # Check if constraint data is non-empty when needed
        if not restricted_age_correlation:
            if (not all_constraint_ages_a or not all_constraint_depths_a or 
                not all_constraint_ages_b or not all_constraint_depths_b or
                not all_constraint_pos_errors_a or not all_constraint_pos_errors_b or
                not all_constraint_neg_errors_a or not all_constraint_neg_errors_b):
                raise ValueError("Complete age constraint data must be provided when restricted_age_correlation is False")
    else:
        # Set all constraint variables to None when age_consideration is False
        all_constraint_ages_a = None
        all_constraint_depths_a = None
        all_constraint_pos_errors_a = None
        all_constraint_neg_errors_a = None
        age_constraint_a_source_cores = None
        all_constraint_ages_b = None
        all_constraint_depths_b = None
        all_constraint_pos_errors_b = None
        all_constraint_neg_errors_b = None
        age_constraint_b_source_cores = None
    
    # Calculate full DTW distance matrix for reference
    if not mute_mode:
        print("Calculating full DTW distance matrix...")
    dtw_distance_matrix_full, _ = custom_dtw(log_a, log_b, subseq=False, exponent=1, independent_dtw=independent_dtw, pca_for_dependent_dtw=pca_for_dependent_dtw, sakoe_chiba=False)
    
    # Create all possible segment pairs for evaluation
    all_possible_pairs = []
    detailed_pairs = {}
    
    for i in range(len(segments_a)):
        for j in range(len(segments_b)):
            # Get segment boundaries
            a_start_idx, a_end_idx = segments_a[i]
            b_start_idx, b_end_idx = segments_b[j]
            
            a_start = depth_boundaries_a[a_start_idx]
            a_end = depth_boundaries_a[a_end_idx]
            b_start = depth_boundaries_b[b_start_idx]
            b_end = depth_boundaries_b[b_end_idx]
            
            # Check for internal boundaries
            has_internal_boundary_a = any(a_start < depth_boundaries_a[idx] < a_end 
                                         for idx in range(len(depth_boundaries_a))
                                         if idx != a_start_idx and idx != a_end_idx)
            
            has_internal_boundary_b = any(b_start < depth_boundaries_b[idx] < b_end 
                                         for idx in range(len(depth_boundaries_b))
                                         if idx != b_start_idx and idx != b_end_idx)
            
            # Skip if either segment has internal boundaries
            if has_internal_boundary_a or has_internal_boundary_b:
                continue
                
            # Check for empty segments
            segment_a_len = a_end - a_start + 1
            segment_b_len = b_end - b_start + 1
            
            if (segment_a_len <= 1 and segment_b_len <= 1) or segment_a_len == 0 or segment_b_len == 0:
                continue
            
            # Store detailed segment information
            detailed_pairs[(i, j)] = {
                'a_start': a_start, 'a_end': a_end,
                'b_start': b_start, 'b_end': b_end
            }
            
            all_possible_pairs.append((i, j))
    
    if not mute_mode:
        print(f"Found {len(all_possible_pairs)} valid segment pairs after boundary checks")
    
    # Create a dictionary to store all pairs with their age information and DTW results
    all_pairs_with_dtw = {}
    
    # Process all possible pairs and calculate age criteria
    if age_consideration:        
        for a_idx, b_idx in tqdm(all_possible_pairs, desc="Calculating age bounds for all segment pairs..." if not mute_mode else None, disable=mute_mode):
            # Get segment depths
            a_start_depth = md_a[depth_boundaries_a[segments_a[a_idx][0]]]
            a_end_depth = md_a[depth_boundaries_a[segments_a[a_idx][1]]]
            b_start_depth = md_b[depth_boundaries_b[segments_b[b_idx][0]]]
            b_end_depth = md_b[depth_boundaries_b[segments_b[b_idx][1]]]
            
            # Find age indices
            a_start_age_idx = np.argmin(np.abs(np.array(datum_ages_a['depths']) - a_start_depth))
            a_end_age_idx = np.argmin(np.abs(np.array(datum_ages_a['depths']) - a_end_depth))
            b_start_age_idx = np.argmin(np.abs(np.array(datum_ages_b['depths']) - b_start_depth))
            b_end_age_idx = np.argmin(np.abs(np.array(datum_ages_b['depths']) - b_end_depth))
            
            # Get age bounds with uncertainties
            a_start_age = datum_ages_a['ages'][a_start_age_idx]
            a_end_age = datum_ages_a['ages'][a_end_age_idx]
            b_start_age = datum_ages_b['ages'][b_start_age_idx]
            b_end_age = datum_ages_b['ages'][b_end_age_idx]
            
            a_start_pos_error = datum_ages_a['pos_uncertainties'][a_start_age_idx]
            a_start_neg_error = datum_ages_a['neg_uncertainties'][a_start_age_idx]
            a_end_pos_error = datum_ages_a['pos_uncertainties'][a_end_age_idx]
            a_end_neg_error = datum_ages_a['neg_uncertainties'][a_end_age_idx]
            
            b_start_pos_error = datum_ages_b['pos_uncertainties'][b_start_age_idx]
            b_start_neg_error = datum_ages_b['neg_uncertainties'][b_start_age_idx]
            b_end_pos_error = datum_ages_b['pos_uncertainties'][b_end_age_idx]
            b_end_neg_error = datum_ages_b['neg_uncertainties'][b_end_age_idx]
            
            # Calculate age bounds
            a_lower_bound = min(a_start_age - a_start_neg_error, a_end_age - a_end_neg_error)
            a_upper_bound = max(a_start_age + a_start_pos_error, a_end_age + a_end_pos_error)
            
            b_lower_bound = min(b_start_age - b_start_neg_error, b_end_age - b_end_neg_error)
            b_upper_bound = max(b_start_age + b_start_pos_error, b_end_age + b_end_pos_error)
            
            # Calculate age overlap percentage (without uncertainty)
            a_age_range_start = min(a_start_age, a_end_age)
            a_age_range_end = max(a_start_age, a_end_age)
            b_age_range_start = min(b_start_age, b_end_age)
            b_age_range_end = max(b_start_age, b_end_age)
            
            perc_age_overlap = calculate_age_overlap_percentage(
                a_age_range_start, a_age_range_end, 
                b_age_range_start, b_age_range_end
            )
            
            # Check for range overlap (used in both restricted and flexible modes)
            ranges_overlap = (a_lower_bound <= b_upper_bound and b_lower_bound <= a_upper_bound)
            
            # Check for single-point ages
            a_is_single_point = abs(a_upper_bound - a_lower_bound) < 1e-2
            b_is_single_point = abs(b_upper_bound - b_lower_bound) < 1e-2
            
            single_point_in_range = False
            if a_is_single_point:
                # Check if A's single point falls within B's range
                single_point_in_range = (b_lower_bound <= a_upper_bound <= b_upper_bound)
            elif b_is_single_point:
                # Check if B's single point falls within A's range
                single_point_in_range = (a_lower_bound <= b_upper_bound <= a_upper_bound)
            
            # Check for identical bounds
            has_identical_bounds = (abs(a_lower_bound - b_lower_bound) < 1e-3 and 
                                   abs(a_upper_bound - b_upper_bound) < 1e-3)
            
            # Store age information for this pair
            all_pairs_with_dtw[(a_idx, b_idx)] = {
                'a_idx': a_idx,
                'b_idx': b_idx,
                'age_bounds': {
                    'a_lower': a_lower_bound,
                    'a_upper': a_upper_bound,
                    'b_lower': b_lower_bound,
                    'b_upper': b_upper_bound
                },
                'ranges_overlap': ranges_overlap,
                'single_point_in_range': single_point_in_range,
                'has_identical_bounds': has_identical_bounds,
                'perc_age_overlap': perc_age_overlap
            }
    else:
        # If age consideration is disabled, create empty entries for all pairs
        for a_idx, b_idx in all_possible_pairs:
            all_pairs_with_dtw[(a_idx, b_idx)] = {
                'a_idx': a_idx,
                'b_idx': b_idx,
                'perc_age_overlap': 0.0  # Default value when age consideration is disabled
            }
    
    # Calculate DTW for candidate pairs (parallel or sequential based on n_jobs)
    if n_jobs == 1:
        # Sequential processing
        for a_idx, b_idx in tqdm(all_possible_pairs, desc="Calculating DTW for segment pairs..." if not mute_mode else None, disable=mute_mode):
            pair_info = all_pairs_with_dtw[(a_idx, b_idx)]
            _, _, dtw_info = _process_segment_pair_worker(
                a_idx, b_idx, pair_info, log_a, log_b,
                segments_a, segments_b, depth_boundaries_a, depth_boundaries_b,
                independent_dtw, pca_for_dependent_dtw, dtw_distance_threshold, debug, mute_mode
            )
            pair_info.update(dtw_info)
    else:
        # Parallel processing
        if not mute_mode:
            print(f"Calculating DTW for {len(all_possible_pairs)} segment pairs using {n_jobs if n_jobs > 0 else 'all available'} cores...")
        
        # Prepare arguments for parallel processing
        parallel_args = [
            (a_idx, b_idx, all_pairs_with_dtw[(a_idx, b_idx)], log_a, log_b,
             segments_a, segments_b, depth_boundaries_a, depth_boundaries_b,
             independent_dtw, pca_for_dependent_dtw, dtw_distance_threshold, debug, mute_mode)
            for a_idx, b_idx in all_possible_pairs
        ]
        
        # Run in parallel
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_segment_pair_worker)(*args) 
            for args in tqdm(parallel_args, desc="Calculating DTW for segment pairs..." if not mute_mode else None, disable=mute_mode)
        )
        
        # Update all_pairs_with_dtw with results
        for a_idx, b_idx, dtw_info in results:
            all_pairs_with_dtw[(a_idx, b_idx)].update(dtw_info)
    
    # Determine which pairs are valid based on age and DTW criteria
    valid_dtw_pairs = set()
    final_dtw_results = {}
    
    if not age_consideration:
        # Age consideration is disabled - use only DTW distance
        for (a_idx, b_idx), pair_info in all_pairs_with_dtw.items():
            if pair_info['passes_distance']:
                valid_dtw_pairs.add((a_idx, b_idx))
                final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
        
        if not mute_mode:
            print(f"\nFound {len(valid_dtw_pairs)} valid segment pairs based on DTW distance")
        
    elif restricted_age_correlation:
        if not mute_mode:
            print(f"\nAssessing age compatibility for segment pairs...(restricted age correlation mode)")
        # Restricted mode - only accept pairs with overlapping age ranges
        # Separate pairs into overlapping and non-overlapping
        overlapping_pairs = {}
        
        for (a_idx, b_idx), pair_info in all_pairs_with_dtw.items():
            if not pair_info['passes_distance']:
                continue
                
            # Check if ranges overlap or single point is in range
            basic_criteria_met = pair_info['ranges_overlap'] or pair_info['single_point_in_range']
            
            if basic_criteria_met:
                overlapping_pairs[(a_idx, b_idx)] = pair_info
        
        # Process overlapping pairs with tqdm
        for (a_idx, b_idx), pair_info in tqdm(overlapping_pairs.items(), desc=f"Checking {len(overlapping_pairs)} segment pairs with overlapping age range..." if not mute_mode else None, disable=mute_mode):
            # For overlapping ranges, check age constraint compatibility
            if pair_info['ranges_overlap']:
                # Get segment age bounds with uncertainty
                a_lower_bound = pair_info['age_bounds']['a_lower']
                a_upper_bound = pair_info['age_bounds']['a_upper']
                b_lower_bound = pair_info['age_bounds']['b_lower']
                b_upper_bound = pair_info['age_bounds']['b_upper']
                
                # Convert constraint data to numpy arrays
                constraint_ages_a = np.array(all_constraint_ages_a)
                constraint_ages_b = np.array(all_constraint_ages_b)
                constraint_pos_errors_a = np.array(all_constraint_pos_errors_a)
                constraint_pos_errors_b = np.array(all_constraint_pos_errors_b)
                constraint_neg_errors_a = np.array(all_constraint_neg_errors_a)
                constraint_neg_errors_b = np.array(all_constraint_neg_errors_b)
                
                # Check compatibility with broader constraint approach
                compatible = check_age_constraint_compatibility(
                    a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound,
                    constraint_ages_a, constraint_ages_b,
                    constraint_pos_errors_a, constraint_pos_errors_b,
                    constraint_neg_errors_a, constraint_neg_errors_b,
                    datum_ages_a=datum_ages_a, datum_ages_b=datum_ages_b
                )
                
                # Accept pairs that meet both criteriaf
                if compatible:
                    valid_dtw_pairs.add((a_idx, b_idx))
                    final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
            else:
                # Single point in range without overlapping ranges - accept without constraint check
                valid_dtw_pairs.add((a_idx, b_idx))
                final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
        
        if not mute_mode:
            print(f"Found {len(valid_dtw_pairs)}/{len(overlapping_pairs)} age-overlapping segment pairs that are compatible with age constraints")
        
    else:
            # Flexible mode - accept overlapping pairs AND compatible non-overlapping pairs
            if not mute_mode:
                print(f"\nAssessing age compatibility for segment pairs...(loose age correlation mode)")
            
            # Process all pairs that pass DTW distance threshold
            overlapping_pairs = {}
            non_overlapping_pairs = {}
            
            for (a_idx, b_idx), pair_info in all_pairs_with_dtw.items():
                if not pair_info['passes_distance']:
                    continue
                    
                # Separate pairs by age overlap status
                basic_criteria_met = pair_info['ranges_overlap'] or pair_info['single_point_in_range']
                
                if basic_criteria_met:
                    overlapping_pairs[(a_idx, b_idx)] = pair_info
                else:
                    non_overlapping_pairs[(a_idx, b_idx)] = pair_info
            
            # Process overlapping pairs (same as restricted mode)
            for (a_idx, b_idx), pair_info in tqdm(overlapping_pairs.items(), 
                                                desc=f"Checking {len(overlapping_pairs)} overlapping segment pairs..." if not mute_mode else None, disable=mute_mode):
                if pair_info['ranges_overlap']:
                    # Get segment age bounds with uncertainty
                    a_lower_bound = pair_info['age_bounds']['a_lower']
                    a_upper_bound = pair_info['age_bounds']['a_upper']
                    b_lower_bound = pair_info['age_bounds']['b_lower']
                    b_upper_bound = pair_info['age_bounds']['b_upper']
                    
                    # Convert constraint data to numpy arrays
                    constraint_ages_a = np.array(all_constraint_ages_a)
                    constraint_ages_b = np.array(all_constraint_ages_b)
                    constraint_pos_errors_a = np.array(all_constraint_pos_errors_a)
                    constraint_pos_errors_b = np.array(all_constraint_pos_errors_b)
                    constraint_neg_errors_a = np.array(all_constraint_neg_errors_a)
                    constraint_neg_errors_b = np.array(all_constraint_neg_errors_b)
                    
                    # Check compatibility with broader constraint approach
                    compatible = check_age_constraint_compatibility(
                        a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound,
                        constraint_ages_a, constraint_ages_b,
                        constraint_pos_errors_a, constraint_pos_errors_b,
                        constraint_neg_errors_a, constraint_neg_errors_b,
                        datum_ages_a=datum_ages_a, datum_ages_b=datum_ages_b
                    )
                    
                    # Accept pairs that meet both criteria
                    if compatible:
                        valid_dtw_pairs.add((a_idx, b_idx))
                        final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
                else:
                    # Single point in range without overlapping ranges - accept without constraint check
                    valid_dtw_pairs.add((a_idx, b_idx))
                    final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
            
            # Process non-overlapping pairs with age constraint compatibility
            compatible_non_overlapping = 0
            for (a_idx, b_idx), pair_info in tqdm(non_overlapping_pairs.items(), 
                                                desc=f"Checking {len(non_overlapping_pairs)} non-overlapping segment pairs..." if not mute_mode else None, disable=mute_mode):
                # Get segment age bounds
                a_lower_bound = pair_info['age_bounds']['a_lower']
                a_upper_bound = pair_info['age_bounds']['a_upper']
                b_lower_bound = pair_info['age_bounds']['b_lower']
                b_upper_bound = pair_info['age_bounds']['b_upper']
                
                # Convert constraint data to numpy arrays
                constraint_ages_a = np.array(all_constraint_ages_a)
                constraint_ages_b = np.array(all_constraint_ages_b)
                constraint_pos_errors_a = np.array(all_constraint_pos_errors_a)
                constraint_pos_errors_b = np.array(all_constraint_pos_errors_b)
                constraint_neg_errors_a = np.array(all_constraint_neg_errors_a)
                constraint_neg_errors_b = np.array(all_constraint_neg_errors_b)
                
                # Check compatibility with age constraints (flexible approach)
                compatible = check_age_constraint_compatibility(
                    a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound,
                    constraint_ages_a, constraint_ages_b,
                    constraint_pos_errors_a, constraint_pos_errors_b,
                    constraint_neg_errors_a, constraint_neg_errors_b,
                    datum_ages_a=datum_ages_a, datum_ages_b=datum_ages_b
                )
                
                if compatible:
                    valid_dtw_pairs.add((a_idx, b_idx))
                    final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
                    compatible_non_overlapping += 1
            
            if not mute_mode:
                print(f"Found {len(overlapping_pairs)} overlapping segment pairs")
                print(f"Found {compatible_non_overlapping}/{len(non_overlapping_pairs)} compatible non-overlapping segment pairs")
                print(f"Total valid pairs: {len(valid_dtw_pairs)}")
    
    # Check connectivity before dead-end filtering
    if not mute_mode:
        print("\n=== CONNECTIVITY CHECK ===")

    has_paths = has_complete_paths(valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b)

    if not has_paths:
        if not mute_mode:
            print("No complete paths - searching for gap-filling pairs...")
        
        # Find depth gaps
        gaps_a, gaps_b = find_depth_gaps(valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b)
        
        if not mute_mode:
            if gaps_a:
                print(f"Core A gaps: {len(gaps_a)} ranges")
            if gaps_b:
                print(f"Core B gaps: {len(gaps_b)} ranges")
        
        # Find gap-filling candidates with flexible criteria
        detailed_pairs = {}
        for a_idx, b_idx in valid_dtw_pairs:
            a_start = depth_boundaries_a[segments_a[a_idx][0]]
            a_end = depth_boundaries_a[segments_a[a_idx][1]]
            b_start = depth_boundaries_b[segments_b[b_idx][0]]
            b_end = depth_boundaries_b[segments_b[b_idx][1]]
            
            detailed_pairs[(a_idx, b_idx)] = {
                'a_start': a_start, 'a_end': a_end,
                'b_start': b_start, 'b_end': b_end,
                'a_len': a_end - a_start + 1,
                'b_len': b_end - b_start + 1
            }

        max_depth_a = max(depth_boundaries_a)
        max_depth_b = max(depth_boundaries_b)

        successors, predecessors = build_connectivity_graph(valid_dtw_pairs, detailed_pairs)
        top_segments, bottom_segments, dead_ends, orphans, _, _ = identify_special_segments(
            valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
        )

        # Find pairs that could connect dead ends to existing segments
        gap_candidates = []

        for dead_end in dead_ends:
            dead_a_end = detailed_pairs[dead_end]['a_end'] 
            dead_b_end = detailed_pairs[dead_end]['b_end']
            
            # Only check segments that could be immediate successors
            for a_idx in range(len(segments_a)):
                for b_idx in range(len(segments_b)):
                    if (a_idx, b_idx) in valid_dtw_pairs:
                        continue
                        
                    candidate_a_start = depth_boundaries_a[segments_a[a_idx][0]]
                    candidate_b_start = depth_boundaries_b[segments_b[b_idx][0]]
                    
                    # Only add if it could directly follow the dead end
                    if (abs(candidate_a_start - dead_a_end) < 1e-6 and 
                        abs(candidate_b_start - dead_b_end) < 1e-6):
                        gap_candidates.append((a_idx, b_idx))

        # Remove duplicates
        gap_candidates = list(set(gap_candidates))
        
        # Find lowest depth of existing valid pairs
        if valid_dtw_pairs:
            lowest_a_depth = max(depth_boundaries_a[segments_a[pair_a_idx][1]] for pair_a_idx, pair_b_idx in valid_dtw_pairs)
            lowest_b_depth = max(depth_boundaries_b[segments_b[pair_b_idx][1]] for pair_a_idx, pair_b_idx in valid_dtw_pairs)
        else:
            lowest_a_depth = 0
            lowest_b_depth = 0

        # Add bottom-reaching pairs
        bottom_added = force_bottom_segment_pairing(
            valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b,
            log_a, log_b, final_dtw_results, independent_dtw=independent_dtw, mute_mode=mute_mode,
            pca_for_dependent_dtw=pca_for_dependent_dtw
        )

        # Add gap-filling pairs 
        added_pairs = 0
        for pair in gap_candidates:
            a_idx, b_idx = pair
            
            # Extract segments
            a_start = depth_boundaries_a[segments_a[a_idx][0]]
            a_end = depth_boundaries_a[segments_a[a_idx][1]]
            b_start = depth_boundaries_b[segments_b[b_idx][0]]
            b_end = depth_boundaries_b[segments_b[b_idx][1]]
            
            log_a_segment = log_a[a_start:a_end+1]
            log_b_segment = log_b[b_start:b_end+1]
            
            try:
                # Calculate full DTW with quality indicators (same as regular pairs)
                D_sub, wp, QIdx = custom_dtw(log_a_segment, log_b_segment, subseq=False, exponent=1, 
                                        QualityIndex=True, independent_dtw=independent_dtw,
                                        pca_for_dependent_dtw=pca_for_dependent_dtw, sakoe_chiba=True)
                
                # Adjust warping path coordinates to match full log coordinates
                adjusted_wp = wp.copy()
                adjusted_wp[:, 0] += a_start
                adjusted_wp[:, 1] += b_start
                
                # Add age overlap percentage if available
                if (a_idx, b_idx) in all_pairs_with_dtw:
                    pair_info = all_pairs_with_dtw[(a_idx, b_idx)]
                    if 'perc_age_overlap' in pair_info:
                        QIdx['perc_age_overlap'] = pair_info['perc_age_overlap']
                else:
                    # Default value for gap-filling pairs not in age analysis
                    QIdx['perc_age_overlap'] = 0.0
                
                # Store complete DTW results (same format as regular pairs)
                valid_dtw_pairs.add(pair)
                final_dtw_results[pair] = ([adjusted_wp], [], [QIdx])
                added_pairs += 1  # ← ADD THIS LINE!
                
            except Exception as e:
                if debug and not mute_mode:
                    print(f"Error calculating DTW for gap-filling pair ({a_idx}, {b_idx}): {e}")
                continue
        
        # Update total count
        total_added = added_pairs + bottom_added
        
        if not mute_mode:
            print(f"WARNING: {total_added} gap-filling pairs are added...")
            print(f"These gap-filling pairs may not comply with given age constraints...")
        
    elif not mute_mode:
        print("Complete paths exist")

    # Create detailed_pairs for dead-end filtering
    detailed_pairs = {}
    for a_idx, b_idx in valid_dtw_pairs:
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        detailed_pairs[(a_idx, b_idx)] = {
            'a_start': a_start, 'a_end': a_end,
            'b_start': b_start, 'b_end': b_end,
            'a_len': a_end - a_start + 1,
            'b_len': b_end - b_start + 1
        }

    # NOW do dead-end filtering (after gap-filling)
    if exclude_deadend:
        if not mute_mode:
            print("\n=== DEAD-END FILTERING ===")
        
        max_depth_a = max(depth_boundaries_a)
        max_depth_b = max(depth_boundaries_b)
        
        filtered_valid_pairs = filter_dead_end_pairs(
            valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=debug and not mute_mode
        )
        
        removed_pairs = valid_dtw_pairs - filtered_valid_pairs
        for pair in removed_pairs:
            if pair in final_dtw_results:
                del final_dtw_results[pair]
        
        valid_dtw_pairs = filtered_valid_pairs
        
        if not mute_mode:
            print(f"Retained {len(valid_dtw_pairs)} segments after filtering")
    

    # Create dtw matrix if requested
    if create_dtw_matrix:
        # Import here to avoid circular imports
        from ..utils.matrix_plots import plot_dtw_matrix_with_paths
        
        # Prepare age data dictionaries for matrix plotting
        if age_consideration and core_a_age_data is not None:
            matrix_core_a_age_data = core_a_age_data
        else:
            matrix_core_a_age_data = None
            
        if age_consideration and core_b_age_data is not None:
            matrix_core_b_age_data = core_b_age_data
        else:
            matrix_core_b_age_data = None
        
        dtwmatrix_output_file = plot_dtw_matrix_with_paths(
                                dtw_distance_matrix_full, 
                                mode='segment_paths',
                                valid_dtw_pairs=valid_dtw_pairs, 
                                dtw_results=final_dtw_results, 
                                segments_a=segments_a, 
                                segments_b=segments_b,
                                depth_boundaries_a=depth_boundaries_a, 
                                depth_boundaries_b=depth_boundaries_b,
                                output_filename=dtwmatrix_output_filename,
                                visualize_pairs=visualize_pairs,
                                visualize_segment_labels=visualize_segment_labels,
                                # Age constraint data (simplified)
                                core_a_age_data=matrix_core_a_age_data,
                                core_b_age_data=matrix_core_b_age_data,
                                md_a=md_a,
                                md_b=md_b,
                                core_a_name=core_a_name,
                                core_b_name=core_b_name,
                                dpi=dpi
                            )
        if not mute_mode:
            print(f"Generated DTW matrix with paths of all segment pairs at: {dtwmatrix_output_file}")

    # Create animation if requested
    if creategif:
        # Import here to avoid circular imports
        from ..utils.animation import create_segment_dtw_animation
        
        if not mute_mode:
            print("\nCreating GIF animation of all segment pairs...")
        
        # Prepare temporary unified dictionary for animation (before final return)
        temp_dtw_result = {
            'dtw_correlation': final_dtw_results,
            'valid_dtw_pairs': valid_dtw_pairs,
            'segments_a': segments_a,
            'segments_b': segments_b,
            'depth_boundaries_a': depth_boundaries_a,
            'depth_boundaries_b': depth_boundaries_b,
            'dtw_distance_matrix_full': dtw_distance_matrix_full
        }
        
        gif_output_file = create_segment_dtw_animation(
            temp_dtw_result, log_a, log_b, md_a, md_b,
            max_frames=max_frames,
            parallel=True,
            debug=debug and not mute_mode,
            color_interval_size=color_interval_size,
            keep_frames=keep_frames,
            output_filename=gif_output_filename,
            age_consideration=age_consideration,
            datum_ages_a=datum_ages_a,
            datum_ages_b=datum_ages_b,
            restricted_age_correlation=restricted_age_correlation,
            all_constraint_depths_a=all_constraint_depths_a,
            all_constraint_depths_b=all_constraint_depths_b,
            all_constraint_ages_a=all_constraint_ages_a,
            all_constraint_ages_b=all_constraint_ages_b,
            all_constraint_pos_errors_a=all_constraint_pos_errors_a,
            all_constraint_pos_errors_b=all_constraint_pos_errors_b,
            all_constraint_neg_errors_a=all_constraint_neg_errors_a,
            all_constraint_neg_errors_b=all_constraint_neg_errors_b,
            dpi=dpi
        )
        
        if not mute_mode:
            print(f"Generated GIF animation of all segment pairs at: {gif_output_file}")

        # Display only if created
        if gif_output_file:
            display(IPImage(filename=gif_output_file))
    else:
        if not mute_mode:
            print("\nSkipping GIF animation creation as creategif=False")

    # Clean up memory before returning
    plt.close('all')
    
    # Make sure all figures are closed
    for fig_num in plt.get_fignums():
        plt.close(fig_num)
    
    # Display DTW matrix output figure if available
    if create_dtw_matrix and dtwmatrix_output_file and os.path.exists(dtwmatrix_output_file):
        if not mute_mode:
            print(f"\nDisplaying DTW matrix visualization from: {dtwmatrix_output_file}")
        try:
            # Check if file is SVG format which cannot be embedded
            if dtwmatrix_output_file.lower().endswith('.svg'):
                if not mute_mode:
                    print(f"SVG format detected. File saved at: {dtwmatrix_output_file}")
            else:
                display(IPImage(filename=dtwmatrix_output_file))
        except ValueError as e:
            if "Cannot embed" in str(e) and "image format" in str(e):
                if not mute_mode:
                    print(f"Image format not supported for display. File saved at: {dtwmatrix_output_file}")
            else:
                pass
        except:
            pass

    gc.collect()
    
    # Return unified dictionary containing all DTW analysis results
    dtw_result = {
        'dtw_correlation': final_dtw_results,
        'valid_dtw_pairs': valid_dtw_pairs,
        'segments_a': segments_a,
        'segments_b': segments_b,
        'depth_boundaries_a': depth_boundaries_a,
        'depth_boundaries_b': depth_boundaries_b,
        'dtw_distance_matrix_full': dtw_distance_matrix_full
    }
    
    return dtw_result

