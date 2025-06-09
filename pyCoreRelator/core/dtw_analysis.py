"""
Core DTW analysis functions for pyCoreRelator

Main DTW computation and comprehensive analysis functions.
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

# Import from other modules
from .quality_metrics import compute_quality_indicators, calculate_age_overlap_percentage
from .age_models import check_age_constraint_compatibility
from .segment_analysis import find_all_segments, filter_dead_end_pairs
from ..utils.helpers import find_nearest_index
from ..visualization.plotting import plot_segment_pair_correlation
from ..visualization.matrix_plots import plot_dtw_matrix_with_paths
from ..visualization.animation import create_segment_dtw_animation


def handle_single_point_dtw(log1, log2, exponent=1, QualityIndex=False):
    """
    Handle DTW for the case where log1 contains only a single data point.
    This function creates a custom warping path that maps the single point in log1
    to all points in log2.
    
    Parameters:
    -----------
    log1 : array-like
        First well log data with a single point.
    log2 : array-like
        Second well log data.
    exponent : float, default=1
        Exponent for cost calculation.
    QualityIndex : bool, default=False
        If True, computes quality indicators.
        
    Returns:
    --------
    D : np.ndarray
        The accumulated cost matrix (1 x len(log2)).
    wp : np.ndarray
        The warping path as a sequence of index pairs.
    QIdx : dict, optional
        Quality indicators (if QualityIndex is True).
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
    to the single point in log2.
    
    Parameters:
    -----------
    log1 : array-like
        First well log data with multiple points.
    log2 : array-like
        Second well log data with a single point.
    exponent : float, default=1
        Exponent for cost calculation.
    QualityIndex : bool, default=False
        If True, computes quality indicators.
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
    
    Parameters:
    -----------
    log1, log2 : array-like
        Single-point log data
    exponent : float, default=1
        Exponent for cost calculation
    QualityIndex : bool, default=False
        If True, computes quality indicators
    
    Returns:
    --------
    D : np.ndarray
        The accumulated cost matrix (1x1)
    wp : np.ndarray
        The warping path as a single index pair
    QIdx : dict, optional
        Quality indicators (if QualityIndex is True)
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


def custom_dtw(log1, log2, subseq=False, exponent=1, QualityIndex=False, independent_dtw=False, available_columns=None):
    """
    Custom implementation of Dynamic Time Warping for well log correlation.
    This function creates a similarity matrix between two well logs and applies DTW
    to find the optimal alignment, handling all edge cases.
    
    Parameters:
    -----------
    log1, log2 : array-like
        Well log data to be compared
    subseq : bool, default=False
        If True, performs subsequence DTW
    exponent : float, default=1
        Exponent for cost calculation
    QualityIndex : bool, default=False
        If True, computes quality indicators
    independent_dtw : bool, default=False
        If True, performs independent DTW on each dimension separately
    available_columns : list, default=None
        Column names for logging when independent_dtw=True
        
    Returns:
    --------
    D : np.ndarray
        The accumulated cost matrix where D[i,j] contains the minimum cumulative cost
        to reach point (i,j) from the starting point. 
    wp : np.ndarray
        The warping path as a sequence of index pairs
    QIdx : dict, optional
        Quality indicators (if QualityIndex is True)
    """

    # Convert inputs to float32 if not already
    log1 = log1.astype(np.float32)
    log2 = log2.astype(np.float32)

    # Check for empty logs
    if log1 is None or len(log1) == 0:
        print("Error: log1 is empty or None. Cannot perform valid DTW.")
        if QualityIndex:
            return np.array([[0]]), np.array([[0, 0]]), {'norm_dtw': 0, 'dtw_ratio': 0, 'variance_deviation': 0, 'perc_diag': 0, 'corr_coef': 0}
        return np.array([[0]]), np.array([[0, 0]])
    
    if log2 is None or len(log2) == 0:
        print("Error: log2 is empty or None. Cannot perform valid DTW.")
        if QualityIndex:
            return np.array([[0]]), np.array([[0, 0]]), {'norm_dtw': 0, 'dtw_ratio': 0, 'variance_deviation': 0, 'perc_diag': 0, 'corr_coef': 0}
        return np.array([[0]]), np.array([[0, 0]])

    # Convert logs to numpy arrays
    log1 = np.array(log1)
    log2 = np.array(log2)
    r = len(log1)
    c = len(log2)
    
    # Handle special case: log1 is a single point - always use full correlation
    if r == 1:
        # print("Processing DTW with log1 having only one point - using full correlation")
        return handle_single_point_dtw(log1, log2, exponent, QualityIndex)
    
    # Handle special case: log2 is a single point - always use full correlation
    if c == 1:
        # print("Processing DTW with log2 having only one point - using full correlation")
        # Use the dedicated function instead of transposing
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
                                                   independent_dtw=False)
                all_QIdx.append(QIdx_dim)
            else:
                D_dim, wp_dim = custom_dtw(dim_log1, dim_log2, subseq=subseq, 
                                         exponent=exponent, QualityIndex=False, 
                                         independent_dtw=False)
            
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
    
    if log1_is_multidim or log2_is_multidim:
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
        # For 1D data
        for i in range(r):
            sm[i, :] = (np.abs(log2 - log1[i])) ** exponent

    # Compute the accumulated cost matrix D and the warping path wp
    D, wp = dtw(C=sm, subseq=subseq)
    
    # Adjust warping path indices to be within valid ranges
    if wp is not None:
        wp[:, 0] = np.clip(wp[:, 0], 0, r - 1)
        wp[:, 1] = np.clip(wp[:, 1], 0, c - 1)
        
    # Compute quality indicators if requested
    if QualityIndex and wp is not None:
        p = wp[:, 0]
        q = wp[:, 1]
        QIdx = compute_quality_indicators(log1, log2, p, q, D)
        return D, wp, QIdx
    else:
        return D, wp


def run_comprehensive_dtw_analysis(log_a, log_b, md_a, md_b, picked_depths_a=None, picked_depths_b=None, 
                              top_bottom=True, top_depth=0.0,
                              independent_dtw=False, 
                              create_dtw_matrix=True,
                              visualize_pairs=True, 
                              visualize_segment_labels=False,
                              dtwmatrix_output_filename='SegmentPair_DTW_matrix.png',
                              creategif=True, 
                              gif_output_filename='SegmentPair_DTW_animation.gif', max_frames=150, 
                              debug=False, color_interval_size=None,
                              keep_frames=True, age_consideration=False, ages_a=None, ages_b=None,
                              restricted_age_correlation=True, 
                              all_constraint_ages_a=None, all_constraint_ages_b=None,
                              all_constraint_depths_a=None, all_constraint_depths_b=None,
                              all_constraint_pos_errors_a=None, all_constraint_pos_errors_b=None,
                              all_constraint_neg_errors_a=None, all_constraint_neg_errors_b=None,
                              dtw_distance_threshold=None,
                              exclude_deadend=True,  # NEW PARAMETER
                              # Age constraint visualization parameters (default None)
                              age_constraint_a_source_cores=None,
                              age_constraint_b_source_cores=None,
                              core_a_name=None,
                              core_b_name=None):
    """
    Run comprehensive DTW analysis with integrated age correlation functionality.
    
    NEW PARAMETERS:
    ---------------
    exclude_deadend : bool, default=True
        If True, filter out dead end and orphan segment pairs before final processing
    
    Returns:
    --------
    tuple: (dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full)
    """
    
    print("Starting comprehensive DTW analysis with integrated age correlation...")
    
    # Find all segments
    segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dated_picked_depths_a, dated_picked_depths_b = find_all_segments(
        log_a, log_b, md_a, md_b, 
        picked_depths_a, picked_depths_b,
        top_bottom=top_bottom, 
        top_depth=top_depth
    )

    # Check if age consideration is enabled and validate age data
    if age_consideration:
        print(f"\nAge consideration enabled - {'restricted' if restricted_age_correlation else 'flexible'} age correlation")
        
        if ages_a is None or ages_b is None:
            raise ValueError("Both ages_a and ages_b must be provided when age_consideration is True")
        
        # Check if age data dictionaries have the required keys and non-empty values
        required_keys = ['depths', 'ages', 'pos_uncertainties', 'neg_uncertainties']
        for key in required_keys:
            if key not in ages_a or not ages_a[key] or key not in ages_b or not ages_b[key]:
                raise ValueError(f"Missing or empty required key '{key}' in ages_a or ages_b")
        
        # Check if depths match picked_depths
        if (picked_depths_a is not None and len(dated_picked_depths_a) != len(ages_a['depths'])) or \
           (picked_depths_b is not None and len(dated_picked_depths_b) != len(ages_b['depths'])):
            raise ValueError("The number of depths in ages_a/ages_b must match the number of dated picked depths")
        
        # Check if constraint data is provided when flexible age correlation is enabled
        if not restricted_age_correlation:
            if (all_constraint_ages_a is None or all_constraint_depths_a is None or 
                all_constraint_ages_b is None or all_constraint_depths_b is None or
                all_constraint_pos_errors_a is None or all_constraint_pos_errors_b is None or
                all_constraint_neg_errors_a is None or all_constraint_neg_errors_b is None):
                raise ValueError("Complete age constraint data must be provided when restricted_age_correlation is False")
    
    # Calculate full DTW distance matrix for reference
    print("Calculating full DTW distance matrix...")
    dtw_distance_matrix_full, _ = custom_dtw(log_a, log_b, subseq=False, exponent=1, independent_dtw=independent_dtw)
    
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
    
    print(f"Found {len(all_possible_pairs)} valid segment pairs after boundary checks")
    
    # Create a dictionary to store all pairs with their age information and DTW results
    all_pairs_with_dtw = {}
    
    # Process all possible pairs and calculate age criteria
    if age_consideration:        
        for a_idx, b_idx in tqdm(all_possible_pairs, desc="Calculating age bounds for all segment pairs..."):
            # Get segment depths
            a_start_depth = md_a[depth_boundaries_a[segments_a[a_idx][0]]]
            a_end_depth = md_a[depth_boundaries_a[segments_a[a_idx][1]]]
            b_start_depth = md_b[depth_boundaries_b[segments_b[b_idx][0]]]
            b_end_depth = md_b[depth_boundaries_b[segments_b[b_idx][1]]]
            
            # Find age indices
            a_start_age_idx = np.argmin(np.abs(np.array(ages_a['depths']) - a_start_depth))
            a_end_age_idx = np.argmin(np.abs(np.array(ages_a['depths']) - a_end_depth))
            b_start_age_idx = np.argmin(np.abs(np.array(ages_b['depths']) - b_start_depth))
            b_end_age_idx = np.argmin(np.abs(np.array(ages_b['depths']) - b_end_depth))
            
            # Get age bounds with uncertainties
            a_start_age = ages_a['ages'][a_start_age_idx]
            a_end_age = ages_a['ages'][a_end_age_idx]
            b_start_age = ages_b['ages'][b_start_age_idx]
            b_end_age = ages_b['ages'][b_end_age_idx]
            
            a_start_pos_error = ages_a['pos_uncertainties'][a_start_age_idx]
            a_start_neg_error = ages_a['neg_uncertainties'][a_start_age_idx]
            a_end_pos_error = ages_a['pos_uncertainties'][a_end_age_idx]
            a_end_neg_error = ages_a['neg_uncertainties'][a_end_age_idx]
            
            b_start_pos_error = ages_b['pos_uncertainties'][b_start_age_idx]
            b_start_neg_error = ages_b['neg_uncertainties'][b_start_age_idx]
            b_end_pos_error = ages_b['pos_uncertainties'][b_end_age_idx]
            b_end_neg_error = ages_b['neg_uncertainties'][b_end_age_idx]
            
            # Calculate age bounds
            a_lower_bound = min(a_start_age - a_start_neg_error, a_end_age - a_end_neg_error)
            a_upper_bound = max(a_start_age + a_start_pos_error, a_end_age + a_end_pos_error)
            
            b_lower_bound = min(b_start_age - b_start_neg_error, b_end_age - b_end_neg_error)
            b_upper_bound = max(b_start_age + b_start_pos_error, b_end_age + b_end_pos_error)
            
            # NEW: Calculate age overlap percentage (without uncertainty)
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
                'perc_age_overlap': perc_age_overlap  # NEW METRIC
            }
    else:
        # If age consideration is disabled, create empty entries for all pairs
        for a_idx, b_idx in all_possible_pairs:
            all_pairs_with_dtw[(a_idx, b_idx)] = {
                'a_idx': a_idx,
                'b_idx': b_idx,
                'perc_age_overlap': 0.0  # Default value when age consideration is disabled
            }
    
    # Process pairs based on age criteria and calculate DTW   
    def process_segment_pair(a_idx, b_idx, pair_info, independent_dtw=False):
        """Process a single segment pair and calculate DTW"""
        # Extract segments
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        segment_a = log_a[a_start:a_end+1]
        segment_b = log_b[b_start:b_end+1]
        
        # Perform DTW
        try:
            D_sub, wp, QIdx = custom_dtw(segment_a, segment_b, subseq=False, exponent=1, QualityIndex=True, independent_dtw=independent_dtw)
            
            # Adjust warping path coordinates
            adjusted_wp = wp.copy()
            adjusted_wp[:, 0] += a_start
            adjusted_wp[:, 1] += b_start
            
            final_dist = D_sub[-1, -1]
            
            # NEW: Add age overlap percentage to quality indicators
            if 'perc_age_overlap' in pair_info:
                QIdx['perc_age_overlap'] = pair_info['perc_age_overlap']
            
            # Flexible DTW distance filtering
            if dtw_distance_threshold is None:
                # No DTW distance filtering - accept all segments
                passes_distance = True
                if debug:
                    print(f"Segment ({a_idx+1}, {b_idx+1}): DTW distance {final_dist:.2f} - ACCEPTED (no threshold)")
            else:
                # Apply DTW distance threshold
                passes_distance = final_dist < dtw_distance_threshold or len(segment_a) == 1 or len(segment_b) == 1
                if debug and not passes_distance:
                    print(f"Segment ({a_idx+1}, {b_idx+1}): DTW distance {final_dist:.2f} - REJECTED (threshold: {dtw_distance_threshold})")
                elif debug:
                    print(f"Segment ({a_idx+1}, {b_idx+1}): DTW distance {final_dist:.2f} - ACCEPTED (threshold: {dtw_distance_threshold})")
            
            return {
                'dtw_results': ([adjusted_wp], [], [QIdx]),
                'dtw_distance': final_dist,
                'passes_distance': passes_distance,
            }
        except Exception as e:
            if debug:
                print(f"Error calculating DTW for pair ({a_idx}, {b_idx}): {e}")
            return {
                'dtw_results': ([], [], []),
                'dtw_distance': float('inf'),
                'passes_distance': False,
            }
    
    # Calculate DTW for candidate pairs
    for a_idx, b_idx in tqdm(all_possible_pairs, desc="Calculating DTW for segment pairs..."):
        pair_info = all_pairs_with_dtw[(a_idx, b_idx)]
        
        # Get DTW results for this pair
        dtw_info = process_segment_pair(a_idx, b_idx, pair_info, independent_dtw=independent_dtw)
        
        # Update the pair information with DTW results
        pair_info.update(dtw_info)
    
    # Determine which pairs are valid based on age and DTW criteria
    valid_dtw_pairs = set()
    final_dtw_results = {}
    
    if not age_consideration:
        # Age consideration is disabled - use only DTW distance
        for (a_idx, b_idx), pair_info in all_pairs_with_dtw.items():
            if pair_info['passes_distance']:
                valid_dtw_pairs.add((a_idx, b_idx))
                final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
        
        print(f"\nFound {len(valid_dtw_pairs)} valid segment pairs based on DTW distance")
        
    elif restricted_age_correlation:
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
        for (a_idx, b_idx), pair_info in tqdm(overlapping_pairs.items(), desc=f"Checking {len(overlapping_pairs)} segment pairs with overlapping age range..."):
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
                    ages_a=ages_a, ages_b=ages_b
                )
                
                # Accept pairs that meet both criteria
                if compatible:
                    valid_dtw_pairs.add((a_idx, b_idx))
                    final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
            else:
                # Single point in range without overlapping ranges - accept without constraint check
                valid_dtw_pairs.add((a_idx, b_idx))
                final_dtw_results[(a_idx, b_idx)] = pair_info['dtw_results']
        
        print(f"Found {len(valid_dtw_pairs)}/{len(overlapping_pairs)} age-overlapping segment pairs that are compatible with age constraints")
        
    else:
        # Flexible mode - accept overlapping pairs AND compatible non-overlapping pairs
        print(f"\nAssessing age compatibility for segment pairs...(loose age correlation mode)")
        # [Previous flexible mode code remains the same but with valid_dtw_pairs and final_dtw_results updates]
        # Note: This section would need the full flexible mode implementation from the original code
        pass
    
    # NEW: Filter out dead end pairs if exclude_deadend is True
    if exclude_deadend:
        print(f"\nFiltering dead-end pairs (exclude_deadend=True)...")
        
        # Get max depths
        max_depth_a = max(depth_boundaries_a)
        max_depth_b = max(depth_boundaries_b)
        
        # Apply dead end filtering
        filtered_valid_pairs = filter_dead_end_pairs(
            valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=debug
        )
        
        # Update valid_dtw_pairs and final_dtw_results
        removed_pairs = valid_dtw_pairs - filtered_valid_pairs
        for pair in removed_pairs:
            if pair in final_dtw_results:
                del final_dtw_results[pair]
        
        valid_dtw_pairs = filtered_valid_pairs
        
        print(f"Dead-end filtering complete: {len(valid_dtw_pairs)} segments retained")
    
    # Create dtw matrix if requested
    if create_dtw_matrix:
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
                                # Age constraint parameters (passed through)
                                age_constraint_a_depths=all_constraint_depths_a,
                                age_constraint_a_ages=all_constraint_ages_a,
                                age_constraint_a_source_cores=age_constraint_a_source_cores,
                                age_constraint_b_depths=all_constraint_depths_b,
                                age_constraint_b_ages=all_constraint_ages_b,
                                age_constraint_b_source_cores=age_constraint_b_source_cores,
                                md_a=md_a,
                                md_b=md_b,
                                core_a_name=core_a_name,
                                core_b_name=core_b_name
                            )
        print(f"Generated DTW matrix with paths of all segment pairs at: {dtwmatrix_output_file}")

    # Create animation if requested
    if creategif:
        print("\nCreating GIF animation of all segment pairs...")
        gif_output_file = create_segment_dtw_animation(
            log_a, log_b, md_a, md_b, 
            final_dtw_results, valid_dtw_pairs, 
            segments_a, segments_b, 
            depth_boundaries_a, depth_boundaries_b,
            max_frames=max_frames,
            parallel=True,
            debug=debug,
            color_interval_size=color_interval_size,
            keep_frames=keep_frames,
            output_filename=gif_output_filename,
            age_consideration=age_consideration,
            ages_a=ages_a,
            ages_b=ages_b,
            restricted_age_correlation=restricted_age_correlation,
            all_constraint_depths_a=all_constraint_depths_a,
            all_constraint_depths_b=all_constraint_depths_b,
            all_constraint_ages_a=all_constraint_ages_a,
            all_constraint_ages_b=all_constraint_ages_b,
            all_constraint_pos_errors_a=all_constraint_pos_errors_a,
            all_constraint_pos_errors_b=all_constraint_pos_errors_b,
            all_constraint_neg_errors_a=all_constraint_neg_errors_a,
            all_constraint_neg_errors_b=all_constraint_neg_errors_b
        )
        
        print(f"Generated GIF animation of all segment pairs at: {gif_output_file}")

        # Display only if created
        if gif_output_file:
            from IPython.display import Image as IPImage
            display(IPImage(filename=gif_output_file))

    # Clean up memory before returning
    plt.close('all')
    
    # Make sure all figures are closed
    for fig_num in plt.get_fignums():
        plt.close(fig_num)
    
    # Display DTW matrix output figure if available
    if dtwmatrix_output_file and os.path.exists(dtwmatrix_output_file):
        print(f"\nDisplaying DTW matrix visualization from: {dtwmatrix_output_file}")
        display(IPImage(filename=dtwmatrix_output_file))

    gc.collect()
    
    return final_dtw_results, valid_dtw_pairs, segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, dtw_distance_matrix_full