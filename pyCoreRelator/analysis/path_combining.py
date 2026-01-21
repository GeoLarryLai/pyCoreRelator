"""
DTW path combining functions for pyCoreRelator.

Included Functions:
- combine_segment_dtw_results: Combine DTW results from multiple segment pairs
- compute_combined_path_metrics: Compute quality metrics from combined warping paths
 from CSV files
 relationships between paths
 against existing filtered paths
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


from .quality import compute_quality_indicators

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
    segment_wps_ordered = []  # Keep warping paths in segment order for sectional calc
    valid_segment_pairs = []  # Track which segment pairs have valid paths
    
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
        segment_wps_ordered.append(paths[0])
        valid_segment_pairs.append((a_idx, b_idx))
        
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
    
    # Calculate combined quality metrics (including sectional metrics)
    if all_quality_indicators:
        age_overlap_values = []
        for qi in all_quality_indicators:
            if 'perc_age_overlap' in qi:
                age_overlap_values.append(float(qi['perc_age_overlap']))
        
        combined_quality = compute_combined_path_metrics(
            combined_wp, log_a, log_b, all_quality_indicators, dtw_distance_matrix_full, age_overlap_values, 
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            segment_wps=segment_wps_ordered,
            path_segment_pairs=valid_segment_pairs,
            segments_a=segments_a,
            segments_b=segments_b,
            depth_boundaries_a=depth_boundaries_a,
            depth_boundaries_b=depth_boundaries_b
        )
    else:
        combined_quality = None
        
    return combined_wp, combined_quality



def compute_combined_path_metrics(combined_wp, log_a, log_b, segment_quality_indicators, dtw_distance_matrix_full, age_overlap_values=None, pca_for_dependent_dtw=False, 
                                   segment_wps=None, path_segment_pairs=None, segments_a=None, segments_b=None, 
                                   depth_boundaries_a=None, depth_boundaries_b=None, metrics_to_compute=None):
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
    segment_wps : list, optional
        List of individual segment warping paths (for sectional metrics)
    path_segment_pairs : list, optional
        List of (a_idx, b_idx) segment pairs in order (for sectional metrics)
    segments_a, segments_b : list, optional
        Segment definitions for cores A and B
    depth_boundaries_a, depth_boundaries_b : list, optional
        Depth boundaries for cores A and B
    metrics_to_compute : list or str, optional
        List of metrics to compute. 
        If None, computes default metrics: ['norm_dtw', 'corr_coef', 'norm_dtw_sect', 'corr_coef_sect'].
        If 'ALL', computes all available metrics.
        Available options:
        - 'norm_dtw': Normalized DTW distance (lower is better)
        - 'corr_coef': Correlation coefficient (higher is better)
        - 'norm_dtw_sect': Sectional normalized DTW (excludes pinch-outs)
        - 'corr_coef_sect': Sectional correlation coefficient (excludes pinch-outs)
        - 'dtw_ratio': DTW warping ratio (lower is better)
        - 'perc_diag': Path diagonality percentage (higher is better)
        - 'dtw_warp_eff': DTW warping efficiency
        - 'perc_age_overlap': Age overlap percentage (higher is better)
        Invalid metrics in the list are silently skipped.
    
    Returns
    -------
    dict
        Combined quality metrics including normalized DTW distance, correlation
        coefficient, path characteristics, age overlap percentage, and sectional metrics
    """
    from .quality import compute_quality_indicators
    
    # Define all available metrics
    all_metrics = ['norm_dtw', 'dtw_ratio', 'perc_diag', 'dtw_warp_eff', 'corr_coef', 'perc_age_overlap', 'norm_dtw_sect', 'corr_coef_sect']
    default_metrics = ['norm_dtw', 'corr_coef', 'norm_dtw_sect', 'corr_coef_sect']
    
    # Handle metrics_to_compute parameter
    if metrics_to_compute is None:
        # Default to the 4 primary metrics
        metrics_to_compute = default_metrics
    elif isinstance(metrics_to_compute, str) and metrics_to_compute.upper() == 'ALL':
        # Compute all available metrics
        metrics_to_compute = all_metrics
    else:
        # Filter to only valid metrics, skip invalid ones silently
        metrics_to_compute = [m for m in metrics_to_compute if m in all_metrics]
        if not metrics_to_compute:
            # If no valid metrics, fall back to default
            metrics_to_compute = default_metrics
    
    # Initialize metrics dictionary
    metrics = {
        'norm_dtw': 0.0,
        'dtw_ratio': 0.0,
        'perc_diag': 0.0,
        'dtw_warp_eff': 0.0,
        'corr_coef': 0.0,
        'perc_age_overlap': 0.0,
        'norm_dtw_sect': 0.0,
        'corr_coef_sect': 0.0
    }
    
    # Helper function to calculate DTW step costs along a path
    def get_path_dtw_cost_efficient(wp, dtw_matrix):
        """Extract step costs only at path coordinates"""
        if dtw_matrix is None or wp is None or len(wp) == 0:
            return 0.0
            
        total_cost = 0.0
        
        for i in range(len(wp)):
            a_idx = int(wp[i, 0])
            b_idx = int(wp[i, 1])
            
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
    
    # Compute metrics using the combined warping path (only compute requested metrics)
    if combined_wp is not None and len(combined_wp) > 1:
        # Extract and validate indices from combined warping path
        p_indices = combined_wp[:, 0].astype(int)
        q_indices = combined_wp[:, 1].astype(int)
        
        p_indices = np.clip(p_indices, 0, len(log_a) - 1)
        q_indices = np.clip(q_indices, 0, len(log_b) - 1)
        
        # Determine which computations are needed
        needs_path_cost = any(m in metrics_to_compute for m in ['norm_dtw', 'norm_dtw_sect'])
        needs_quality_indicators = any(m in metrics_to_compute for m in ['dtw_ratio', 'dtw_warp_eff', 'corr_coef', 'perc_diag'])
        needs_sectional = any(m in metrics_to_compute for m in ['norm_dtw_sect', 'corr_coef_sect'])
        
        # Compute path cost only if needed
        path_cost = None
        if needs_path_cost or needs_quality_indicators:
            path_cost = get_path_dtw_cost_efficient(combined_wp, dtw_distance_matrix_full)
        
        # Calculate norm_dtw if requested
        if 'norm_dtw' in metrics_to_compute and path_cost is not None:
            metrics['norm_dtw'] = path_cost / (dtw_distance_matrix_full.shape[0] + dtw_distance_matrix_full.shape[1])
        
        # Compute quality indicators only if needed
        if needs_quality_indicators and path_cost is not None:
            dummy_D = np.array([[path_cost]])
            combined_metrics = compute_quality_indicators(log_a, log_b, p_indices, q_indices, dummy_D, pca_for_dependent_dtw=pca_for_dependent_dtw)
            
            # Update only requested metrics from compute_quality_indicators
            if 'dtw_ratio' in metrics_to_compute:
                metrics['dtw_ratio'] = float(combined_metrics.get('dtw_ratio', 0.0))
            if 'dtw_warp_eff' in metrics_to_compute:
                metrics['dtw_warp_eff'] = float(combined_metrics.get('dtw_warp_eff', 0.0))
            if 'corr_coef' in metrics_to_compute:
                metrics['corr_coef'] = float(combined_metrics.get('corr_coef', 0.0))
            if 'perc_diag' in metrics_to_compute:
                metrics['perc_diag'] = float(combined_metrics.get('perc_diag', 0.0))
        
        # Compute sectional metrics only if requested
        if needs_sectional:
            if (segment_wps is not None and path_segment_pairs is not None and 
                segments_a is not None and segments_b is not None and
                depth_boundaries_a is not None and depth_boundaries_b is not None):
                
                sect_wp, sect_len_a, sect_len_b = _compute_sectional_warping_path(
                    segment_wps, path_segment_pairs, segments_a, segments_b,
                    depth_boundaries_a, depth_boundaries_b
                )
                
                if sect_wp is not None and len(sect_wp) > 1:
                    # Compute sectional norm_dtw if requested
                    if 'norm_dtw_sect' in metrics_to_compute:
                        sect_path_cost = get_path_dtw_cost_efficient(sect_wp, dtw_distance_matrix_full)
                        metrics['norm_dtw_sect'] = sect_path_cost / (sect_len_a + sect_len_b) if (sect_len_a + sect_len_b) > 0 else 0.0
                    
                    # Compute sectional corr_coef if requested
                    if 'corr_coef_sect' in metrics_to_compute:
                        sect_p_indices = sect_wp[:, 0].astype(int)
                        sect_q_indices = sect_wp[:, 1].astype(int)
                        sect_p_indices = np.clip(sect_p_indices, 0, len(log_a) - 1)
                        sect_q_indices = np.clip(sect_q_indices, 0, len(log_b) - 1)
                        
                        sect_path_cost_for_corr = get_path_dtw_cost_efficient(sect_wp, dtw_distance_matrix_full) if 'norm_dtw_sect' not in metrics_to_compute else sect_path_cost
                        sect_dummy_D = np.array([[sect_path_cost_for_corr]])
                        sect_metrics = compute_quality_indicators(log_a, log_b, sect_p_indices, sect_q_indices, sect_dummy_D, pca_for_dependent_dtw=pca_for_dependent_dtw)
                        metrics['corr_coef_sect'] = float(sect_metrics.get('corr_coef', 0.0))
                else:
                    # No meaningful section found, use full metrics
                    if 'norm_dtw_sect' in metrics_to_compute:
                        metrics['norm_dtw_sect'] = metrics['norm_dtw']
                    if 'corr_coef_sect' in metrics_to_compute:
                        metrics['corr_coef_sect'] = metrics['corr_coef']
            else:
                # No segment info provided, use full metrics
                if 'norm_dtw_sect' in metrics_to_compute:
                    metrics['norm_dtw_sect'] = metrics['norm_dtw']
                if 'corr_coef_sect' in metrics_to_compute:
                    metrics['corr_coef_sect'] = metrics['corr_coef']
    
    # Average age overlap values across segments (only if requested)
    if 'perc_age_overlap' in metrics_to_compute and age_overlap_values:
        metrics['perc_age_overlap'] = float(sum(age_overlap_values) / len(age_overlap_values))
    
    return metrics


def _compute_sectional_warping_path(segment_wps, path_segment_pairs, segments_a, segments_b,
                                     depth_boundaries_a, depth_boundaries_b):
    """
    Compute the sectional warping path by excluding pinch-out segments at top and bottom.
    
    Pinch-out segments are those where one side has length=1 (single point) and 
    connects to the boundary (top=0 or bottom=max_depth).
    
    Parameters
    ----------
    segment_wps : list
        List of individual segment warping paths in order
    path_segment_pairs : list
        List of (a_idx, b_idx) segment pairs in the same order as segment_wps
    segments_a, segments_b : list
        Segment definitions [(start_idx, end_idx), ...]
    depth_boundaries_a, depth_boundaries_b : list
        Depth boundary values
    
    Returns
    -------
    tuple
        (sectional_wp, sectional_len_a, sectional_len_b) or (None, 0, 0) if no valid section
    """
    if not segment_wps or not path_segment_pairs:
        return None, 0, 0
    
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    def is_pinchout_segment(a_idx, b_idx):
        """Check if a segment pair is a pinch-out (one side has length 1 at boundary)."""
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        a_len = a_end - a_start + 1
        b_len = b_end - b_start + 1
        
        # Top pinch-out: one side has length 1 and starts at 0
        is_top_pinch_a = (a_len == 1 and a_start == 0)
        is_top_pinch_b = (b_len == 1 and b_start == 0)
        
        # Bottom pinch-out: one side has length 1 and ends at max depth
        is_bottom_pinch_a = (a_len == 1 and a_end == max_depth_a)
        is_bottom_pinch_b = (b_len == 1 and b_end == max_depth_b)
        
        return (is_top_pinch_a or is_top_pinch_b, is_bottom_pinch_a or is_bottom_pinch_b)
    
    # Identify which segments are pinch-outs
    n_segments = len(path_segment_pairs)
    is_top_pinch = [False] * n_segments
    is_bottom_pinch = [False] * n_segments
    
    for i, (a_idx, b_idx) in enumerate(path_segment_pairs):
        top_p, bottom_p = is_pinchout_segment(a_idx, b_idx)
        is_top_pinch[i] = top_p
        is_bottom_pinch[i] = bottom_p
    
    # Find consecutive top pinch-outs from the start
    first_non_pinch_top = 0
    for i in range(n_segments):
        if is_top_pinch[i]:
            first_non_pinch_top = i + 1
        else:
            break
    
    # Find consecutive bottom pinch-outs from the end
    last_non_pinch_bottom = n_segments - 1
    for i in range(n_segments - 1, -1, -1):
        if is_bottom_pinch[i]:
            last_non_pinch_bottom = i - 1
        else:
            break
    
    # Check if there's any meaningful section left
    if first_non_pinch_top > last_non_pinch_bottom:
        return None, 0, 0
    
    # Collect warping paths for the meaningful section
    sectional_wps = []
    for i in range(first_non_pinch_top, last_non_pinch_bottom + 1):
        if i < len(segment_wps):
            sectional_wps.append(segment_wps[i])
    
    if not sectional_wps:
        return None, 0, 0
    
    # Combine sectional warping paths
    combined_sect_wp = np.vstack(sectional_wps)
    combined_sect_wp = np.unique(combined_sect_wp, axis=0)
    combined_sect_wp = combined_sect_wp[combined_sect_wp[:, 0].argsort()]
    
    # Calculate sectional log lengths for normalization
    # These are the index ranges covered ONLY by the sectional warping path (excluding pinch-outs)
    # Used to normalize norm_dtw_sect by the sectional length, not the full log length
    sect_min_a = int(combined_sect_wp[:, 0].min())
    sect_max_a = int(combined_sect_wp[:, 0].max())
    sect_min_b = int(combined_sect_wp[:, 1].min())
    sect_max_b = int(combined_sect_wp[:, 1].max())
    
    sect_len_a = sect_max_a - sect_min_a + 1
    sect_len_b = sect_max_b - sect_min_b + 1
    
    return combined_sect_wp, sect_len_a, sect_len_b


