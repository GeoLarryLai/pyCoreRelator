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
    from .quality import compute_quality_indicators
    
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


