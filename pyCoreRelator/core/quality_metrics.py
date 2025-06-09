"""
Quality metrics computation for DTW analysis.

Included Functions:
- compute_quality_indicators: Compute comprehensive quality metrics for DTW alignment
- calculate_age_overlap_percentage: Calculate overlap percentage between two age intervals

This module provides functions to compute various quality indicators for Dynamic Time Warping (DTW)
alignment results and calculate age overlap percentages between geological intervals.
"""

import numpy as np
from scipy import stats

def compute_quality_indicators(log1, log2, p, q, D):
    """
    Compute quality indicators for the DTW alignment.
    
    This function calculates comprehensive quality metrics to evaluate the performance
    of a DTW alignment between two log sequences. It provides various measures including
    normalized DTW distance, correlation coefficients, and path characteristics.
    
    Parameters
    ----------
    log1 : array-like
        The first original log array
    log2 : array-like
        The second original log array
    p : array-like
        The warping path indices for log1
    q : array-like
        The warping path indices for log2
    D : numpy.ndarray
        The accumulated cost matrix from DTW computation
    
    Returns
    -------
    dict
        A dictionary containing quality indicators:
        
        - norm_dtw : float
            Normalized DTW distance (total cost divided by path length)
        - dtw_ratio : float
            Ratio of DTW distance to Euclidean distance
        - variance_deviation : float
            Variance of index differences in warping path (measures path deviation)
        - perc_diag : float
            Percentage indicating diagonality of the warping path (higher = more diagonal)
        - corr_coef : float
            Correlation coefficient between aligned sequences
        - match_min : float
            Minimum value of the matching function
        - match_mean : float
            Mean value of the matching function
    
    Examples
    --------
    >>> import numpy as np
    >>> log1 = [1.0, 2.0, 3.0, 4.0]
    >>> log2 = [1.1, 2.1, 3.1, 4.1]
    >>> p = [0, 1, 2, 3]
    >>> q = [0, 1, 2, 3]
    >>> D = np.array([[0.1, 0.2, 0.3, 0.4],
    ...               [0.2, 0.1, 0.2, 0.3],
    ...               [0.3, 0.2, 0.1, 0.2],
    ...               [0.4, 0.3, 0.2, 0.1]])
    >>> metrics = compute_quality_indicators(log1, log2, p, q, D)
    >>> print(f"Correlation: {metrics['corr_coef']:.3f}")
    Correlation: 1.000
    """

    # Handle edge case: single pair warping path
    if len(p) <= 1:
        norm_dtw = D[-1, -1] if D.size > 0 else 0.0
        aligned_log1 = np.array(log1)[np.array(p)]
        aligned_log2 = np.array(log2)[np.array(q)]
        euclidean_dist = np.linalg.norm(aligned_log1 - aligned_log2) if aligned_log1.size and aligned_log2.size else 0.0
        dtw_ratio = norm_dtw / (euclidean_dist + 1e-10)
        return {
            'norm_dtw': norm_dtw,
            'dtw_ratio': dtw_ratio,
            'variance_deviation': 0.0,
            'perc_diag': 0.0,
            'corr_coef': 0.0,
            'match_min': norm_dtw,
            'match_mean': norm_dtw
        }
    
    # Main quality indicator computation
    try:
        # Normalized DTW distance
        norm_dtw = D[-1, -1] / float(len(p))
        
        # Extract aligned sequences using warping path indices
        aligned_log1 = np.array(log1)[np.array(p)]
        aligned_log2 = np.array(log2)[np.array(q)]
        
        # Calculate Euclidean distance and DTW ratio
        euclidean_dist = np.linalg.norm(aligned_log1 - aligned_log2)
        dtw_ratio = D[-1, -1] / (euclidean_dist + 1e-10)
        
        # Calculate warping path deviation from diagonal
        diff_indices = np.abs(np.array(p) - np.array(q))
        variance_deviation = np.var(diff_indices)
        
        # Calculate diagonality measure
        unique_p = len(np.unique(p))
        unique_q = len(np.unique(q))
        
        if unique_p <= 1 or unique_q <= 1:
            perc_diag = 0.0
        else:
            # Compute actual path length vs ideal diagonal length
            path_length = len(p) - 1
            ideal_length = max(unique_p - 1, unique_q - 1)
            diagonality_ratio = ideal_length / path_length
            perc_diag = diagonality_ratio * 100
        
        # Calculate correlation coefficient between aligned sequences
        if len(aligned_log1) < 2 or len(aligned_log2) < 2:
            corr_coef = 0.0
        else:
            # Check for constant values which would make correlation undefined
            if (np.all(aligned_log1 == aligned_log1[0]) or 
                np.all(aligned_log2 == aligned_log2[0])):
                corr_coef = 0.0
            else:
                try:
                    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(
                        aligned_log1, aligned_log2
                    )
                    corr_coef = r_value
                    if np.isnan(corr_coef):
                        corr_coef = 0.0
                except Exception:
                    corr_coef = 0.0
        
        # Calculate matching function statistics
        if D.shape[0] == len(log1):
            matching_function = D[-1, :] / float(len(log1))
            match_min = np.min(matching_function)
            match_mean = np.mean(matching_function)
        else:
            match_min = norm_dtw
            match_mean = norm_dtw
        
        return {
            'norm_dtw': norm_dtw,
            'dtw_ratio': dtw_ratio,
            'variance_deviation': variance_deviation,
            'perc_diag': perc_diag,
            'corr_coef': corr_coef,
            'match_min': match_min,
            'match_mean': match_mean
        }
    except Exception as e:
        print(f"Warning: Error calculating quality indicators: {e}")
        return {
            'norm_dtw': D[-1, -1] if D.size > 0 else 0.0,
            'dtw_ratio': 1.0,
            'variance_deviation': 0.0,
            'perc_diag': 0.0,
            'corr_coef': 0.0,
            'match_min': D[-1, -1] if D.size > 0 else 0.0,
            'match_mean': D[-1, -1] if D.size > 0 else 0.0
        }


def calculate_age_overlap_percentage(a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound):
    """
    Calculate the percentage of overlap between two age intervals.
    
    This function computes how much two age ranges overlap as a percentage
    of their total combined range (union). This is useful for assessing
    compatibility between age constraints from different cores or methods.
    
    Parameters
    ----------
    a_lower_bound : float
        Lower bound of the first age interval
    a_upper_bound : float
        Upper bound of the first age interval
    b_lower_bound : float
        Lower bound of the second age interval
    b_upper_bound : float
        Upper bound of the second age interval
    
    Returns
    -------
    float
        Percentage of overlap relative to the union of both intervals.
        Returns 0.0 if no overlap exists, 100.0 if both ranges are identical points.
    
    Examples
    --------
    >>> # Example 1: Partial overlap
    >>> overlap_pct = calculate_age_overlap_percentage(100, 200, 150, 250)
    >>> print(f"Overlap: {overlap_pct:.1f}%")
    Overlap: 33.3%
    
    >>> # Example 2: No overlap
    >>> overlap_pct = calculate_age_overlap_percentage(100, 150, 200, 250)
    >>> print(f"Overlap: {overlap_pct:.1f}%")
    Overlap: 0.0%
    
    >>> # Example 3: Complete overlap (one interval inside another)
    >>> overlap_pct = calculate_age_overlap_percentage(100, 200, 120, 180)
    >>> print(f"Overlap: {overlap_pct:.1f}%")
    Overlap: 60.0%
    """
    # Calculate overlap boundaries
    overlap_start = max(a_lower_bound, b_lower_bound)
    overlap_end = min(a_upper_bound, b_upper_bound)
    
    # Check if overlap exists
    if overlap_end <= overlap_start:
        return 0.0
    
    # Calculate overlap and union lengths
    overlap_length = overlap_end - overlap_start
    union_start = min(a_lower_bound, b_lower_bound)
    union_end = max(a_upper_bound, b_upper_bound)
    union_length = union_end - union_start
    
    # Return percentage of overlap relative to union
    if union_length > 0:
        return (overlap_length / union_length) * 100.0
    else:
        return 100.0