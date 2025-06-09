"""
Quality metrics computation for DTW analysis
"""

import numpy as np
from scipy import stats

def compute_quality_indicators(log1, log2, p, q, D):
    """
    Compute quality indicators for the DTW alignment.
    
    Parameters:
    - log1, log2: The original log arrays.
    - p, q: The warping path indices for log1 and log2 respectively.
    - D: The accumulated cost matrix.
    
    Returns:
    --------
    dict: A dictionary of quality indicators including normalized DTW distance, DTW ratio,
          variance of index differences, percentage of diagonal moves, and correlation coefficient.
    """

    # If the warping path consists of a single pair, compute quality indices on that one pair.
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
    
    # For normal cases, calculate quality indicators
    try:
        # Normalized DTW distance
        norm_dtw = D[-1, -1] / float(len(p))
        
        # Extract aligned sequences - NO NaN filtering for repeated indices
        aligned_log1 = np.array(log1)[np.array(p)]
        aligned_log2 = np.array(log2)[np.array(q)]
        
        # Calculate Euclidean distance and DTW ratio
        euclidean_dist = np.linalg.norm(aligned_log1 - aligned_log2)
        dtw_ratio = D[-1, -1] / (euclidean_dist + 1e-10)
        
        # Calculate warping path deviation
        diff_indices = np.abs(np.array(p) - np.array(q))
        variance_deviation = np.var(diff_indices)
        
        # Calculate diagonality
        unique_p = len(np.unique(p))
        unique_q = len(np.unique(q))
        
        # Check for trivial cases
        if unique_p <= 1 or unique_q <= 1:
            perc_diag = 0.0
        else:
            # Compute the path length in index space
            path_length = len(p) - 1
            
            # Calculate the ideal diagonal length (Euclidean distance in index space)
            ideal_length = max(unique_p - 1, unique_q - 1)
            
            # A perfectly diagonal path would be the shortest possible path
            # The ratio between ideal and actual path length is our diagonality measure
            diagonality_ratio = ideal_length / path_length
            
            # Convert to percentage
            perc_diag = diagonality_ratio * 100
        
        # Calculate correlation coefficient - REVISED: No NaN filtering
        # Use all aligned values including repeated indices
        if len(aligned_log1) < 2 or len(aligned_log2) < 2:
            corr_coef = 0.0
        else:
            # Check for constant values
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
        
        # Calculate matching function
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
    # Calculate overlap
    overlap_start = max(a_lower_bound, b_lower_bound)
    overlap_end = min(a_upper_bound, b_upper_bound)
    
    # If no overlap, return 0%
    if overlap_end <= overlap_start:
        return 0.0
    
    # Calculate overlap length
    overlap_length = overlap_end - overlap_start
    
    # Calculate total combined range (union)
    union_start = min(a_lower_bound, b_lower_bound)
    union_end = max(a_upper_bound, b_upper_bound)
    union_length = union_end - union_start
    
    # Return percentage of overlap relative to union
    if union_length > 0:
        return (overlap_length / union_length) * 100.0
    else:
        return 100.0  # If both ranges are identical single points