"""
Quality metrics computation for DTW analysis.

Included Functions:
- compute_quality_indicators: Compute comprehensive quality metrics for DTW alignment
- calculate_age_overlap_percentage: Calculate overlap percentage between two age intervals
- find_best_mappings: Find the best DTW mappings based on multiple quality metrics
  (supports both standard best mappings and boundary correlation filtering modes)

This module provides functions to compute various quality indicators for Dynamic Time Warping (DTW)
alignment results and calculate age overlap percentages between geological intervals.
Compatible with both original and ML-processed core data.
"""

import numpy as np
import pandas as pd
import csv
from scipy import stats

def compute_quality_indicators(log1, log2, p, q, D, pca_for_dependent_dtw=False):
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
    pca_for_dependent_dtw : bool, default=True
        Whether to use PCA for dependent multidimensional DTW correlation calculation
    
    Returns
    -------
    dict
        A dictionary containing quality indicators:
        
        - norm_dtw : float
            Normalized DTW distance (total cost divided by path length)
        - dtw_ratio : float
            Ratio of DTW distance to Euclidean distance of linear alignment
        - perc_diag : float
            Geometric diagonality percentage (45-degree straightness, higher = more diagonal)
        - dtw_warp_eff : float
            Warping efficiency percentage (path efficiency vs theoretical minimum)
        - corr_coef : float
            Correlation coefficient between aligned sequences
    
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
            'perc_diag': 0.0,  
            'dtw_warp_eff': 0.0, 
            'corr_coef': 0.0
        }
    
    # Main quality indicator computation
    try:
        # Normalized DTW distance
        norm_dtw = D[-1, -1] / (len(log1) + len(log2))
        
        # Extract aligned sequences using warping path indices
        aligned_log1 = np.array(log1)[np.array(p)]
        aligned_log2 = np.array(log2)[np.array(q)]
        
        # Calculate DTW ratio using linear alignment as baseline
        # Create a linear (diagonal) alignment between the original sequences
        len1, len2 = len(log1), len(log2)
        
        if len1 == 1 and len2 == 1:
            # Special case: both sequences have single points
            linear_euclidean_dist = abs(log1[0] - log2[0])
        elif len1 == 1:
            # log1 has single point, map to all points in log2
            linear_euclidean_dist = np.mean([abs(log1[0] - log2[i]) for i in range(len2)])
        elif len2 == 1:
            # log2 has single point, map to all points in log1
            linear_euclidean_dist = np.mean([abs(log1[i] - log2[0]) for i in range(len1)])
        else:
            # Both sequences have multiple points - create linear alignment
            # Generate linear indices that map proportionally from one sequence to another
            linear_p = np.linspace(0, len1-1, max(len1, len2)).astype(int)
            linear_q = np.linspace(0, len2-1, max(len1, len2)).astype(int)
            
            # Clip indices to ensure they're within bounds
            linear_p = np.clip(linear_p, 0, len1-1)
            linear_q = np.clip(linear_q, 0, len2-1)
            
            # Extract linearly aligned sequences
            linear_aligned_log1 = np.array(log1)[linear_p]
            linear_aligned_log2 = np.array(log2)[linear_q]
            
            # Calculate Euclidean distance for linear alignment
            linear_euclidean_dist = np.linalg.norm(linear_aligned_log1 - linear_aligned_log2)
        
        # Calculate DTW ratio: DTW distance vs linear alignment distance
        dtw_ratio = D[-1, -1] / (linear_euclidean_dist + 1e-10)
        
        # Calculate geometric diagonality (45-degree straightness)
        if len1 > 1 and len2 > 1:
            # Normalize path positions to 0-1 range
            a_positions = np.array(p) / (len1 - 1)
            b_positions = np.array(q) / (len2 - 1)
            
            # Calculate deviations from perfect diagonal
            diagonal_deviations = np.abs(a_positions - b_positions)
            avg_deviation = np.mean(diagonal_deviations)
            perc_diag = (1 - avg_deviation) * 100
        else:
            # Single point cases are perfectly diagonal
            perc_diag = 0.0
            dtw_warp_eff = 0.0
        
        # Calculate warping efficiency (path efficiency vs theoretical minimum)
        theoretical_min_path = max(len1, len2) - 1
        actual_path = len(p) - 1
        dtw_warp_eff = (theoretical_min_path / actual_path) * 100 if actual_path > 0 else 100.0
        
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
                    # Handle both multidimensional and single dimension cases
                    if aligned_log1.ndim > 1 and aligned_log2.ndim > 1:
                        # MULTIDIMENSIONAL CASE
                        # Check if both logs have the same number of dimensions
                        if aligned_log1.shape[1] != aligned_log2.shape[1]:
                            corr_coef = 0.0
                        else:
                            # Trim to same length if necessary
                            min_length = min(len(aligned_log1), len(aligned_log2))
                            if len(aligned_log1) != len(aligned_log2):
                                aligned_log1 = aligned_log1[:min_length]
                                aligned_log2 = aligned_log2[:min_length]
                            
                            # Check minimum length requirement
                            if min_length < 2:
                                corr_coef = 0.0
                            else:
                                if pca_for_dependent_dtw:
                                    # Use PCA to find the main trend direction
                                    try:
                                        # Combine both sequences for consistent PCA transformation
                                        combined_data = np.vstack([aligned_log1, aligned_log2])
                                        
                                        # Check if combined data has enough variation for PCA
                                        if np.var(combined_data, axis=0).sum() < 1e-10:
                                            corr_coef = 0.0
                                        else:
                                            # Simple PCA implementation (avoiding sklearn dependency)
                                            # Center the data
                                            mean_combined = np.mean(combined_data, axis=0)
                                            centered_combined = combined_data - mean_combined
                                            
                                            # Calculate covariance matrix
                                            cov_matrix = np.cov(centered_combined.T)
                                            
                                            # Get first principal component (eigenvector with largest eigenvalue)
                                            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                            pc1_direction = eigenvectors[:, -1]  # Last column = largest eigenvalue
                                            
                                            # Project aligned sequences onto first principal component
                                            centered_log1 = aligned_log1 - mean_combined
                                            centered_log2 = aligned_log2 - mean_combined
                                            
                                            pc1_log1 = np.dot(centered_log1, pc1_direction)
                                            pc1_log2 = np.dot(centered_log2, pc1_direction)
                                            
                                            # Check for constant values in PC1 projections
                                            if (np.all(pc1_log1 == pc1_log1[0]) or 
                                                np.all(pc1_log2 == pc1_log2[0])):
                                                corr_coef = 0.0
                                            else:
                                                # Calculate Pearson correlation on PC1 scores
                                                slope, intercept, r_value, p_value, slope_std_error = stats.linregress(
                                                    pc1_log1, pc1_log2)
                                                corr_coef = r_value
                                    except Exception:
                                        corr_coef = 0.0
                                else:
                                    # CONVENTIONAL MULTIDIMENSIONAL CORRELATION CALCULATION
                                    # Average correlations across dimensions
                                    try:
                                        dim_correlations = []
                                        for dim in range(aligned_log1.shape[1]):
                                            dim_log1 = aligned_log1[:, dim]
                                            dim_log2 = aligned_log2[:, dim]
                                            
                                            # Check for constant values in this dimension
                                            if (np.all(dim_log1 == dim_log1[0]) or 
                                                np.all(dim_log2 == dim_log2[0])):
                                                continue  # Skip this dimension
                                            
                                            # Calculate correlation for this dimension
                                            slope, intercept, r_value, p_value, slope_std_error = stats.linregress(
                                                dim_log1, dim_log2)
                                            dim_correlations.append(r_value)
                                        
                                        if len(dim_correlations) > 0:
                                            # Average correlations across valid dimensions
                                            corr_coef = np.mean(dim_correlations)
                                        else:
                                            corr_coef = 0.0
                                    except Exception:
                                        corr_coef = 0.0
                    else:
                        # SINGLE DIMENSIONAL CASE
                        # Flatten if necessary
                        if aligned_log1.ndim > 1:
                            aligned_log1 = aligned_log1.flatten()
                        if aligned_log2.ndim > 1:
                            aligned_log2 = aligned_log2.flatten()
                        
                        # Calculate Pearson correlation coefficient
                        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(
                            aligned_log1, aligned_log2)
                        corr_coef = r_value
                except Exception:
                    corr_coef = 0.0
        
        return {
            'norm_dtw': norm_dtw,
            'dtw_ratio': dtw_ratio,
            'perc_diag': perc_diag,
            'dtw_warp_eff': dtw_warp_eff,
            'corr_coef': corr_coef
        }
        
    except Exception as e:
        # Fallback values if computation fails
        return {
            'norm_dtw': 0.0,
            'dtw_ratio': 0.0,
            'perc_diag': 0.0,
            'dtw_warp_eff': 0.0,
            'corr_coef': 0.0
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
