"""
Quality metrics computation for DTW analysis.

Included Functions:
- compute_quality_indicators: Compute comprehensive quality metrics for DTW alignment
- calculate_age_overlap_percentage: Calculate overlap percentage between two age intervals
- find_best_mappings: Find the best DTW mappings based on multiple quality metrics

This module provides functions to compute various quality indicators for Dynamic Time Warping (DTW)
alignment results and calculate age overlap percentages between geological intervals.
"""

import numpy as np
import pandas as pd
import csv
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
                        # MULTIDIMENSIONAL CASE - Use PCA approach
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

def find_best_mappings(csv_file_path, 
                       top_n=5, 
                       filter_shortest_dtw=True,
                       metric_weight=None):
    """
    Find the best DTW mappings based on multiple quality metrics with configurable scoring.
    
    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing DTW results
    top_n : int, optional
        Number of top mappings to return (default: 5)
    filter_shortest_dtw : bool, optional
        If True, only consider mappings with shortest DTW path length (default: True)
    metric_weight : dict, optional
        Dictionary defining metric weights for scoring
        Format: {metric_name: weight_value}
        If None, uses default weights
    
    Returns
    -------
    tuple
        - top_mapping_id : list
            List of top mapping IDs in order
        - top_mapping_pairs : list
            List of valid_pairs_to_combine for each top mapping ID
        - top_mapping_df : pandas.DataFrame
            DataFrame containing the top N mappings sorted by combined score
    
    Examples
    --------
    >>> # Using default weights
    >>> top_ids, top_pairs, top_df = find_best_mappings('dtw_results.csv', top_n=3)
    
    >>> # Using custom weights
    >>> custom_weights = {
    ...     'corr_coef': 5.0,
    ...     'perc_diag': 2.0,
    ...     'norm_dtw': 1.0
    ... }
    >>> top_ids, top_pairs, top_df = find_best_mappings('dtw_results.csv', 
    ...                                       metric_weight=custom_weights)
    """
    
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    # Default metrics configuration with fixed higher_is_better values
    default_weights = {
        'perc_diag': 0.5,
        'norm_dtw': 1.0,
        'dtw_ratio': 0.0,
        'corr_coef': 2.5,
        'dtw_warp_eff': 1.0,
        'perc_age_overlap': 1.0
    }
    
    # Fixed higher_is_better configuration (cannot be changed)
    higher_is_better_config = {
        'perc_diag': True,
        'norm_dtw': False,
        'dtw_ratio': False,
        'corr_coef': True,
        'dtw_warp_eff': True,
        'perc_age_overlap': True,
    }
    
    # Use provided weights or default weights
    weights = metric_weight if metric_weight is not None else default_weights
    
    # Load and clean the data
    dtw_results_df = pd.read_csv(csv_file_path)
    required_cols = ['corr_coef', 'perc_diag', 'perc_age_overlap']
    dtw_results_df = dtw_results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)
    
    print(f"=== Top {top_n} Overall Best Mappings ===")
    
    # Filter for shortest DTW path length if requested
    if filter_shortest_dtw and 'length' in dtw_results_df.columns:
        dtw_results_df['dtw_path_length'] = dtw_results_df['length']
        min_length = dtw_results_df['dtw_path_length'].min()
        shortest_mappings = dtw_results_df[dtw_results_df['dtw_path_length'] == min_length]
        
        print(f"Filtering for shortest DTW path length: {min_length}")
        print(f"Number of mappings considered: {len(shortest_mappings)} out of {len(dtw_results_df)}")
    else:
        shortest_mappings = dtw_results_df
        if filter_shortest_dtw:
            print("Warning: No 'length' column found. Using all mappings.")
    
    # Create a copy for scoring calculations
    df_for_ranking = shortest_mappings.copy()
    
    # Initialize the combined score column
    df_for_ranking['combined_score'] = 0.0
    total_weight = 0.0
    
    # Calculate scores for each metric and add to combined score
    for metric, weight in weights.items():
        if metric in df_for_ranking.columns:
            # Make sure we have valid data to work with
            valid_data = df_for_ranking[~df_for_ranking[metric].isna()]
            
            if len(valid_data) > 0:
                min_val = valid_data[metric].min()
                max_val = valid_data[metric].max()
                
                # Only normalize if there's a range
                if max_val > min_val:
                    if higher_is_better_config.get(metric, True):
                        # For metrics where higher values are better
                        df_for_ranking[f'{metric}_score'] = (df_for_ranking[metric] - min_val) / (max_val - min_val)
                    else:
                        # For metrics where lower values are better
                        df_for_ranking[f'{metric}_score'] = 1 - ((df_for_ranking[metric] - min_val) / (max_val - min_val))
                else:
                    # If all values are the same, assign a score of 1
                    df_for_ranking[f'{metric}_score'] = 1.0
                    
                # Add to weighted sum
                df_for_ranking['combined_score'] += df_for_ranking[f'{metric}_score'].fillna(0) * weight
                total_weight += weight
    
    # Normalize the combined score by the total weight
    if total_weight > 0:
        df_for_ranking['combined_score'] = df_for_ranking['combined_score'] / total_weight
    else:
        print("Warning: No valid metrics found for scoring.")
        df_for_ranking['combined_score'] = 0.0
    
    # Handle NaN values in combined score
    if df_for_ranking['combined_score'].isna().any():
        print("Warning: NaN values detected in combined scores. Replacing with zeros.")
        df_for_ranking['combined_score'] = df_for_ranking['combined_score'].fillna(0)
    
    # Get top N mappings by combined score
    top_mappings = df_for_ranking.sort_values(by='combined_score', ascending=False).head(top_n)
    
    # Parse paths and create valid_pairs_to_combine for each mapping
    top_mapping_ids = []
    top_mapping_pairs = []
    
    # Print detailed results
    print(f"\nTop {top_n} mappings by combined score:")
    for idx, row in top_mappings.iterrows():
        if 'mapping_id' in row:
            mapping_id = int(row['mapping_id'])
            top_mapping_ids.append(mapping_id)
            print(f"Mapping ID {mapping_id}: Combined Score={row['combined_score']:.3f}")
            
            # Parse the path and convert to valid_pairs_to_combine
            if 'path' in row:
                target_data_row = parse_compact_path(row['path'])
                # Convert 1-based to 0-based indices for python
                valid_pairs_to_combine = [(a-1, b-1) for a, b in target_data_row]
                top_mapping_pairs.append(valid_pairs_to_combine)
            else:
                # If no path column, append empty list
                top_mapping_pairs.append([])
        
        # Print all available metrics
        metric_outputs = []
        if 'dtw_path_length' in row:
            metric_outputs.append(f"dtw_path_length={row['dtw_path_length']:.1f}")
        if 'corr_coef' in row:
            metric_outputs.append(f"correlation coefficient r={row['corr_coef']:.3f}")
        if 'perc_diag' in row:
            metric_outputs.append(f"perc_diag={row['perc_diag']:.1f}%")
        if 'norm_dtw' in row:
            metric_outputs.append(f"norm_dtw={row['norm_dtw']:.3f}")
        if 'dtw_ratio' in row:
            metric_outputs.append(f"dtw_ratio={row['dtw_ratio']:.3f}")
        if 'dtw_warp_eff' in row:
            metric_outputs.append(f"dtw_warp_eff={row['dtw_warp_eff']:.1f}%")
        if 'perc_age_overlap' in row:
            metric_outputs.append(f"perc_age_overlap={row['perc_age_overlap']:.1f}%")
        
        # Print metrics with proper indentation
        for metric_output in metric_outputs:
            print(f"  {metric_output}")
        
        # Print valid_pairs_to_combine at the end of each mapping
        if len(top_mapping_pairs) > 0 and len(top_mapping_pairs[-1]) > 0:
            print(f"  valid_pairs_to_combine={top_mapping_pairs[-1]}")
        else:
            print(f"  valid_pairs_to_combine=[]")
        print("")
    
    # Handle case where no valid mappings found
    if not top_mappings.empty:
        print(f"Best mapping ID: {top_mapping_ids[0] if top_mapping_ids else 'None'}")
    else:
        print("Warning: No valid mappings found")
    
    return top_mapping_ids, top_mapping_pairs, top_mappings