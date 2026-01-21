"""
Quality metrics computation and utility functions for DTW analysis.

Included Functions:
- compute_quality_indicators: Compute comprehensive quality metrics for DTW alignment
- calculate_age_overlap_percentage: Calculate overlap percentage between two age intervals
- find_best_mappings: Find the best DTW mappings based on multiple quality metrics
  (supports both standard best mappings and boundary correlation filtering modes)
- find_nearest_index: Find the index in depth_array closest to a given depth value
- cohens_d: Calculate Cohen's d effect size between two groups

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


def find_nearest_index(depth_array, depth_value):
    """
    Find the index in depth_array that has the closest depth value to the given depth_value.
    
    This function is commonly used when converting between depth values and array indices
    in core log data analysis, particularly for finding segment boundaries and correlation points.
    
    Parameters
    ----------
    depth_array : array-like
        Array of depth values, typically measured depths from core logs
    depth_value : float
        Target depth value to find the nearest match for
        
    Returns
    -------
    int
        Index in depth_array with the closest value to depth_value
        
    Example
    -------
    >>> import numpy as np
    >>> depths = np.array([10.5, 11.2, 12.1, 13.0, 14.5])
    >>> target_depth = 12.5
    >>> idx = find_nearest_index(depths, target_depth)
    >>> print(f"Nearest index: {idx}, depth: {depths[idx]}")
    Nearest index: 2, depth: 12.1
    """
    return np.abs(np.array(depth_array) - depth_value).argmin()


def cohens_d(x, y):
    """
    Calculate Cohen's d for effect size between two samples.
    
    Cohen's d is a measure of effect size that indicates the standardized difference
    between two means. It is commonly used in statistical analysis to quantify the
    magnitude of differences between groups.
    
    Parameters
    ----------
    x : array-like
        First sample group
    y : array-like
        Second sample group
        
    Returns
    -------
    float
        Cohen's d effect size value
        
    Example
    -------
    >>> import numpy as np
    >>> group1 = np.array([1, 2, 3, 4, 5])
    >>> group2 = np.array([2, 3, 4, 5, 6])
    >>> d = cohens_d(group1, group2)
    """
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    # Cohen's d
    d = (np.mean(x) - np.mean(y)) / s_pooled
    return d


def find_best_mappings(input_mapping_csv, 
                       top_n=10, 
                       filter_shortest_dtw=True,
                       metric_weight=None,
                       core_a_picked_datums=None,
                       core_b_picked_datums=None,
                       core_a_interpreted_beds=None,
                       core_b_interpreted_beds=None,
                       dtw_result=None):
    """
    Find the best DTW mappings based on multiple quality metrics with configurable scoring.
    
    If boundary correlation parameters are provided, filters mappings that comply with 
    boundary correlations between cores. If those parameters are not provided or no 
    matching boundaries are found, behaves as standard best mappings finder.
    
    Parameters:
    -----------
    input_mapping_csv : str
        Path to CSV file containing DTW mapping results
    top_n : int, default=10
        Number of top mappings to return
    filter_shortest_dtw : bool, default=True
        Whether to filter for shortest DTW path length first
    metric_weight : dict, optional
        Dictionary of weights for different quality metrics
    core_a_picked_datums : list, optional
        Picked depth values for core A (for boundary correlation mode)
    core_b_picked_datums : list, optional
        Picked depth values for core B (for boundary correlation mode)
    core_a_interpreted_beds : list, optional
        Interpreted bed names for core A boundaries (for boundary correlation mode)
    core_b_interpreted_beds : list, optional
        Interpreted bed names for core B boundaries (for boundary correlation mode)
    dtw_result : dict, optional
        Dictionary containing DTW analysis results from run_comprehensive_dtw_analysis().
        Expected keys: 'valid_dtw_pairs', 'segments_a', 'segments_b'.
        Required only for boundary correlation mode.
    """
    
    # Extract variables from unified dictionary if provided (for boundary correlation mode)
    if dtw_result is not None:
        valid_dtw_pairs = dtw_result['valid_dtw_pairs']
        segments_a = dtw_result['segments_a']
        segments_b = dtw_result['segments_b']
    else:
        valid_dtw_pairs = None
        segments_a = None
        segments_b = None
    
    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]
    
    def clean_name(name):
        """Helper function to clean bed names (remove "?")"""
        if pd.isna(name) or name == '':
            return ''
        return name.replace('?', '').strip()
    
    def _calculate_combined_scores(df_input, weights, higher_is_better_config):
        """Calculate combined scores for each mapping"""
        df_for_ranking = df_input.copy()
        
        # Initialize combined score
        df_for_ranking['combined_score'] = 0.0
        
        # Calculate weighted combined score
        total_weight = 0
        for metric, weight in weights.items():
            if weight != 0 and metric in df_for_ranking.columns:
                # Normalize metric values to 0-1 range
                metric_values = df_for_ranking[metric].copy()
                metric_min = metric_values.min()
                metric_max = metric_values.max()
                
                if metric_max != metric_min:
                    normalized_values = (metric_values - metric_min) / (metric_max - metric_min)
                    
                    # Flip values if lower is better
                    if not higher_is_better_config.get(metric, True):
                        normalized_values = 1 - normalized_values
                    
                    df_for_ranking['combined_score'] += weight * normalized_values
                    total_weight += weight
        
        # Normalize by total weight if weights were applied
        if total_weight > 0:
            df_for_ranking['combined_score'] /= total_weight
        
        # Handle NaN values in combined score
        if df_for_ranking['combined_score'].isna().any():
            print("Warning: NaN values detected in combined scores. Replacing with zeros.")
            df_for_ranking['combined_score'] = df_for_ranking['combined_score'].fillna(0)
        
        return df_for_ranking
    
    def _print_results(top_mappings, weights, mode_title):
        """Print detailed results for mappings"""
        top_mapping_ids = []
        top_mapping_pairs = []
        
        print(f"\nTop {len(top_mappings)} mappings by combined score:")
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
            if 'dtw_path_length' in row and weights.get('dtw_path_length', 0) != 0:
                metric_outputs.append(f"dtw_path_length={row['dtw_path_length']:.1f}")
            if 'length' in row and weights.get('length', 0) != 0:
                metric_outputs.append(f"dtw_path_length={row['length']:.1f}")
            # Print corr_coef and corr_coef_sect on same line
            if 'corr_coef' in row and weights.get('corr_coef', 0) != 0:
                corr_line = f"r={row['corr_coef']:.3f}"
                if 'corr_coef_sect' in row and weights.get('corr_coef_sect', 0) != 0:
                    corr_line += f"; r_sect={row['corr_coef_sect']:.3f}"
                metric_outputs.append(corr_line)
            if 'perc_diag' in row and weights.get('perc_diag', 0) != 0:
                metric_outputs.append(f"perc_diag={row['perc_diag']:.1f}%")
            # Print norm_dtw and norm_dtw_sect on same line
            if 'norm_dtw' in row and weights.get('norm_dtw', 0) != 0:
                dtw_line = f"norm_dtw={row['norm_dtw']:.3f}"
                if 'norm_dtw_sect' in row and weights.get('norm_dtw_sect', 0) != 0:
                    dtw_line += f"; norm_dtw_sect={row['norm_dtw_sect']:.3f}"
                metric_outputs.append(dtw_line)
            if 'dtw_ratio' in row and weights.get('dtw_ratio', 0) != 0:
                metric_outputs.append(f"dtw_ratio={row['dtw_ratio']:.3f}")
            if 'dtw_warp_eff' in row and weights.get('dtw_warp_eff', 0) != 0:
                metric_outputs.append(f"dtw_warp_eff={row['dtw_warp_eff']:.1f}%")
            if 'perc_age_overlap' in row and weights.get('perc_age_overlap', 0) != 0:
                metric_outputs.append(f"perc_age_overlap={row['perc_age_overlap']:.1f}%")
            
            # Print metrics with proper indentation
            for metric_output in metric_outputs:
                print(f"  {metric_output}")
            
            # Print valid_pairs_to_combine with display numbering (export values + 1)
            if len(top_mapping_pairs) > 0 and len(top_mapping_pairs[-1]) > 0:
                # Convert export values to display format (add 1 more)
                pairs_display = [(a+1, b+1) for a, b in top_mapping_pairs[-1]]
                print(f"  valid_pairs_to_combine={pairs_display}")
            else:
                print(f"  valid_pairs_to_combine=[]")
            
            # Print matched datums if available
            if 'matched_datums' in row and row['matched_datums']:
                datums_list = row['matched_datums'].split(',')
                print(f"  matched_datums={datums_list}")
            
            print("")
        
        # Handle case where no valid mappings found
        if not top_mappings.empty:
            best_mapping_id = top_mapping_ids[0] if top_mapping_ids else 'None'
            print(f"Best mapping ID: {best_mapping_id}")
        else:
            print("Warning: No valid mappings found")
        
        return top_mapping_ids, top_mapping_pairs
    
    # Default metrics configuration with fixed higher_is_better values
    # Uses norm_dtw, corr_coef, norm_dtw_sect, and corr_coef_sect evenly weighted
    default_weights = {
        'perc_diag': 0.0,
        'norm_dtw': 1.0,
        'dtw_ratio': 0.0,
        'corr_coef': 1.0,
        'dtw_warp_eff': 0.0,
        'perc_age_overlap': 0.0,
        'norm_dtw_sect': 1.0,
        'corr_coef_sect': 1.0
    }
    
    # Fixed higher_is_better configuration (cannot be changed)
    higher_is_better_config = {
        'perc_diag': True,
        'norm_dtw': False,
        'dtw_ratio': False,
        'corr_coef': True,
        'dtw_warp_eff': True,
        'perc_age_overlap': True,
        'norm_dtw_sect': False,
        'corr_coef_sect': True,
    }
    
    # Use provided weights or default weights
    weights = metric_weight if metric_weight is not None else default_weights
    
    # Load and clean the data
    if isinstance(input_mapping_csv, str):
        dtw_results_df = pd.read_csv(input_mapping_csv)
    else:
        dtw_results_df = input_mapping_csv
    
    # Always reset ranking columns after loading CSV (overwrite any existing values)
    dtw_results_df['Ranking'] = ''
    dtw_results_df['Ranking_datums'] = ''
    dtw_results_df['matched_datums'] = ''
    dtw_results_df['combined_score'] = ''
   
    # Check if we should use boundary correlation mode
    use_boundary_mode = all(param is not None for param in [
        core_a_picked_datums, core_b_picked_datums, core_a_interpreted_beds, 
        core_b_interpreted_beds, valid_dtw_pairs, segments_a, segments_b
    ])
    
    target_mappings_df = None
    matching_boundary_names = set()
    matching_details = []
    
    if use_boundary_mode:
        # Check if all interpreted bed names are empty
        all_beds_empty = (
            all(pd.isna(name) or clean_name(name) == '' for name in core_a_interpreted_beds) and
            all(pd.isna(name) or clean_name(name) == '' for name in core_b_interpreted_beds)
        )
        
        if not all_beds_empty:
            # Step 1: Find all bed names that appear in both core_a_interpreted_beds and core_b_interpreted_beds
            bed_names_a = set()
            bed_names_b = set()
            
            for name in core_a_interpreted_beds:
                cleaned = clean_name(name)
                if cleaned:
                    bed_names_a.add(cleaned)
            
            for name in core_b_interpreted_beds:
                cleaned = clean_name(name)
                if cleaned:
                    bed_names_b.add(cleaned)
            
            # Find common bed names
            common_bed_names = bed_names_a.intersection(bed_names_b)
            
            # Step 2: The common bed names ARE our matching datums
            matching_boundary_names = common_bed_names.copy()
            
            # Find segment pairs that contain any of these common bed names
            # (for finding correlations, but the matching datums are already identified)
            matching_pairs = []
            
            if common_bed_names:
                for pair in valid_dtw_pairs:
                    seg_a_idx, seg_b_idx = pair
                    
                    # Get the actual segment tuples
                    seg_a = segments_a[seg_a_idx]
                    seg_b = segments_b[seg_b_idx]
                    
                    # Only check consecutive segments (i, i+1) - these represent intervals between boundaries
                    if seg_a[1] == seg_a[0] + 1 and seg_b[1] == seg_b[0] + 1:
                        start_idx_a, end_idx_a = seg_a
                        start_idx_b, end_idx_b = seg_b
                        
                        # Check if boundaries exist in core_a_interpreted_beds arrays
                        if (start_idx_a < len(core_a_interpreted_beds) and end_idx_a < len(core_a_interpreted_beds) and
                            start_idx_b < len(core_b_interpreted_beds) and end_idx_b < len(core_b_interpreted_beds)):
                            
                            # Get boundary names for this segment
                            start_name_a = clean_name(core_a_interpreted_beds[start_idx_a])
                            end_name_a = clean_name(core_a_interpreted_beds[end_idx_a])
                            start_name_b = clean_name(core_b_interpreted_beds[start_idx_b])
                            end_name_b = clean_name(core_b_interpreted_beds[end_idx_b])
                            
                            # Check for matching boundaries (top-to-top or bottom-to-bottom)
                            top_match = False
                            bottom_match = False
                            matched_names = []
                            
                            # Check if top boundaries match
                            if start_name_a and start_name_b and start_name_a == start_name_b and start_name_a in common_bed_names:
                                top_match = True
                                matched_names.append(f"top:{start_name_a}")
                            
                            # Check if bottom boundaries match
                            if end_name_a and end_name_b and end_name_a == end_name_b and end_name_a in common_bed_names:
                                bottom_match = True
                                matched_names.append(f"bottom:{end_name_a}")
                            
                            # Only include if at least one boundary matches
                            if top_match or bottom_match:
                                matching_pairs.append(pair)
                                
                                matching_details.append({
                                    'sort_key': start_idx_a,
                                    'description': f"Segment pair ({seg_a_idx+3},{seg_b_idx+3}): [{','.join(matched_names)}]"
                                })
                
                # Sort matching details from top to bottom (by boundary index)
                matching_details.sort(key=lambda x: x['sort_key'])

            # Convert matching_pairs to set for faster lookup
            if matching_pairs:
                valid_pairs_set = set(matching_pairs)

        # Print matching datums and segment pairs
        print(f"Matching datums: {sorted(matching_boundary_names)}")
        if matching_boundary_names:
            print(f"Datum matching correlations:")
            for detail in matching_details:
                print(f"  {detail['description']}")

    # Step 2: Find DTW results that cover all matching datums
    if matching_pairs:
        target_mappings = []
        
        # Convert boundary pairs to 1-based format for comparison with CSV path values
        boundary_pairs_1based = set((seg_a_idx + 3, seg_b_idx + 3) for seg_a_idx, seg_b_idx in valid_pairs_set)
        
        # Extract all datum names from matching_details to check coverage
        import re
        all_datums_in_pairs = set()
        for detail in matching_details:
            description = detail['description']
            bed_names = re.findall(r'(?:top|bottom):(\w+)', description)
            all_datums_in_pairs.update(bed_names)
        
        for _, mapping_row in dtw_results_df.iterrows():
            path_str = mapping_row.get('path')
            
            if pd.isna(path_str) or path_str == '':
                continue
            
            # Parse path: "2,2;4,4;6,6" -> {(2,2), (4,4), (6,6)}
            try:
                path_pairs = set()
                for pair_str in path_str.split(';'):
                    a, b = pair_str.split(',')
                    path_pairs.add((int(a), int(b)))
            except:
                continue  # Skip if parsing fails
            
            # Check which segment pairs from our list are in this path
            matched_pairs = boundary_pairs_1based.intersection(path_pairs)
            
            # Extract datums covered by the matched segment pairs
            covered_datums = set()
            for i, detail in enumerate(matching_details):
                seg_pair_display = detail['description'].split(':')[0].strip().replace('Segment pair ', '')
                # Check if this segment pair is in the matched set
                if matching_pairs[i] in valid_pairs_set:
                    seg_a_idx, seg_b_idx = matching_pairs[i]
                    if (seg_a_idx + 3, seg_b_idx + 3) in matched_pairs:
                        # Extract datums from this matched pair
                        bed_names = re.findall(r'(?:top|bottom):(\w+)', detail['description'])
                        covered_datums.update(bed_names)
            
            # Check if ALL datums are covered
            if all_datums_in_pairs.issubset(covered_datums):
                # Store the matched datums with this mapping
                mapping_dict = mapping_row.to_dict()
                mapping_dict['matched_datums'] = ','.join(sorted(covered_datums))
                target_mappings.append(mapping_dict)
        
        # Convert to DataFrame if mappings found
        if target_mappings:
            target_mappings_df = pd.DataFrame(target_mappings).reset_index(drop=True)
    
    # Determine which dataframe to use for ranking
    if target_mappings_df is not None and not target_mappings_df.empty:
        working_df = target_mappings_df
        mode_title = "Target Mappings (Boundary Correlation)"
        print(f"\n{len(target_mappings_df)}/{len(dtw_results_df)} mappings found with all {len(matching_boundary_names)} matched datums ({len(matching_pairs)} segment pairs)")
    elif use_boundary_mode and matching_pairs:
        # No mappings contain ALL segment pairs, but we have segment pairs
        # Find mappings that contain the MOST segment pairs from our list
        print("No mappings found with matched datums. Searching for mappings with most matched segment pairs.")
        
        # Convert boundary pairs to 1-based format for comparison with CSV path values
        boundary_pairs_1based = set((seg_a_idx + 3, seg_b_idx + 3) for seg_a_idx, seg_b_idx in valid_pairs_set)
        
        # Extract all datum names for counting
        import re
        all_datums_in_pairs = set()
        for detail in matching_details:
            description = detail['description']
            bed_names = re.findall(r'(?:top|bottom):(\w+)', description)
            all_datums_in_pairs.update(bed_names)
        
        # Count how many matching segment pairs and datums each mapping contains
        mapping_scores = []
        for _, mapping_row in dtw_results_df.iterrows():
            path_str = mapping_row.get('path')
            
            if pd.isna(path_str) or path_str == '':
                continue
            
            # Parse path: "2,2;4,4;6,6" -> {(2,2), (4,4), (6,6)}
            try:
                path_pairs = set()
                for pair_str in path_str.split(';'):
                    a, b = pair_str.split(',')
                    path_pairs.add((int(a), int(b)))
            except:
                continue  # Skip if parsing fails
            
            # Count how many of our target segment pairs are in this mapping
            matched_pairs = boundary_pairs_1based.intersection(path_pairs)
            matching_count = len(matched_pairs)
            
            # Count how many datums are covered
            covered_datums = set()
            for i, detail in enumerate(matching_details):
                if matching_pairs[i] in valid_pairs_set:
                    seg_a_idx, seg_b_idx = matching_pairs[i]
                    if (seg_a_idx + 3, seg_b_idx + 3) in matched_pairs:
                        bed_names = re.findall(r'(?:top|bottom):(\w+)', detail['description'])
                        covered_datums.update(bed_names)
            datum_count = len(covered_datums)
            
            if matching_count > 0:
                # Add the mapping with its matching count
                mapping_dict = mapping_row.to_dict()
                mapping_dict['matching_segment_count'] = matching_count
                mapping_dict['matching_datum_count'] = datum_count
                mapping_dict['matched_datums'] = ','.join(sorted(covered_datums))
                mapping_scores.append(mapping_dict)
        
        if mapping_scores:
            # Convert to DataFrame and find mappings with the highest datum count first, then segment pairs
            scored_df = pd.DataFrame(mapping_scores)
            max_datum_count = scored_df['matching_datum_count'].max()
            best_by_datum = scored_df[scored_df['matching_datum_count'] == max_datum_count]
            max_segment_count = best_by_datum['matching_segment_count'].max()
            best_partial_mappings = best_by_datum[best_by_datum['matching_segment_count'] == max_segment_count]
            
            working_df = best_partial_mappings
            mode_title = f"Best Partial Mappings ({max_datum_count}/{len(all_datums_in_pairs)} datums, {max_segment_count}/{len(boundary_pairs_1based)} segment pairs)"
            print(f"\n{len(best_partial_mappings)}/{len(dtw_results_df)} mappings found with {max_datum_count}/{len(all_datums_in_pairs)} matched datums ({max_segment_count}/{len(boundary_pairs_1based)} segment pairs)")
        else:
            # No mappings contain any of our segment pairs - fall back to standard
            working_df = dtw_results_df
            mode_title = "Overall Best Mappings"
            print(f"=== Top {top_n} Overall Best Mappings ===")
            print("No mappings found with any matched segment pairs. Falling back to standard best mappings mode.")
    else:
        working_df = dtw_results_df
        mode_title = "Overall Best Mappings"
        print(f"=== Top {top_n} Overall Best Mappings ===")
        
        if use_boundary_mode and not matching_boundary_names:
            print("No matching datums found. Falling back to standard best mappings mode.")
        elif use_boundary_mode:
            print("No segment pairs found. Falling back to standard best mappings mode.")
    
    # Step 1: ALWAYS compute standard mode ranking for all mappings first
    # Clean the data for standard ranking
    standard_working_df = dtw_results_df.copy()
    required_cols = ['corr_coef', 'perc_diag', 'perc_age_overlap']
    existing_cols = [col for col in required_cols if col in standard_working_df.columns]
    if existing_cols:
        standard_working_df = standard_working_df.replace([np.inf, -np.inf], np.nan).dropna(subset=existing_cols)
    
    # Filter for shortest DTW path length for standard mode (only when not in boundary mode)
    if filter_shortest_dtw and not use_boundary_mode and 'length' in standard_working_df.columns:
        standard_working_df['dtw_path_length'] = standard_working_df['length']
        min_length = standard_working_df['dtw_path_length'].min()
        standard_shortest = standard_working_df[standard_working_df['dtw_path_length'] == min_length]
        print(f"Filtering for shortest DTW path length: {min_length}")
    else:
        standard_shortest = standard_working_df
        if filter_shortest_dtw and not use_boundary_mode and 'length' not in standard_working_df.columns:
            print("Warning: No 'length' column found. Using all mappings.")
    
    # Calculate combined scores for standard mode
    standard_df_for_ranking = _calculate_combined_scores(standard_shortest.copy(), weights, higher_is_better_config)
    
    # Always calculate and append standard mode ranking to 'Ranking' column for ALL rows
    standard_ranked_df = standard_df_for_ranking.sort_values(by='combined_score', ascending=False)
    # Rank ALL rows, not just top_n
    for i, (idx, row) in enumerate(standard_ranked_df.iterrows(), 1):
        if 'mapping_id' in row:
            mapping_id = int(row['mapping_id'])
            # Update the original dtw_results_df with standard ranking and combined_score
            dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'Ranking'] = i
            dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'combined_score'] = row['combined_score']
    
    # Get top_n for display purposes
    top_n_standard = standard_ranked_df.head(top_n)
    
    # Step 2: Check if we should proceed with boundary mode
    if not use_boundary_mode or not matching_boundary_names:
        # No boundary mode or no matching names - return standard mode results
        print(f"=== Top {top_n} Overall Best Mappings ===")
        if use_boundary_mode and not matching_boundary_names:
            print("No matching datums found. Using standard best mappings mode.")
        
        # Merge ranking columns from dtw_results_df back into standard_ranked_df
        ranking_cols = ['Ranking', 'Ranking_datums', 'matched_datums', 'combined_score']
        for col in ranking_cols:
            if col in dtw_results_df.columns and 'mapping_id' in standard_ranked_df.columns:
                id_to_value = dtw_results_df.set_index('mapping_id')[col].to_dict()
                standard_ranked_df[col] = standard_ranked_df['mapping_id'].map(id_to_value)
        
        top_mapping_ids, top_mapping_pairs = _print_results(top_n_standard, weights, "Overall Best Mappings")
        
        # Save the updated CSV if it was loaded from a file path
        if isinstance(input_mapping_csv, str):
            dtw_results_df.to_csv(input_mapping_csv, index=False)
        
        return top_mapping_ids, top_mapping_pairs, standard_ranked_df
    
    # Step 3: Proceed with boundary mode processing since we have matching names
    # The working_df and mode_title were already set above (lines 740-800)
    # Do NOT overwrite them here - they contain the filtered mappings we want to use
    
    # Clean the working dataframe for boundary mode
    existing_cols = [col for col in required_cols if col in working_df.columns]
    if existing_cols:
        working_df = working_df.replace([np.inf, -np.inf], np.nan).dropna(subset=existing_cols)
    
    # In boundary mode, don't filter for shortest DTW path length
    shortest_mappings = working_df
    
    # Create a copy for scoring calculations
    df_for_ranking = shortest_mappings.copy()
    
    # Calculate combined scores for boundary mode
    df_for_ranking = _calculate_combined_scores(df_for_ranking, weights, higher_is_better_config)
    
    # Get top N mappings by combined score for boundary mode
    top_mappings_df = df_for_ranking.sort_values(by='combined_score', ascending=False)
    
    # Add datums ranking if we have target mappings (matched boundary names) or partial mappings
    if target_mappings_df is not None and not target_mappings_df.empty:
        # Calculate combined scores for target mappings if not already done
        if 'combined_score' not in target_mappings_df.columns:
            target_mappings_df = _calculate_combined_scores(target_mappings_df, weights, higher_is_better_config)
        
        # Rank ALL target mappings (those with matched datums) for 'Ranking_datums' column
        target_ranked = target_mappings_df.sort_values(by='combined_score', ascending=False)
        for i, (idx, row) in enumerate(target_ranked.iterrows(), 1):
            if 'mapping_id' in row:
                mapping_id = int(row['mapping_id'])
                # Update the original dtw_results_df with datums ranking and matched_datums
                dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'Ranking_datums'] = i
                if 'matched_datums' in row and row['matched_datums']:
                    dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'matched_datums'] = row['matched_datums']
    elif use_boundary_mode and matching_pairs and "Best Partial Mappings" in mode_title:
        # Rank ALL partial mappings (those with some matched segment pairs) for 'Ranking_datums' column
        partial_ranked = top_mappings_df.sort_values(by='combined_score', ascending=False)
        for i, (idx, row) in enumerate(partial_ranked.iterrows(), 1):
            if 'mapping_id' in row:
                mapping_id = int(row['mapping_id'])
                # Update the original dtw_results_df with datums ranking and matched_datums
                dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'Ranking_datums'] = i
                if 'matched_datums' in row and row['matched_datums']:
                    dtw_results_df.loc[dtw_results_df['mapping_id'] == mapping_id, 'matched_datums'] = row['matched_datums']
    
    # Merge ranking columns from dtw_results_df back into top_mappings_df
    ranking_cols = ['Ranking', 'Ranking_datums', 'matched_datums', 'combined_score']
    for col in ranking_cols:
        if col in dtw_results_df.columns and 'mapping_id' in top_mappings_df.columns:
            # Create a mapping from mapping_id to the column value
            id_to_value = dtw_results_df.set_index('mapping_id')[col].to_dict()
            top_mappings_df[col] = top_mappings_df['mapping_id'].map(id_to_value)
    
    # Print detailed results and return boundary mode results
    top_mapping_ids, top_mapping_pairs = _print_results(top_mappings_df.head(top_n), weights, mode_title)
    
    # Save the updated CSV if it was loaded from a file path
    if isinstance(input_mapping_csv, str):
        dtw_results_df.to_csv(input_mapping_csv, index=False)
    
    return top_mapping_ids, top_mapping_pairs, top_mappings_df