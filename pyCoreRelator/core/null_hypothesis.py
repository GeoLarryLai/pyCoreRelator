"""
Null hypothesis testing functions for pyCoreRelator

Included Functions:
- create_segment_pool_from_available_cores: Extract segments from all available cores
- generate_synthetic_core_pair: Generate synthetic log pairs for null hypothesis testing
- compute_pycorerelator_null_hypothesis: Compute null hypothesis r-value distribution
- create_synthetic_picked_depths: Create picked depths for synthetic core

This module provides null hypothesis testing functionality specifically designed for
pyCoreRelator's segment-wise DTW correlation approach. It generates synthetic log
pairs from real segment pools and computes correlation distributions for significance
testing without relying on geological correlations.
"""

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import warnings
from scipy import stats

# Import from other pyCoreRelator modules
from .dtw_analysis import run_comprehensive_dtw_analysis
from .quality_metrics import compute_quality_indicators


def create_segment_pool_from_available_cores(all_cores_data, all_boundaries_data):
    """
    Extract all segments from available cores to create synthesis pool.
    
    This function processes all available core data to extract individual segments
    that can be used for generating synthetic logs. Each segment maintains its
    original characteristics while being detached from its geological context.
    
    Parameters
    ----------
    all_cores_data : dict
        Dictionary with core names as keys, values containing:
        - 'log_data': numpy array of log measurements
        - 'md_data': numpy array of measured depth values
    all_boundaries_data : dict
        Dictionary with core names as keys, values containing:
        - 'depth_boundaries': list of depth boundary indices
        - 'segments': list of (start_idx, end_idx) segment tuples
    
    Returns
    -------
    list
        List of dictionaries, each containing:
        - 'log_data': numpy array of segment log values
        - 'depth_span': float, segment length in depth units
        - 'source_core': str, originating core name
        - 'dimensions': int, number of log dimensions
        - 'segment_id': str, unique identifier for the segment
    
    Examples
    --------
    >>> cores = {
    ...     'CORE_A': {'log_data': np.random.randn(100, 2), 'md_data': np.arange(100)},
    ...     'CORE_B': {'log_data': np.random.randn(120, 2), 'md_data': np.arange(120)}
    ... }
    >>> boundaries = {
    ...     'CORE_A': {'depth_boundaries': [0, 50, 100], 'segments': [(0, 1), (1, 2)]},
    ...     'CORE_B': {'depth_boundaries': [0, 60, 120], 'segments': [(0, 1), (1, 2)]}
    ... }
    >>> pool = create_segment_pool_from_available_cores(cores, boundaries)
    >>> print(f"Created pool with {len(pool)} segments")
    Created pool with 4 segments
    """
    segment_pool = []
    segment_counter = 0
    
    print("Extracting segments from available cores...")
    
    for core_name, core_data in all_cores_data.items():
        if core_name not in all_boundaries_data:
            print(f"Warning: No boundary data found for core {core_name}, skipping...")
            continue
            
        log_data = core_data['log_data']
        md_data = core_data['md_data']
        depth_boundaries = all_boundaries_data[core_name]['depth_boundaries']
        segments = all_boundaries_data[core_name]['segments']
        
        # Determine dimensionality
        if log_data.ndim == 1:
            dimensions = 1
        else:
            dimensions = log_data.shape[1]
        
        # Extract each segment
        for i, (start_seg_idx, end_seg_idx) in enumerate(segments):
            try:
                # Convert segment indices to depth boundary indices
                start_depth_idx = depth_boundaries[start_seg_idx]
                end_depth_idx = depth_boundaries[end_seg_idx]
                
                # Extract segment data
                if start_depth_idx == end_depth_idx:
                    # Single point segment
                    segment_log = log_data[start_depth_idx:start_depth_idx+1]
                    segment_md = md_data[start_depth_idx:start_depth_idx+1]
                else:
                    # Multi-point segment
                    segment_log = log_data[start_depth_idx:end_depth_idx+1]
                    segment_md = md_data[start_depth_idx:end_depth_idx+1]
                
                # Calculate depth span
                depth_span = segment_md[-1] - segment_md[0] if len(segment_md) > 1 else 0.0
                
                # Create segment dictionary
                segment_info = {
                    'log_data': segment_log.copy(),
                    'depth_span': depth_span,
                    'source_core': core_name,
                    'dimensions': dimensions,
                    'segment_id': f"{core_name}_seg_{segment_counter:04d}",
                    'original_segment_idx': i,
                    'length': len(segment_log)
                }
                
                segment_pool.append(segment_info)
                segment_counter += 1
                
            except (IndexError, ValueError) as e:
                print(f"Warning: Error processing segment {i} from core {core_name}: {e}")
                continue
    
    print(f"Successfully extracted {len(segment_pool)} segments from {len(all_cores_data)} cores")
    
    # Print summary statistics
    if segment_pool:
        lengths = [seg['length'] for seg in segment_pool]
        depth_spans = [seg['depth_span'] for seg in segment_pool]
        print(f"Segment lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
        print(f"Depth spans: min={min(depth_spans):.1f}, max={max(depth_spans):.1f}, mean={np.mean(depth_spans):.1f}")
    
    return segment_pool


def generate_synthetic_core_pair(segment_pool, target_length_a, target_length_b, target_dimensions):
    """
    Generate synthetic log pairs matching target core characteristics.
    
    This function randomly samples segments from the pool to create synthetic
    logs that match the length and dimensionality of target cores, ensuring
    no geological correlation between the synthetic pair.
    
    Parameters
    ----------
    segment_pool : list
        Pool of available segments from create_segment_pool_from_available_cores
    target_length_a : int
        Target number of data points for synthetic core A
    target_length_b : int
        Target number of data points for synthetic core B
    target_dimensions : int
        Number of log dimensions to maintain
    
    Returns
    -------
    tuple
        (synthetic_log_a, synthetic_log_b, synthetic_md_a, synthetic_md_b, 
         synthetic_boundaries_a, synthetic_boundaries_b)
        
        - synthetic_log_a/b: numpy arrays of synthetic log data
        - synthetic_md_a/b: numpy arrays of accumulated depth values
        - synthetic_boundaries_a/b: lists of segment boundary indices
    
    Examples
    --------
    >>> pool = [{'log_data': np.array([1, 2]), 'depth_span': 10, 'dimensions': 1}]
    >>> log_a, log_b, md_a, md_b, bounds_a, bounds_b = generate_synthetic_core_pair(
    ...     pool, target_length_a=100, target_length_b=120, target_dimensions=1
    ... )
    >>> print(f"Generated logs: A={len(log_a)}, B={len(log_b)}")
    Generated logs: A=100, B=120
    """
    
    # Filter segments by dimensionality
    compatible_segments = [seg for seg in segment_pool if seg['dimensions'] == target_dimensions]
    
    if not compatible_segments:
        raise ValueError(f"No segments found with {target_dimensions} dimensions in the pool")
    
    def build_synthetic_log(target_length, core_id):
        """Helper function to build a single synthetic log"""
        synthetic_log = []
        synthetic_md = []
        boundaries = [0]  # Start with boundary at index 0
        current_depth = 0.0
        current_length = 0
        
        while current_length < target_length:
            # Randomly select a segment
            segment = random.choice(compatible_segments)
            segment_log = segment['log_data']
            segment_depth_span = max(segment['depth_span'], len(segment_log))  # Ensure positive depth span
            
            # Determine how much of the segment to use
            remaining_length = target_length - current_length
            segment_length = min(len(segment_log), remaining_length)
            
            # Extract the portion we need
            if segment_length == len(segment_log):
                # Use entire segment
                log_portion = segment_log
            else:
                # Use partial segment (from beginning)
                log_portion = segment_log[:segment_length]
            
            # Calculate depth increment (proportional to used portion)
            depth_increment = segment_depth_span * (segment_length / len(segment_log))
            
            # Append to synthetic log
            if current_length == 0:
                # First segment
                synthetic_log.extend(log_portion)
                depth_values = np.linspace(current_depth, current_depth + depth_increment, len(log_portion))
                synthetic_md.extend(depth_values)
            else:
                # Subsequent segments
                synthetic_log.extend(log_portion)
                depth_values = np.linspace(current_depth, current_depth + depth_increment, len(log_portion))
                synthetic_md.extend(depth_values)
            
            # Update counters
            current_length += len(log_portion)
            current_depth += depth_increment
            
            # Add boundary (except for the last iteration)
            if current_length < target_length:
                boundaries.append(current_length)
        
        # Ensure we have the final boundary
        if boundaries[-1] != current_length:
            boundaries.append(current_length)
        
        # Convert to numpy arrays
        synthetic_log = np.array(synthetic_log)
        synthetic_md = np.array(synthetic_md)
        
        # Reshape if multidimensional
        if target_dimensions > 1:
            # If segments have different dimensions, pad or truncate as needed
            if synthetic_log.ndim == 1:
                synthetic_log = synthetic_log.reshape(-1, 1)
                # Replicate to match target dimensions
                synthetic_log = np.repeat(synthetic_log, target_dimensions, axis=1)
            elif synthetic_log.shape[1] != target_dimensions:
                # Adjust dimensions by padding or truncating
                if synthetic_log.shape[1] < target_dimensions:
                    # Pad with noise
                    padding = np.random.normal(0, 0.1, (len(synthetic_log), target_dimensions - synthetic_log.shape[1]))
                    synthetic_log = np.hstack([synthetic_log, padding])
                else:
                    # Truncate
                    synthetic_log = synthetic_log[:, :target_dimensions]
        
        return synthetic_log, synthetic_md, boundaries
    
    # Generate both synthetic cores independently
    synthetic_log_a, synthetic_md_a, synthetic_boundaries_a = build_synthetic_log(target_length_a, 'A')
    synthetic_log_b, synthetic_md_b, synthetic_boundaries_b = build_synthetic_log(target_length_b, 'B')
    
    return (synthetic_log_a, synthetic_log_b, synthetic_md_a, synthetic_md_b, 
            synthetic_boundaries_a, synthetic_boundaries_b)


def compute_pycorerelator_null_hypothesis(segment_pool, core_a_config, core_b_config, 
                                        n_iterations=10000, exponent=0.3, 
                                        dtw_distance_threshold=None, progress_bar=True):
    """
    Compute null hypothesis r-value distribution using pyCoreRelator methodology.
    
    This function generates multiple synthetic core pairs and runs the same DTW
    analysis pipeline as real cores to establish a null hypothesis distribution
    for statistical significance testing.
    
    Parameters
    ----------
    segment_pool : list
        Pool of segments from create_segment_pool_from_available_cores
    core_a_config : dict
        Configuration for synthetic core A:
        - 'target_length': int, number of data points
        - 'target_dimensions': int, number of log dimensions
    core_b_config : dict
        Configuration for synthetic core B:
        - 'target_length': int, number of data points  
        - 'target_dimensions': int, number of log dimensions
    n_iterations : int, default=10000
        Number of synthetic pairs to generate for null hypothesis
    exponent : float, default=0.3
        DTW exponent parameter (matching real analysis)
    dtw_distance_threshold : float, optional
        DTW distance threshold for filtering (matching real analysis)
    progress_bar : bool, default=True
        Whether to show progress bar during computation
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'r_values_distribution': numpy array of correlation coefficients
        - 'distribution_stats': dict with statistical summary
        - 'successful_iterations': int, number of successful analyses
        - 'failed_iterations': int, number of failed analyses
        - 'metadata': dict with analysis parameters
    
    Examples
    --------
    >>> pool = create_segment_pool_from_available_cores(cores_data, boundaries_data)
    >>> config_a = {'target_length': 100, 'target_dimensions': 2}
    >>> config_b = {'target_length': 120, 'target_dimensions': 2}
    >>> results = compute_pycorerelator_null_hypothesis(pool, config_a, config_b, 
    ...                                               n_iterations=1000)
    >>> print(f"Mean r-value: {results['distribution_stats']['mean']:.3f}")
    Mean r-value: 0.023
    """
    
    print(f"Computing pyCoreRelator null hypothesis with {n_iterations} iterations...")
    print(f"Target Core A: {core_a_config['target_length']} points, {core_a_config['target_dimensions']} dimensions")
    print(f"Target Core B: {core_b_config['target_length']} points, {core_b_config['target_dimensions']} dimensions")
    
    r_values_distribution = []
    successful_iterations = 0
    failed_iterations = 0
    
    # Create progress bar
    iterator = tqdm(range(n_iterations), desc="Null hypothesis iterations") if progress_bar else range(n_iterations)
    
    for i in iterator:
        try:
            # Generate synthetic core pair
            (synthetic_log_a, synthetic_log_b, synthetic_md_a, synthetic_md_b, 
             synthetic_boundaries_a, synthetic_boundaries_b) = generate_synthetic_core_pair(
                segment_pool, 
                core_a_config['target_length'], 
                core_b_config['target_length'],
                core_a_config['target_dimensions']
            )
            
            # Convert boundaries to depth values for DTW analysis
            synthetic_depths_a = [synthetic_md_a[min(idx, len(synthetic_md_a)-1)] for idx in synthetic_boundaries_a]
            synthetic_depths_b = [synthetic_md_b[min(idx, len(synthetic_md_b)-1)] for idx in synthetic_boundaries_b]
            
            # Run DTW analysis (no age constraints for null hypothesis)
            dtw_results, valid_dtw_pairs, segments_a, segments_b, _, _, _ = run_comprehensive_dtw_analysis(
                synthetic_log_a, synthetic_log_b, 
                synthetic_md_a, synthetic_md_b,
                picked_depths_a=synthetic_depths_a,
                picked_depths_b=synthetic_depths_b,
                top_bottom=True,
                exponent=exponent,
                dtw_distance_threshold=dtw_distance_threshold,
                age_consideration=False,  # No age constraints for null hypothesis
                exclude_deadend=True
            )
            
            # Extract correlation coefficients from valid segment pairs
            segment_r_values = []
            for pair_key in valid_dtw_pairs:
                if pair_key in dtw_results:
                    _, _, quality_metrics = dtw_results[pair_key]
                    if 'corr_coef' in quality_metrics and not np.isnan(quality_metrics['corr_coef']):
                        segment_r_values.append(quality_metrics['corr_coef'])
            
            # Compute overall r-value for this iteration
            if segment_r_values:
                # Use mean of segment correlations (can be adjusted to other aggregation methods)
                overall_r_value = np.mean(segment_r_values)
                r_values_distribution.append(overall_r_value)
                successful_iterations += 1
            else:
                failed_iterations += 1
                
        except Exception as e:
            failed_iterations += 1
            if progress_bar:
                tqdm.write(f"Iteration {i} failed: {str(e)}")
            continue
    
    # Convert to numpy array
    r_values_distribution = np.array(r_values_distribution)
    
    # Compute distribution statistics
    if len(r_values_distribution) > 0:
        distribution_stats = {
            'mean': np.mean(r_values_distribution),
            'std': np.std(r_values_distribution),
            'median': np.median(r_values_distribution),
            'min': np.min(r_values_distribution),
            'max': np.max(r_values_distribution),
            'percentile_95': np.percentile(r_values_distribution, 95),
            'percentile_97_5': np.percentile(r_values_distribution, 97.5),
            'percentile_99': np.percentile(r_values_distribution, 99),
            'count': len(r_values_distribution)
        }
    else:
        distribution_stats = {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'min': np.nan, 'max': np.nan,
            'percentile_95': np.nan, 'percentile_97_5': np.nan, 'percentile_99': np.nan,
            'count': 0
        }
    
    # Compile results
    results = {
        'r_values_distribution': r_values_distribution,
        'distribution_stats': distribution_stats,
        'successful_iterations': successful_iterations,
        'failed_iterations': failed_iterations,
        'metadata': {
            'n_iterations': n_iterations,
            'exponent': exponent,
            'dtw_distance_threshold': dtw_distance_threshold,
            'core_a_config': core_a_config,
            'core_b_config': core_b_config,
            'segment_pool_size': len(segment_pool)
        }
    }
    
    print(f"\nNull hypothesis computation complete:")
    print(f"Successful iterations: {successful_iterations}/{n_iterations}")
    print(f"Failed iterations: {failed_iterations}/{n_iterations}")
    if successful_iterations > 0:
        print(f"Mean r-value: {distribution_stats['mean']:.4f} ± {distribution_stats['std']:.4f}")
        print(f"95th percentile: {distribution_stats['percentile_95']:.4f}")
        print(f"97.5th percentile: {distribution_stats['percentile_97_5']:.4f}")
    
    return results

def create_synthetic_picked_depths(synthetic_md, segment_info):
    """
    Create picked depths for synthetic core based on segment boundaries.
    
    Parameters:
    - synthetic_md: depth array for synthetic core
    - segment_info: list of dictionaries containing segment information from generate_synthetic_core_pair
    
    Returns:
    - picked_depths: list of tuples (depth, category) for synthetic core
    """
    picked_depths = []
    cumulative_length = 0
    
    # Add boundary at the start
    picked_depths.append((synthetic_md[0], 1))
    
    # Add boundaries at segment junctions
    for i, seg_info in enumerate(segment_info[:-1]):  # Exclude last segment
        cumulative_length += seg_info['length']
        if cumulative_length < len(synthetic_md):
            boundary_depth = synthetic_md[cumulative_length]
            picked_depths.append((boundary_depth, 1))
    
    # Add boundary at the end
    picked_depths.append((synthetic_md[-1], 1))
    
    return picked_depths