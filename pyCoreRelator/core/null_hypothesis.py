"""
Null hypothesis testing functions for pyCoreRelator

Included Functions:
- create_segment_pool_from_available_cores: Extract segments from all available cores
- generate_synthetic_core_pair: Generate synthetic log pairs for null hypothesis testing
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