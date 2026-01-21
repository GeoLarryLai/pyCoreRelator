"""
Synthetic stratigraphy functions for pyCoreRelator

This module provides functions for generating synthetic core data and running
null hypothesis tests for correlation analysis. It includes segment pool management,
synthetic log generation, and visualization tools.

Functions:
- load_segment_pool: Load segment pool data from turbidite database
- modify_segment_pool: Remove unwanted segments from the pool data
- create_synthetic_log: Create synthetic log using turbidite database approach with picked depths at turbidite bases
- create_synthetic_core_pair: Generate synthetic core pair and optionally plot the results
- plot_segment_pool: Plot all segments from the pool in a grid layout
- plot_synthetic_log: Plot a single synthetic log with turbidite boundaries
- synthetic_correlation_quality: Generate DTW correlation quality analysis for synthetic core pairs with multiple iterations
- plot_synthetic_correlation_quality: Plot synthetic correlation quality distributions from saved CSV files
- generate_constraint_subsets: Generate all possible subsets of constraints (2^n combinations)
- run_multi_parameter_analysis: Run comprehensive multi-parameter analysis for core correlation
"""

# Data manipulation and analysis
import os
import gc
import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import from other pyCoreRelator modules
from ..utils.data_loader import load_log_data
from .dtw_core import run_comprehensive_dtw_analysis
from .path_finding import find_complete_core_paths
from .age_models import calculate_interpolated_ages
# Note: plot_correlation_distribution is imported inside functions to avoid circular imports


def load_segment_pool(core_names, log_data_csv, log_data_type, picked_datum, 
                     depth_column, alternative_column_names=None, boundary_category=None, neglect_topbottom=True):
    """
    Load segment pool data from turbidite database.
    
    Parameters:
    - core_names: list of core names to process
    - log_data_csv: dict mapping core names to log file paths
    - log_data_type: list of log column names to load
    - picked_datum: dict mapping core names to picked depth file paths
    - depth_column: name of depth column
    - alternative_column_names: dict of alternative column names (optional)
    - boundary_category: category number for turbidite boundaries (default: None). 
                        If None, uses category 1 if available, otherwise uses the lowest available category
    - neglect_topbottom: if True, skip the first and last segments of each core (default: True)
    
    Returns:
    - seg_pool_metadata: dict containing loaded core data
    - seg_logs: list of turbidite log segments
    - seg_depths: list of turbidite depth segments
    """
    
    seg_pool_metadata = {}
    seg_logs = []
    seg_depths = []
    
    print("Loading segment pool from available cores...")
    
    for core_name in core_names:
        print(f"Processing {core_name}...")
        
        try:
            # Load data for segment pool
            log_data, md_data = load_log_data(
                log_data_csv[core_name],
                log_columns=log_data_type,
                depth_column=depth_column,
                normalize=True
            )
            
            # Check if data was successfully loaded
            if len(md_data) == 0:
                print(f"  Skipping {core_name}: Failed to load log data")
                continue
            
            # Store core data
            seg_pool_metadata[core_name] = {
                'log_data': log_data,
                'md_data': md_data,
                'available_columns': log_data_type
            }
            
            # Load turbidite boundaries for this core
            picked_file = picked_datum[core_name]
            try:
                picked_df = pd.read_csv(picked_file)
                
                # Determine boundary_category to use
                effective_category = boundary_category
                if effective_category is None:
                    # Get all available categories
                    available_categories = sorted(picked_df['category'].unique())
                    if len(available_categories) == 0:
                        raise ValueError(f"No categories found in picked datum file for {core_name}")
                    
                    # Use category 1 if available, otherwise use the lowest available category
                    if 1 in available_categories:
                        effective_category = 1
                    else:
                        effective_category = available_categories[0]
                    print(f"  Using boundary_category={effective_category} (auto-detected)")
                
                # Filter for specified category boundaries only
                category_depths = picked_df[picked_df['category'] == effective_category]['picked_depths_cm'].values
                
                if len(category_depths) == 0:
                    raise ValueError(f"No boundaries found for category {effective_category} in {core_name}")
                
                category_depths = np.sort(category_depths)  # Ensure sorted order
                
                # Create turbidite segments (from boundary to boundary)
                # Determine range based on neglect_topbottom parameter
                if neglect_topbottom and len(category_depths) > 2:
                    # Skip first and last segments
                    start_range = 1
                    end_range = len(category_depths) - 2
                else:
                    # Include all segments
                    start_range = 0
                    end_range = len(category_depths) - 1
                
                for i in range(start_range, end_range):
                    start_depth = category_depths[i]
                    end_depth = category_depths[i + 1]
                    
                    # Find indices corresponding to these depths
                    start_idx = np.argmin(np.abs(md_data - start_depth))
                    end_idx = np.argmin(np.abs(md_data - end_depth))
                    
                    if end_idx > start_idx:
                        # Extract turbidite segment
                        turb_segment = log_data[start_idx:end_idx]
                        turb_depth = md_data[start_idx:end_idx] - md_data[start_idx]  # Relative depths
                        
                        seg_logs.append(turb_segment)
                        seg_depths.append(turb_depth)
                
            except Exception as e:
                print(f"Warning: Could not load turbidite boundaries for {core_name}: {e}")
            
            print(f"  Loaded: {len(log_data)} points, columns: {log_data_type}")
            
        except Exception as e:
            print(f"Error loading {core_name}: {e}")
    
    # Set target dimensions based on segment pool
    target_dimensions = seg_logs[0].shape[1] if len(seg_logs) > 0 and seg_logs[0].ndim > 1 else 1
    
    print(f"Segment pool created with {len(seg_logs)} turbidites")
    print(f"Total cores processed: {len(seg_pool_metadata)}")
    print(f"Target dimensions: {target_dimensions}")
    
    return seg_logs, seg_depths, seg_pool_metadata



def modify_segment_pool(segment_logs, segment_depths, remove_list=None):
    """
    Remove unwanted segments from the pool data and return the modified pool.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - remove_list: list of 1-based segment numbers to remove (optional)
                  If None or empty, no segments are removed
    
    Returns:
    - modified_segment_logs: list of remaining log data arrays
    - modified_segment_depths: list of remaining depth arrays
    """
    
    # If remove_list is None or empty, return original data
    if not remove_list:
        print("No segments to remove. Returning original pool data.")
        return segment_logs.copy(), segment_depths.copy()
    
    # Convert remove_list to 0-based indices and validate
    remove_indices = []
    for item in remove_list:
        try:
            # Convert to int (handle both string and int inputs)
            segment_num = int(item)
            if 1 <= segment_num <= len(segment_logs):
                remove_indices.append(segment_num - 1)  # Convert to 0-based
            else:
                print(f"Warning: Segment number {segment_num} is out of range (1-{len(segment_logs)}). Skipping.")
        except (ValueError, TypeError):
            print(f"Warning: Invalid segment number '{item}'. Skipping.")
    
    # Remove duplicates and sort
    remove_indices = sorted(set(remove_indices))
    
    if not remove_indices:
        print("No valid segments to remove. Returning original pool data.")
        return segment_logs.copy(), segment_depths.copy()
    
    # Create modified lists by excluding specified indices
    modified_segment_logs = []
    modified_segment_depths = []
    
    for i, (segment_log, segment_depth) in enumerate(zip(segment_logs, segment_depths)):
        if i not in remove_indices:
            modified_segment_logs.append(segment_log)
            modified_segment_depths.append(segment_depth)
    
    # Print summary of changes
    removed_segments_1based = [idx + 1 for idx in remove_indices]
    print(f"Removed segments: {removed_segments_1based}")
    print(f"Original pool size: {len(segment_logs)} segments")
    print(f"Modified pool size: {len(modified_segment_logs)} segments")
    
    return modified_segment_logs, modified_segment_depths

def create_synthetic_log(target_thickness, segment_logs, segment_depths, exclude_inds=None, repetition=False):
    """Create synthetic log using turbidite database approach with picked depths at turbidite bases.
    
    Parameters:
    - target_thickness: target thickness for the synthetic log
    - segment_logs: list of turbidite log segments
    - segment_depths: list of corresponding depth arrays
    - exclude_inds: indices to exclude from selection (optional)
    - repetition: if True, allow reusing turbidite segments; if False, each segment can only be used once (default: False)
    
    Returns:
    - tuple: (log, d, valid_picked_depths, inds)
      - log: synthetic log data array
      - d: depth values array
      - valid_picked_depths: list of boundary depth values (just depths, no category info)
      - inds: list of indices of segments used from the pool
    """
    # Determine target dimensions from the first available segment
    target_dimensions = segment_logs[0].shape[1] if len(segment_logs) > 0 and segment_logs[0].ndim > 1 else 1
    
    fake_log = np.array([]).reshape(0, target_dimensions) if target_dimensions > 1 else np.array([])
    md_log = np.array([])
    max_depth = 0
    inds = []
    picked_depths = []
    
    # Initialize available indices for selection
    if repetition:
        # If repetition is allowed, always use the full range
        available_inds = list(range(len(segment_logs)))
    else:
        # If no repetition, start with all indices and remove as we use them
        available_inds = list(range(len(segment_logs)))
        if exclude_inds is not None:
            available_inds = [ind for ind in available_inds if ind not in exclude_inds]
    
    # Add initial boundary
    picked_depths.append((0, 1))
    
    while max_depth <= target_thickness:
        # Check if we have available indices
        if not repetition and len(available_inds) == 0:
            print("Warning: No more unique turbidite segments available. Stopping log generation.")
            break
            
        if repetition:
            # Original behavior: select from full range, excluding only exclude_inds
            potential_inds = [ind for ind in range(len(segment_logs)) if exclude_inds is None or ind not in exclude_inds]
            if not potential_inds:
                print("Warning: No available turbidite segments after exclusions. Stopping log generation.")
                break
            ind = random.choices(potential_inds, k=1)[0]
        else:
            # New behavior: select from available indices and remove after use
            ind = random.choices(available_inds, k=1)[0]
            available_inds.remove(ind)  # Remove from available list to prevent reuse
            
        inds.append(ind)
        
        # Get turbidite segment from database
        turb_segment = segment_logs[ind]
        turb_depths = segment_depths[ind]
        
        # Ensure turbidite has proper dimensions
        if turb_segment.ndim == 1:
            turb_segment = turb_segment.reshape(-1, 1)
        
        # Ensure proper dimensions match target
        if turb_segment.shape[1] < target_dimensions:
            # Pad with noise if needed
            padding = np.random.normal(0, 0.1, (len(turb_segment), target_dimensions - turb_segment.shape[1]))
            turb_segment = np.hstack([turb_segment, padding])
        elif turb_segment.shape[1] > target_dimensions:
            # Truncate if needed
            turb_segment = turb_segment[:, :target_dimensions]
        
        # Append log data
        if target_dimensions > 1:
            if len(fake_log) == 0:
                fake_log = turb_segment.copy()
            else:
                fake_log = np.vstack((fake_log, turb_segment))
        else:
            fake_log = np.hstack((fake_log, turb_segment.flatten()))
        
        # Append depth data
        if len(md_log) == 0:
            md_log = np.hstack((md_log, 1 + turb_depths))
        else:
            md_log = np.hstack((md_log, 1 + md_log[-1] + turb_depths))
            
        max_depth = md_log[-1]
        
        # Add picked depth at the base of this turbidite (current max_depth)
        if max_depth <= target_thickness:
            picked_depths.append((max_depth, 1))
    
    # Truncate to target thickness
    valid_indices = md_log <= target_thickness
    if target_dimensions > 1:
        log = fake_log[valid_indices]
    else:
        log = fake_log[valid_indices]
    d = md_log[valid_indices]
    
    # Filter picked depths to only include those within the valid range
    # Extract just the depth values (no category info)
    valid_picked_depths = [depth for depth, category in picked_depths if depth <= target_thickness]
    
    # Ensure we have an end boundary
    if len(valid_picked_depths) == 0 or valid_picked_depths[-1] != d[-1]:
        valid_picked_depths.append(d[-1])
    
    return log, d, valid_picked_depths, inds



def generate_constraint_subsets(n_constraints):
    """Generate all possible subsets of constraints (2^n combinations)"""
    all_subsets = []
    for r in range(n_constraints + 1):  # 0 to n_constraints
        for subset in combinations(range(n_constraints), r):
            all_subsets.append(list(subset))
    return all_subsets


def _process_single_parameter_combination(
    idx, params, 
    log_a, log_b, md_a, md_b,
    picked_datum_a, picked_datum_b,
    datum_ages_a, datum_ages_b,
    core_a_age_data, core_b_age_data,
    target_quality_indices,
    output_csv_filenames,
    synthetic_csv_filenames,
    pca_for_dependent_dtw,
    test_age_constraint_removal,
    output_metric_only=True
):
    """Process a single parameter combination with parallel computation inside."""
    
    # Import here to avoid circular imports in workers
    from .dtw_core import run_comprehensive_dtw_analysis
    from .path_finding import find_complete_core_paths
    from ..utils.plotting import plot_correlation_distribution
    
    # Generate a random suffix for temporary files in this iteration
    random_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=8))

    # Initialize temp_mapping_file in Downloads folder to avoid cloud sync overhead
    temp_dir = os.path.expanduser('~/Downloads')
    temp_mapping_file = os.path.join(temp_dir, f'temp_mappings_{random_suffix}.pkl')

    # Extract parameters
    age_consideration = params['age_consideration']
    restricted_age_correlation = params['restricted_age_correlation']
    shortest_path_search = params['shortest_path_search']
    
    # Generate parameter labels
    if age_consideration:
        if restricted_age_correlation:
            age_label = 'restricted_age'
        else:
            age_label = 'loose_age'
    else:
        age_label = 'no_age'
    
    search_label = 'optimal' if shortest_path_search else 'random'
    combo_id = f"{age_label}_{search_label}"
    
    # Cache for synthetic CSV DataFrames to avoid redundant reads
    synthetic_df_cache = {}
    
    # Check if this scenario exists in synthetic CSV files (if provided)
    if synthetic_csv_filenames:
        scenario_exists = False
        for quality_index in target_quality_indices:
            if quality_index in synthetic_csv_filenames:
                synthetic_csv_file = synthetic_csv_filenames[quality_index]
                if os.path.exists(synthetic_csv_file):
                    try:
                        synthetic_df = pd.read_csv(synthetic_csv_file)
                        # Cache the DataFrame for later use
                        synthetic_df_cache[quality_index] = synthetic_df
                        # Check if this combination_id exists in the synthetic CSV
                        if 'combination_id' in synthetic_df.columns:
                            if combo_id in synthetic_df['combination_id'].values:
                                scenario_exists = True
                                break
                    except Exception:
                        pass
        
        # If synthetic CSV exists but this scenario is not in it, skip processing
        if not scenario_exists:
            return True, combo_id, {}  # Return success with empty results to skip without error
    
    try:
        # Validate input age data when age consideration is enabled
        if age_consideration:
            # Check datum_ages_a and datum_ages_b
            if datum_ages_a is None or datum_ages_b is None:
                return False, combo_id, "datum_ages_a or datum_ages_b is None when age_consideration=True"
            
            # Check required keys in datum_ages
            required_age_keys = ['depths', 'ages', 'pos_uncertainties', 'neg_uncertainties']
            for key in required_age_keys:
                if key not in datum_ages_a or not datum_ages_a[key]:
                    return False, combo_id, f"Missing or empty key '{key}' in datum_ages_a"
                if key not in datum_ages_b or not datum_ages_b[key]:
                    return False, combo_id, f"Missing or empty key '{key}' in datum_ages_b"
            
            # Check age_data constraints
            required_constraint_keys = ['in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors']
            for key in required_constraint_keys:
                if key not in core_a_age_data:
                    return False, combo_id, f"Missing key '{key}' in core_a_age_data"
                if key not in core_b_age_data:
                    return False, combo_id, f"Missing key '{key}' in core_b_age_data"
                    
                # Check for all NaN values in constraint data
                try:
                    if core_a_age_data[key] and len(core_a_age_data[key]) > 0:
                        valid_values_a = [val for val in core_a_age_data[key] if not (np.isnan(val) if isinstance(val, (int, float)) else False)]
                        if len(valid_values_a) == 0:
                            return False, combo_id, f"All values are NaN in core_a_age_data['{key}']"
                    
                    if core_b_age_data[key] and len(core_b_age_data[key]) > 0:
                        valid_values_b = [val for val in core_b_age_data[key] if not (np.isnan(val) if isinstance(val, (int, float)) else False)]
                        if len(valid_values_b) == 0:
                            return False, combo_id, f"All values are NaN in core_b_age_data['{key}']"
                except Exception as e:
                    return False, combo_id, f"Error validating age_data['{key}']: {str(e)}"
        
        # Run comprehensive DTW analysis with original constraints
        dtw_result = run_comprehensive_dtw_analysis(
            log_a, log_b, md_a, md_b,
            picked_datum_a=picked_datum_a,
            picked_datum_b=picked_datum_b,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=True,
            top_depth=0.0,
            exclude_deadend=True,
            mute_mode=True,
            age_consideration=age_consideration,
            datum_ages_a=datum_ages_a if age_consideration else None,
            datum_ages_b=datum_ages_b if age_consideration else None,
            restricted_age_correlation=restricted_age_correlation if age_consideration else False,
            core_a_age_data=core_a_age_data if age_consideration else None,
            core_b_age_data=core_b_age_data if age_consideration else None,
            n_jobs=-1  # Use all available cores
        )
        
        # Check if DTW analysis returned None
        if dtw_result is None:
            return False, combo_id, "run_comprehensive_dtw_analysis returned None"
        
        # Validate DTW results before proceeding (check required keys exist)
        required_keys = ['dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 
                        'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full']
        if not all(key in dtw_result for key in required_keys):
            return False, combo_id, "DTW analysis returned incomplete dictionary"
        
        if any(dtw_result[key] is None for key in required_keys):
            return False, combo_id, "DTW analysis returned None values"
        
        # Find complete core paths
        if shortest_path_search:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=True, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        else:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=False, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        
        # Process quality indices and collect results
        results = {}
        for quality_index in target_quality_indices:
            
            # Extract bin size information from cached synthetic CSV if available
            targeted_binsize = None
            if quality_index in synthetic_df_cache:
                try:
                    synthetic_df = synthetic_df_cache[quality_index]
                    if not synthetic_df.empty and 'bins' in synthetic_df.columns:
                        # Parse the first row's bins to get the bin structure
                        bins_str = synthetic_df.iloc[0]['bins']
                        if pd.notna(bins_str):
                            synthetic_bins = np.fromstring(bins_str.strip('[]'), sep=' ')
                            bin_width_synthetic = np.mean(np.diff(synthetic_bins))
                            targeted_binsize = (synthetic_bins, bin_width_synthetic)
                except Exception:
                    pass  # Use default binning if extraction fails
            
            fit_params = plot_correlation_distribution(
                mapping_csv=f'{temp_mapping_file}',
                quality_index=quality_index,
                save_png=False, pdf_method='normal',
                kde_bandwidth=0.05, mute_mode=True, targeted_binsize=targeted_binsize
            )
            
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                
                # Remove kde_object as it can't be serialized to CSV
                fit_params_copy.pop('kde_object', None)
                
                # Add metadata fields
                fit_params_copy['combination_id'] = combo_id
                fit_params_copy['age_consideration'] = age_consideration
                fit_params_copy['restricted_age_correlation'] = restricted_age_correlation
                fit_params_copy['shortest_path_search'] = shortest_path_search
                
                # Add constraint tracking columns
                fit_params_copy['core_a_constraints_count'] = len(core_a_age_data['in_sequence_ages']) if age_consideration else 0
                fit_params_copy['core_b_constraints_count'] = len(core_b_age_data['in_sequence_ages']) if age_consideration else 0
                fit_params_copy['constraint_scenario_description'] = 'all_original_constraints_remained' if age_consideration else 'no_age_constraints_used'
                
                results[quality_index] = fit_params_copy
        
        # Clean up
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        
        del dtw_result
        gc.collect()
        
        return True, combo_id, results
        
    except Exception as e:
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        gc.collect()
        return False, combo_id, str(e)


def _process_single_constraint_scenario(
    param_idx, params, constraint_subset, in_sequence_indices,
    log_a, log_b, md_a, md_b,
    picked_datum_a, picked_datum_b,
    datum_ages_a, datum_ages_b,
    core_a_age_data, core_b_age_data,
    uncertainty_method,
    target_quality_indices,
    cached_binsizes,
    pca_for_dependent_dtw,
    output_metric_only=True
):
    """Process a single constraint scenario (exact copy of original loop body)"""
    
    # Import here to avoid circular imports in workers
    from .dtw_core import run_comprehensive_dtw_analysis
    from .path_finding import find_complete_core_paths
    from ..utils.plotting import plot_correlation_distribution
    from .age_models import calculate_interpolated_ages
    
    # Generate a process-safe random suffix for temporary files using numpy
    # This avoids conflicts between parallel workers
    import os
    process_id = os.getpid()
    np.random.seed(None)  # Use current time as seed (non-deterministic)
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    random_suffix = ''.join(np.random.choice(list(chars), size=8))
    
    # Initialize temp_mapping_file in Downloads folder to avoid cloud sync overhead
    temp_dir = os.path.expanduser('~/Downloads')
    temp_mapping_file = os.path.join(temp_dir, f'temp_mappings_{random_suffix}.pkl')
    
    # Reset and reload parameters correctly for each iteration
    age_consideration = params['age_consideration']
    restricted_age_correlation = params['restricted_age_correlation']
    shortest_path_search = params['shortest_path_search']
    
    # Generate parameter labels
    if restricted_age_correlation:
        age_label = 'restricted_age'
    else:
        age_label = 'loose_age'
    
    search_label = 'optimal' if shortest_path_search else 'random'
    combo_id = f"{age_label}_{search_label}"
    
    try:
        # Map subset indices to original constraint indices (in-sequence only)
        # No NaN filtering - user confirmed the data_columns fields are valid
        original_indices = []
        for i in constraint_subset:
            idx = in_sequence_indices[i]
            original_indices.append(idx)
        
        # Check if we have any constraints (should always be true)
        if len(original_indices) == 0:
            return False, f"{combo_id}_no_constraints", "No constraints in subset"
        
        # Create modified core_b_age_data using original indices
        # Convert pandas Series to lists and ensure proper data types
        age_data_b_current = {
            'depths': [float(core_b_age_data['depths'].iloc[i]) if hasattr(core_b_age_data['depths'], 'iloc') else float(core_b_age_data['depths'][i]) for i in original_indices],
            'ages': [float(core_b_age_data['ages'][i]) for i in original_indices],
            'pos_errors': [float(core_b_age_data['pos_errors'][i]) for i in original_indices],
            'neg_errors': [float(core_b_age_data['neg_errors'][i]) for i in original_indices],
            'in_sequence_flags': [core_b_age_data['in_sequence_flags'][i] for i in original_indices],
            'in_sequence_depths': [float(core_b_age_data['depths'].iloc[i]) if hasattr(core_b_age_data['depths'], 'iloc') else float(core_b_age_data['depths'][i]) for i in original_indices],
            'in_sequence_ages': [float(core_b_age_data['ages'][i]) for i in original_indices],
            'in_sequence_pos_errors': [float(core_b_age_data['pos_errors'][i]) for i in original_indices],
            'in_sequence_neg_errors': [float(core_b_age_data['neg_errors'][i]) for i in original_indices],
            'core': [core_b_age_data['core'][i] for i in original_indices]
        }
        
        # Recalculate interpolated ages for core B with reduced constraints
        datum_ages_b_current = calculate_interpolated_ages(
            picked_datum=picked_datum_b,
            age_constraints_depths=np.array(age_data_b_current['depths']),
            age_constraints_ages=np.array(age_data_b_current['ages']),
            age_constraints_pos_errors=np.array(age_data_b_current['pos_errors']),
            age_constraints_neg_errors=np.array(age_data_b_current['neg_errors']),
            age_constraints_in_sequence_flags=age_data_b_current['in_sequence_flags'],
            age_constraint_source_core=age_data_b_current['core'],
            top_bottom=True,
            top_depth=0.0,
            bottom_depth=md_b[-1],
            top_age=0,
            top_age_pos_error=75,
            top_age_neg_error=75,
            uncertainty_method=uncertainty_method,
            n_monte_carlo=10000,
            show_plot=False,
            export_csv=False,
            mute_mode=True
        )
        
        # Validate the interpolated ages result
        if datum_ages_b_current is None:
            return False, f"{combo_id}_invalid_ages", "calculate_interpolated_ages returned None"
        
        # Check if interpolated ages contain valid values
        required_keys = ['depths', 'ages', 'pos_uncertainties', 'neg_uncertainties']
        for key in required_keys:
            if key not in datum_ages_b_current or not datum_ages_b_current[key]:
                return False, f"{combo_id}_missing_age_key", f"Missing or empty key '{key}' in interpolated ages"
            
            # Check for all NaN values
            if all(np.isnan(val) for val in datum_ages_b_current[key]):
                return False, f"{combo_id}_all_nan_ages", f"All values are NaN in interpolated ages key '{key}'"
        
        # Run DTW analysis with reduced constraints
        dtw_result = run_comprehensive_dtw_analysis(
            log_a, log_b, md_a, md_b,
            picked_datum_a=picked_datum_a,
            picked_datum_b=picked_datum_b,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=True,
            top_depth=0.0,
            exclude_deadend=True,
            mute_mode=True,
            age_consideration=age_consideration,
            datum_ages_a=datum_ages_a,  # Use original ages for core A
            datum_ages_b=datum_ages_b_current,  # Use modified ages for core B
            restricted_age_correlation=restricted_age_correlation,
            core_a_age_data=core_a_age_data,  # Original age constraint data for core A
            core_b_age_data=age_data_b_current,  # Modified age constraint data for core B
            n_jobs=-1  # Use all available cores
        )
        
        # Check if DTW analysis returned None
        if dtw_result is None:
            return False, f"{combo_id}_dtw_none", "run_comprehensive_dtw_analysis returned None"
        
        # Validate DTW results before proceeding (check required keys exist)
        required_keys = ['dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 
                        'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full']
        if not all(key in dtw_result for key in required_keys):
            return False, f"{combo_id}_dtw_incomplete", "DTW analysis returned incomplete dictionary"
        
        if any(dtw_result[key] is None for key in required_keys):
            return False, f"{combo_id}_dtw_none", "DTW analysis returned None values"
        
        # Find paths with correct parameters
        if shortest_path_search:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=True, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        else:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=False, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        
        # Process quality indices
        results = {}
        for quality_index in target_quality_indices:
            
            # Use pre-cached bin size information if available
            targeted_binsize = cached_binsizes.get(quality_index, None)
            
            fit_params = plot_correlation_distribution(
                mapping_csv=f'{temp_mapping_file}',
                quality_index=quality_index,
                save_png=False, pdf_method='normal',
                kde_bandwidth=0.05, mute_mode=True, targeted_binsize=targeted_binsize
            )
            
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                
                # Remove kde_object as it can't be serialized to CSV
                fit_params_copy.pop('kde_object', None)
                
                # Convert 0-based indices to 1-based for constraint description
                remaining_indices_1based = [i + 1 for i in sorted(constraint_subset)]
                
                # Add metadata fields
                fit_params_copy['combination_id'] = combo_id
                fit_params_copy['age_consideration'] = age_consideration
                fit_params_copy['restricted_age_correlation'] = restricted_age_correlation
                fit_params_copy['shortest_path_search'] = shortest_path_search
                
                # Add constraint tracking with correct counts
                fit_params_copy['core_a_constraints_count'] = len(core_a_age_data['in_sequence_ages'])  # Original count for core A
                fit_params_copy['core_b_constraints_count'] = len(constraint_subset)  # Modified count for core B
                fit_params_copy['constraint_scenario_description'] = f'constraints_{remaining_indices_1based}_remained'
                
                results[quality_index] = fit_params_copy
        
        # Clean up temporary files
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        
        del dtw_result
        del age_data_b_current, datum_ages_b_current
        gc.collect()
        
        scenario_id = f"{combo_id}_subset_{len(constraint_subset)}"
        return True, scenario_id, results
        
    except Exception as e:
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        # Clean up variables in case of error
        if 'age_data_b_current' in locals():
            del age_data_b_current
        if 'datum_ages_b_current' in locals():
            del datum_ages_b_current
        gc.collect()
        return False, f"{combo_id}_subset_error", str(e)


def run_multi_parameter_analysis(
    # Core data inputs
    log_a, log_b, md_a, md_b,
    picked_datum_a, picked_datum_b,
    datum_ages_a, datum_ages_b,
    core_a_age_data, core_b_age_data,
    uncertainty_method,
    
    # Core identifiers
    core_a_name, 
    core_b_name,
    
    # Output configuration
    output_csv_directory,  # Directory path for output CSV files
    
    # Analysis parameters
    parameter_combinations,
    target_quality_indices=['corr_coef', 'norm_dtw'],
    log_columns=None,  # List of log column names (e.g., ['hiresMS', 'CT', 'Lumin'])
    test_age_constraint_removal = True,

    # Optional parameters
    synthetic_csv_filenames=None,  # Dict with quality_index as key and synthetic CSV filename as value
    pca_for_dependent_dtw=False,
    max_search_per_layer=None,  # Max scenarios per constraint removal layer
    output_metric_only=True  # Only output quality metrics, skip full path info for faster processing
):
    """
    Run comprehensive multi-parameter analysis for core correlation.
    
    Brief summary: This function performs DTW analysis across multiple parameter combinations
    and optionally tests age constraint removal scenarios, generating distribution fit parameters
    for various quality indices.
    
    Parameters:
    -----------
    log_a, log_b : array-like
        Log data for cores A and B
    md_a, md_b : array-like
        Measured depth arrays for cores A and B
    picked_datum_a, picked_datum_b : array-like
        Picked depths of category 1 for cores A and B
    datum_ages_a, datum_ages_b : dict
        Age interpolation results for picked depths
    core_a_age_data, core_b_age_data : dict
        Age constraint data for cores A and B
    uncertainty_method : str
        Method for uncertainty calculation
    core_a_name, core_b_name : str
        Names of cores A and B
    output_csv_directory : str
        Directory path where output CSV files will be saved
    parameter_combinations : list of dict
        List of parameter combinations to test
    target_quality_indices : list, default=['corr_coef', 'norm_dtw']
        Quality indices to analyze (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    log_columns : list or None, default=None
        List of log column names (e.g., ['hiresMS', 'CT', 'Lumin']).
        If provided, will be used in the output directory structure.
    test_age_constraint_removal : bool (default=True)
        Whether to test age constraint removal scenarios
    synthetic_csv_filenames : dict or None, default=None
        Dictionary mapping quality_index to synthetic CSV filename for consistent bin sizing
    pca_for_dependent_dtw : bool
        Whether to use PCA for dependent DTW
    max_search_per_layer : int or None, default=None
        Maximum number of scenarios to process per constraint removal layer.
        If None, processes all scenarios. A layer represents combinations with
        the same number of remaining age constraints.
    output_metric_only : bool, default=True
        If True, only output quality metrics in path finding results, skip storing
        full path information (path sequences and warping paths). This significantly
        reduces memory usage and speeds up processing. Set to False only if you need
        the detailed path information for downstream analysis.
    
    Returns:
    --------
    None
        Results are saved to CSV files in output_csv_directory
    """
    
    # Generate output CSV filenames based on directory and log_columns
    if log_columns is not None and len(log_columns) > 0:
        # If log_columns provided, use subdirectory structure
        log_cols_str = "_".join(log_columns)
        full_output_dir = os.path.join(output_csv_directory, log_cols_str)
    else:
        # If no log_columns, use the directory as is
        full_output_dir = output_csv_directory
    
    # Create output CSV filenames dictionary
    output_csv_filenames = {}
    for quality_index in target_quality_indices:
        output_csv_filenames[quality_index] = os.path.join(
            full_output_dir, 
            f'{quality_index}_fit_params.csv'
        )
    
    # Create directories for output files if needed
    os.makedirs(full_output_dir, exist_ok=True)
    
    # VALIDATE AGE DATA BEFORE PROCESSING
    print("Validating age data for age-based analysis...")
    
    # Check if age data contains valid age values
    def validate_age_data(age_data, core_name):
        """Validate age data for a given core"""
        if age_data is None:
            return False, f"{core_name}: age_data is None"
        
        if 'ages' not in age_data or not age_data['ages']:
            return False, f"{core_name}: no 'ages' found in age_data"
        
        # Check if all ages are NaN or invalid
        valid_ages = [age for age in age_data['ages'] if isinstance(age, (int, float)) and not np.isnan(age)]
        if len(valid_ages) == 0:
            return False, f"{core_name}: all age values in age_data['ages'] are NaN or invalid"
        
        return True, f"{core_name}: age_data validation passed"
    
    def validate_pickeddepth_ages(pickeddepth_ages, core_name):
        """Validate pickeddepth ages for a given core"""
        if pickeddepth_ages is None:
            return False, f"{core_name}: pickeddepth_ages is None"
        
        if 'ages' not in pickeddepth_ages or not pickeddepth_ages['ages']:
            return False, f"{core_name}: no 'ages' found in pickeddepth_ages"
        
        # Check if the last age value is NaN
        ages = pickeddepth_ages['ages']
        if len(ages) > 0 and np.isnan(ages[-1]):
            return False, f"{core_name}: last age value in pickeddepth_ages['ages'] is NaN"
        
        # Check if all ages are NaN
        valid_ages = [age for age in ages if not np.isnan(age)]
        if len(valid_ages) == 0:
            return False, f"{core_name}: all age values in pickeddepth_ages['ages'] are NaN"
        
        return True, f"{core_name}: pickeddepth_ages validation passed"
    
    # Validate age data for both cores
    age_data_a_valid, age_data_a_msg = validate_age_data(core_a_age_data, core_a_name)
    age_data_b_valid, age_data_b_msg = validate_age_data(core_b_age_data, core_b_name)
    pickeddepth_ages_a_valid, pickeddepth_ages_a_msg = validate_pickeddepth_ages(datum_ages_a, core_a_name)
    pickeddepth_ages_b_valid, pickeddepth_ages_b_msg = validate_pickeddepth_ages(datum_ages_b, core_b_name)
    
    # Determine if age-based analysis can be performed
    age_analysis_possible = (age_data_a_valid and age_data_b_valid and 
                           pickeddepth_ages_a_valid and pickeddepth_ages_b_valid)
    
    # Determine which cores have invalid age constraints for specific warning
    core_a_age_valid = age_data_a_valid and pickeddepth_ages_a_valid
    core_b_age_valid = age_data_b_valid and pickeddepth_ages_b_valid
    
    # Print validation results and specific warning
    if not age_analysis_possible:
        # Determine which cores have invalid age data
        invalid_cores = []
        if not core_a_age_valid:
            invalid_cores.append(core_a_name)
        if not core_b_age_valid:
            invalid_cores.append(core_b_name)
        
        if len(invalid_cores) == 2:
            core_warning = f"{core_a_name} & {core_b_name}"
        else:
            core_warning = invalid_cores[0]
        
        print(f"⚠️  WARNING: No valid age constraints in CORE {core_warning}. Only compute results without age consideration.")
    else:
        print("✓ Age data validation passed - age-based analysis is possible")
    
    # Filter parameter combinations based on age data validity
    original_param_count = len(parameter_combinations)
    if age_analysis_possible:
        # Keep all parameter combinations if age analysis is possible
        filtered_parameter_combinations = parameter_combinations
    else:
        # Remove parameter combinations with age_consideration=True if age data is invalid
        filtered_parameter_combinations = [
            params for params in parameter_combinations 
            if not params.get('age_consideration', False)
        ]
    
    filtered_param_count = len(filtered_parameter_combinations)
    
    if filtered_param_count < original_param_count:
        skipped_count = original_param_count - filtered_param_count
        print(f"   Filtered out {skipped_count} parameter combinations with age_consideration=True")
        print(f"   Proceeding with {filtered_param_count} parameter combinations")
    
    # Disable age constraint removal testing if age data is invalid
    effective_test_age_constraint_removal = test_age_constraint_removal and age_analysis_possible
    
    if test_age_constraint_removal and not effective_test_age_constraint_removal:
        print("   Age constraint removal testing has been disabled due to invalid age data")
    
    # Update parameter_combinations for the rest of the function
    parameter_combinations = filtered_parameter_combinations
    test_age_constraint_removal = effective_test_age_constraint_removal
    
    # Loop through all quality indices
    print(f"Running {len(parameter_combinations)} parameter combinations for {len(target_quality_indices)} quality indices...")
    print(f"Processing scenarios sequentially with parallel computation inside each scenario")

    # Reset variables at the beginning
    n_constraints_b = 0
    age_enabled_params = []
    total_additional_scenarios = 0

    if test_age_constraint_removal:
        n_constraints_b = len(core_b_age_data['in_sequence_ages'])
        age_enabled_params = [p for p in parameter_combinations if p['age_consideration']]
        constraint_scenarios_per_param = (2 ** n_constraints_b) - 1  # Exclude empty set
        total_additional_scenarios = len(age_enabled_params) * (constraint_scenarios_per_param - 1)  # Exclude original scenario
        
        print(f"Age constraint removal testing enabled:")
        print(f"- Core B has {n_constraints_b} age constraints")
        
        if max_search_per_layer is None:
            print(f"- Additional scenarios to process: {total_additional_scenarios}")
            # Warning for large number of scenarios
            if total_additional_scenarios > 5000:
                print(f"⚠️  WARNING: Processing {total_additional_scenarios} scenarios will take very long time and use large memory.")
                print(f"   Recommend setting max_search_per_layer to about 200-300 (but >= the number of constraints in core B: {n_constraints_b})")
                print(f"   to reduce the number of scenarios per constraint removal layer to process.")
                print(f"   When max_search_per_layer is set, scenarios are randomly sampled from each layer.")
                print(f"   This provides statistical approximation while maintaining computational feasibility.")
        else:
            # Calculate how many scenarios will actually be processed
            # This is an approximation since it depends on the distribution across layers
            print(f"- Additional {total_additional_scenarios} scenarios exist")
            print(f"- As max_search_per_layer is defined: {max_search_per_layer} scenarios are randomly sampled from each constraint removal layer")
            print(f"- ⚠️  WARNING: Due to random sampling, not every run will yield identical results")
            print(f"-     This just provides statistical approximation while maintaining computational feasibility")
            # Check if max_search_per_layer is too small
            if max_search_per_layer < n_constraints_b:
                print(f"- ⚠️  WARNING: max_search_per_layer ({max_search_per_layer}) is less than the number of constraints ({n_constraints_b})")
                print(f"-     Recommend setting max_search_per_layer >= {n_constraints_b} to ensure all age constraints are evaluated")

    # PHASE 1: Run original parameter combinations
    if test_age_constraint_removal:
        print("\n=== PHASE 1: Running original parameter combinations ===")
    else:
        print("\nRunning parameter combinations...")
    
    if not age_analysis_possible:
        print("Note: Only processing parameter combinations with age_consideration=False due to invalid age data")

    # Parallelism strategy: sequential scenarios with full parallelism inside each
    n_combinations = len(parameter_combinations)

    # Prepare data for processing
    phase1_args = [
        (idx, params, 
         log_a, log_b, md_a, md_b,
         picked_datum_a, picked_datum_b,
         datum_ages_a, datum_ages_b,
         core_a_age_data, core_b_age_data,
         target_quality_indices,
         output_csv_filenames,
         synthetic_csv_filenames,
         pca_for_dependent_dtw,
         test_age_constraint_removal,
         output_metric_only) 
        for idx, params in enumerate(parameter_combinations)
    ]
    
    # Run Phase 1 sequentially with progress bar showing elapsed time
    desc_phase1 = "Phase 1: Parameter combinations" if test_age_constraint_removal else "Processing parameter combinations"
    phase1_results = []
    with tqdm(total=n_combinations, desc=desc_phase1, unit="combo", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for args in phase1_args:
            result = _process_single_parameter_combination(*args)
            phase1_results.append(result)
            pbar.update(1)
    
    # Process Phase 1 results and write to CSV
    # Track which quality indices have had their header written
    header_written = {quality_index: False for quality_index in target_quality_indices}
    
    for idx, (success, combo_id, results) in enumerate(phase1_results):
        if success:
            # Check if results is empty (scenario was skipped)
            if not results:
                print(f"⊘ Skipped {combo_id}: scenario not found in synthetic CSV")
                continue
            
            # Write results to CSV files
            for quality_index in target_quality_indices:
                if quality_index in results:
                    fit_params = results[quality_index]
                    master_csv_filename = output_csv_filenames[quality_index]
                    
                    df_single = pd.DataFrame([fit_params])
                    # Write header only for the first result for each quality index
                    if not header_written[quality_index]:
                        df_single.to_csv(master_csv_filename, mode='w', index=False, header=True)
                        header_written[quality_index] = True
                    else:
                        df_single.to_csv(master_csv_filename, mode='a', index=False, header=False)
                    del df_single
        else:
            print(f"✗ Error in {combo_id}: {results}")

    print("✓ All original parameter combinations processed" if test_age_constraint_removal else "✓ All parameter combinations processed")

    # PHASE 2: Run age constraint removal scenarios (if enabled)
    # Warning for large number of scenarios
    if test_age_constraint_removal:
        print("\n=== PHASE 2: Running age constraint removal scenarios ===")
        
        # Calculate additional scenarios
        n_constraints_b = len(core_b_age_data['in_sequence_ages'])
        age_enabled_params = [p for p in parameter_combinations if p['age_consideration']]
        
        # Check if there are any valid age constraints and age-enabled parameters
        if n_constraints_b == 0 or len(age_enabled_params) == 0:
            print("✓ Phase 2 completed: No age constraint removal scenarios to process")
        else:
            constraint_scenarios_per_param = (2 ** n_constraints_b) - 1  # Exclude empty set
            total_additional_scenarios = len(age_enabled_params) * (constraint_scenarios_per_param - 1)  # Exclude original scenario
           
            print(f"- Core B has {n_constraints_b} age constraints")
            print(f"- Processing {total_additional_scenarios} additional constraint removal scenarios")
            
            # Get indices of only in-sequence constraints from the original data
            in_sequence_indices = []
            for i, flag in enumerate(core_b_age_data['in_sequence_flags']):
                # Handle various flag formats (string, boolean, numeric)
                is_in_sequence = False
                if isinstance(flag, str):
                    is_in_sequence = flag.upper() == 'TRUE'
                elif isinstance(flag, (bool, np.bool_)):
                    is_in_sequence = flag
                else:
                    is_in_sequence = flag == 1
                
                if is_in_sequence:
                    in_sequence_indices.append(i)
            
            n_constraints_b = len(in_sequence_indices)  # Count of in-sequence constraints only

            # Generate subsets from in-sequence constraint indices only
            all_subsets = generate_constraint_subsets(n_constraints_b)
            constraint_subsets = [subset for subset in all_subsets if 0 < len(subset) < n_constraints_b]
            
            # Apply max_search_per_layer limitation if specified
            if max_search_per_layer is not None:
                # Group constraint subsets by layer (number of remaining constraints)
                print(f"max_search_per_layer is defined: randomly sampling up to {max_search_per_layer} scenarios per layer of search")
                layers = {}
                for subset in constraint_subsets:
                    layer_size = len(subset)
                    if layer_size not in layers:
                        layers[layer_size] = []
                    layers[layer_size].append(subset)
                
                # Sample from each layer if it exceeds max_search_per_layer
                limited_constraint_subsets = []
                for layer_size in sorted(layers.keys()):
                    layer_subsets = layers[layer_size]
                    if len(layer_subsets) > max_search_per_layer:
                        # Use numpy random to avoid conflicts and ensure no repeats within layer
                        layer_indices = np.arange(len(layer_subsets))
                        sampled_indices = np.random.choice(layer_indices, size=max_search_per_layer, replace=False)
                        sampled_subsets = [layer_subsets[i] for i in sampled_indices]
                        
                        limited_constraint_subsets.extend(sampled_subsets)
                        print(f"- Layer {layer_size} constraints: {len(layer_subsets)} scenarios → {max_search_per_layer} sampled")
                    else:
                        limited_constraint_subsets.extend(layer_subsets)
                        print(f"- Layer {layer_size} constraints: {len(layer_subsets)} scenarios (all processed)")
                
                constraint_subsets = limited_constraint_subsets
            
            # Pre-cache synthetic CSV bin information to avoid redundant reads in parallel workers
            cached_binsizes = {}
            if synthetic_csv_filenames:
                for quality_index in target_quality_indices:
                    if quality_index in synthetic_csv_filenames:
                        synthetic_csv_file = synthetic_csv_filenames[quality_index]
                        if os.path.exists(synthetic_csv_file):
                            try:
                                synthetic_df = pd.read_csv(synthetic_csv_file)
                                if not synthetic_df.empty and 'bins' in synthetic_df.columns:
                                    bins_str = synthetic_df.iloc[0]['bins']
                                    if pd.notna(bins_str):
                                        synthetic_bins = np.fromstring(bins_str.strip('[]'), sep=' ')
                                        bin_width_synthetic = np.mean(np.diff(synthetic_bins))
                                        cached_binsizes[quality_index] = (synthetic_bins, bin_width_synthetic)
                            except Exception:
                                pass
            
            # Prepare data for Phase 2 parallel processing
            phase2_args = []
            for param_idx, params in enumerate(age_enabled_params):
                for constraint_subset in constraint_subsets:
                    phase2_args.append((
                        param_idx, params, constraint_subset, in_sequence_indices,
                        log_a, log_b, md_a, md_b,
                        picked_datum_a, picked_datum_b,
                        datum_ages_a, datum_ages_b,
                        core_a_age_data, core_b_age_data,
                        uncertainty_method,
                        target_quality_indices,
                        cached_binsizes,
                        pca_for_dependent_dtw,
                        output_metric_only
                    ))
            
            # Run Phase 2 sequentially with progress bar showing elapsed time
            print(f"Processing {len(phase2_args)} scenarios sequentially (parallel inside each scenario)")
            
            phase2_results = []
            with tqdm(total=len(phase2_args), desc="Phase 2: Age constraint removal", unit="scenario",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for args in phase2_args:
                    result = _process_single_constraint_scenario(*args)
                    phase2_results.append(result)
                    pbar.update(1)
            
            # Process Phase 2 results and append to CSV
            for success, scenario_id, results in phase2_results:
                if success:
                    # Append results to CSV files
                    for quality_index in target_quality_indices:
                        if quality_index in results:
                            fit_params = results[quality_index]
                            master_csv_filename = output_csv_filenames[quality_index]
                            
                            df_single = pd.DataFrame([fit_params])
                            df_single.to_csv(master_csv_filename, mode='a', index=False, header=False)
                            del df_single
                else:
                    print(f"✗ Error in {scenario_id}: {results}")

            print("✓ Phase 2 completed: All age constraint removal scenarios processed")

    # Final summary
    print(f"\n✓ All processing completed")
    
    for quality_index in target_quality_indices:
        filename = output_csv_filenames[quality_index]
        print(f"✓ {quality_index} fit_params saved to: {filename}")


def create_synthetic_core_pair(core_a_length, core_b_length, seg_logs, seg_depths, 
                                       log_columns, repetition=False, plot_results=True, save_plot=False, plot_filename=None):
    """
    Generate synthetic core pair (computation only).
    
    Parameters:
    - core_a_length: target length for core A
    - core_b_length: target length for core B
    - seg_logs: list of turbidite log segments
    - seg_depths: list of corresponding depth arrays
    - log_columns: list of log column names for labeling
    - repetition: if True, allow reusing turbidite segments; if False, each segment can only be used once (default: False)
    - plot_results: whether to display the plot
    - save_plot: whether to save the plot to file
    - plot_filename: filename for saving plot (if save_plot=True)
    
    Returns:
    - tuple: (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a,
              synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)
    """
    
    # Generate synthetic logs for cores A and B
    print("Generating synthetic core pair...")

    synthetic_log_a, synthetic_md_a, synthetic_picked_a, inds_a = create_synthetic_log(
        core_a_length, seg_logs, seg_depths, exclude_inds=None, repetition=repetition
    )
    synthetic_log_b, synthetic_md_b, synthetic_picked_b, inds_b = create_synthetic_log(
        core_b_length, seg_logs, seg_depths, exclude_inds=None, repetition=repetition
    )

    # Plot synthetic core pair if requested
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 9))
        
        # Define colors for different log types
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        line_styles = ['-', '-', '-', '-.', '-.', '-.', ':', ':', ':']

        # Plot synthetic core A
        if synthetic_log_a.ndim > 1:
            n_log_types = synthetic_log_a.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_columns[col_idx] if col_idx < len(log_columns) else f'Log_{col_idx}'
                
                ax1.plot(synthetic_log_a[:, col_idx], synthetic_md_a, 
                        color=color, linestyle=line_style, linewidth=1, 
                        label=col_name, alpha=0.8)
            
            # Add legend if multiple log types
            if n_log_types > 1:
                ax1.legend(fontsize=8, loc='upper right')
                
            # Set xlabel to show all log types
            if len(log_columns) > 1:
                ax1.set_xlabel(f'Multiple Logs (normalized)')
            else:
                ax1.set_xlabel(f'{log_columns[0]} (normalized)')
        else:
            ax1.plot(synthetic_log_a, synthetic_md_a, 'b-', linewidth=1)
            ax1.set_xlabel(f'{log_columns[0]} (normalized)')

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_a:
            ax1.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

        ax1.set_ylabel('Depth (cm)')
        ax1.set_title(f'Synthetic Core A\n({len(inds_a)} turbidites)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Plot synthetic core B
        if synthetic_log_b.ndim > 1:
            n_log_types = synthetic_log_b.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_columns[col_idx] if col_idx < len(log_columns) else f'Log_{col_idx}'
                
                ax2.plot(synthetic_log_b[:, col_idx], synthetic_md_b, 
                        color=color, linestyle=line_style, linewidth=1, 
                        label=col_name, alpha=0.8)
            
            # Add legend if multiple log types
            if n_log_types > 1:
                ax2.legend(fontsize=8, loc='upper right')
                
            # Set xlabel to show all log types
            if len(log_columns) > 1:
                ax2.set_xlabel(f'Multiple Logs (normalized)')
            else:
                ax2.set_xlabel(f'{log_columns[0]} (normalized)')
        else:
            ax2.plot(synthetic_log_b, synthetic_md_b, 'g-', linewidth=1)
            ax2.set_xlabel(f'{log_columns[0]} (normalized)')

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_b:
            ax2.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

        ax2.set_ylabel('Depth (cm)')
        ax2.set_title(f'Synthetic Core B\n({len(inds_b)} turbidites)')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plot and plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.show()

    print(f"Synthetic Core A: {len(synthetic_log_a)} points, {len(inds_a)} turbidites, {len(synthetic_picked_a)} boundaries")
    print(f"Synthetic Core B: {len(synthetic_log_b)} points, {len(inds_b)} turbidites, {len(synthetic_picked_b)} boundaries")
    print(f"Turbidite indices used in A: {[int(x) for x in inds_a[:10]]}..." if len(inds_a) > 10 else f"Turbidite indices used in A: {[int(x) for x in inds_a]}")
    print(f"Turbidite indices used in B: {[int(x) for x in inds_b[:10]]}..." if len(inds_b) > 10 else f"Turbidite indices used in B: {[int(x) for x in inds_b]}")
    
    return (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a,
            synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)


def plot_segment_pool(segment_logs, segment_depths, log_data_type, n_cols=8, figsize_per_row=4, 
                     plot_segments=True, save_plot=False, plot_filename=None):
    """
    Plot all segments from the pool in a grid layout.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - log_data_type: list of column names for labeling
    - n_cols: number of columns in the subplot grid
    - figsize_per_row: height per row in the figure
    - plot_segments: whether to plot the segments (default True)
    - save_plot: whether to save the plot to file (default False)
    - plot_filename: filename for saving plot (optional)
    
    Returns:
    - None
    """
    print(f"Plotting {len(segment_logs)} segments from the pool...")
    
    if not plot_segments:
        return
    
    # Create subplot grid
    n_segments = len(segment_logs)
    n_rows = int(np.ceil(n_segments / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, figsize_per_row * n_rows))
    axes = axes.flatten() if n_segments > 1 else [axes]
    
    # Define colors for different log types
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    line_styles = ['-', '-', '-', '-.', '-.', '-.', ':', ':', ':']
    
    for i, (segment, depth) in enumerate(zip(segment_logs, segment_depths)):
        ax = axes[i]
        
        # Plot segment
        if segment.ndim > 1:
            # Multi-dimensional data - plot all columns
            n_log_types = segment.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_data_type[col_idx] if col_idx < len(log_data_type) else f'Log_{col_idx}'
                
                ax.plot(segment[:, col_idx], depth, 
                       color=color, linestyle=line_style, linewidth=1, 
                       label=col_name, alpha=0.8)
            
            # Set xlabel to show all log types
            if len(log_data_type) > 1:
                ax.set_xlabel(f'Multiple Logs: {", ".join(log_data_type[:n_log_types])} (normalized)')
            else:
                ax.set_xlabel(f'{log_data_type[0]} (normalized)')
                
            # Add legend if multiple log types
            if n_log_types > 1:
                ax.legend(fontsize=8, loc='best')
                
        else:
            # 1D data
            ax.plot(segment, depth, 'b-', linewidth=1)
            ax.set_xlabel(f'{log_data_type[0]} (normalized)')
        
        ax.set_ylabel('Relative Depth (cm)')
        ax.set_title(f'Segment {i+1}\n({len(segment)} pts, {depth[-1]:.1f} cm)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Depth increases downward
    
    # Hide unused subplots
    for i in range(n_segments, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Update title to reflect multiple log types if present
    if len(log_data_type) > 1:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments, {len(log_data_type)} log types)', 
                     y=1.02, fontsize=16)
    else:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments)', y=1.02, fontsize=16)
    
    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()


def plot_synthetic_log(synthetic_log, synthetic_md, synthetic_picked_datum, log_data_type, 
                      title="Synthetic Log", save_plot=False, plot_filename=None):
    """
    Plot a single synthetic log with turbidite boundaries.

    Parameters:
    - synthetic_log: numpy array of log values (can be 1D or 2D for multiple log types)
    - synthetic_md: numpy array of depth values
    - synthetic_picked_datum: list of turbidite boundary depths
    - log_data_type: name(s) of the log column(s) for labeling (string or list)
    - title: title for the plot (default: "Synthetic Log")
    - save_plot: whether to save the plot to file (default: False)
    - plot_filename: filename for saving plot (if save_plot=True)

    Returns:
    - fig, ax: matplotlib figure and axis objects
    """

    # Convert log_data_type to list if it's a string
    if isinstance(log_data_type, str):
        log_data_type = [log_data_type]

    # Increase width for legend outside plot
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 8))

    # Define colors and line styles for different log types (matching old code)
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    line_styles = ['-', ':', '--', '-.', '-', ':', '--', '-.']

    # Plot the synthetic log
    legend_handle = None
    if synthetic_log.ndim > 1:
        # Multi-dimensional data - plot all columns
        n_log_types = synthetic_log.shape[1]

        for col_idx in range(n_log_types):
            color = colors[col_idx % len(colors)]
            line_style = line_styles[col_idx % len(line_styles)]

            # Get column name for label
            col_name = log_data_type[col_idx] if col_idx < len(log_data_type) else f'Log_{col_idx}'

            ax.plot(synthetic_log[:, col_idx], synthetic_md,
                    color=color, linestyle=line_style, linewidth=1,
                    label=col_name, alpha=0.8)

        # Add legend if multiple log types, outside the plot at best position
        if n_log_types > 1:
            # Shrink current axis by 20% on the right to fit the legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend_handle = ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Set xlabel to show all log types
        if len(log_data_type) > 1:
            ax.set_xlabel(f'Multiple Logs\n(Normalized)')
        else:
            ax.set_xlabel(f'{log_data_type[0]}\n(normalized)')
    else:
        # 1D data
        ax.plot(synthetic_log, synthetic_md, 'b-', linewidth=1)
        ax.set_xlabel(f'{log_data_type[0]} (normalized)')

    # Add picked depths as horizontal lines
    for depth in synthetic_picked_datum:
        ax.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

    ax.set_ylabel('Depth (cm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_plot and plot_filename:
        # If legend exists, pass it to savefig so it is not cut off
        if legend_handle is not None:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend_handle,))
        else:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")

    plt.show()

    return fig, ax

def plot_synthetic_correlation_quality(
    input_csv, 
    quality_indices=['corr_coef', 'norm_dtw'],
    bin_width=None,
    plot_individual_pdf=False,
    save_plot=False,
    plot_filename=None
):
    """
    Plot synthetic correlation quality distributions from saved CSV files.
    
    This function reads fit parameters from CSV files generated by synthetic_correlation_quality
    and creates visualization plots showing either individual PDFs from each iteration or a 
    combined distribution.
    
    Parameters:
    -----------
    input_csv : str
        Path to the CSV file containing fit parameters. Can include {quality_index} placeholder
        which will be replaced for each quality index.
        Example: 'outputs/synthetic_PDFs_hiresMS_CT_Lumin_{quality_index}.csv'
    quality_indices : list, default=['corr_coef', 'norm_dtw']
        List of quality indices to plot
    bin_width : float or None, default=None
        Bin width for histogram. If None, uses quality-specific defaults:
        - corr_coef: 0.025
        - norm_dtw: 0.0025
        - others: auto-determined using Freedman-Diaconis rule
    plot_individual_pdf : bool, default=False
        If True, plots all individual iteration PDFs overlaid (following Cell 9 approach)
        If False, plots combined distribution across all iterations (following Cell 10 approach)
    save_plot : bool, default=False
        Whether to save the plot to file
    plot_filename : str or None, default=None
        Filename for saving plot. Can include {quality_index} placeholder.
        If save_plot=True but plot_filename=None, uses default naming based on plot type
    
    Returns:
    --------
    None
        Displays and optionally saves plots
    """
    
    from scipy import stats
    
    for targeted_quality_index in quality_indices:
        print(f"\nPlotting distribution for {targeted_quality_index}...")
        
        # Replace placeholder in input_csv with actual quality index
        input_csv_filename = input_csv.replace('{quality_index}', targeted_quality_index).replace('{targeted_quality_index}', targeted_quality_index)
        
        # Check if file exists
        if not os.path.exists(input_csv_filename):
            print(f"Error: File {input_csv_filename} does not exist. Skipping {targeted_quality_index}.")
            continue
        
        # Load fit params from CSV
        df_fit_params = pd.read_csv(input_csv_filename)
        
        # Reconstruct raw data for each iteration
        all_fit_params = []
        for _, row in df_fit_params.iterrows():
            # Extract binned data for reconstruction
            bins = np.fromstring(row['bins'].strip('[]'), sep=' ') if 'bins' in row and pd.notna(row['bins']) else None
            hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ') if 'hist' in row and pd.notna(row['hist']) else None
            n_points = row['n_points'] if 'n_points' in row and pd.notna(row['n_points']) else None
            
            # Reconstruct raw data from histogram
            raw_data = []
            if bins is not None and hist_percentages is not None and n_points is not None:
                # Normalize histogram to get proper proportions
                hist_sum = np.sum(hist_percentages)
                if hist_sum > 0:
                    raw_counts = (hist_percentages / hist_sum) * n_points
                else:
                    raw_counts = np.zeros_like(hist_percentages)
                
                # Reconstruct data points by sampling from each bin
                for i, count in enumerate(raw_counts):
                    if count > 0:
                        n_samples = int(round(count))
                        if n_samples > 0:
                            bin_samples = np.random.uniform(bins[i], bins[i+1], n_samples)
                            raw_data.extend(bin_samples)
            
            # Store fit params with reconstructed raw data
            fit_params = {
                'raw_data': np.array(raw_data),
                'bins': bins,
                'mean': row['mean'] if 'mean' in row else None,
                'std': row['std'] if 'std' in row else None
            }
            all_fit_params.append(fit_params)
        
        # Determine bin width
        if bin_width is None:
            if targeted_quality_index == 'corr_coef':
                current_bin_width = 0.025
            elif targeted_quality_index == 'norm_dtw':
                current_bin_width = 0.0025
            else:
                # Use bin_width from first iteration as fallback
                if len(all_fit_params) > 0 and all_fit_params[0]['bins'] is not None:
                    first_bins = all_fit_params[0]['bins']
                    current_bin_width = first_bins[1] - first_bins[0]
                else:
                    current_bin_width = 0.025  # Default fallback
        else:
            current_bin_width = bin_width
        
        if plot_individual_pdf:
            # ===== CELL 9 APPROACH: Plot individual PDFs =====
            
            # Find data range across ALL iterations to create consistent bins
            all_min = min([fp['raw_data'].min() for fp in all_fit_params if len(fp['raw_data']) > 0])
            all_max = max([fp['raw_data'].max() for fp in all_fit_params if len(fp['raw_data']) > 0])
            
            # Create explicit bin edges with consistent bin width
            bin_start = np.floor(all_min / current_bin_width) * current_bin_width
            bin_end = np.ceil(all_max / current_bin_width) * current_bin_width
            consistent_bins = np.arange(bin_start, bin_end + current_bin_width, current_bin_width)
            
            # Plot all distribution curves and histogram bars
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Plot histogram bars for each iteration
            for fit_params in all_fit_params:
                raw_data = fit_params.get('raw_data')
                if raw_data is not None and len(raw_data) > 0:
                    ax.hist(raw_data, bins=consistent_bins, alpha=0.1, color='gray', 
                            density=False, edgecolor='none',
                            weights=np.ones(len(raw_data)) * 100 / len(raw_data))
            
            # Plot all PDF curves as transparent red lines
            for fit_params in all_fit_params:
                mean_val = fit_params.get('mean')
                std_val = fit_params.get('std')
                raw_data = fit_params.get('raw_data')
                
                if mean_val is not None and std_val is not None and raw_data is not None and len(raw_data) > 0:
                    # Generate PDF curve with proper scaling
                    x_min = raw_data.min()
                    x_max = raw_data.max()
                    x = np.linspace(x_min, x_max, 1000)
                    # Scale PDF by bin_width * 100 to match histogram percentage scale
                    y = stats.norm.pdf(x, mean_val, std_val) * current_bin_width * 100
                    ax.plot(x, y, 'r-', linewidth=.7, alpha=0.3)
            
            # Formatting based on quality index
            if targeted_quality_index == 'corr_coef':
                ax.set_xlabel("Pearson's r\n(Correlation Coefficient)")
                ax.set_xlim(0, 1.0)
            elif targeted_quality_index == 'norm_dtw':
                ax.set_xlabel("Normalized DTW Distance")
            elif targeted_quality_index == 'dtw_ratio':
                ax.set_xlabel("DTW Ratio")
            elif targeted_quality_index == 'perc_diag':
                ax.set_xlabel("Percentage Diagonal (%)")
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Synthetic Core {targeted_quality_index.replace("_", " ").title()}: {len(all_fit_params)} Iterations\n[Optimal (shortest path) search; no age consideration)]')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                if plot_filename:
                    output_filename = plot_filename.replace('{quality_index}', targeted_quality_index).replace('{targeted_quality_index}', targeted_quality_index)
                else:
                    output_filename = f'synthetic_iterations_{targeted_quality_index}.png'
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {output_filename}")
            
            plt.show()
            
        else:
            # ===== CELL 10 APPROACH: Combined distribution =====
            
            # Initialize lists to collect all raw data points
            all_raw_data = []
            
            # Process each iteration to reconstruct raw data from binned data
            for fit_params in all_fit_params:
                raw_data = fit_params.get('raw_data')
                if raw_data is not None and len(raw_data) > 0:
                    all_raw_data.extend(raw_data)
            
            # Convert to numpy array
            combined_data = np.array(all_raw_data)
            
            print(f"Combined {len(combined_data)} data points from {len(df_fit_params)} iterations")
            
            # Calculate combined statistics
            combined_mean = np.mean(combined_data)
            combined_std = np.std(combined_data)
            combined_median = np.median(combined_data)
            
            # Determine bin width for combined data
            if bin_width is None:
                if targeted_quality_index == 'corr_coef':
                    final_bin_width = 0.025
                elif targeted_quality_index == 'norm_dtw':
                    final_bin_width = 0.0025
                else:
                    # Automatically determine bin width using Freedman-Diaconis rule
                    data_range = combined_data.max() - combined_data.min()
                    if data_range > 0:
                        if len(combined_data) > 1:
                            q75, q25 = np.percentile(combined_data, [75, 25])
                            iqr = q75 - q25
                            if iqr > 0:
                                final_bin_width = 2 * iqr / (len(combined_data) ** (1/3))
                            else:
                                # Fallback to simple rule if IQR is 0
                                final_bin_width = data_range / max(10, min(int(np.sqrt(len(combined_data))), 100))
                        else:
                            final_bin_width = 0.1  # Default for single value
                    else:
                        final_bin_width = 0.1  # Default for zero range
            else:
                final_bin_width = bin_width
            
            # Calculate number of bins based on bin_width
            data_range = combined_data.max() - combined_data.min()
            if data_range > 0 and final_bin_width > 0:
                n_bins = max(1, int(np.ceil(data_range / final_bin_width)))
                # Constrain to reasonable range
                n_bins = max(10, min(n_bins, 200))
            else:
                n_bins = 10  # Default fallback
            
            # Create new histogram from combined data as PERCENTAGES
            hist_combined, bins_combined = np.histogram(combined_data, bins=n_bins, density=False)
            # Convert counts to percentages
            hist_percentages = (hist_combined / len(combined_data)) * 100
            actual_bin_width = bins_combined[1] - bins_combined[0]
            
            # Fit normal distribution to combined data
            fitted_mean, fitted_std = stats.norm.fit(combined_data)
            
            # Generate fitted curve and scale to percentage
            x_fitted = np.linspace(combined_data.min(), combined_data.max(), 1000)
            # PDF must be scaled by: actual_bin_width * 100 (to convert density to percentage per bin)
            y_fitted = stats.norm.pdf(x_fitted, fitted_mean, fitted_std) * actual_bin_width * 100
            
            # Verify histogram normalization
            total_percentage = np.sum(hist_percentages)
            print(f"Histogram total percentage: {total_percentage:.2f}% (should be 100%)")
            print(f"Number of bins used: {n_bins}, Bin width: {actual_bin_width:.6f}")
            
            # Verify PDF curve integration
            dx = x_fitted[1] - x_fitted[0]
            pdf_area = np.trapz(y_fitted, dx=dx)
            print(f"PDF curve area: {pdf_area:.2f}% (should be ~100%)")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Plot combined histogram in gray bars as PERCENTAGES
            ax.hist(combined_data, bins=n_bins, alpha=0.5, color='gray', density=False, 
                    weights=np.ones(len(combined_data)) * 100 / len(combined_data), 
                    label=f'Combined Data (n = {len(combined_data):,})')
            
            # Plot fitted normal curve as red line
            ax.plot(x_fitted, y_fitted, 'r-', linewidth=2, alpha=0.8,
                    label=f'Normal Fit (μ={fitted_mean:.3f}, σ={fitted_std:.3f})')
            
            # Formatting based on quality index
            if targeted_quality_index == 'corr_coef':
                ax.set_xlabel("Pearson's r\n(Correlation Coefficient)")
                ax.set_xlim(0, 1.0)
            elif targeted_quality_index == 'norm_dtw':
                ax.set_xlabel("Normalized DTW Distance")
            elif targeted_quality_index == 'dtw_ratio':
                ax.set_xlabel("DTW Ratio")
            elif targeted_quality_index == 'perc_diag':
                ax.set_xlabel("Percentage Diagonal (%)")
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Combined {targeted_quality_index.replace("_", " ").title()} Distribution from All {len(df_fit_params)} Iterations\n[Synthetic Core Analysis - Null Hypothesis]')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_plot:
                if plot_filename:
                    output_filename = plot_filename.replace('{quality_index}', targeted_quality_index).replace('{targeted_quality_index}', targeted_quality_index)
                else:
                    output_filename = f'combined_synthetic_distribution_{targeted_quality_index}.png'
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {output_filename}")
            
            plt.show()
            
            # Print comprehensive summary
            print(f"\nCombined Distribution Summary for {targeted_quality_index}:")
            print(f"{'='*50}")
            print(f"Total data points: {len(combined_data):,}")
            print(f"Number of iterations combined: {len(df_fit_params)}")
            print(f"Combined Mean: {combined_mean:.4f}")
            print(f"Combined Median: {combined_median:.4f}")
            print(f"Combined Std Dev: {combined_std:.4f}")
            print(f"Data Range: {combined_data.min():.4f} to {combined_data.max():.4f}")
            print(f"\nFitted Normal Distribution:")
            print(f"Fitted Mean (μ): {fitted_mean:.4f}")
            print(f"Fitted Std Dev (σ): {fitted_std:.4f}")
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            pct_values = np.percentile(combined_data, percentiles)
            print(f"\nPercentiles:")
            for pct, val in zip(percentiles, pct_values):
                print(f"{pct}th percentile: {val:.4f}")


def synthetic_correlation_quality(
    segment_logs, 
    segment_depths, 
    log_data_type, 
    quality_indices=['corr_coef', 'norm_dtw'], 
    number_of_iterations=20, 
    core_a_length=400, 
    core_b_length=400,
    repetition=False, 
    pca_for_dependent_dtw=False, 
    output_csv_dir=None,
    max_search_path=10000,
    mute_mode=True,
    append_mode=False,
    combination_id=None,
    max_paths_for_metrics=None,
    n_jobs=-1
):
    """
    Run DTW correlation quality measurement analysis for synthetic core pairs over multiple iterations.

    This function generates synthetic core pairs, computes selected correlation quality metrics (such as DTW and correlation coefficient) for each pair, and saves the resulting quality distributions for later plotting and assessment. Distribution parameters for each quality metric are saved across all iterations.

    Parameters
    ----------
    segment_logs : list
        List of turbidite log segments (numpy arrays or compatible).
    segment_depths : list
        List of turbidite depth segments (numpy arrays or compatible).
    log_data_type : list of str
        List of log column names to consider in the analysis.
    quality_indices : list of str, default ['corr_coef', 'norm_dtw']
        List of quality indices (metrics) to compute for each synthetic pair.
    number_of_iterations : int, default 20
        Number of synthetic core pairs to generate/analyze.
    core_a_length : int, default 400
        Target length (in cm) for synthetic core A.
    core_b_length : int, default 400
        Target length (in cm) for synthetic core B.
    repetition : bool, default False
        Whether to allow resampling (reuse) of turbidite segments when assembling synthetic cores.
    pca_for_dependent_dtw : bool, default False
        If True, applies PCA transformation to logs for dependent DTW analysis.
    output_csv_dir : str or None, default None
        Output directory path for saving quality metric CSV files.
        If None, files are saved in the current directory.
        If provided, directory is created if it does not exist.
        Output files will be named 'synthetic_PDFs_{log_columns}_{quality_index}.csv'.
    max_search_path : int, default 10000
        Maximum allowable search path for the DTW algorithm.
    mute_mode : bool, default True
        If True, suppresses detailed informational output.
    append_mode : bool, default False
        If True, appends results to existing CSV files instead of overwriting.
        Useful when running multiple combinations of core lengths.
    combination_id : int or None, default None
        Optional identifier for the current combination of core lengths.
        If provided, will be saved in the output CSV.
    max_paths_for_metrics : int or None, default None
        Maximum paths to compute metrics for. If total paths exceed this, a random
        sample is used. Useful for distribution fitting where full enumeration is
        unnecessary. If None, uses max_search_path value (no additional sampling).
    n_jobs : int, default -1
        Number of parallel jobs for metric computation. -1 uses all available cores.

    Returns
    -------
    dict
        Mapping from each quality index to its corresponding output CSV filename.
    """
    
    # Create output directory if specified
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
    
    # Prepare output filenames
    output_files = {}
    for targeted_quality_index in quality_indices:
        # Construct filename
        filename = f'synthetic_PDFs_{"_".join(log_data_type)}_{targeted_quality_index}.csv'
        if output_csv_dir:
            output_files[targeted_quality_index] = os.path.join(output_csv_dir, filename)
        else:
            output_files[targeted_quality_index] = filename
    
    # Define temp CSV path
    temp_filename = f'temp_synthetic_{"_".join(log_data_type)}_core_pair_metrics.csv'
    if output_csv_dir:
        temp_csv = os.path.join(output_csv_dir, temp_filename)
    else:
        temp_csv = temp_filename
    
    print(f"Starting synthetic correlation analysis with {number_of_iterations} iterations...")
    print(f"Quality indices: {quality_indices}")
    print(f"Core lengths: A={core_a_length}cm, B={core_b_length}cm")
    
    # Run iterations with progress bar
    for iteration in tqdm(range(number_of_iterations), desc="Running synthetic analysis"):
        
        # Generate synthetic core pair
        syn_log_a, syn_md_a, syn_picked_a, inds_a = create_synthetic_log(
            target_thickness=core_a_length, 
            segment_logs=segment_logs, 
            segment_depths=segment_depths, 
            exclude_inds=None, 
            repetition=repetition
        )
        syn_log_b, syn_md_b, syn_picked_b, inds_b = create_synthetic_log(
            target_thickness=core_b_length, 
            segment_logs=segment_logs, 
            segment_depths=segment_depths, 
            exclude_inds=None, 
            repetition=repetition
        )
        
        # Run DTW analysis (syn_picked_a and syn_picked_b are already just depth values)
        dtw_result = run_comprehensive_dtw_analysis(
            syn_log_a, syn_log_b, syn_md_a, syn_md_b,
            picked_datum_a=syn_picked_a,
            picked_datum_b=syn_picked_b,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=False,
            mute_mode=mute_mode
        )
        
        # Find complete core paths with optimizations:
        # - metrics_to_compute: Only compute the quality indices we need
        # - max_paths_for_metrics: Sample paths if too many (for distribution fitting)
        # - return_dataframe: Get results in memory (no file I/O)
        # - n_jobs: Parallel metric computation
        path_result = find_complete_core_paths(
            dtw_result,
            syn_log_a, 
            syn_log_b,
            output_csv=temp_csv,
            output_metric_only=True,
            shortest_path_search=True,
            shortest_path_level=2,
            max_search_path=max_search_path,
            mute_mode=mute_mode,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            metrics_to_compute=quality_indices,
            max_paths_for_metrics=max_paths_for_metrics,
            return_dataframe=True,
            n_jobs=n_jobs
        )
        
        # Get metrics DataFrame directly from result (Solution 2: no file I/O)
        metrics_df = path_result.get('metrics_dataframe')
        
        # Iterate through each quality index to extract fit_params
        for targeted_quality_index in quality_indices:
            
            output_csv_filename = output_files[targeted_quality_index]

            # Compute distribution params directly from DataFrame (no file read)
            if metrics_df is not None and targeted_quality_index in metrics_df.columns:
                quality_values = metrics_df[targeted_quality_index].dropna().values
                
                if len(quality_values) > 0:
                    # Compute normal distribution fit params directly
                    mean_val = float(np.mean(quality_values))
                    std_val = float(np.std(quality_values))
                    fit_params = {
                        'mean': mean_val,
                        'std': std_val,
                        'count': len(quality_values),
                        'min': float(np.min(quality_values)),
                        'max': float(np.max(quality_values))
                    }
                else:
                    fit_params = None
            else:
                fit_params = None
            
            # Store fit_params with iteration number and incrementally save to CSV
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                fit_params_copy['iteration'] = iteration
                fit_params_copy['core_a_length'] = core_a_length
                fit_params_copy['core_b_length'] = core_b_length
                if combination_id is not None:
                    fit_params_copy['combination_id'] = combination_id
                
                # Incrementally save to CSV
                df_single = pd.DataFrame([fit_params_copy])
                
                # Determine write mode based on append_mode and iteration
                if append_mode:
                    # In append mode, check if file exists to decide on header
                    file_exists = os.path.exists(output_csv_filename)
                    df_single.to_csv(output_csv_filename, mode='a', index=False, header=not file_exists)
                else:
                    if iteration == 0:
                        # Write header for first iteration
                        df_single.to_csv(output_csv_filename, mode='w', index=False, header=True)
                    else:
                        # Append subsequent iterations without header
                        df_single.to_csv(output_csv_filename, mode='a', index=False, header=False)
                
                del df_single, fit_params_copy
            
            del fit_params
        
        # Clear memory after each iteration
        del syn_log_a, syn_md_a, inds_a, syn_picked_a
        del syn_log_b, syn_md_b, inds_b, syn_picked_b
        del dtw_result, path_result, metrics_df
        
        gc.collect()

    # Remove temporary CSV file if it exists (for backward compatibility)
    # Note: With return_dataframe=True, no temp file is created
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print(f"\nCompleted {number_of_iterations} iterations for all quality indices: {quality_indices}")
    
    for targeted_quality_index in quality_indices:
        print(f"Distribution parameters for {targeted_quality_index} saved to: {output_files[targeted_quality_index]}")
