"""
Data loading functions for pyCoreRelator.

Included Functions:
- load_log_data: Load log data from CSV files and resample to common depth scale
- resample_datasets: Resample multiple datasets to a common depth scale
- load_age_constraints_from_csv: Load age constraints from a single CSV file
- combine_age_constraints: Combine multiple age constraint dictionaries
- load_core_age_constraints: Helper function to load age constraints for a single core

This module provides utilities for loading core log data from CSV files.
It handles multiple log types, data normalization, and resampling to common depth scales.
Compatible with both original and ML-processed/gap-filled core data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats


def load_log_data(log_paths, img_paths=None, log_columns=None, depth_column='SB_DEPTH_cm', normalize=True, column_alternatives=None):
    """
    Load log data and images for a core.
    
    This function loads multiple log datasets from CSV files, resamples them to a common
    depth scale, and optionally loads RGB and CT images. It supports alternative column
    names and automatic data normalization.
    
    Parameters
    ----------
    log_paths : dict
        Dictionary mapping log names to file paths
    img_paths : dict, optional
        Dictionary mapping image types ('rgb', 'ct') to file paths
    log_columns : list of str
        List of log column names to load from the CSV files
    depth_column : str, default='SB_DEPTH_cm'
        Name of the depth column in the CSV files
    normalize : bool, default=True
        Whether to normalize each log to the range [0, 1]
    column_alternatives : dict, optional
        Dictionary mapping log column names to lists of alternative column names
        to try if the primary column name is not found
    
    Returns
    -------
    log : numpy.ndarray
        Log data with shape (n_samples, n_logs) for multiple logs or (n_samples,) for single log
    md : numpy.ndarray
        Measured depths array
    available_columns : list of str
        Names of the logs that were successfully loaded
    rgb_img : numpy.ndarray or None
        RGB image array if available, None otherwise
    ct_img : numpy.ndarray or None
        CT image array if available, None otherwise
    
    Example
    -------
    >>> log_paths = {'MS': 'data/core1_ms.csv', 'Lumin': 'data/core1_lumin.csv'}
    >>> img_paths = {'rgb': 'data/core1_rgb.jpg', 'ct': 'data/core1_ct.jpg'}
    >>> log_columns = ['MS', 'Lumin']
    >>> log, md, cols, rgb_img, ct_img = load_log_data(log_paths, img_paths, log_columns)
    """
    # Check if log_paths is provided
    if log_paths is None:
        raise ValueError("log_paths must be provided")
    
    # Check if log_columns is provided
    if log_columns is None:
        raise ValueError("log_columns must be provided")
    
    # Initialize lists to store data
    datasets = []
    available_columns = []
    
    # Try to load images only if img_paths is provided
    rgb_img = None
    ct_img = None
    
    if img_paths is not None:
        if 'rgb' in img_paths:
            try:
                rgb_img = plt.imread(img_paths['rgb'])
                print(f"Loaded RGB image")
            except Exception as e:
                print(f"RGB image not available: {e}")
        
        if 'ct' in img_paths:
            try:
                ct_img = plt.imread(img_paths['ct'])
                print(f"Loaded CT image")
            except Exception as e:
                print(f"CT image not available: {e}")
    
    # Process each log column
    for log_column in log_columns:
        if log_column not in log_paths:
            print(f"Warning: No path defined for {log_column}. Skipping.")
            continue
            
        log_path = log_paths[log_column]
        
        # Try to load the data
        try:
            df = pd.read_csv(log_path)
            
            # Check if the required depth column exists
            if depth_column not in df.columns:
                print(f"Warning: Depth column {depth_column} not found in {log_path}. Skipping.")
                continue
                
            # Find the effective column name
            effective_column = log_column
            if log_column not in df.columns:
                # Try alternative column names if provided
                found = False
                if column_alternatives and log_column in column_alternatives:
                    for alt in column_alternatives[log_column]:
                        if alt in df.columns:
                            effective_column = alt
                            found = True
                            print(f"Using alternative column {alt} for {log_column}")
                            break
                
                if not found:
                    print(f"Warning: Log column {log_column} not found in {log_path}. Available columns: {list(df.columns)}. Skipping.")
                    continue
            
            # Extract data
            data = {}
            data['depth'] = np.array(df[depth_column])
            data[log_column] = np.array(df[effective_column])
            
            # Normalize the log data if requested
            if normalize:
                min_val = np.min(data[log_column])
                max_val = np.max(data[log_column])
                if max_val > min_val:  # Avoid division by zero
                    data[log_column] = (data[log_column] - min_val) / (max_val - min_val)
                else:
                    data[log_column] = np.zeros_like(data[log_column])
            
            # Add to datasets and available columns
            datasets.append(data)
            available_columns.append(log_column)
            
        except Exception as e:
            print(f"Error loading {log_path}: {e}")
    
    # If no datasets were loaded, return empty arrays
    if not datasets:
        print(f"No log datasets were loaded")
        return np.array([]), np.array([]), [], rgb_img, ct_img
    
    # Resample all datasets to a common depth scale
    resampled_data = resample_datasets(datasets)
    
    # Stack the selected columns
    log = np.column_stack([resampled_data[col] for col in available_columns])
    md = resampled_data['depth']
    
    # If only one log column was loaded, return as 1D array for backward compatibility
    if len(available_columns) == 1:
        log = log.flatten()
    
    return log, md, available_columns, rgb_img, ct_img

def resample_datasets(datasets, target_resolution_factor=2):
    """
    Resample multiple datasets to a common depth scale.
    
    This function takes multiple datasets with potentially different depth sampling
    and resamples them all to a common high-resolution depth scale using linear
    interpolation.
    
    Parameters
    ----------
    datasets : list of dict
        List of dictionaries, each containing 'depth' array and data arrays
    target_resolution_factor : float, default=2
        Factor to divide the lowest resolution by to create target resolution
    
    Returns
    -------
    dict
        Dictionary with resampled data arrays and common 'depth' scale
    
    Example
    -------
    >>> datasets = [
    ...     {'depth': np.array([0, 10, 20]), 'MS': np.array([0.1, 0.5, 0.9])},
    ...     {'depth': np.array([0, 5, 15, 20]), 'Lumin': np.array([0.2, 0.4, 0.6, 0.8])}
    ... ]
    >>> resampled = resample_datasets(datasets)
    """
    if not datasets:
        return {'depth': np.array([])}
        
    # Find depth ranges across all datasets
    min_depths = [np.min(dataset['depth']) for dataset in datasets]
    max_depths = [np.max(dataset['depth']) for dataset in datasets]
    start_depth = min(min_depths)
    end_depth = max(max_depths)
    
    # Calculate resolutions for each dataset
    def calculate_resolution(depth_array):
        return (depth_array[-1] - depth_array[0]) / len(depth_array)
    
    resolutions = [calculate_resolution(dataset['depth']) for dataset in datasets]
    lowest_resolution = max(resolutions)  # Largest value = lowest resolution
    
    # Create target depth array with improved resolution
    target_resolution = lowest_resolution / target_resolution_factor
    num_points = int((end_depth - start_depth) / target_resolution) + 1
    target_depth = np.linspace(start_depth, end_depth, num_points)
    
    # Resample all data to the target depth using linear interpolation
    resampled_data = {'depth': target_depth}
    
    for dataset in datasets:
        for key, values in dataset.items():
            if key != 'depth':  # Skip depth as we already have the target depth
                resampled_data[key] = np.interp(target_depth, dataset['depth'], values)
    
    return resampled_data


def load_age_constraints_from_csv(csv_file_path, data_columns, mute_mode=False):
    """
    Load age constraints from a single CSV file with configurable column mapping.
    
    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing age constraint data
    data_columns : dict
        Dictionary mapping standard column names to actual CSV column names.
        Expected keys: 'age', 'pos_error', 'neg_error', 'min_depth', 'max_depth', 
                      'in_sequence', 'core', 'interpreted_bed'
        This parameter is required and must be provided.
    
    Returns
    -------
    dict
        Dictionary containing all age constraint data with standardized structure
        
    Example
    -------
    >>> data_columns = {
    ...     'age': 'calib502_agebp',
    ...     'pos_error': 'calib502_2sigma_pos',
    ...     'neg_error': 'calib502_2sigma_neg',
    ...     'min_depth': 'mindepth_cm',
    ...     'max_depth': 'maxdepth_cm',
    ...     'in_sequence': 'in_sequence',
    ...     'core': 'core',
    ...     'interpreted_bed': 'interpreted_bed'
    ... }
    >>> age_data = load_age_constraints_from_csv('age_constraints.csv', data_columns)
    """
    # Check if data_columns is provided
    if data_columns is None:
        raise ValueError("data_columns parameter is required. "
                        "Provide a dictionary mapping: {'age': 'column_name', 'pos_error': 'column_name', "
                        "'neg_error': 'column_name', 'min_depth': 'column_name', 'max_depth': 'column_name', "
                        "'in_sequence': 'column_name', 'core': 'column_name', 'interpreted_bed': 'column_name'}")
    
    # Initialize result containers
    result = {
        'depths': [],
        'ages': [],
        'pos_errors': [],
        'neg_errors': [],
        'in_sequence_flags': [],
        'in_sequence_depths': [],
        'in_sequence_ages': [],
        'in_sequence_pos_errors': [],
        'in_sequence_neg_errors': [],
        'out_sequence_depths': [],
        'out_sequence_ages': [],
        'out_sequence_pos_errors': [],
        'out_sequence_neg_errors': [],
        'core': [],
        'interpreted_bed': []
    }
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Warning: CSV file not found: {csv_file_path}")
        return result
    
    # Load CSV data
    try:
        data = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return result
    
    # Check which columns are available and create a working dataset
    working_data = pd.DataFrame()
    available_mappings = {}
    
    for standard_col, csv_col in data_columns.items():
        if csv_col in data.columns:
            # Only keep rows where this column has data (except for interpreted_bed which is optional)
            if standard_col != 'interpreted_bed':
                valid_mask = data[csv_col].notna()
                if working_data.empty:
                    working_data = data[valid_mask].copy()
                else:
                    working_data = working_data[working_data[csv_col].notna()]
            else:
                # For interpreted_bed, don't filter rows - just mark as available
                if working_data.empty:
                    working_data = data.copy()
            available_mappings[standard_col] = csv_col
        else:
            print(f"Warning: Column '{csv_col}' not found in {csv_file_path}")
    
    if working_data.empty:
        print(f"Warning: No valid data found in {csv_file_path}")
        return result
    
    if not mute_mode:
        print(f"Loaded {len(working_data)} age constraints from {os.path.basename(csv_file_path)}")
    
    # Extract data with NA handling
    if 'min_depth' in available_mappings and 'max_depth' in available_mappings:
        result['depths'] = (working_data[available_mappings['min_depth']] + working_data[available_mappings['max_depth']]) / 2
    elif 'min_depth' in available_mappings:
        result['depths'] = working_data[available_mappings['min_depth']]
    else:
        result['depths'] = pd.Series([np.nan] * len(working_data))
    
    result['ages'] = working_data[available_mappings['age']].tolist() if 'age' in available_mappings else [np.nan] * len(working_data)
    result['pos_errors'] = working_data[available_mappings['pos_error']].tolist() if 'pos_error' in available_mappings else [np.nan] * len(working_data)
    result['neg_errors'] = working_data[available_mappings['neg_error']].tolist() if 'neg_error' in available_mappings else [np.nan] * len(working_data)
    result['in_sequence_flags'] = working_data[available_mappings['in_sequence']].tolist() if 'in_sequence' in available_mappings else [np.nan] * len(working_data)
    result['core'] = working_data[available_mappings['core']].tolist() if 'core' in available_mappings else ['Unknown'] * len(working_data)
    result['interpreted_bed'] = working_data[available_mappings['interpreted_bed']].fillna('Unknown').replace('', 'Unknown').tolist() if 'interpreted_bed' in available_mappings else ['Unknown'] * len(working_data)
    
    # Separate in-sequence and out-of-sequence constraints
    for i in range(len(result['in_sequence_flags'])):
        flag_value = result['in_sequence_flags'][i]
        
        # Check if flag indicates in-sequence (handle string, numeric, and boolean values)
        is_in_sequence = False
        if not pd.isna(flag_value):
            if isinstance(flag_value, str):
                is_in_sequence = flag_value.upper() == 'TRUE'
            elif isinstance(flag_value, (bool, np.bool_)):
                is_in_sequence = flag_value
            else:
                is_in_sequence = flag_value == 1
        
        if is_in_sequence:
            result['in_sequence_depths'].append(result['depths'].iloc[i] if isinstance(result['depths'], pd.Series) else result['depths'][i])
            result['in_sequence_ages'].append(result['ages'][i])
            result['in_sequence_pos_errors'].append(result['pos_errors'][i])
            result['in_sequence_neg_errors'].append(result['neg_errors'][i])
        else:
            result['out_sequence_depths'].append(result['depths'].iloc[i] if isinstance(result['depths'], pd.Series) else result['depths'][i])
            result['out_sequence_ages'].append(result['ages'][i])
            result['out_sequence_pos_errors'].append(result['pos_errors'][i])
            result['out_sequence_neg_errors'].append(result['neg_errors'][i])
    
    return result


def combine_age_constraints(age_constraint_list):
    """
    Combine multiple age constraint dictionaries into a single one.
    
    Parameters
    ----------
    age_constraint_list : list of dict
        List of age constraint dictionaries from load_age_constraints_from_csv
        
    Returns
    -------
    dict
        Combined age constraint dictionary
        
    Example
    -------
    >>> age_data1 = load_age_constraints_from_csv('core1_ages.csv', data_columns)
    >>> age_data2 = load_age_constraints_from_csv('core2_ages.csv', data_columns)
    >>> combined = combine_age_constraints([age_data1, age_data2])
    """
    if not age_constraint_list:
        return {
            'depths': [],
            'ages': [],
            'pos_errors': [],
            'neg_errors': [],
            'in_sequence_flags': [],
            'in_sequence_depths': [],
            'in_sequence_ages': [],
            'in_sequence_pos_errors': [],
            'in_sequence_neg_errors': [],
            'out_sequence_depths': [],
            'out_sequence_ages': [],
            'out_sequence_pos_errors': [],
            'out_sequence_neg_errors': [],
            'core': [],
            'interpreted_bed': []
        }
    
    # Initialize combined result with first dataset structure
    combined = {key: [] for key in age_constraint_list[0].keys()}
    
    # Combine all datasets
    all_data_rows = []
    for age_data in age_constraint_list:
        if len(age_data['depths']) > 0:
            # Convert depths to list if it's a pandas Series
            depths = age_data['depths'].tolist() if isinstance(age_data['depths'], pd.Series) else age_data['depths']
            
            for i in range(len(depths)):
                all_data_rows.append({
                    'depth': depths[i],
                    'age': age_data['ages'][i],
                    'pos_error': age_data['pos_errors'][i],
                    'neg_error': age_data['neg_errors'][i],
                    'in_sequence_flag': age_data['in_sequence_flags'][i],
                    'core': age_data['core'][i],
                    'interpreted_bed': age_data['interpreted_bed'][i]
                })
    
    # Sort combined data by depth
    all_data_rows.sort(key=lambda x: x['depth'] if not pd.isna(x['depth']) else float('inf'))
    
    # Extract combined data
    if all_data_rows:
        combined['depths'] = [row['depth'] for row in all_data_rows]
        combined['ages'] = [row['age'] for row in all_data_rows]
        combined['pos_errors'] = [row['pos_error'] for row in all_data_rows]
        combined['neg_errors'] = [row['neg_error'] for row in all_data_rows]
        combined['in_sequence_flags'] = [row['in_sequence_flag'] for row in all_data_rows]
        combined['core'] = [row['core'] for row in all_data_rows]
        combined['interpreted_bed'] = [row['interpreted_bed'] for row in all_data_rows]
        
        # Separate in-sequence and out-of-sequence constraints
        for i, in_seq_flag in enumerate(combined['in_sequence_flags']):
            # Check if flag indicates in-sequence (handle string, numeric, and boolean values)
            is_in_sequence = False
            if not pd.isna(in_seq_flag):
                if isinstance(in_seq_flag, str):
                    is_in_sequence = in_seq_flag.upper() == 'TRUE'
                elif isinstance(in_seq_flag, (bool, np.bool_)):
                    is_in_sequence = in_seq_flag
                else:
                    is_in_sequence = in_seq_flag == 1
            
            if is_in_sequence:
                combined['in_sequence_depths'].append(combined['depths'][i])
                combined['in_sequence_ages'].append(combined['ages'][i])
                combined['in_sequence_pos_errors'].append(combined['pos_errors'][i])
                combined['in_sequence_neg_errors'].append(combined['neg_errors'][i])
            else:
                combined['out_sequence_depths'].append(combined['depths'][i])
                combined['out_sequence_ages'].append(combined['ages'][i])
                combined['out_sequence_pos_errors'].append(combined['pos_errors'][i])
                combined['out_sequence_neg_errors'].append(combined['neg_errors'][i])
    
    return combined


def load_core_age_constraints(core_name, age_base_path, consider_adjacent_core=False, data_columns=None, mute_mode=False):
    """
    Helper function to load age constraints for a single core.
    
    Parameters
    ----------
    core_name : str
        Name of the core to load age constraints for
    age_base_path : str
        Base directory path containing age constraint CSV files
    consider_adjacent_core : bool, default=False
        Whether to also load age constraints from similar core names (e.g., adjacent cores)
    data_columns : dict
        Dictionary mapping standard column names to actual CSV column names.
        Expected keys: 'age', 'pos_error', 'neg_error', 'min_depth', 'max_depth', 
                      'in_sequence', 'core', 'interpreted_bed'
    mute_mode : bool, default=False
        If True, suppress all print statements
        
    Returns
    -------
    dict
        Age constraint dictionary for the specified core
        
    Example
    -------
    >>> data_columns = {
    ...     'age': 'calib502_agebp',
    ...     'pos_error': 'calib502_2sigma_pos',
    ...     'neg_error': 'calib502_2sigma_neg',
    ...     'min_depth': 'mindepth_cm',
    ...     'max_depth': 'maxdepth_cm',
    ...     'in_sequence': 'in_sequence',
    ...     'core': 'core',
    ...     'interpreted_bed': 'interpreted_bed'
    ... }
    >>> age_data = load_core_age_constraints('M9907-23PC', '/path/to/age/data', 
    ...                                      consider_adjacent_core=True, 
    ...                                      data_columns=data_columns)
    """
    csv_files = []
    
    if not os.path.exists(age_base_path):
        if not mute_mode:
            print(f"Warning: Directory not found: {age_base_path}")
        return combine_age_constraints([])
    
    # First, find CSV files that contain the whole core name
    for file in os.listdir(age_base_path):
        if file.endswith('.csv') and core_name in file:
            csv_files.append(f'{age_base_path}/{file}')
    
    # If consider_adjacent_core is True, also search for similar core names
    if consider_adjacent_core:
        core_base = core_name[:-2]  # Remove last two characters
        for file in os.listdir(age_base_path):
            if file.endswith('.csv') and file.startswith(f'{core_base}') and core_name not in file:
                csv_files.append(f'{age_base_path}/{file}')
    
    # Load all CSV files
    age_constraints = []
    for csv_file in csv_files:
        age_data = load_age_constraints_from_csv(csv_file, data_columns, mute_mode=mute_mode)
        if len(age_data['depths']) > 0:
            age_constraints.append(age_data)
    
    # Return single dataset or combine multiple datasets
    if len(age_constraints) == 1:
        return age_constraints[0]
    elif len(age_constraints) > 1:
        return combine_age_constraints(age_constraints)
    else:
        # Return empty structure if no data found
        return combine_age_constraints([])


# The plot_core_data function has been moved to pyCoreRelator.visualization.core_plots


def reconstruct_raw_data_from_histogram(bins, hist_percentages, n_points):
    """Reconstruct raw data points from histogram bins and percentages
    
    Note: hist_percentages may not be true percentages due to legacy normalization issues.
    This function handles both true percentages and raw histogram counts.
    """
    raw_data = []
    
    # Handle incorrectly normalized histogram data from legacy CSV files
    # First, normalize the histogram values so they represent proper proportions
    hist_sum = np.sum(hist_percentages)
    if hist_sum > 0:
        # If the sum is approximately 100, treat as percentages; otherwise renormalize
        if 90 <= hist_sum <= 110:  # Allow some tolerance for true percentages
            raw_counts = (hist_percentages * n_points) / 100
        else:
            # Renormalize incorrectly scaled histogram data
            raw_counts = (hist_percentages / hist_sum) * n_points
    else:
        raw_counts = np.zeros_like(hist_percentages)
    
    # Generate data points for each bin
    for i, count in enumerate(raw_counts):
        if count > 0:
            n_samples = int(round(count))
            if n_samples > 0:
                # Sample uniformly within the bin
                bin_samples = np.random.uniform(bins[i], bins[i+1], n_samples)
                raw_data.extend(bin_samples)
    
    return np.array(raw_data)


def load_pickeddepth_ages_from_csv(pickeddepth_age_csv):
    """
    Load pickeddepth ages data from CSV file.
    
    This function loads age-depth model data for picked depth intervals from a CSV file.
    The data includes depths, ages, and associated uncertainties for correlation analysis.
    
    Parameters
    ----------
    pickeddepth_age_csv : str
        Full path to the CSV file containing pickeddepth ages data
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'depths': list of depth values
        - 'ages': list of age values
        - 'pos_uncertainties': list of positive uncertainty values
        - 'neg_uncertainties': list of negative uncertainty values
    
    Example
    -------
    >>> pickeddepth_ages = load_pickeddepth_ages_from_csv('path/to/M9907-23PC_pickeddepth_ages.csv')
    >>> print(f"Loaded {len(pickeddepth_ages['depths'])} depth-age pairs")
    """
    # Check if file exists
    if not os.path.exists(pickeddepth_age_csv):
        raise FileNotFoundError(f"Pickeddepth ages file not found: {pickeddepth_age_csv}")
    
    try:
        # Load CSV file
        df = pd.read_csv(pickeddepth_age_csv)
        
        # Convert to dictionary format matching the expected structure
        # Use the actual column names from the CSV file or column indices
        pickeddepth_ages = {
            'depths': df.iloc[:, 0].tolist(),  # First column
            'ages': df.iloc[:, 1].tolist(),    # Second column
            'pos_uncertainties': df.iloc[:, 2].tolist(),  # Third column
            'neg_uncertainties': df.iloc[:, 3].tolist()   # Fourth column
        }
        
        print(f"Loaded {len(pickeddepth_ages['depths'])} pickeddepth ages from {os.path.basename(pickeddepth_age_csv)}")
        
        return pickeddepth_ages
        
    except Exception as e:
        raise Exception(f"Error loading pickeddepth ages from {pickeddepth_age_csv}: {e}")


def load_and_prepare_quality_data(target_quality_indices, master_csv_filenames, synthetic_csv_filenames, 
                                  CORE_A, CORE_B, debug=True):
    """
    Load and prepare quality data from master and synthetic CSV files for plotting.
    
    Brief summary: This function loads quality index data from CSV files, applies filters,
    and prepares the data for visualization using the stored histogram and PDF data directly
    from CSV files (similar to plot_correlation_distribution approach).
    
    Parameters:
    -----------
    target_quality_indices : list
        List of quality indices to process (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    master_csv_filenames : dict
        Dictionary mapping quality_index to master CSV filename paths
    synthetic_csv_filenames : dict  
        Dictionary mapping quality_index to synthetic CSV filename paths
    CORE_A : str
        Name of core A
    CORE_B : str
        Name of core B
    debug : bool, default True
        If True, only print essential messages. If False, print all detailed messages.
    
    Returns:
    --------
    dict
        Dictionary with quality_index as keys and data dictionaries as values.
        Each data dictionary contains:
        - 'df_all_params': filtered master data DataFrame
        - 'combined_data': synthetic histogram data (for display only)
        - 'fitted_mean': fitted normal distribution mean from synthetic data
        - 'fitted_std': fitted normal distribution standard deviation from synthetic data
        - 'max_core_a_constraints': maximum core A constraints count
        - 'max_core_b_constraints': maximum core B constraints count
        - 'unique_combinations': unique parameter combinations
    """
    
    # Define which categories to load - you can customize these filters
    load_filters = {
        'age_consideration': [True, False],
        'restricted_age_correlation': [True, False],
        'shortest_path_search': [True]
    }
    
    quality_data = {}
    
    # Loop through all quality indices
    for quality_index in target_quality_indices:
        
        # Load all fit_params from master CSV
        master_csv_filename = master_csv_filenames[quality_index]
        
        # Check if master CSV exists
        if not os.path.exists(master_csv_filename):
            if debug:
                print(f"✗ Error: Master CSV file not found: {master_csv_filename}")
            else:
                print(f"✗ Error: Master CSV file not found: {master_csv_filename}")
                print(f"   Skipping {quality_index} and moving to next index...")
            continue
        
        try:
            df_all_params = pd.read_csv(master_csv_filename)
            if not debug:
                print(f"✓ Loaded master CSV: {master_csv_filename}")
        except Exception as e:
            if debug:
                print(f"✗ Error loading master CSV {master_csv_filename}: {str(e)}")
            else:
                print(f"✗ Error loading master CSV {master_csv_filename}: {str(e)}")
                print(f"   Skipping {quality_index} and moving to next index...")
            continue

        # Apply filters with NaN handling
        mask = pd.Series([True] * len(df_all_params))
        for column, values in load_filters.items():
            if values is not None:
                if None in values:
                    mask &= (df_all_params[column].isin([v for v in values if v is not None]) | 
                            df_all_params[column].isna())
                else:
                    mask &= df_all_params[column].isin(values)

        df_all_params = df_all_params[mask]
        if not debug:
            print(f"Loaded {len(df_all_params)} rows after filtering")

        # Load synthetic fit params from CSV for background
        synthetic_csv_filename = synthetic_csv_filenames[quality_index]
        
        # Check if synthetic CSV exists
        if not os.path.exists(synthetic_csv_filename):
            if debug:
                print(f"✗ Error: Synthetic CSV file not found: {synthetic_csv_filename}")
            else:
                print(f"✗ Error: Synthetic CSV file not found: {synthetic_csv_filename}")
                print(f"   Skipping {quality_index} and moving to next index...")
            continue
        
        try:
            df_synthetic_params = pd.read_csv(synthetic_csv_filename)
            if not debug:
                print(f"✓ Loaded synthetic CSV: {synthetic_csv_filename}")
        except Exception as e:
            if debug:
                print(f"✗ Error loading synthetic CSV {synthetic_csv_filename}: {str(e)}")
            else:
                print(f"✗ Error loading synthetic CSV {synthetic_csv_filename}: {str(e)}")
                print(f"   Skipping {quality_index} and moving to next index...")
            continue

        # Reconstruct synthetic (null hypothesis) data from all synthetic rows
        # This matches the approach in calculate_quality_comparison_t_statistics
        combined_data = []
        fitted_mean = None
        fitted_std = None
        
        for _, row in df_synthetic_params.iterrows():
            try:
                bins = np.fromstring(row['bins'].strip('[]'), sep=' ')
                hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ')
                n_points = row['n_points']
                raw_data_points = reconstruct_raw_data_from_histogram(bins, hist_percentages, n_points)
                combined_data.extend(raw_data_points)
                
                # Store fitted parameters from first row for display purposes
                if fitted_mean is None:
                    fitted_mean = row['mean'] if 'mean' in row and pd.notna(row['mean']) else 0.5
                    fitted_std = row['std'] if 'std' in row and pd.notna(row['std']) else 0.1
            except:
                continue
        
        combined_data = np.array(combined_data)
        if len(combined_data) == 0:
            # Fallback: create synthetic data based on fitted parameters
            if fitted_mean is None:
                fitted_mean = 0.5
                fitted_std = 0.1
            combined_data = np.random.normal(fitted_mean, fitted_std, 10000)

        # Get unique combinations available in the CSV
        unique_combinations = df_all_params.drop_duplicates(
            subset=['age_consideration', 'restricted_age_correlation', 'shortest_path_search']
        )[['age_consideration', 'restricted_age_correlation', 'shortest_path_search']].to_dict('records')

        if not debug:
            print(f"Found {len(unique_combinations)} unique parameter combinations in CSV:")
            for combo in unique_combinations:
                print(f"  {combo}")

        # Find max constraint counts for determining which curves to highlight
        max_core_a_constraints = df_all_params['core_a_constraints_count'].max()
        max_core_b_constraints = df_all_params['core_b_constraints_count'].max()
        
        if not debug:
            print(f"Number of the age constraints available in {CORE_A}: {max_core_a_constraints}")
            print(f"Number of the age constraints available in {CORE_B}: {max_core_b_constraints}")
        
        # Store processed data for this quality index
        quality_data[quality_index] = {
            'df_all_params': df_all_params,
            'combined_data': combined_data,
            'fitted_mean': fitted_mean,
            'fitted_std': fitted_std,
            'max_core_a_constraints': max_core_a_constraints,
            'max_core_b_constraints': max_core_b_constraints,
            'unique_combinations': unique_combinations
        }
    
    return quality_data