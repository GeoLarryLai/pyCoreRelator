"""
Data loading functions for pyCoreRelator.

Included Functions:
- load_log_data: Load log data from CSV files and resample to common depth scale
- resample_datasets: Resample multiple datasets to a common depth scale

This module provides utilities for loading core log data from CSV files.
It handles multiple log types, data normalization, and resampling to common depth scales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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


# The plot_core_data function has been moved to pyCoreRelator.visualization.core_plots